import librosa
import os
import tensorflow as tf
import numpy as np
import soundfile as sf

class CachedStemMixTFDataset:
    def __init__(self, all_paths_tensor, stems=('vocals', 'drums', 'bass', 'other'), swap_prob=0.0):
        self.all_paths_tensor = all_paths_tensor
        self.stems = stems
        self.swap_prob = swap_prob
        self.num_stems = len(stems)
        self.num_segments = tf.shape(all_paths_tensor)[0]

    def get_tf_dataset(self, batch_size=16, shuffle=True, augment=False):
        ds = tf.data.Dataset.from_tensor_slices(tf.range(self.num_segments))

        if shuffle:
            ds = ds.shuffle(buffer_size=1000)

        def load_sample(index):
            original_paths = self.all_paths_tensor[index]  # shape: (num_stems,)

            def maybe_swap(i):
                def do_swap():
                    rand_index = tf.random.uniform([], minval=0, maxval=self.num_segments, dtype=tf.int32)
                    return tf.cond(
                        tf.not_equal(rand_index, index),
                        lambda: self.all_paths_tensor[rand_index, i],
                        lambda: original_paths[i]
                    )
                return tf.cond(
                    tf.random.uniform([]) < self.swap_prob,
                    do_swap,
                    lambda: original_paths[i]
                )

            swapped_paths = tf.stack([maybe_swap(i) for i in range(self.num_stems)])

            # Load and decode each audio file
            def load_audio(path):
                audio_bytes = tf.io.read_file(path)
                audio, _ = tf.audio.decode_wav(audio_bytes, desired_channels=2)
                return audio  # shape: (T, 2)

            stem_waveforms = tf.map_fn(load_audio, swapped_paths, dtype=tf.float32)

            # Find max length and pad
            lengths = tf.map_fn(lambda x: tf.shape(x)[0], stem_waveforms, dtype=tf.int32)
            max_len = tf.reduce_max(lengths)

            def pad_to_max(wav):
                pad_amt = max_len - tf.shape(wav)[0]
                return tf.pad(wav, [[0, pad_amt], [0, 0]])

            stem_waveforms = tf.map_fn(pad_to_max, stem_waveforms)

            if augment:
                stem_waveforms, mixture = self.apply_augmentations(stem_waveforms)
            else:
                mixture = tf.reduce_sum(stem_waveforms, axis=0)
                mixture = tf.clip_by_value(mixture, -1.0, 1.0)

            # Target: (T, num_stems * 2)
            target = tf.transpose(stem_waveforms, perm=[1, 0, 2])
            target = tf.reshape(target, [max_len, self.num_stems * 2])

            return mixture, target

        ds = ds.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds

    def apply_augmentations(self, stems):
        def random_gain(audio):
            gain_db = tf.random.uniform([], -6.0, 6.0)
            gain = tf.pow(10.0, gain_db / 20.0)
            return audio * gain

        def fade_in_out(audio):
            T = tf.shape(audio)[0]
            fade_len = tf.minimum(4410, T // 10)  # 100ms or 10% of audio length
            fade_in = tf.linspace(0.0, 1.0, fade_len)
            fade_out = tf.linspace(1.0, 0.0, fade_len)
            ones = tf.ones([T - 2 * fade_len])
            envelope = tf.concat([fade_in, ones, fade_out], axis=0)
            envelope = tf.expand_dims(envelope, axis=-1)
            return audio * envelope

        def reverse(audio):
            return tf.reverse(audio, axis=[0])

        # Augment each stem independently
        def augment_stem(stem):
            if tf.random.uniform([]) < 0.5:
                stem = random_gain(stem)
            if tf.random.uniform([]) < 0.5:
                stem = fade_in_out(stem)
            if tf.random.uniform([]) < 0.3:
                stem = reverse(stem)
            return stem

        stems = tf.map_fn(augment_stem, stems)

        # Sum stems to create mixture
        mixture = tf.reduce_sum(stems, axis=0)

        # Add noise only to mixture
        if tf.random.uniform([]) < 0.5:
            noise = tf.random.normal(tf.shape(mixture), mean=0.0, stddev=0.005)
            mixture += noise

        # Clip to [-1, 1]
        stems = tf.clip_by_value(stems, -1.0, 1.0)
        mixture = tf.clip_by_value(mixture, -1.0, 1.0)

        return stems, mixture

def is_corrupt_or_invalid(stem):
    """
    Returns True if the stem is structurally corrupt (NaNs, wrong shape, etc.)
    """
    return (
        np.isnan(stem).any() or
        np.isinf(stem).any() or
        stem.ndim != 2 or
        stem.shape[0] != 2
    )

def is_noise_only(stem, sr, flat_thresh=0.8, energy_thresh=1e-5):
    """
    Returns True if the stem is just noise (low energy or very flat)
    """
    mono = np.mean(stem, axis=0)
    energy = np.mean(mono ** 2)
    if energy < energy_thresh:
        return True

    flatness = librosa.feature.spectral_flatness(y=mono)[0]
    avg_flatness = np.mean(flatness)
    return avg_flatness > flat_thresh

def segment_track(list_track, chunk, filename, sr, n_song, flat_thresh=0.8, energy_thresh=1e-5):
    len_track = list_track[0].shape[1]
    chunk_samples = chunk

    for stem_name in ['drums', 'bass', 'other', 'vocals']:
        os.makedirs(os.path.join(filename, stem_name), exist_ok=True)

    count = 0
    for i in range(0, len_track - chunk_samples + 1, chunk_samples):
        segment = [track[:, i:i+chunk_samples] for track in list_track]

        if segment[0].shape[1] < chunk_samples:
            pad_width = ((0, 0), (0, chunk_samples - segment[0].shape[1]))
            segment = [np.pad(stem, pad_width, mode='constant') for stem in segment]

        # Check for corruption — if any stem is corrupt, skip
        if any(is_corrupt_or_invalid(stem) for stem in segment):
            continue

        # Silence stems that are noisy (but not corrupt)
        cleaned_segment = []
        silent_count = 0
        for stem in segment:
            if is_noise_only(stem, sr, flat_thresh, energy_thresh):
                cleaned_segment.append(np.zeros_like(stem))  # silence this stem
                silent_count += 1
            else:
                cleaned_segment.append(stem)
        
        # Skip the segment if ALL stems are noise
        if silent_count >= 3:
            continue

        stem_names = ['drums', 'bass', 'other', 'vocals']
        for stem_name, audio in zip(stem_names, cleaned_segment):
            sf.write(
                os.path.join(filename, stem_name, f'{n_song}_{count+1:04d}.wav'),
                audio.T,
            )
        count += 1

    print(f'{filename} Complete')

def create_wav(subset, data, sr, window_size):
    song_count = 1
    for file in data:
        directory = file.name
        path = os.path.join(subset, directory)
        os.makedirs(path, exist_ok=True)
        _, drums, bass, other, vocals = file.stems
        
        drums, bass, other, vocals = drums.T, bass.T, other.T, vocals.T  # shape: (channels, samples)

        segment_track(
            list_track=[
                drums,
                bass,
                other,
                vocals
            ],
            chunk=window_size,
            filename=path,
            sr=sr,
            n_song=song_count
        )
        song_count += 1
    return print(f'{subset} subset complete process {song_count - 1}')

def list_stem_files(data_root, stem_names=['vocals', 'drums', 'bass', 'other']):
    song_dirs = [os.path.join(data_root, d) for d in tf.io.gfile.listdir(data_root)]
    all_paths = []

    for song in song_dirs:
        if not tf.io.gfile.isdir(song):
            continue

        stem_file_lists = {}
        valid = True
        for stem in stem_names:
            stem_dir = os.path.join(song, stem)
            if not tf.io.gfile.exists(stem_dir):
                valid = False
                break
            files = sorted(tf.io.gfile.listdir(stem_dir))
            stem_file_lists[stem] = [os.path.join(stem_dir, f) for f in files]

        if not valid:
            continue

        n_chunks = len(stem_file_lists[stem_names[0]])
        for i in range(n_chunks):
            chunk_paths = [stem_file_lists[stem][i] for stem in stem_names]
            all_paths.append(chunk_paths)

    return all_paths