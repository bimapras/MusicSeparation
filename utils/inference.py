import tensorflow as tf
import os
from tqdm import tqdm
import soundfile as sf
from modules.layers_wrapper import TFLiteWrapper

class AudioInference:
    def __init__(
        self,
        model,
        sample_rate=44100,
        segment_length=88064,
        overlap=0.5,
        batch_size=4,
        use_wiener=False,
        stft_frame_length=2048,
        stft_frame_step=512,
        wiener_iterations=3,
    ):
        self.model = TFLiteWrapper(model)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.overlap = overlap
        self.batch_size = batch_size
        self.use_wiener = use_wiener
        self.stft_frame_length = stft_frame_length
        self.stft_frame_step = stft_frame_step
        self.wiener_iterations = wiener_iterations

    @tf.function
    def _wiener_filter_tf(self, stems, mixture):
        eps = 1e-10
        stems = tf.cast(stems, tf.float32)
        mixture = tf.cast(mixture, tf.float32)

        mix_T = tf.transpose(mixture, [1, 0])
        stft_mix = tf.signal.stft(
            mix_T, self.stft_frame_length, self.stft_frame_step,
            window_fn=tf.signal.hann_window
        )
        stft_mix = tf.transpose(stft_mix, [1, 2, 0])

        stems_perm = tf.transpose(stems, [1, 2, 0])
        stems_flat = tf.reshape(stems_perm, [8, -1])
        stft_stems = tf.signal.stft(
            stems_flat, self.stft_frame_length, self.stft_frame_step,
            window_fn=tf.signal.hann_window
        )
        frames = tf.shape(stft_stems)[1]
        fftbins = tf.shape(stft_stems)[2]
        stft_stems = tf.reshape(stft_stems, [4, 2, frames, fftbins])
        stft_stems = tf.transpose(stft_stems, [2, 3, 0, 1])

        source_mag = tf.maximum(tf.abs(stft_stems), eps)
        Y = stft_mix
        Y_mag = tf.abs(Y)
        denom = tf.cast(tf.maximum(Y_mag, eps), tf.complex64)
        Y_phase = Y / denom

        def cond(i, sm):
            return i < self.wiener_iterations

        def body(i, sm):
            power = tf.square(sm)
            total_power = tf.reduce_sum(power, axis=2, keepdims=True) + eps
            gain = power / total_power
            sm = gain * tf.expand_dims(Y_mag, axis=2)
            return i + 1, sm

        _, source_mag = tf.while_loop(cond, body, [0, source_mag])

        source_est = tf.cast(source_mag, tf.complex64) * tf.expand_dims(Y_phase, axis=2)
        source_est = tf.transpose(source_est, [2, 3, 0, 1])
        source_est = tf.reshape(source_est, [8, frames, fftbins])

        time_sources = tf.signal.inverse_stft(
            source_est, self.stft_frame_length, self.stft_frame_step,
            window_fn=tf.signal.hann_window
        )
        time_sources = tf.transpose(time_sources, [1, 0])
        return tf.reshape(time_sources, [tf.shape(time_sources)[0], 4, 2])

    def _predict_full_batch_tflite(self, audio):
        hop = int(round(self.segment_length * (1.0 - self.overlap)))
        n = audio.shape[0]
        n_segments = int(tf.math.ceil((n - self.segment_length) / hop)) + 1
        total_len = (n_segments - 1) * hop + self.segment_length
        pad = total_len - n
        audio_padded = tf.pad(audio, [[0, pad], [0, 0]])

        left = tf.signal.frame(audio_padded[:, 0], self.segment_length, hop)
        right = tf.signal.frame(audio_padded[:, 1], self.segment_length, hop)
        frames = tf.stack([left, right], axis=-1)
        window = tf.signal.hann_window(self.segment_length, periodic=True, dtype=tf.float32)
        frames_windowed = frames * tf.reshape(window, [1, self.segment_length, 1])

        preds_list = []
        for start_idx in tqdm(range(0, n_segments, self.batch_size), desc="Inference Progress"):
            end_idx = min(start_idx + self.batch_size, n_segments)
            batch = frames_windowed[start_idx:end_idx]
            preds = self.model.predict_batch(batch)
            preds_list.append(preds)

        predictions = tf.concat(preds_list, axis=0)[:n_segments]
        preds_reshaped = tf.reshape(predictions, [n_segments, self.segment_length, 4, 2])

        stems_reshaped = tf.reshape(preds_reshaped, [n_segments, self.segment_length, 8])
        stems_T = tf.transpose(stems_reshaped, [2, 0, 1])
        recon = tf.map_fn(lambda ch: tf.signal.overlap_and_add(ch, hop), stems_T)
        recon = tf.transpose(recon, [1, 0])[:n]
        stems_out = tf.reshape(recon, [n, 4, 2])

        if self.use_wiener:
            stems_out = self._wiener_filter_tf(stems_out, audio)

        return stems_out

    def predict(self, audio, segment_length_sec=None, export=False, export_dir="stems"):
        audio = tf.cast(audio, tf.float32)
        if segment_length_sec is not None and segment_length_sec > 0:
            segment_len_samples = int(segment_length_sec * self.sample_rate)
            n = audio.shape[0]
            outputs = []
            for start in range(0, n, segment_len_samples):
                end = min(start + segment_len_samples, n)
                seg = audio[start:end]
                out = self._predict_full_batch_tflite(seg)
                outputs.append(out)
            pred = tf.concat(outputs, axis=0)
        else:
            pred = self._predict_full_batch_tflite(audio)

        if export:
            os.makedirs(export_dir, exist_ok=True)
            estimates = {
                'vocals': pred[:, 0, :].numpy(),
                'drums':  pred[:, 1, :].numpy(),
                'bass':   pred[:, 2, :].numpy(),
                'other':  pred[:, 3, :].numpy(),
            }
            for name, data in estimates.items():
                out_path = os.path.join(export_dir, f"{name}.wav")
                sf.write(out_path, data, self.sample_rate)

            print('Export Wav Complete')
        return pred