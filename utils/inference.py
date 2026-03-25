import tensorflow as tf
import os
import soundfile as sf
from modules.layers_wrapper import TFLiteWrapper, ONNXWrapper
from tqdm import tqdm

class AudioInference:
    def __init__(
        self,
        model_path,
        sample_rate=44100,
        segment_length=88064,
        overlap=0.25,
        use_wiener=False,
        stft_frame_length=2048,
        stft_frame_step=512,
        wiener_iterations=1,
    ):

        self.format = model_path.split(".")[-1].lower()
        if self.format == "tflite":
            self.model = TFLiteWrapper(model_path)
        elif self.format == "onnx":
            self.model = ONNXWrapper(model_path)
        elif self.format in ["h5", "keras"]:
            self.format = "keras"
            self.model = tf.keras.models.load_model(model_path, compile=False)
        else:
            raise ValueError("Unsupported model format")

        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.overlap = overlap
        self.hop = int(round(segment_length * (1 - overlap)))
        self.use_wiener = use_wiener
        self.stft_frame_length = stft_frame_length
        self.stft_frame_step = stft_frame_step
        self.wiener_iterations = wiener_iterations
        self.window = tf.signal.hann_window(segment_length, periodic=True, dtype=tf.float32)

    def _predict_batch(self, batch):
        if self.format == "keras":
            return self.model(batch, training=False)
        return self.model.predict_batch(batch)

    def _predict_stream(self, audio):
        n = audio.shape[0]
        n_channels = audio.shape[-1]

        if n < self.segment_length:
            pad = self.segment_length - n
            audio = tf.pad(audio, [[0, pad], [0, 0]])
            n = tf.shape(audio)[0]

        # n-segments
        n_segments = (n - 1) // self.hop + 1
        total_len = self.hop * (n_segments - 1) + self.segment_length
        pad_total = total_len - n
        audio_pad = tf.pad(audio, [[0, pad_total], [0, 0]])

        # Output buffer
        output = tf.zeros([total_len, 4, 2], dtype=tf.float32)
        window_sum = tf.zeros([total_len], dtype=tf.float32)

        for i in tqdm(range(n_segments), desc="Inference Progress"):
            start = i * self.hop
            end = start + self.segment_length
            frame_len = end - start

            frame = audio_pad[start:end] * tf.reshape(self.window[:frame_len], [-1, 1])
            frame = tf.expand_dims(frame, axis=0)
            pred = self._predict_batch(frame)[0]  # [frame_len, 8]
            pred = tf.reshape(pred, [frame_len, 4, 2])
            pred = pred * tf.reshape(self.window[:frame_len], [-1, 1, 1])

            output = tf.tensor_scatter_nd_add(
                output,
                tf.reshape(tf.range(start, end), [-1, 1]),
                pred
            )
            window_sum = tf.tensor_scatter_nd_add(
                window_sum,
                tf.reshape(tf.range(start, end), [-1, 1]),
                self.window[:frame_len]
            )

        output = output / tf.reshape(tf.maximum(window_sum, 1e-8), [-1, 1, 1])
        return output[:n]

    @tf.function
    def _wiener_filter_tf(self, stems, mixture):
        eps = 1e-10
        stems = tf.cast(stems, tf.float32)
        mixture = tf.cast(mixture, tf.float32)
        n_samples = tf.shape(mixture)[0]
        pad = self.stft_frame_length

        mixture_pad = tf.pad(mixture, [[pad, pad], [0, 0]])
        stems_pad = tf.pad(stems, [[pad, pad], [0, 0], [0, 0]])

        # STFT mixture
        stft_mix = tf.signal.stft(tf.transpose(mixture_pad, [1, 0]),
                                  frame_length=self.stft_frame_length,
                                  frame_step=self.stft_frame_step,
                                  window_fn=tf.signal.hann_window)
        mix_mag = tf.abs(stft_mix)
        expanded_mix_mag = tf.expand_dims(mix_mag, axis=-1)
        expanded_mix_mag = tf.transpose(expanded_mix_mag, [1, 2, 3, 0])
        mix_phase = stft_mix / tf.cast(tf.maximum(mix_mag, eps), tf.complex64)
        mix_phase = tf.transpose(mix_phase, [1, 2, 0])
        mix_phase = tf.expand_dims(mix_phase, axis=2)

        # STFT stems
        stems_flat = tf.transpose(stems_pad, [1, 2, 0])
        stems_flat = tf.reshape(stems_flat, [8, -1])
        stft_stems = tf.signal.stft(stems_flat,
                                    frame_length=self.stft_frame_length,
                                    frame_step=self.stft_frame_step,
                                    window_fn=tf.signal.hann_window)
        stft_stems_mag = tf.abs(stft_stems)
        frames = tf.shape(stft_stems_mag)[1]
        fftbins = tf.shape(stft_stems_mag)[2]
        stft_stems_mag = tf.reshape(stft_stems_mag, [4, 2, frames, fftbins])
        stft_stems_mag = tf.transpose(stft_stems_mag, [2, 3, 0, 1])
        source_mag = tf.maximum(stft_stems_mag, eps)

        # Wiener iterations
        def cond(i, sm): return i < self.wiener_iterations
        def body(i, sm):
            power = tf.square(sm)
            total_power = tf.reduce_sum(power, axis=2, keepdims=True) + eps
            gain = power / total_power
            sm = gain * expanded_mix_mag
            return i + 1, sm

        _, source_mag = tf.while_loop(cond, body, [0, source_mag])
        source_est = tf.cast(source_mag, tf.complex64) * mix_phase

        # ISTFT
        source_est = tf.transpose(source_est, [2, 3, 0, 1])
        source_est = tf.reshape(source_est, [8, frames, fftbins])
        time_sources = tf.signal.inverse_stft(source_est,
                                              frame_length=self.stft_frame_length,
                                              frame_step=self.stft_frame_step,
                                              window_fn=tf.signal.hann_window)
        time_sources = tf.transpose(time_sources, [1, 0])
        time_sources = time_sources[pad:pad + n_samples]
        return tf.reshape(time_sources, [n_samples, 4, 2])

    def predict(self, audio, export=False, export_dir="stems"):
        audio = tf.cast(audio, tf.float32)
        if len(audio.shape) == 1:
            audio = tf.expand_dims(audio, -1)
        if audio.shape[-1] == 1:
            audio = tf.repeat(audio, 2, axis=-1)

        stems = self._predict_stream(audio)
        if self.use_wiener:
            stems = self._wiener_filter_tf(stems, audio)

        if export:
            os.makedirs(export_dir, exist_ok=True)
            estimates = {
                "vocals": stems[:, 0, :].numpy(),
                "drums": stems[:, 1, :].numpy(),
                "bass": stems[:, 2, :].numpy(),
                "other": stems[:, 3, :].numpy(),
            }
            for name, data in estimates.items():
                out_path = os.path.join(export_dir, f"{name}.wav")
                sf.write(out_path, data, self.sample_rate)
            print("Export Wav Complete")
        return stems