import tensorflow as tf
from layers_wrapper import TFLiteWrapper

# Wiener filter
@tf.function
def wiener_filter_tf(stems, mixture, frame_length, frame_step, iterations=1):
    eps = 1e-10
    stems = tf.cast(stems, tf.float32)
    mixture = tf.cast(mixture, tf.float32)

    mix_T = tf.transpose(mixture, [1, 0])
    stft_mix = tf.signal.stft(mix_T, frame_length=frame_length, frame_step=frame_step,
                              window_fn=tf.signal.hann_window)
    stft_mix = tf.transpose(stft_mix, [1, 2, 0])

    stems_perm = tf.transpose(stems, [1, 2, 0])
    stems_flat = tf.reshape(stems_perm, [8, -1])
    stft_stems = tf.signal.stft(stems_flat, frame_length=frame_length, frame_step=frame_step,
                                window_fn=tf.signal.hann_window)
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
        return i < iterations

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

    time_sources = tf.signal.inverse_stft(source_est, frame_length=frame_length, frame_step=frame_step,
                                          window_fn=tf.signal.hann_window)
    time_sources = tf.transpose(time_sources, [1, 0])
    return tf.reshape(time_sources, [tf.shape(time_sources)[0], 4, 2])


# _predict_full_batch Keras model (tf.function)
@tf.function
def _predict_full_batch_tf(
    audio,
    model,
    segment_length,
    overlap,
    batch_size,
    use_wiener,
    stft_frame_length,
    stft_frame_step,
    wiener_iterations,
):
    audio = tf.cast(audio, tf.float32)
    hop = tf.cast(tf.round(tf.cast(segment_length, tf.float32) * (1.0 - overlap)), tf.int32)
    n = tf.shape(audio)[0]

    n_segments = tf.cast(tf.math.ceil((tf.cast(n - segment_length, tf.float32)) / tf.cast(hop, tf.float32)), tf.int32) + 1
    total_len = (n_segments - 1) * hop + segment_length
    pad = total_len - n
    audio_padded = tf.pad(audio, [[0, pad], [0, 0]])

    left = tf.signal.frame(audio_padded[:, 0], frame_length=segment_length, frame_step=hop)
    right = tf.signal.frame(audio_padded[:, 1], frame_length=segment_length, frame_step=hop)
    frames = tf.stack([left, right], axis=-1)

    window = tf.signal.hann_window(segment_length, periodic=True, dtype=tf.float32)
    frames_windowed = frames * tf.reshape(window, [1, segment_length, 1])

    def batch_pred(start_idx):
        end_idx = tf.minimum(start_idx + batch_size, n_segments)
        batch = frames_windowed[start_idx:end_idx]
        preds = model(batch, training=False)
        return preds

    starts = tf.range(0, n_segments, batch_size)
    preds_nested = tf.map_fn(batch_pred, starts, fn_output_signature=tf.float32)

    preds_flat = tf.reshape(preds_nested, [-1, segment_length, 8])[:n_segments]
    preds_reshaped = tf.reshape(preds_flat, [n_segments, segment_length, 4, 2])

    stems_reshaped = tf.reshape(preds_reshaped, [n_segments, segment_length, 8])
    stems_T = tf.transpose(stems_reshaped, [2, 0, 1])
    recon = tf.map_fn(lambda ch: tf.signal.overlap_and_add(ch, hop), stems_T)
    recon = tf.transpose(recon, [1, 0])[:n]
    stems_out = tf.reshape(recon, [n, 4, 2])

    if use_wiener:
        stems_out = wiener_filter_tf(stems_out, audio, stft_frame_length, stft_frame_step, wiener_iterations)

    return stems_out


# Inference TFLite
def _predict_full_batch_tflite(
    audio,
    tflite_wrapper: TFLiteWrapper,
    segment_length,
    overlap,
    batch_size,
    use_wiener,
    stft_frame_length,
    stft_frame_step,
    wiener_iterations,
):
    audio = tf.cast(audio, tf.float32)
    hop = int(round(segment_length * (1.0 - overlap)))
    n = audio.shape[0]

    n_segments = int(tf.math.ceil((n - segment_length) / hop)) + 1
    total_len = (n_segments - 1) * hop + segment_length
    pad = total_len - n
    audio_padded = tf.pad(audio, [[0, pad], [0, 0]])

    left = tf.signal.frame(audio_padded[:, 0], frame_length=segment_length, frame_step=hop)
    right = tf.signal.frame(audio_padded[:, 1], frame_length=segment_length, frame_step=hop)
    frames = tf.stack([left, right], axis=-1)

    window = tf.signal.hann_window(segment_length, periodic=True, dtype=tf.float32)
    frames_windowed = frames * tf.reshape(window, [1, segment_length, 1])

    preds_list = []
    for start_idx in range(0, n_segments, batch_size):
        end_idx = min(start_idx + batch_size, n_segments)
        batch = frames_windowed[start_idx:end_idx]
        preds = tflite_wrapper.predict_batch(batch)
        preds_list.append(preds)

    predictions = tf.concat(preds_list, axis=0)[:n_segments]

    preds_reshaped = tf.reshape(predictions, [n_segments, segment_length, 4, 2])

    stems_reshaped = tf.reshape(preds_reshaped, [n_segments, segment_length, 8])
    stems_T = tf.transpose(stems_reshaped, [2, 0, 1])
    recon = tf.map_fn(lambda ch: tf.signal.overlap_and_add(ch, hop), stems_T)
    recon = tf.transpose(recon, [1, 0])[:n]
    stems_out = tf.reshape(recon, [n, 4, 2])

    if use_wiener:
        stems_out = wiener_filter_tf(stems_out, audio, stft_frame_length, stft_frame_step, wiener_iterations)

    return stems_out

# Wrapper long audio inference
def predict_long_audio_tf(
    audio,
    model,
    segment_length=88064,
    overlap=0.5,
    batch_size=4,
    use_wiener=False,
    stft_frame_length=2048,
    stft_frame_step=512,
    wiener_iterations=1,
    segment_length_sec=None,
    sample_rate=44100,
    is_tflite=False,
):
    if segment_length_sec is not None and segment_length_sec > 0:
        segment_len_samples = int(segment_length_sec * sample_rate)
        n = audio.shape[0]
        outputs = []
        for start in range(0, n, segment_len_samples):
            end = min(start + segment_len_samples, n)
            seg = audio[start:end]
            if is_tflite:
                out = _predict_full_batch_tflite(
                    seg,
                    model,
                    segment_length,
                    overlap,
                    batch_size,
                    use_wiener,
                    stft_frame_length,
                    stft_frame_step,
                    wiener_iterations,
                )
            else:
                out = _predict_full_batch_tf(
                    seg,
                    model,
                    segment_length,
                    overlap,
                    batch_size,
                    use_wiener,
                    stft_frame_length,
                    stft_frame_step,
                    wiener_iterations,
                )
            outputs.append(out)
        return tf.concat(outputs, axis=0)

    else:
        if is_tflite:
            return _predict_full_batch_tflite(
                audio,
                model,
                segment_length,
                overlap,
                batch_size,
                use_wiener,
                stft_frame_length,
                stft_frame_step,
                wiener_iterations,
            )
        else:
            return _predict_full_batch_tf(
                audio,
                model,
                segment_length,
                overlap,
                batch_size,
                use_wiener,
                stft_frame_length,
                stft_frame_step,
                wiener_iterations,
            )