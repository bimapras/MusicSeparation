import tensorflow as tf

# Loss functions for mono channels
@tf.keras.utils.register_keras_serializable()
def stft_mag(y, frame_length, frame_step, fft_length=None):
    stft = tf.signal.stft(y, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    return tf.abs(stft)

@tf.keras.utils.register_keras_serializable()
def stft_loss(y_true, y_pred):
    """
    Multi-resolution STFT loss
    """
    resolutions = [
        (2048, 512),
        (1024, 256),
        (512, 128),
    ]
    loss = 0
    for frame_length, frame_step in resolutions:
        S_true = stft_mag(y_true, frame_length, frame_step)
        S_pred = stft_mag(y_pred, frame_length, frame_step)
        loss += tf.reduce_mean(tf.abs(S_true - S_pred))
    return loss / len(resolutions)

@tf.keras.utils.register_keras_serializable()
def demucs_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    """
    Combined L1 + STFT loss for mono channels
    """
    true_channels = tf.unstack(y_true, axis=-1)
    pred_channels = tf.unstack(y_pred, axis=-1)

    losses = []
    for true_c, pred_c in zip(true_channels, pred_channels):
        time_loss = tf.reduce_mean(tf.abs(true_c - pred_c))
        freq_loss = stft_loss(true_c, pred_c)
        losses.append(alpha * time_loss + beta * freq_loss)

    total_loss = tf.add_n(losses) / len(losses)
    return total_loss

# Loss function for stereo channels
@tf.keras.utils.register_keras_serializable()
def new_stft_mag(x, frame_length, frame_step):
    """
    Short-time Fourier transform magnitude for stereo channels
    """
    batch_size = tf.shape(x)[0]
    time_len = tf.shape(x)[1]

    x = tf.reshape(x, [batch_size, time_len, -1, 2])
    x = tf.transpose(x, [0, 2, 3, 1])
    x = tf.reshape(x, [-1, time_len])

    x_stft = tf.signal.stft(x, frame_length=frame_length, frame_step=frame_step)
    return tf.abs(x_stft) + 1e-8

@tf.keras.utils.register_keras_serializable()
def new_stft_loss(y_true, y_pred):
    resolutions = [
        (2048, 512),
        (1024, 256),
        (512, 128),
    ]
    loss = 0.0
    for frame_length, frame_step in resolutions:
        S_true = new_stft_mag(y_true, frame_length, frame_step)
        S_pred = new_stft_mag(y_pred, frame_length, frame_step)
        loss += tf.reduce_mean(tf.abs(S_true - S_pred))
    return loss / len(resolutions)

@tf.keras.utils.register_keras_serializable()
def new_demucs_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    time_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    freq_loss = new_stft_loss(y_true, y_pred)
    return alpha * time_loss + beta * freq_loss

# Custom SDR metric
@tf.keras.utils.register_keras_serializable()
def compute_sdr(y_true, y_pred, eps=1e-8):
    true_energy = tf.reduce_sum(tf.square(y_true), axis=-1)
    noise_energy = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    sdr = 10.0 * tf.math.log((true_energy + eps) / (noise_energy + eps)) / tf.math.log(10.0)
    return sdr