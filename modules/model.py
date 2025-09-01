import tensorflow as tf
from modules.layers_wrapper import *

class SeparatorModel:
    def __init__(
        self,
        input_length=88064,
        n_filter=64,
        n_kernel=8,
        stride=4,
        t_layer=3,
        n_chunk=4,
        bias=False,
        n_stems=4,
        n_depth=4
    ):
        self.input_length = input_length
        self.n_filter = n_filter
        self.n_kernel = n_kernel
        self.stride = stride
        self.t_layer = t_layer
        self.bias = bias
        self.n_chunk = n_chunk
        self.n_stems = n_stems
        self.n_depth = n_depth

    def encoder(self, x, num_filters, kernel_size, stride, depth, padding='same', bias=True):
        list_enc = []
        list_dilation = []
        dilation = 1
        for _ in range(depth):
            x = tf.keras.layers.Conv1D(num_filters, kernel_size, strides=stride, padding=padding, use_bias=bias)(x)
            list_enc.append(x)
            list_dilation.append(dilation)
            x = tf.keras.layers.ReLU()(x)
            x = GLU(num_filters, kernel_size, dilation)(x)
            num_filters += num_filters
            dilation += dilation

        return x, list_enc, list_dilation

    def decoder(self, x, list_enc, list_dilation, kernel_size, stride, n_tcn=4, n_chunk=4, padding='same', bias=True, n_stems=4):
        rev_enc = list_enc[::-1]
        depth = len(rev_enc)
        count = 0

        for enc in rev_enc:
            l_enc = LocalTCN(num_filters=x.shape[-1] // 2,
                             out_filters=x.shape[-1],
                             kernel_size=kernel_size,
                             n_layer=n_tcn,
                             n_chunks=n_chunk,
                             padding='causal',
                             bias=bias)(enc)
            g_enc = GlobalTCN(num_filters=x.shape[-1] // 2,
                              out_filters=x.shape[-1],
                              kernel_size=kernel_size,
                              n_layer=n_tcn,
                              padding='causal',
                              bias=bias)(l_enc)
            x = tf.keras.layers.Add()([x, g_enc + enc])
            x = GLU(x.shape[-1], kernel_size, list_dilation.pop(-1))(x)
            n_filter = x.shape[-1] // 2
            if count + 1 == depth:
                x = tf.keras.layers.Conv1DTranspose(n_stems * 2, kernel_size, strides=stride, padding=padding, use_bias=bias)(x)
            else:
                x = tf.keras.layers.Conv1DTranspose(n_filter, kernel_size, strides=stride, padding=padding, use_bias=bias)(x)
            count += 1
        return x

    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=(self.input_length, 2), name='input_wave')
        x_norm, x_mean, x_std = DemucsNormalize()(input_layer)

        x, list_enc, list_dilation = self.encoder(x_norm, self.n_filter, self.n_kernel, self.stride, self.n_depth, padding='same', bias=self.bias)

        x = self.decoder(x, 
                         list_enc,
                         list_dilation,
                         kernel_size=self.n_kernel,
                         stride=self.stride,
                         padding='same',
                         bias=self.bias,
                         n_tcn=self.t_layer,
                         n_chunk=self.n_chunk,
                         n_stems=self.n_stems)
        x_denorm = DemucsNormalize()(x, reverse=True, mean=x_mean, std=x_std)

        model = tf.keras.models.Model(inputs=input_layer, outputs=x_denorm, name='separator_model')
        return model