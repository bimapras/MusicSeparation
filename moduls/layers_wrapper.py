import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='GlobalTCN')
class GlobalTCN(tf.keras.layers.Layer):
    def __init__(self, num_filters, out_filters, kernel_size, n_layer=5,
                 padding='causal', bias=False, **kwargs):
        super(GlobalTCN, self).__init__(**kwargs)
        self.num_filters = int(num_filters)
        self.out_filters = int(out_filters)
        self.kernel_size = int(kernel_size)
        self.n_layer = int(n_layer)
        self.padding = padding
        self.bias = bias

        self.input_proj = None
        self.conv_blocks = []
        self.norms = []
        self.gates = []
        self.skip_projs = []
        self.skip_add = tf.keras.layers.Add()
        self.pointwise = None

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("GlobalTCN expects input shape (batch, length, channels)")

        # Input linear projection
        self.input_proj = tf.keras.layers.Dense(self.num_filters, use_bias=self.bias)

        for i in range(self.n_layer):
            conv = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=self.num_filters,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              dilation_rate=2**i,
                              use_bias=self.bias,
                              kernel_initializer='he_normal'),
                tf.keras.layers.ReLU()
            ])
            self.conv_blocks.append(conv)
            self.norms.append(tf.keras.layers.LayerNormalization())
            self.gates.append(
                tf.keras.layers.Conv1D(filters=self.num_filters,
                              kernel_size=1,
                              padding='same',
                              activation='sigmoid',
                              use_bias=True)
            )
            self.skip_projs.append(
                tf.keras.layers.Dense(self.out_filters, use_bias=self.bias)
            )

        self.pointwise = tf.keras.layers.Conv1D(self.out_filters, 
                                       kernel_size=1, 
                                       padding='same', 
                                       use_bias=self.bias)

        super(GlobalTCN, self).build(input_shape)

    def call(self, inputs, training=None):
        x = self.input_proj(inputs)
        _skip = []

        for i in range(self.n_layer):
            res = x

            x = self.conv_blocks[i](x)
            x = self.norms[i](x)

            skip = self.skip_projs[i](x)
            _skip.append(skip)

            gate_input = tf.concat([x, res], axis=-1)
            z = self.gates[i](gate_input)
            x = z * x + (1 - z) * res
            x = tf.nn.relu(x)

        skip_sum = self.skip_add(_skip)

        gate_out = self.pointwise(skip_sum)

        return gate_out

    def get_config(self):
        config = super(GlobalTCN, self).get_config()
        config.update({
            'num_filters': self.num_filters,
            'out_filters': self.out_filters,
            'kernel_size': self.kernel_size,
            'n_layer': self.n_layer,
            'padding': self.padding,
            'bias': self.bias
        })
        return config

@tf.keras.utils.register_keras_serializable(package='LocalTCN')
class LocalTCN(tf.keras.layers.Layer):
    def __init__(self, num_filters, out_filters, kernel_size, n_layer=5,
                 padding='causal', bias=False, n_chunks=4, **kwargs):
        super(LocalTCN, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.n_layer = n_layer
        self.padding = padding
        self.bias = bias
        self.n_chunks = n_chunks

        self.global_tcn = GlobalTCN(num_filters, out_filters, kernel_size, 
                                    n_layer=n_layer, padding=padding, bias=bias)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("LocalTCN expects input shape (batch, length, channels)")

        self.seq_len = input_shape[1]
        if self.seq_len % self.n_chunks != 0:
            raise ValueError(f"Input length {self.seq_len} tidak bisa dibagi rata oleh n_chunks={self.n_chunks}")

        # Chunk size non-overlapping
        self.chunk_size = self.seq_len // self.n_chunks

        # Add overlapping 50%
        self.step = self.chunk_size // 2
        if self.step == 0:
            raise ValueError("Step size jadi 0, coba kecilkan n_chunks atau tambah panjang input")

        super(LocalTCN, self).build(input_shape)

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]

        output_buffer = tf.zeros((batch_size, self.seq_len, self.out_filters), dtype=inputs.dtype)
        weight_buffer = tf.zeros((batch_size, self.seq_len, 1), dtype=inputs.dtype)
        
        starts = list(range(0, self.seq_len - self.chunk_size + 1, self.step))

        for i, start in enumerate(starts):
            end = start + self.chunk_size
            chunk = inputs[:, start:end, :]
            out_chunk = self.global_tcn(chunk, training=training)

            paddings_before = [[0, 0], [start, self.seq_len - end], [0, 0]]
            padded_out = tf.pad(out_chunk, paddings_before)
            output_buffer += padded_out

            weight = tf.ones((batch_size, end - start, 1), dtype=inputs.dtype)
            padded_weight = tf.pad(weight, paddings_before)
            weight_buffer += padded_weight

        # Avoid division by zero
        weight_buffer = tf.where(weight_buffer == 0, tf.ones_like(weight_buffer), weight_buffer)

        output_final = output_buffer / weight_buffer

        return output_final

    def get_config(self):
        config = super(LocalTCN, self).get_config()
        config.update({
            'num_filters': self.num_filters,
            'out_filters': self.out_filters,
            'kernel_size': self.kernel_size,
            'n_layer': self.n_layer,
            'padding': self.padding,
            'bias': self.bias,
            'n_chunks': self.n_chunks
        })
        return config
    

@tf.keras.utils.register_keras_serializable(package='TCN')
class TCN(tf.keras.layers.Layer):
    def __init__(self, num_filters, out_filters, kernel_size, n_layer=5,
                 padding='causal', bias=False, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.num_filters = int(num_filters)
        self.out_filters = int(out_filters)
        self.kernel_size = int(kernel_size)
        self.n_layer = int(n_layer)
        self.padding = padding
        self.bias = bias

        self.input_proj = None
        self.conv_blocks = []
        self.norms = []
        self.gates = []
        self.skip_projs = []
        self.skip_add = tf.keras.layers.Add()
        self.pointwise = None

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("TCN expects input shape (batch, length, channels)")

        self.input_proj = tf.keras.layers.Dense(self.num_filters, use_bias=self.bias)

        for i in range(self.n_layer):
            conv = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=self.num_filters,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              dilation_rate=3**i,
                              use_bias=self.bias,
                              kernel_initializer='he_normal'),
                tf.keras.layers.ReLU()
            ])
            self.conv_blocks.append(conv)
            self.norms.append(tf.keras.layers.LayerNormalization())
            self.gates.append(
                tf.keras.layers.Conv1D(filters=self.num_filters,
                              kernel_size=1,
                              padding='same',
                              activation='sigmoid',
                              use_bias=True)
            )
            self.skip_projs.append(
                tf.keras.layers.Dense(self.out_filters, use_bias=self.bias)
            )

        self.pointwise = tf.keras.layers.Conv1D(self.out_filters, 
                                       kernel_size=1, 
                                       padding='same', 
                                       use_bias=self.bias)

        super(TCN, self).build(input_shape)

    def call(self, inputs, training=None):
        x = self.input_proj(inputs)
        _skip = []

        for i in range(self.n_layer):
            res = x

            x = self.conv_blocks[i](x)
            x = self.norms[i](x)

            skip = self.skip_projs[i](x)
            _skip.append(skip)

            gate_input = tf.concat([x, res], axis=-1)
            z = self.gates[i](gate_input)
            x = z * x + (1 - z) * res
            x = tf.nn.relu(x)

        skip_sum = self.skip_add(_skip)

        gate_out = self.pointwise(skip_sum)
        output = inputs * gate_out
        return output

    def get_config(self):
        config = super(TCN, self).get_config()
        config.update({
            'num_filters': self.num_filters,
            'out_filters': self.out_filters,
            'kernel_size': self.kernel_size,
            'n_layer': self.n_layer,
            'padding': self.padding,
            'bias': self.bias
        })
        return config

@tf.keras.utils.register_keras_serializable(package='WeightedFusionGate')
class WeightedFusionGate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedFusionGate, self).__init__(**kwargs)
        self.gate = None

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 3:
            raise ValueError("Fusion expects 3 inputs: [x, tcn_out, cbam_out]")

        self.gate = tf.keras.layers.Dense(2, activation="sigmoid")
        super(WeightedFusionGate, self).build(input_shape)

    def call(self, inputs):
        x, tcn_out, cbam_out = inputs

        context = tf.reduce_mean(tf.concat([tcn_out, cbam_out], axis=-1), axis=1)

        weights = self.gate(context)
        alpha, beta = tf.split(weights, 2, axis=-1)

        alpha = tf.expand_dims(alpha, axis=1)
        beta  = tf.expand_dims(beta, axis=1)

        return x + alpha * tcn_out + beta * cbam_out

    def get_config(self):
        config = super(WeightedFusionGate, self).get_config()
        return config

@tf.keras.utils.register_keras_serializable(package='GLU')
class GLU(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, dilation, use_bias=False, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.dilation = dilation
        self.conv = None
        
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("GLU expects input shape (batch, length, channels)")
        self.conv = tf.keras.layers.Conv1D(self.num_filters * 2, 
                                           kernel_size=self.kernel_size, 
                                           strides=1, 
                                           dilation_rate = self.dilation,
                                           padding='same',
                                           use_bias=self.use_bias)

    def call(self, inputs):
        x = self.conv(inputs)
        linear, gate = tf.split(x, num_or_size_splits=2, axis=-1)
        return linear * tf.sigmoid(gate)

    def get_config(self):
        config = super(GLU, self).get_config()
        config.update({
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "dilation": self.dilation,
            "use_bias": self.use_bias
        })
        return config

@tf.keras.utils.register_keras_serializable(package='DEMUCSNORM')
class DemucsNormalize(tf.keras.layers.Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, inputs, reverse=False, mean=None, std=None):
        # inputs: (batch, time, channels)
        if not reverse:
            mono = tf.reduce_mean(inputs, axis=-1, keepdims=True)  # shape: (B, T, 1)
            mean = tf.reduce_mean(mono, axis=1, keepdims=True)     # (B, 1, 1)
            std = tf.math.reduce_std(mono, axis=1, keepdims=True)  # (B, 1, 1)
            std = tf.maximum(std, self.eps)

            normalized = (inputs - mean) / std
            return normalized, mean, std
        else:
            if mean is None or std is None:
                raise ValueError("Must provide mean and std for denormalization")
            return inputs * std + mean

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config
    
# Wrapper TFLite
class TFLiteWrapper:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.batch_size = None

    def allocate(self, batch_size, input_shape):
        if self.batch_size != batch_size:
            new_shape = list(input_shape)
            new_shape[0] = batch_size
            self.interpreter.resize_tensor_input(self.input_details[0]['index'], new_shape)
            self.interpreter.allocate_tensors()
            self.batch_size = batch_size

    def predict_batch(self, batch: tf.Tensor) -> tf.Tensor:
        batch_size = batch.shape[0]
        self.allocate(batch_size, self.input_details[0]['shape'])

        input_np = batch.numpy().astype(self.input_details[0]['dtype'])
        self.interpreter.set_tensor(self.input_details[0]['index'], input_np)
        self.interpreter.invoke()
        output_np = self.interpreter.get_tensor(self.output_details[0]['index'])
        return tf.convert_to_tensor(output_np, dtype=tf.float32)