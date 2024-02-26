import numpy as np


'''
class ComplexReservoirLayer(tf.keras.layers.Layer):
    def __init__(self, reservoir_size, gamma=0.95, **kwargs):
        super(ComplexReservoirLayer, self).__init__(**kwargs)
        self.reservoir_size = reservoir_size
        self.gamma = gamma


    def build(self, input_shape):
        feature_size = input_shape[-1]  # Infer input size dynamically

        self.input_weights = self.add_weight("input_weights",
                                             shape=(self.reservoir_size, (feature_size*2)**2),
                                             initializer=tf.keras.initializers.RandomNormal(),
                                             trainable=False)

        self.recurrent_weights = self.add_weight("recurrent_weights",
                                                 shape=(self.reservoir_size, self.reservoir_size),
                                                 initializer=tf.keras.initializers.RandomNormal(),
                                                 trainable=False)

        self.bias = self.add_weight("bias",
                                    shape=(self.reservoir_size, 1),
                                    initializer=tf.keras.initializers.RandomNormal(),
                                    trainable=False)

        self.reservoir_start = self.add_weight("reservoir_start",
                                               shape=(self.reservoir_size, 1),
                                               trainable=False,
                                               initializer=tf.keras.initializers.RandomNormal(),
                                               )

        super(ComplexReservoirLayer, self).build(input_shape)

    def call(self, inputs):
        outputs = []
        reservoir_cal = self.reservoir_start
        time_dim = tf.unstack(inputs, axis=1)

        for nr, tensor in enumerate(time_dim):  # Iterate over the temporal dimension
                             # Take input at each time step
            if nr == 0:
                mid_var = tf.concat((tensor, time_dim[nr]), axis=1)
            else:
                mid_var = tf.concat((tensor, time_dim[nr-1]), axis=1)
            print(mid_var.shape)
            double = tf.matmul(mid_var, tf.reshape(mid_var, shape=[mid_var.shape[1], -1]))  # Element-wise multiplication
            print(double.shape)
            fisk = tf.reshape(double,
                              shape=(-1, (tensor.shape[1]*2)**2, 1))  # Reshape to match the concatenation in NumPy code
            # Update reservoir state
            reservoir_cal = ((1 - self.gamma) * reservoir_cal + self.gamma *
                             tf.math.tanh(tf.matmul(self.recurrent_weights, reservoir_cal) +
                                          tf.matmul(self.input_weights, fisk) + self.bias))

            outputs.append(tf.identity(reservoir_cal))  # Append the reservoir state at each time step

        stacked_outputs = tf.squeeze(tf.stack(outputs, axis=1), axis=-1)  # Stack the outputs along the temporal dimension
        print(stacked_outputs.shape)
        return stacked_outputs  # Return both the stacked outputs and intermediate tensor fisk

    def compute_output_shape(self, input_shape):
        return None, input_shape[1], self.reservoir_size
'''


def activation_function(x):
    return np.tanh(x)


def update_reservoir(reservoir, x_var):
    return ((1 - gamma) * reservoir + gamma *
            activation_function(np.matmul(recurrent_weights, reservoir) + np.matmul(input_weights, x_var) + bias))


def output_function(reservoir):
    return np.matmul(output_weights, reservoir)


if '__main__' == __name__:
    input_size = 16
    reservoir_size = 20
    output_size = 7
    time_steps = 5
    batch_size = 10

    input_var = np.zeros((batch_size, time_steps + 1, input_size))
    reservoir_state = np.zeros((batch_size, time_steps + 1, reservoir_size))

    gamma = 0.95

    bias = np.random.randn(reservoir_size)

    recurrent_weights = np.random.randn(reservoir_size, reservoir_size)

    input_weights = np.random.randn(reservoir_size, (input_size * 2) ** 2)

    output_weights = np.random.randn(output_size, reservoir_size)
    output_var = np.zeros((batch_size, time_steps, output_size))

    for i in range(1, time_steps + 1):
        for j in range(input_size):
            input_var[i, j] = i - 1 + (j + 1) / 2

    print('input', input_var.shape)
    print()

    mid_cal = [np.concatenate([input_var[0], input_var[0]])]

    for i in range(1, time_steps + 1):
        mid_cal.append(np.concatenate([input_var[i - 1], input_var[i]]))
    mid_cal = np.array(mid_cal)

    print('mid input', mid_cal.shape)
    print()

    fisk = []

    for inp in mid_cal:
        double = inp * inp[:, None]  # Use broadcasting for element-wise multiplication
        result = np.concatenate(double)
        fisk.append(result)

    com = np.array(fisk)
    print('o2', com.shape)
    print()

    for i in range(time_steps):
        reservoir_state[i + 1] = update_reservoir(reservoir_state[i], x_var=com[i])
        output_var[i, :] = output_function(reservoir_state[i + 1])

    print('reservoir', reservoir_state.shape)
    print()
    print('output', output_var.shape)
