import numpy as np
import tensorflow as tf


class ReservoirLayer(tf.keras.layers.Layer):
    def __init__(self, reservoir_size, gamma=0.95, **kwargs):
        super(ReservoirLayer, self).__init__(**kwargs)
        self.reservoir_size = reservoir_size
        self.gamma = gamma

        self.reservoir_state = None

    def build(self, input_shape):
        feature_size = input_shape[-1]  # Infer input size dynamically

        self.input_weights = self.add_weight("input_weights",
                                             shape=(self.reservoir_size, feature_size),
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

        self.reservoir_states = self.add_weight("reservoir_state",
                                                shape=(self.reservoir_size, 1),
                                                trainable=False,
                                                initializer=tf.keras.initializers.RandomNormal(),
                                                )

        super(ReservoirLayer, self).build(input_shape)

    def call(self, inputs):
        outputs = []
        reservoir_cal = self.reservoir_state or self.reservoir_states
        time_dim = tf.unstack(inputs, axis=1)

        for i in time_dim:  # Iterate over the temporal dimension

            input_var = tf.expand_dims(i, -1)  # Take input at each time step

            # Update reservoir state
            reservoir_cal = ((1 - self.gamma) * reservoir_cal + self.gamma *
                             tf.math.tanh(tf.matmul(self.recurrent_weights, reservoir_cal) +
                                          tf.matmul(self.input_weights, input_var) + self.bias))

            outputs.append(tf.identity(reservoir_cal))  # Append the reservoir state at each time step
        return tf.squeeze(tf.stack(outputs, axis=1), axis=-1)  # Stack the outputs along the temporal dimension

    def compute_output_shape(self, input_shape):
        return None, input_shape[1], self.reservoir_size


def activation_function(x):
    return np.tanh(x)


def update_reservoir(reservoir):
    return ((1 - gamma) * reservoir + gamma *
            activation_function(np.matmul(recurrent_weights, reservoir) + np.matmul(input_weights, input_var) + bias))


def output_function(reservoir):
    return np.matmul(output_weights, reservoir)


if '__main__' == __name__:
    input_size = 3
    reservoir_size = 2
    output_size = 4
    timesteps = 5

    reservoir_state = np.zeros((timesteps + 1, reservoir_size))
    reservoir_state[0] = np.random.randn(reservoir_size)

    gamma = 0.95

    bias = np.random.randn(reservoir_size)
    input_var = np.random.randn(input_size)

    recurrent_weights = np.random.randn(reservoir_size, reservoir_size)

    input_weights = np.random.randn(reservoir_size, input_size)
    output_weights = np.random.randn(output_size, reservoir_size)

    for i in range(timesteps):
        reservoir_state[i + 1] = update_reservoir(reservoir_state[i])
        print(f'{reservoir_state}')
        print()
