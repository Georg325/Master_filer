import numpy as np
import tensorflow as tf
import scipy as sp

import matplotlib.pyplot as plt

rng = np.random.default_rng()

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


class BrainLayer(tf.keras.layers.Layer):
    def __init__(self, reservoir_size, gamma=0.95, make_weights=False, half_output=False, half_input=False, **kwargs):
        super(BrainLayer, self).__init__(**kwargs)

        self.reservoir_size = reservoir_size
        self.gamma = gamma

        self.reservoir_state = None
        self.reservoir_start = None
        self.bias = None
        self.recurrent_weights = None
        self.input_weights = None
        self.made_weights = None

        self.in_cor = None
        self.out_cor = None

        if make_weights:
            self.made_weights = make_rec_weights(reservoir_size, thickness=20)

    def build(self, input_shape):
        feature_size = input_shape[-1]  # Infer input size dynamically

        self.input_weights = self.add_weight("input_weights",
                                             shape=(self.reservoir_size, feature_size),
                                             initializer=tf.keras.initializers.GlorotNormal(),
                                             trainable=False)

        if self.made_weights is None:
            self.recurrent_weights = self.add_weight("recurrent_weights",
                                                     shape=(self.reservoir_size, self.reservoir_size),
                                                     initializer=tf.keras.initializers.Orthogonal(),
                                                     trainable=False)
        else:
            self.recurrent_weights = self.add_weight("recurrent_weights",
                                                     shape=(self.reservoir_size, self.reservoir_size),
                                                     initializer=tf.constant_initializer(value=self.made_weights),
                                                     trainable=False)

        self.bias = self.add_weight("bias",
                                    shape=(self.reservoir_size, 1),
                                    initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                    trainable=False)

        self.reservoir_start = self.add_weight("reservoir_state",
                                               shape=(self.reservoir_size, 1),
                                               trainable=False,
                                               initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
                                               )

        out_cor = np.zeros((self.reservoir_size // 2, self.reservoir_size // 2), dtype=np.float32)
        dig = np.diag(np.ones((self.reservoir_size // 2)))
        out_cor = np.concatenate((out_cor, dig), axis=1, dtype=np.float32)

        self.in_cor = np.diag(np.concatenate(
            (np.ones(self.reservoir_size // 2), np.zeros(self.reservoir_size // 2)), axis=0, dtype=np.float32))
        self.out_cor = out_cor

        super(BrainLayer, self).build(input_shape)

    def call(self, inputs):
        outputs = []
        reservoir_cal = self.reservoir_start
        time_dim = tf.unstack(inputs, axis=1)

        for i in time_dim:  # Iterate over the temporal dimension

            input_var = tf.expand_dims(i, -1)  # Take input at each time step

            start_step = tf.matmul(self.input_weights, input_var)

            # Update reservoir state
            reservoir_cal = ((1 - self.gamma) * reservoir_cal + self.gamma *
                             tf.math.tanh(tf.matmul(self.recurrent_weights, reservoir_cal) +
                                          tf.matmul(self.in_cor, start_step) + self.bias))

            mid_step = tf.matmul(self.out_cor, reservoir_cal)

            outputs.append(tf.identity(mid_step))  # Append the reservoir state at each time step

        return tf.squeeze(tf.stack(outputs, axis=1), axis=-1)  # Stack the outputs along the temporal dimension

    def compute_output_shape(self, input_shape):
        return None, input_shape[1], self.reservoir_size


def activation_function(x):
    return np.tanh(x)


def update_reservoir(reservoir, x_var):
    return ((1 - gamma) * reservoir + gamma *
            activation_function(np.matmul(recurrent_weights, reservoir) + np.matmul(input_weights, x_var) + bias))


def output_function(reservoir):
    return np.matmul(output_weights, reservoir)


def make_rec_weights(size, thickness=1, info=False, show=False, num=None, shuffle=True):
    if size % 2 == 1:
        raise ValueError('Size must be an even number')

    rt = sp.stats.ortho_group.rvs(dim=size // 2)
    lb = sp.stats.ortho_group.rvs(dim=size // 2)
    zer = np.zeros((size // 2, size // 2))

    a = np.concatenate((np.concatenate((rt, zer), axis=1), np.concatenate((zer, lb), axis=1)), axis=0)

    if num is None:
        num = min(a.max()*0.2, 0.4)

    for i in range(thickness):
        dig = np.random.choice((num, -num), size=size//2 - i)
        if i == 0:
            a[size // 2:, :size // 2] += np.rot90(np.diag(dig))

            a[:size // 2, size // 2:] += np.rot90(np.diag(dig))
        else:
            a[size // 2:, :size // 2] += np.rot90(np.diag(dig, k=i))
            a[size // 2:, :size // 2] -= np.rot90(np.diag(dig, k=-i))

            a[:size // 2, size // 2:] += np.rot90(np.diag(dig, k=i))
            a[:size // 2, size // 2:] -= np.rot90(np.diag(dig, k=-i))

    if shuffle:
        rng.shuffle(a[size // 2:, :size // 2], axis=0)
        rng.shuffle(a[size // 2:, :size // 2], axis=1)

        rng.shuffle(a[:size // 2, size // 2:], axis=1)
        rng.shuffle(a[:size // 2, size // 2:], axis=0)

    stability = np.absolute(sp.linalg.eigvals(a)).round(3)

    if stability.max() > 1.04 or stability.min() < 0.94:
        print(f'not stable: max={stability.max()}, min={stability.min()}')
        # a = make_weights(size, num=num-0.1)

    if info:
        print('num:', num)
        print(stability.max(), stability.min())
        print(sum(stability) - size)

    if show:
        plt.imshow(a, cmap='gray')
        plt.colorbar()
        plt.title('Example weights:')
        plt.show()
    else:
        plt.imshow(a)
        plt.colorbar()
        plt.savefig('last_weights.png')

    return a


# make_rec_weights(100, show=True, shuffle=True, thickness=5).round(3)

if '__ma in__' == __name__:
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
