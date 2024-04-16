import numpy as np
import tensorflow as tf
import keras as ks

from abc import ABC


class DataGenerator(ks.utils.Sequence, ABC):
    """Generates data for Keras"""

    def __init__(self, mat_obj, batch_size=500, num_batch=2, val=False):
        """Initialization"""
        super().__init__()
        self.mat_obj = mat_obj
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.val = val

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.val:
            return 1
        else:
            return self.num_batch

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data
        x, y, _ = self.mat_obj.create_matrix_in_list(self.batch_size, self.val)
        # self.mat_obj.randomize_kernel((0.7, 0.4))
        return np.expand_dims(x, -1), np.expand_dims(y, -1)


class IoUMaker(tf.keras.metrics.Metric):
    def __init__(self, n, **kwargs):
        super(IoUMaker, self).__init__(name=f'IoU{n}', **kwargs)
        self.n = n
        self.threshold = 0.5
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_timestep = tf.cast(y_true[:, self.n, :, :], tf.float32)
        y_pred_timestep = tf.cast(y_pred[:, self.n, :, :], tf.float32)
        if self.threshold <= 0.1:
            self.intersection.assign_add(tf.reduce_sum(tf.math.multiply(y_true_timestep, y_pred_timestep)))
            self.union.assign_add(tf.reduce_sum(tf.math.add(y_true_timestep, y_pred_timestep)))
        else:
            y_true_timestep = tf.cast(tf.math.greater_equal(y_true_timestep, self.threshold), tf.float32)
            y_pred_timestep = tf.cast(tf.math.greater_equal(y_pred_timestep, self.threshold), tf.float32)

            self.intersection.assign_add(tf.reduce_sum(tf.math.multiply(y_true_timestep, y_pred_timestep)))
            self.union.assign_add(tf.reduce_sum(tf.math.add(y_true_timestep, y_pred_timestep)))

    def result(self):
        iou_timestep = ks.backend.maximum(1e-10, self.intersection / (self.union - self.intersection))
        return iou_timestep

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class ReservoirLayer(tf.keras.layers.Layer):
    def __init__(self, reservoir_size, gamma=0.95, special=False, **kwargs):
        super(ReservoirLayer, self).__init__(**kwargs)

        self.reservoir_size = reservoir_size
        self.gamma = gamma

        self.reservoir_state = None
        self.reservoir_start = None
        self.bias = None
        self.recurrent_weights = None
        self.input_weights = None
        self.trainables = False
        self.special = special

    def build(self, input_shape):
        feature_size = input_shape[-1]  # Infer input size dynamically

        self.input_weights = self.add_weight("input_weights",
                                             shape=(self.reservoir_size, feature_size),
                                             initializer=tf.keras.initializers.GlorotNormal(),
                                             trainable=self.special)

        self.recurrent_weights = self.add_weight("recurrent_weights",
                                                 shape=(self.reservoir_size, self.reservoir_size),
                                                 initializer=tf.keras.initializers.Orthogonal(),
                                                 trainable=self.trainables)

        self.bias = self.add_weight("bias",
                                    shape=(self.reservoir_size, 1),
                                    initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                    trainable=self.trainables)

        self.reservoir_start = self.add_weight("reservoir_state",
                                               shape=(self.reservoir_size, 1),
                                               trainable=self.trainables,
                                               initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
                                               )

        super(ReservoirLayer, self).build(input_shape)

    def call(self, inputs):
        outputs = []
        reservoir_cal = self.reservoir_start
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


def custom_weighted_loss(y_true, y_pred, scaling=1.0):
    """
    Custom loss function with emphasis on errors for values that are 1 in y_true using mean squared error.
    Weighting factor is dependent on the fill factor of y_true.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels

    Returns:
    - Weighted mean squared error loss
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    # Calculate fill factor (proportion of non-zero values)
    fill_factor = tf.keras.backend.mean(tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32))
    # firstly makes a boolean tensor of non-zero, then makes the boolean to numbers and then takes the mean

    # Calculates the normal squared errors
    squared_errors = tf.math.square(y_true - y_pred)

    # lets the fill factor be dampened or amped
    weight_factor = scaling / fill_factor

    # applies the weight factor only to where the object is
    weighted_squared_errors = (y_true * (weight_factor * squared_errors)
                               + (1 - y_true) * squared_errors)

    # Calculates the mean over all elements, to make it rank 0
    loss = tf.keras.backend.mean(weighted_squared_errors)

    return loss


def custom_iou(y_true, y_pred):
    """
    Custom IoU (Intersection over Union) evaluation function for each timestep.

    Parameters:
    - y_true: True labels with shape [batch_size, time, row, column, color]
    - y_pred: Predicted labels with shape [batch_size, time, row, column, color]

    Returns:
    - IoU for each timestep
    """
    iou_per_timestep = []

    # Iterate over time steps
    for t in range(y_true.shape[1]):
        # Extract labels for the current timestep
        y_true_timestep = tf.cast(y_true[:, t, :, :], tf.float32)

        # Extract predictions for the current timestep
        y_pred_timestep = tf.cast(y_pred[:, t, :, :], tf.float32)

        # Calculate Intersection and Union for the current timestep
        intersection = tf.reduce_sum(tf.math.multiply(y_true_timestep, y_pred_timestep))
        union = tf.reduce_sum(tf.math.add(y_true_timestep, y_pred_timestep))

        # Calculate IoU for the current timestep
        iou_timestep = intersection / (union - intersection)

        # Append IoU for the current timestep to the list
        iou_per_timestep.append(iou_timestep)

    return np.array(iou_per_timestep)
