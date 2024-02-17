import numpy as np
import tensorflow as tf
import keras as ks

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, TimeDistributed, Input, GRU, LSTM

from abc import ABC
from resoviarfuntion import ReservoirLayer


class DataGenerator(ks.utils.Sequence, ABC):
    """Generates data for Keras"""

    def __init__(self, mat_obj, model, batch_size=500, num_batch=2):
        """Initialization"""
        super().__init__()
        self.mat_obj = mat_obj
        self.model = model
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.mat_obj.clear_score()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.num_batch

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data
        x, y, _ = self.mat_obj.create_matrix_in_list(self.batch_size)
        # self.mat_obj.randomize_kernel((0.7, 0.4))
        return np.expand_dims(x, -1), np.expand_dims(y, -1)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.mat_obj.eval(self.model, self.batch_size)
        return


def build_model(model_type, parameters):
    if model_type == 'cnn_gru':
        return build_cnn_gru(parameters)
    elif model_type == 'cnn':
        return build_cnn(parameters)
    elif model_type == 'res':
        return build_resor(parameters)

    print('error')


def build_cnn_gru(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = Sequential()

    model.add(Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(TimeDistributed(Conv2D(filter_base, kernel_size=(3, 3), padding='same', activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))

    model.add(TimeDistributed(Conv2D(filter_base * 2, kernel_size=(2, 2), padding='same', activation='tanh')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))

    # model.add(TimeDistributed(Conv2D(filter_base * 4, kernel_size=(2, 2), padding='same', activation='tanh')))

    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(neuron_base, activation='tanh', return_sequences=True, recurrent_regularizer='l2'))
    model.add(LSTM(neuron_base * 2, activation='tanh', return_sequences=True))

    # Fully connected layer
    model.add(TimeDistributed(Dense(mat_size[0] * mat_size[1], activation='sigmoid')))

    # Reshape to the desired output shape
    model.add(Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def build_cnn(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = Sequential()

    model.add(Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))

    model.add(TimeDistributed(Conv2D(filter_base * 2, kernel_size=(3, 3), padding='same', activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))

    model.add(TimeDistributed(Conv2D(filter_base * 4, kernel_size=(2, 2), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(1, kernel_size=(2, 2), padding='same', activation='tanh')))

    # Reshape to the desired output shape
    model.add(Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def custom_weighted_loss(y_true, y_pred, scaling=1.5):
    """
    Custom loss function with emphasis on errors for values that are 1 in y_true using mean squared error.
    Weighting factor is dependent on the fill factor of y_true.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels

    Returns:
    - Weighted mean squared error loss
    """
    # Calculate fill factor (proportion of non-zero values)
    fill_factor = tf.keras.backend.mean(tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32))

    # Calculate squared errors
    squared_errors = tf.math.square(tf.cast(y_true, dtype=tf.float32) - y_pred)

    # Calculate weight factor based on fill factor
    weight_factor = scaling * fill_factor

    # Apply weights to positive class
    weighted_squared_errors = (tf.cast(y_true, dtype=tf.float32) * (weight_factor * squared_errors)
                               + tf.cast((1 - y_true), dtype=tf.float32) * squared_errors)

    # Calculate mean loss over all elements
    loss = tf.keras.backend.mean(weighted_squared_errors)

    return loss


def f1_score(y_true, y_pred):
    """
    Custom F1 score calculation.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels

    Returns:
    - F1 score
    """
    true_positives = tf.reduce_sum(tf.cast(y_true * tf.round(y_pred), tf.float32))
    predicted_positives = tf.reduce_sum(tf.round(y_pred))
    actual_positives = tf.reduce_sum(y_true)

    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    return f1


def custom_accuracy(y_true, y_pred):
    """
    Custom F1 score evaluation function for each timestep.

    Parameters:
    - y_true: True labels with shape [batch_size, time, row, column]
    - y_pred: Predicted labels with shape [batch_size, time, row, column]

    Returns:
    - F1 score for each timestep
    """
    f1_per_timestep = []

    # Iterate over timesteps
    for t in range(10):
        # Extract labels for the current timestep
        y_true_timestep = tf.cast(y_true[:, t, :, :], tf.float32)

        # Extract predictions for the current timestep
        y_pred_timestep = tf.cast(y_pred[:, t, :, :], tf.float32)

        # Calculate F1 score for the current timestep
        f1_timestep = f1_score(y_true_timestep, y_pred_timestep)

        # Append F1 score for the current timestep to the list
        f1_per_timestep.append(f1_timestep)

    return np.array(f1_per_timestep)


def custom_iou(y_true, y_pred):
    """
    Custom IoU (Intersection over Union) evaluation function for each timestep.

    Parameters:
    - y_true: True labels with shape [batch_size, time, row, column]
    - y_pred: Predicted labels with shape [batch_size, time, row, column]

    Returns:
    - IoU for each timestep
    """
    iou_per_timestep = []

    # Iterate over timesteps
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


def cnn_block(model, parameters, kernel_size, blocks=1, pooling=False):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    for i in range(blocks):
        model.add(
            TimeDistributed(Conv2D(filter_base * 2 ** i, kernel_size=kernel_size, padding='same', activation='relu')))
        if pooling:
            model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))
    return model


def build_resor(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = Sequential()

    model.add(Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(Reshape((pic_per_mat, np.prod(mat_size))))
    model.add(ReservoirLayer(neuron_base))
    model.add(TimeDistributed(Dense(np.prod(mat_size), activation='sigmoid')))

    # Reshape to the desired output shape
    model.add(TimeDistributed(Reshape((mat_size[0], mat_size[1], 1))))
    return model
