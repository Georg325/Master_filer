from abc import ABC

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, TimeDistributed, Input, LSTM

import tensorflow as tf
from keras import backend as k_back
import keras as ks

import numpy as np


class DataGenerator(ks.utils.Sequence, ABC):
    """Generates data for Keras"""

    def __init__(self, mat_obj, tot_epoch, batch_size=128, num_batch=2):
        """Initialization"""
        self.mat_obj = mat_obj
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.tot_epoch = tot_epoch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.num_batch

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data
        x, y, _ = self.mat_obj.create_matrix_in_list(self.batch_size)
        return np.expand_dims(x, -1), np.expand_dims(y, -1)


def predict_neural_network(model, in_data):
    input_data = np.expand_dims(np.array([matrix for matrix in in_data]), -1)
    return model.predict(input_data)


def build_model(mat_size, filter_base, neuron_base, pic_per_mat):
    model = Sequential()

    model.add(Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(TimeDistributed(Conv2D(filter_base, kernel_size=(3, 3), padding='same', activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))

    model.add(TimeDistributed(Conv2D(filter_base * 2, kernel_size=(2, 2), padding='same', activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))

    model.add(TimeDistributed(Conv2D(filter_base * 2, kernel_size=(2, 2), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(neuron_base, activation='relu', return_sequences=True))
    model.add(LSTM(neuron_base, activation='relu', return_sequences=True))

    # Fully connected layer
    model.add(TimeDistributed(Dense(mat_size[0]*mat_size[1], activation='sigmoid')))

    # Reshape to the desired output shape
    model.add(Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def custom_weighted_loss(y_true, y_pred, weight_factor=5.0):
    """
    Custom loss function with emphasis on errors for values that are 1 in y_true using mean squared error.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - weight_factor: Weighting factor for positive class (default is 2.0)

    Returns:
    - Weighted mean squared error loss
    """

    # Calculate squared errors
    squared_errors = k_back.square(tf.cast(y_true, dtype=tf.float32) - y_pred)

    # Apply weights to positive class
    weighted_squared_errors = (tf.cast(y_true, dtype=tf.float32) * (weight_factor * squared_errors)
                               + tf.cast((1 - y_true), dtype=tf.float32) * squared_errors)

    # Calculate mean loss over all elements
    loss = k_back.mean(weighted_squared_errors)

    return loss
