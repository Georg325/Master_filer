from abc import ABC

from keras.models import Sequential
from keras.layers import Dense, GRU, Conv2D, Flatten, MaxPooling2D, Dropout, Reshape, TimeDistributed, Input, LSTM

import tensorflow as tf
from keras import backend as K
import keras as ks

import numpy as np


class DataGenerator(ks.utils.Sequence, ABC):
    """Generates data for Keras"""

    def __init__(self, mat_obj, batch_size=32, fades_per_mat=8, num_of_mat=100):
        """Initialization"""
        self.mat_obj = mat_obj
        self.batch_size = batch_size
        self.fades_per_mat = fades_per_mat
        self.num_of_mat = num_of_mat

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return round(self.num_of_mat * self.fades_per_mat / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data
        x, y, _ = self.mat_obj.create_matrix_in_list(self.num_of_mat)

        return np.expand_dims(x, -1), np.expand_dims(y, -1)


def predict_neural_network(model, in_data):
    input_data = np.expand_dims(np.array([matrix for matrix in in_data]), -1)
    return model.predict(input_data)


def build_model(row_len, col_len, filter_base, pic_per_mat):
    model = Sequential()

    model.add(Input(shape=(pic_per_mat, row_len, col_len, 1)))
    model.add(TimeDistributed(Conv2D(filter_base, kernel_size=(3, 3), padding='same', activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))

    model.add(TimeDistributed(Conv2D(filter_base * 2, kernel_size=(2, 2), padding='same', activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))

    model.add(TimeDistributed(Conv2D(filter_base * 2, kernel_size=(2, 2), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu'))

    # Fully connected layer
    model.add(Dense(pic_per_mat * row_len * col_len, activation='sigmoid'))

    # Reshape to the desired output shape
    model.add(Reshape((pic_per_mat, row_len, col_len, 1)))
    return model


def custom_loss(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    # Calculate 1 - Precision as the loss
    loss_1 = 1 - precision

    # Define a mask to identify positions where y_true is 1
    mask = tf.cast(y_true, dtype=tf.bool)
    penalty = tf.where(mask, tf.math.square(tf.maximum(0.5 - y_pred, 0)), 0)

    # Calculate mean squared error
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Combine the mean squared error with the custom penalty
    combined_loss = mse_loss + 500 * tf.reduce_mean(penalty) + 10*loss_1

    return combined_loss


def custom_weighted_loss(y_true, y_pred):
    """
    Custom loss function with emphasis on errors for values that are 1 in y_true.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels

    Returns:
    - Weighted binary cross-entropy loss
    """
    # Calculate fill factor (proportion of 1s in y_true)
    fill_factor = K.sum(y_true) / K.cast(K.prod(K.shape(y_true)), K.floatx())

    # Calculate binary cross-entropy loss
    binary_loss = K.binary_crossentropy(y_true, y_pred)

    # Apply weights based on fill factor
    weighted_binary_loss = y_true * (fill_factor * binary_loss) + (1 - y_true) * binary_loss

    # Calculate mean loss over all elements
    loss = K.mean(weighted_binary_loss)

    return loss
