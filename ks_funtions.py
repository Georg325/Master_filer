from abc import ABC

from keras.models import Sequential
from keras.layers import Dense, GRU, Conv2D, Flatten, MaxPooling2D, Dropout, Reshape, TimeDistributed

import tensorflow as tf
import keras as ks

import numpy as np


class DataGenerator(ks.utils.Sequence, ABC):
    """Generates data for Keras"""
    def __init__(self, mat_obj):
        """Initialization"""
        self.mat_obj = mat_obj

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return 2

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data
        x, y, _ = self.mat_obj.create_matrix_in_list(2*512)

        return np.expand_dims(x, -1), np.expand_dims(y, -1)


def predict_neural_network(model, in_data):
    input_data = np.expand_dims(np.array([matrix for matrix in in_data]), -1)
    return model.predict(input_data)


def build_model(row_len, col_len, filter_base, num_neuron):
    model = Sequential()

    # Apply Conv2D and Flatten to each time step
    model.add(Conv2D(filter_base, kernel_size=(1, 3), padding='same', activation='relu'))
    model.add(Conv2D(filter_base, kernel_size=(3, 1), padding='same', activation='relu'))
    model.add(Conv2D(filter_base*2, kernel_size=(2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filter_base*4, kernel_size=(2, 2), padding='same', activation='tanh'))
    model.add(MaxPooling2D(pool_size=2))

    # Apply SimpleRNN to the output of Conv2D and Flatten
    model.add(TimeDistributed(Flatten()))
    model.add(Dense(row_len * col_len, activation='tanh'))
    model.add(GRU(num_neuron, activation='relu', return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(num_neuron, activation='relu'))
    model.add(Dense(num_neuron, activation='tanh'))

    # Fully connected layer
    model.add(Dense(row_len * col_len, activation='sigmoid'))

    # Reshape to the desired output shape
    model.add(Reshape((row_len, col_len, 1)))
    return model


def custom_loss(y_true, y_pred):
    # Define a mask to identify positions where y_true is 1
    mask = tf.cast(y_true, dtype=tf.bool)

    # Calculate mean squared error
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Penalize predictions less than 0.5 when y_true is 1
    penalty = tf.where(mask, tf.math.square(tf.maximum(0.5 - y_pred, 0)), 0)

    # Combine the mean squared error with the custom penalty
    combined_loss = mse_loss + 500*tf.reduce_mean(penalty)

    return combined_loss
