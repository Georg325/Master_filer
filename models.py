import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, TimeDistributed, Input, GRU, LSTM

from resoviarfuntion import ReservoirLayer, ComplexReservoirLayer


def build_model(model_type, parameters):
    if model_type == 'cnn_rnn':
        return build_cnn_rnn(parameters)
    elif model_type == 'cnn':
        return build_cnn(parameters)
    elif model_type == 'res':
        return build_res(parameters)
    elif model_type == 'dense':
        return dense(parameters)
    print('error')


def build_cnn_rnn(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = Sequential()

    model.add(Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(TimeDistributed(Conv2D(filter_base, kernel_size=(3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(filter_base * 2, kernel_size=(2, 2), padding='same', activation='tanh')))

    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(neuron_base, activation='tanh', return_sequences=True, recurrent_regularizer='l2'))
    model.add(LSTM(neuron_base * 2, activation='tanh', return_sequences=True))

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


def dense(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = Sequential()

    model.add(Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(Reshape((pic_per_mat, np.prod(mat_size))))
    model.add(TimeDistributed(Dense(neuron_base, activation='linear')))
    model.add(TimeDistributed(Dense(np.prod(mat_size), activation='sigmoid')))

    # Reshape to the desired output shape
    model.add(TimeDistributed(Reshape((mat_size[0], mat_size[1], 1))))
    return model


def build_res(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = Sequential()

    model.add(Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(Reshape((pic_per_mat, np.prod(mat_size))))
    model.add(ReservoirLayer(750))
    model.add(TimeDistributed(Dense(np.prod(mat_size), activation='tanh')))

    # Reshape to the desired output shape
    model.add(TimeDistributed(Reshape((mat_size[0], mat_size[1], 1)), name='jon'))
    return model


def cnn_block(model, parameters, kernel_size, blocks=1, pooling=False):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    for i in range(blocks):
        model.add(
            TimeDistributed(Conv2D(filter_base * 2 ** i, kernel_size=kernel_size, padding='same', activation='relu')))
        if pooling:
            model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))
    return model
