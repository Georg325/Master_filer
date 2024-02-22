import numpy as np
from keras import models as m
from keras import layers as la
import tensorflow as tf

from resoviarfuntion import ReservoirLayer, ComplexReservoirLayer


def build_model(model_type, parameters):
    model_type = model_type.lower()
    if model_type == 'cnn_rnn':
        return build_cnn_rnn(parameters)
    elif model_type == 'cnn':
        return build_cnn(parameters)
    elif model_type == 'res':
        return build_res(parameters)
    elif model_type == 'dense':
        return la.Dense(parameters)
    elif model_type == 'rnn':
        return build_rnn(parameters)
    elif model_type == 'cnn_lstm':
        return build_cnn_int(parameters)
    elif model_type == 'unet':
        return build_unet(parameters)
    elif model_type == 'cnn_res':
        return build_cnn_res(parameters)
    breakpoint('error')


def build_cnn_rnn(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.TimeDistributed(la.Conv2D(filter_base, kernel_size=(3, 3), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.Conv2D(filter_base * 2, kernel_size=(2, 2), padding='same', activation='tanh')))

    model.add(la.TimeDistributed(la.MaxPooling2D(pool_size=(2, 2), strides=2)))

    model.add(la.TimeDistributed(la.Flatten()))

    model.add(la.LSTM(neuron_base, activation='tanh', return_sequences=True, recurrent_regularizer='l2'))
    model.add(la.LSTM(neuron_base * 2, activation='tanh', return_sequences=True))

    model.add(la.TimeDistributed(la.Dense(mat_size[0] * mat_size[1], activation='sigmoid')))

    # la.Reshape to the desired output shape
    model.add(la.Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def build_cnn_int(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))

    model.add(la.ConvLSTM2D(filter_base * 2, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True))

    model.add(la.ConvLSTM2D(filter_base * 4, kernel_size=(2, 2), padding='same', activation='relu', return_sequences=True))
    model.add(la.ConvLSTM2D(1, kernel_size=(2, 2), padding='same', activation='tanh', return_sequences=True))

    # la.Reshape to the desired output shape
    model.add(la.Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def build_cnn(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.TimeDistributed(la.Conv2D(filter_base, kernel_size=(3, 3), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.Conv2D(filter_base * 2, kernel_size=(3, 3), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.Conv2D(filter_base * 4, kernel_size=(2, 2), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.Conv2D(1, kernel_size=(2, 2), padding='same', activation='tanh')))

    # la.Reshape to the desired output shape
    model.add(la.Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def Dense(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.Reshape((pic_per_mat, np.prod(mat_size))))
    model.add(la.TimeDistributed(la.Dense(64, activation='tanh')))
    model.add(la.TimeDistributed(la.Dense(64*2, activation='tanh')))
    model.add(la.TimeDistributed(la.Dense(np.prod(mat_size), activation='sigmoid')))

    # Reshape to the desired output shape
    model.add(la.TimeDistributed(la.Reshape((mat_size[0], mat_size[1], 1))))
    return model


def build_res(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.Reshape((pic_per_mat, np.prod(mat_size))))
    model.add(ReservoirLayer(750))
    model.add(la.TimeDistributed(la.Dense(np.prod(mat_size), activation='tanh')))

    # Reshape to the desired output shape
    model.add(la.TimeDistributed(la.Reshape((mat_size[0], mat_size[1], 1)), name='jon'))
    return model


def build_cnn_res(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.TimeDistributed(la.Conv2D(filter_base, kernel_size=(2, 2), padding='same', activation='relu')))
    model.add(la.Reshape((pic_per_mat, np.prod(mat_size)*filter_base)))
    model.add(ReservoirLayer(750))
    model.add(la.TimeDistributed(la.Dense(np.prod(mat_size), activation='tanh')))

    # Reshape to the desired output shape
    model.add(la.TimeDistributed(la.Reshape((mat_size[0], mat_size[1], 1)), name='jon'))
    return model


def build_rnn(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.TimeDistributed(la.Flatten()))
    model.add(la.LSTM(neuron_base, activation='tanh', return_sequences=True, recurrent_regularizer='l2'))
    model.add(la.LSTM(neuron_base, activation='tanh', return_sequences=True, recurrent_regularizer='l2'))
    model.add(la.LSTM(neuron_base * 2, activation='tanh', return_sequences=True))

    model.add(la.TimeDistributed(la.Dense(mat_size[0] * mat_size[1], activation='sigmoid')))

    # Reshape to the desired output shape
    model.add(la.Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def build_unet(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    le = 2
    # inputs
    inputs = la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1))
    downlist = [la.TimeDistributed(la.Conv2D(filter_base, kernel_size=(2, 2), activation='relu'))(inputs)]
    for i in range(1, le+1):
        downlist.append(la.TimeDistributed(la.Conv2D(filter_base * 2 ** (i + 1), kernel_size=(2, 2), activation='relu'))(downlist[i - 1]))
    bottle = la.TimeDistributed(la.Conv2D(filter_base * 2 ** (le + 2), kernel_size=(2, 2), activation='relu'))(downlist[-1])

    # decoder: expanding path - up sample
    x = la.TimeDistributed(la.Conv2DTranspose(filter_base * 2 ** (le + 1), kernel_size=(2, 2)))(bottle)

    uplist = [la.concatenate([x, downlist[-1]])]
    for i in range(1, le+1):
        x = la.TimeDistributed(la.Conv2DTranspose(filter_base * 2 ** (le + 1 - i), kernel_size=(2, 2)))(uplist[i-1])
        uplist.append(la.concatenate([x, downlist[le - i]]))

    x = la.TimeDistributed(la.Conv2DTranspose(filter_base, kernel_size=(2, 2)))(uplist[- 1])

    # outputs
    outputs = la.Conv2D(1, 1, padding='same', activation="softmax")(x)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model
