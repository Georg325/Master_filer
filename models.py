import numpy as np
from keras import models as m
from keras import layers as la
import tensorflow as tf

from ml_funtions import ReservoirLayer
from resoviarfuntion import BrainLayer


def build_model(model_type, parameters):
    model_type = model_type.lower()
    if model_type == 'cnn-rnn':
        return build_cnn_rnn(parameters)
    elif model_type == 'cnn':
        return build_cnn(parameters)
    elif model_type == 'res':
        return build_res(parameters)
    elif model_type == 'dense':
        return Dense(parameters)
    elif model_type == 'rnn':
        return build_rnn(parameters)
    elif model_type == 'cnn-lstm':
        return build_cnn_lstm(parameters)
    elif model_type == 'unet':
        return build_unet(parameters)
    elif model_type == 'cnn-res':
        return build_cnn_res(parameters)
    elif model_type == 'unet-rnn':
        return build_unet_rnn(parameters)
    elif model_type == 'res-dense':
        return build_res_dense(parameters)
    elif model_type == 'deep-res':
        return build_deep_res(parameters)
    elif model_type == 'brain':
        return build_brain(parameters)
    elif model_type == 'cnn-brain':
        return build_cnn_brain(parameters)
    breakpoint('error')


def build_cnn(parameters):
    mat_size, cnn_scaling, neuron_base, pic_per_mat = parameters
    filter_base = round(32 * cnn_scaling)
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.TimeDistributed(la.Conv2D(filter_base, kernel_size=(7, 7), activation='relu')))
    model.add(la.TimeDistributed(la.Conv2D(filter_base * 2, kernel_size=(5, 5), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.Conv2D(filter_base * 4, kernel_size=(3, 3), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.Conv2D(filter_base * 4, kernel_size=(2, 2), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.Conv2D(1, kernel_size=(2, 2), padding='same', activation='tanh')))

    # la.Reshape to the desired output shape
    model.add(la.Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def build_cnn_rnn(parameters):
    mat_size, cnn_scaling, rnn_scaling, pic_per_mat = parameters
    filter_base = round(32 * cnn_scaling)
    neuron_base = round(64 * rnn_scaling)
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.TimeDistributed(la.Conv2D(filter_base, kernel_size=(3, 3), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.Conv2D(filter_base * 2, kernel_size=(2, 2), padding='same', activation='tanh')))
    model.add(la.TimeDistributed(la.Conv2D(filter_base, kernel_size=(2, 2), padding='same', activation='tanh')))
    model.add(la.TimeDistributed(la.MaxPooling2D(pool_size=(2, 2), strides=2)))

    model.add(la.TimeDistributed(la.Flatten()))

    model.add(la.LSTM(neuron_base * 2, activation='tanh', return_sequences=True, recurrent_regularizer='l2'))
    model.add(la.LSTM(neuron_base, activation='tanh', return_sequences=True))
    model.add(la.LSTM(neuron_base, activation='tanh', return_sequences=True, recurrent_regularizer='l2'))

    model.add(la.TimeDistributed(la.Dense(mat_size[0] * mat_size[1], activation='sigmoid')))

    # la.Reshape to the desired output shape
    model.add(la.Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def build_cnn_lstm(parameters):
    mat_size, cnn_scaling, rnn_scaling, pic_per_mat = parameters
    filter_ = round(32 * cnn_scaling)
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))

    model.add(la.TimeDistributed(la.Conv2D(filter_, kernel_size=(3, 3), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.Conv2D(filter_ * 2, kernel_size=(3, 3), padding='same', activation='relu')))
    model.add(la.ConvLSTM2D(filter_ * 2, kernel_size=(2, 2), padding='same', activation='relu', return_sequences=True))
    model.add(la.ConvLSTM2D(filter_ * 4, kernel_size=(2, 2), padding='same', activation='relu', return_sequences=True))
    model.add(la.ConvLSTM2D(1, kernel_size=(2, 2), padding='same', activation='tanh', return_sequences=True))

    # la.Reshape to the desired output shape
    model.add(la.Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def build_res_dense(parameters):
    mat_size, cnn_scaling, rnn_scaling, pic_per_mat = parameters
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.Reshape((pic_per_mat, np.prod(mat_size))))
    model.add(ReservoirLayer(500))
    model.add(la.TimeDistributed(la.Dense(500, activation='tanh')))
    model.add(ReservoirLayer(500))
    model.add(la.TimeDistributed(la.Dense(np.prod(mat_size), activation='tanh')))

    # Reshape to the desired output shape
    model.add(la.TimeDistributed(la.Reshape((mat_size[0], mat_size[1], 1))))
    return model


def build_deep_res(parameters):
    mat_size, cnn_scaling, rnn_scaling, pic_per_mat = parameters

    inputs = la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1))
    x = [la.Reshape((pic_per_mat, np.prod(mat_size)))(inputs)]

    for _ in range(3):
        x.append(ReservoirLayer(250)(x[-1]))
    con = la.concatenate(x[1:])
    den = la.TimeDistributed(la.Dense(np.prod(mat_size), activation='tanh'))(con)
    output = la.TimeDistributed(la.Reshape((mat_size[0], mat_size[1], 1)))(den)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model


def Dense(parameters):
    mat_size, cnn_scaling, rnn_scaling, pic_per_mat = parameters

    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.Reshape((pic_per_mat, np.prod(mat_size))))
    model.add(la.TimeDistributed(la.Dense(100, activation='tanh')))
    model.add(la.TimeDistributed(la.Dense(100, activation='tanh')))
    model.add(la.TimeDistributed(la.Dense(np.prod(mat_size), activation='sigmoid')))

    # Reshape to the desired output shape
    model.add(la.TimeDistributed(la.Reshape((mat_size[0], mat_size[1], 1))))
    return model


def build_res(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.Reshape((pic_per_mat, np.prod(mat_size))))
    model.add(ReservoirLayer(700))
    model.add(la.TimeDistributed(la.Dense(np.prod(mat_size), activation='tanh')))

    # Reshape to the desired output shape
    model.add(la.TimeDistributed(la.Reshape((mat_size[0], mat_size[1], 1))))
    return model


def build_cnn_res(parameters):
    mat_size, cnn_scaling, rnn_scaling, pic_per_mat = parameters
    cnn_filter = round(16 * cnn_scaling)
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.TimeDistributed(la.Conv2D(cnn_filter, kernel_size=(2, 2), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.MaxPooling2D(pool_size=(2, 2), strides=2)))
    model.add(la.Reshape((pic_per_mat, np.prod(mat_size) * cnn_filter//4)))
    model.add(ReservoirLayer(550))
    model.add(la.TimeDistributed(la.Dense(np.prod(mat_size), activation='tanh')))

    # Reshape to the desired output shape
    model.add(la.TimeDistributed(la.Reshape((mat_size[0], mat_size[1], 1))))
    return model


def build_cnn_brain(parameters):
    mat_size, cnn_scaling, rnn_scaling, pic_per_mat = parameters
    cnn_filter = round(16 * cnn_scaling)
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.TimeDistributed(la.Conv2D(cnn_filter, kernel_size=(2, 2), padding='same', activation='relu')))
    model.add(la.TimeDistributed(la.MaxPooling2D(pool_size=(2, 2), strides=2)))
    model.add(la.Reshape((pic_per_mat, np.prod(mat_size) * cnn_filter//4)))
    model.add(BrainLayer(550))
    model.add(la.TimeDistributed(la.Dense(np.prod(mat_size), activation='tanh')))

    # Reshape to the desired output shape
    model.add(la.TimeDistributed(la.Reshape((mat_size[0], mat_size[1], 1))))
    return model


def build_rnn(parameters):
    mat_size, cnn_scaling, rnn_scaling, pic_per_mat = parameters
    neuron_base = round(64 * rnn_scaling)
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.TimeDistributed(la.Flatten()))
    model.add(la.LSTM(neuron_base, activation='tanh', return_sequences=True, recurrent_regularizer='l2'))
    model.add(la.LSTM(neuron_base * 2, activation='tanh', return_sequences=True, recurrent_regularizer='l2'))
    model.add(la.LSTM(neuron_base * 4, activation='tanh', return_sequences=True))

    model.add(la.TimeDistributed(la.Dense(mat_size[0] * mat_size[1], activation='sigmoid')))

    # Reshape to the desired output shape
    model.add(la.Reshape((pic_per_mat, mat_size[0], mat_size[1], 1)))
    return model


def build_unet(parameters):
    mat_size, cnn_scaling, rnn_scaling, pic_per_mat = parameters
    filter_base = round(32 * cnn_scaling)
    le = 1
    # inputs
    inputs = la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1))
    down_list = [la.TimeDistributed(la.Conv2D(filter_base, kernel_size=(2, 2), activation='relu'))(inputs)]
    for i in range(1, le + 1):
        down_list.append(
            la.TimeDistributed(la.Conv2D(filter_base * 2 ** (i + 1), kernel_size=(2, 2), activation='relu'))(
                down_list[i - 1]))
    bottle = la.TimeDistributed(la.Conv2D(filter_base * 2 ** (le + 2), kernel_size=(2, 2), activation='relu'))(
        down_list[-1])

    # decoder: expanding path - up sample
    x = la.TimeDistributed(la.Conv2DTranspose(filter_base * 2 ** (le + 1), kernel_size=(2, 2)))(bottle)

    uplist = [la.concatenate([x, down_list[-1]])]
    for i in range(1, le + 1):
        x = la.TimeDistributed(la.Conv2DTranspose(filter_base * 2 ** (le + 1 - i), kernel_size=(2, 2)))(uplist[i - 1])
        uplist.append(la.concatenate([x, down_list[le - i]]))

    x = la.TimeDistributed(la.Conv2DTranspose(filter_base, kernel_size=(2, 2)))(uplist[- 1])

    # outputs
    outputs = la.Conv2D(1, 1, padding='same', activation="sigmoid")(x)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


def build_unet_rnn(parameters):
    mat_size, cnn_scaling, rnn_scaling, pic_per_mat = parameters
    filter_base = round(32 * cnn_scaling)
    le = 1
    # inputs
    inputs = la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1))
    down_list = [la.TimeDistributed(la.Conv2D(filter_base, kernel_size=(2, 2), activation='relu'))(inputs)]
    for i in range(1, le + 1):
        down_list.append(
            la.TimeDistributed(la.Conv2D(filter_base * 2 ** (i + 1), kernel_size=(2, 2), activation='relu'))(
                down_list[i - 1]))
    bottle = la.ConvLSTM2D(filter_base * 2 ** (le + 2), kernel_size=(2, 2), activation='relu', return_sequences=True)(
        down_list[-1])

    # decoder: expanding path - up sample
    x = la.TimeDistributed(la.Conv2DTranspose(filter_base * 2 ** (le + 1), kernel_size=(2, 2)))(bottle)

    uplist = [la.concatenate([x, down_list[-1]])]
    for i in range(1, le + 1):
        x = la.TimeDistributed(la.Conv2DTranspose(filter_base * 2 ** (le + 1 - i), kernel_size=(2, 2)))(uplist[i - 1])
        uplist.append(la.concatenate([x, down_list[le - i]]))

    x = la.TimeDistributed(la.Conv2DTranspose(filter_base, kernel_size=(2, 2)))(uplist[- 1])

    # outputs
    outputs = la.Conv2D(1, 1, padding='same', activation="sigmoid")(x)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


def build_brain(parameters):
    mat_size, filter_base, neuron_base, pic_per_mat = parameters
    model = m.Sequential()

    model.add(la.Input(shape=(pic_per_mat, mat_size[0], mat_size[1], 1)))
    model.add(la.Reshape((pic_per_mat, np.prod(mat_size))))
    model.add(BrainLayer(700, make_weights=False))
    model.add(la.TimeDistributed(la.Dense(np.prod(mat_size), activation='tanh')))

    # Reshape to the desired output shape
    model.add(la.TimeDistributed(la.Reshape((mat_size[0], mat_size[1], 1))))
    return model
