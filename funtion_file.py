import numpy as np
import scipy as sp
import pandas as pd
import random as rd

import tensorflow as tf
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ks_funtions import predict_neural_network

rng = rd.SystemRandom()


class MatrixLister:
    def __init__(self, row_len, col_len, kernel_size, min_max_line_size,
                 rotate, num_of_mat, num_per_mat, new_background, triangle):
        self.row_len = row_len
        self.col_len = col_len
        self.kernel_size = kernel_size
        self.min_max_line_size = min_max_line_size
        self.rotate = rotate
        self.num_of_mat = num_of_mat
        self.num_per_mat = num_per_mat
        self.new_background = new_background
        self.triangle = triangle

        self.con_matrix, self.line_pos_mat, self.con_alfa = None, None, None

    def create_matrix_in_list(self, numb_of_time_series):
        list_matrix = []
        list_pos_mat = []
        list_alfa = []

        for k in range(0, numb_of_time_series):
            line_size = rotater((
                rng.randint(self.min_max_line_size[0][0], self.min_max_line_size[1][0]),
                rng.randint(self.min_max_line_size[0][1], self.min_max_line_size[1][1])
            ))
            if self.triangle:
                sfb = matrix_triangle_maker(self.row_len, self.col_len, self.kernel_size, line_size,
                                            self.num_per_mat,
                                            new_background=self.new_background)
            else:
                sfb = matrix_maker(self.row_len, self.col_len, self.kernel_size, line_size,
                                   self.num_per_mat,
                                   new_background=self.new_background)
            mat, pos, alf = sfb
            list_matrix.append(mat)
            list_pos_mat.append(pos)
            list_alfa.append(alf)

        return tf.convert_to_tensor(list_matrix), list_pos_mat, list_alfa

    def plot_matrices(self, model, num_to_pred, interval=500):

        self.num_of_mat = num_to_pred
        self.con_matrix, self.line_pos_mat, self.con_alfa = self.create_matrix_in_list(
            num_to_pred * self.num_per_mat)

        input_matrix = np.array(self.con_matrix[:num_to_pred * self.num_per_mat])
        true_matrix = self.line_pos_mat[:num_to_pred * self.num_per_mat]

        pred = predict_neural_network(model, input_matrix)

        input_matrix = np.concatenate(input_matrix, axis=0)
        true_matrix = np.concatenate(true_matrix, axis=0)
        pred = np.concatenate(pred, axis=0)

        predicted_line_pos_mat = np.array(pred).reshape(input_matrix.shape)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

        def update(frame):
            # Plot Input Matrix
            im = [axes[0].imshow(input_matrix[frame], interpolation='nearest', aspect='auto', vmin=0, vmax=1)]
            axes[0].set_title('Input Matrix')

            # Plot True Line Position Matrix
            im.append(axes[1].imshow(true_matrix[frame], interpolation='nearest', aspect='auto', vmin=0, vmax=1))
            axes[1].set_title('True Line Position Matrix')

            # Plot Predicted Line Position Matrix
            im.append(
                axes[2].imshow(predicted_line_pos_mat[frame], interpolation='nearest', aspect='auto', vmin=0, vmax=1))
            axes[2].set_title('Predicted Line Position Matrix')

            return im

        animation = FuncAnimation(fig, update, frames=len(input_matrix), interval=interval, repeat=False, blit=True)

        plt.tight_layout()
        return animation

    def unique_lines(self):
        unique_lines = 0
        for i in range(self.min_max_line_size[0][0], self.min_max_line_size[1][0] + 1):
            for j in range(self.min_max_line_size[0][1], self.min_max_line_size[1][1] + 1):
                possible_row = self.row_len - i
                possible_col = self.col_len - j
                unique_lines += possible_row * possible_col

                possible_row_r = self.row_len - j
                possible_col_r = self.col_len - i
                unique_lines += possible_row_r * possible_col_r

        print('Possible lines: ', unique_lines)


def matrix_maker(rows, cols=None, kernel_size=(2, 2), line_size=(1, 2), num_per_mat=3, new_background=False):
    cols = cols or rows

    # smooth
    kernel = np.ones(shape=kernel_size, dtype=float) / np.prod(kernel_size)
    smooth_matrix = sp.ndimage.convolve(np.random.rand(rows, cols), kernel)

    # line_start
    line_start_position = (rng.randint(0, rows - line_size[0]),
                           rng.randint(0, cols - line_size[1]))

    # alfa
    alfa = np.linspace(1, 0, num=num_per_mat)

    # matrix_fade
    matrix_line_fade = []
    line_pos_mat = []

    for i in range(num_per_mat):
        if new_background:
            kernel = np.ones(shape=kernel_size, dtype=float) / np.prod(kernel_size)
            smooth_matrix = sp.ndimage.convolve(np.random.rand(rows, cols), kernel)

        matrix_with_line = np.ones((rows, cols))
        matrix_with_line[line_start_position[0]:line_start_position[0] + line_size[0],
        line_start_position[1]:line_start_position[1] + line_size[1]] = alfa[i]

        matrix_line_fade.append(smooth_matrix * matrix_with_line)

        if alfa[i] == 1:
            line_pos_mat.append(np.zeros((rows, cols)))

        else:
            matrix_with_line[line_start_position[0]:line_start_position[0] + line_size[0],
            line_start_position[1]:line_start_position[1] + line_size[1]] = 0

            line_pos_mat.append(np.logical_not(matrix_with_line).astype(int))

    return tf.convert_to_tensor(matrix_line_fade), tf.convert_to_tensor(line_pos_mat), tf.convert_to_tensor(alfa)


def matrix_triangle_maker(rows, cols=None, kernel_size=(2, 2), line_size=(1, 2), num_per_mat=3, new_background=False,
                          alternative=False):
    cols = cols or rows

    # smooth
    kernel = np.ones(shape=kernel_size, dtype=float) / np.prod(kernel_size)
    smooth_matrix = sp.ndimage.convolve(np.random.rand(rows, cols), kernel)

    # line_start

    a, b, c = ((rng.randint(0, rows - 1), rng.randint(0, cols - 1)),
               (rng.randint(0, rows - 1), rng.randint(0, cols - 1)),
               (rng.randint(0, rows - 1), rng.randint(0, cols - 1)))
    coords = set(full_triangle(a, b, c))
    arr = np.array(rg.render_at((rows, cols), coords).astype(int))
    if alternative:
        arr = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                        # [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    # alfa
    alfa = np.linspace(1, 0, num=num_per_mat)

    # matrix_fade
    matrix_line_fade = []
    line_pos_mat = []

    for i in range(num_per_mat):
        if new_background:
            kernel = np.ones(shape=kernel_size, dtype=float) / np.prod(kernel_size)
            smooth_matrix = sp.ndimage.convolve(np.random.rand(rows, cols), kernel)

        matrix_with_line = np.ones((rows, cols))
        fish = arr * (1 - alfa[i])

        matrix_line_fade.append(smooth_matrix * (matrix_with_line - fish))

        if alfa[i] == 1:
            line_pos_mat.append(np.zeros((rows, cols)))

        else:
            line_pos_mat.append(arr)

    return tf.convert_to_tensor(matrix_line_fade), tf.convert_to_tensor(line_pos_mat), tf.convert_to_tensor(alfa)


def F1_score(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (P + K.epsilon())

    Pred_P = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (Pred_P + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def plot_training_history(training_history_object, list_of_metrics=None, with_val=True):
    """
    Input:
        training_history_object:: Object returned by model.fit() function in keras
        list_of_metrics        :: A list of metrics to be plotted. Use if you only
                                  want to plot a subset of the total set of metrics
                                  in the training history object. By Default it will
                                  plot all of them in individual subplots.
    """

    valid_keys = None
    history_dict = training_history_object.history

    if list_of_metrics is None:
        list_of_metrics = [key for key in list(history_dict.keys()) if 'val_' not in key]

    trainHistDF = pd.DataFrame(history_dict)
    # trainHistDF.head()
    train_keys = list_of_metrics

    if with_val:
        valid_keys = ['val_' + key for key in train_keys]

    nr_plots = len(train_keys)
    fig, ax = plt.subplots(1, nr_plots, figsize=(5 * nr_plots, 4))

    for i in range(len(train_keys)):
        ax[i].plot(np.array(trainHistDF[train_keys[i]]), label='Training')

        if with_val:
            ax[i].plot(np.array(trainHistDF[valid_keys[i]]), label='Validation')

        ax[i].set_xlabel('Epoch')
        ax[i].set_title(train_keys[i])
        ax[i].grid('on')
        ax[i].legend()

    fig.tight_layout()
    plt.show()


def rotater(line):
    if rng.randint(0, 1):
        return line[::-1]
    return line


def plots(matrix, interval=200):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        im = ax.imshow(matrix[frame], interpolation='nearest', aspect='auto', vmin=0, vmax=1)

        return [im]

    animation = FuncAnimation(fig, update, frames=len(matrix), interval=interval, repeat=False, blit=True)
    plt.tight_layout()
    plt.show(block=False)
    plt.show()
    return animation


import raster_geometry as rg


def full_triangle(a, b, c):
    ab = rg.bresenham_line(a, b, endpoint=True)
    for x in set(ab):
        yield from rg.bresenham_line(c, x, endpoint=True)
