from os import path

import numpy as np
import scipy as sp
import pandas as pd
import random as rd

from tensorflow.keras.metrics import Precision, Recall, BinaryIoU

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from models import build_model

from ks_funtions import *
import raster_geometry as rg

rng = rd.SystemRandom()


class MatrixLister:
    def __init__(self, mat_size, strength_kernel, min_max_line_size,
                 rotate, fades_per_mat, new_background, shape):

        self.mat_size = mat_size
        self.strength, self.kernel_size = strength_kernel
        self.min_max_line_size = min_max_line_size
        self.rotate = rotate
        self.fades_per_mat = fades_per_mat
        self.new_background = new_background
        self.shape = shape

        self.con_matrix, self.line_pos_mat, self.con_alfa = None, None, None
        self.kernel = None
        self.model_type = None
        self.alternative = False
        self.ani = None

        self.scores = []

        self.randomize_kernel((self.strength, self.strength))

    @classmethod
    def from_dict(cls, params):
        return cls(**params)

    def generate_pred_data(self, model, num_to_pred, concatenate=True):
        self.con_matrix, self.line_pos_mat, self.con_alfa = self.create_matrix_in_list(num_to_pred)

        input_matrix = np.array(self.con_matrix[:num_to_pred * self.fades_per_mat])
        true_matrix = self.line_pos_mat[:num_to_pred * self.fades_per_mat]

        input_data = np.expand_dims(np.array([matrix for matrix in input_matrix]), -1)
        true_matrix = np.expand_dims(np.array([matrix for matrix in true_matrix]), -1)
        pred = model.predict(input_data)

        if concatenate:
            input_matrix = np.concatenate(input_matrix, axis=0)
            true_matrix = np.concatenate(true_matrix, axis=0)
            pred = np.concatenate(pred, axis=0)

            predicted_line_pos_mat = np.array(pred).reshape(input_matrix.shape)

            return input_matrix, true_matrix, predicted_line_pos_mat
        else:
            return np.array(input_matrix), np.array(true_matrix), np.array(pred)

    def create_matrix_in_list(self, numb_of_time_series):
        list_matrix = []
        list_pos_mat = []
        list_alfa = []

        for k in range(0, numb_of_time_series):
            if self.shape == 'triangle':
                sfb = matrix_triangle_maker(self.mat_size, self.kernel,
                                            self.fades_per_mat,
                                            new_background=self.new_background)

            elif self.shape == 'face':
                sfb = matrix_triangle_maker(self.mat_size, self.kernel,
                                            self.fades_per_mat,
                                            new_background=self.new_background, alternative=True)

            elif self.shape == 'line':
                if self.rotate:  # makes the line
                    line_size = rotater((
                        rng.randint(self.min_max_line_size[0][0], self.min_max_line_size[1][0]),
                        rng.randint(self.min_max_line_size[0][1], self.min_max_line_size[1][1])))

                else:  # does not rotate the line
                    line_size = (
                        rng.randint(self.min_max_line_size[0][0], self.min_max_line_size[1][0]),
                        rng.randint(self.min_max_line_size[0][1], self.min_max_line_size[1][1]))

                sfb = matrix_maker(self.mat_size, self.kernel, line_size,
                                   self.fades_per_mat,
                                   new_background=self.new_background)

            else:
                print(f'{self.shape} is invalid')
                print('Try triangle, face or line')
                return

            mat, pos, alf = sfb
            list_matrix.append(mat)
            list_pos_mat.append(pos)
            list_alfa.append(alf)

        return tf.convert_to_tensor(list_matrix), list_pos_mat, list_alfa

    def plot_matrices(self, model, num_to_pred, interval=500):
        input_matrix, true_matrix, predicted_line_pos_mat = self.generate_pred_data(model, num_to_pred)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

        def update(frame):
            # Plot Input Matrix
            im = [axes[0].imshow(input_matrix[frame], interpolation='nearest', aspect='auto', vmin=0, vmax=1)]
            axes[0].set_title('Input Matrix')
            axes[0].grid(True, color='white', linestyle='-', linewidth=0.5)  # White grid

            # Plot True Line Position Matrix
            im.append(axes[1].imshow(true_matrix[frame], interpolation='nearest', aspect='auto', vmin=0, vmax=1))
            axes[1].set_title('True Line Position Matrix')
            axes[1].grid(True, color='white', linestyle='-', linewidth=0.5)  # White grid
            # Plot Predicted Line Position Matrix
            im.append(
                axes[2].imshow(predicted_line_pos_mat[frame], interpolation='nearest', aspect='auto', vmin=0, vmax=1))
            axes[2].set_title('Predicted Line Position Matrix')
            axes[2].grid(True, color='white', linestyle='-', linewidth=0.5)  # White grid

            return im

        animation = FuncAnimation(fig, update, frames=len(input_matrix), interval=interval, repeat=False, blit=True)

        plt.tight_layout()
        return animation

    def display_frames(self, model, num_frames=1000, num_to_pred=1):

        for _ in range(num_to_pred):
            num_frames = max(min(self.fades_per_mat, num_frames), 2)
            input_matrix, true_matrix, predicted_line_pos_mat = self.generate_pred_data(model, 1)
            fig, axes = plt.subplots(num_frames, 3)  # num_frames rows, 3 columns
            im = []
            for i in range(num_frames):
                # Plot Combined Matrix (True Matrix overlaid on Predicted Matrix)
                im.append(axes[i, 0].imshow(input_matrix[i], interpolation='nearest', aspect='auto', vmin=0, vmax=1))
                if i == 0:
                    axes[i, 0].set_title('Input Matrix')  # Title on the first plot only
                axes[i, 0].set_xticks([])
                axes[i, 0].set_yticks([])
                axes[i, 0].set_xlabel('')
                axes[i, 0].set_ylabel('')

                im.append(axes[i, 1].imshow(true_matrix[i], interpolation='nearest', aspect='auto', vmin=0, vmax=1))
                if i == 0:
                    axes[i, 1].set_title('True line')  # Title on the first plot only
                axes[i, 1].set_xticks([])
                axes[i, 1].set_yticks([])
                axes[i, 1].set_xlabel('')
                axes[i, 1].set_ylabel('')

                # Plot Predicted Line Position Matrix
                im.append(axes[i, 2].imshow(predicted_line_pos_mat[i], interpolation='nearest', aspect='auto', vmin=0,
                                            vmax=1))
                if i == 0:
                    axes[i, 2].set_title('Predicted Line Position Matrix')
                axes[i, 2].set_xticks([])
                axes[i, 2].set_yticks([])
                axes[i, 2].set_xlabel('')
                axes[i, 2].set_ylabel('')

            plt.tight_layout(pad=0.1)
            plt.show(block=False)
        plt.show()

    def unique_lines(self):
        unique_lines = 0
        for i in range(self.min_max_line_size[0][0], self.min_max_line_size[1][0] + 1):
            for j in range(self.min_max_line_size[0][1], self.min_max_line_size[1][1] + 1):
                possible_row = self.mat_size[0] - i + 1
                possible_col = self.mat_size[1] - j + 1
                unique_lines += possible_row * possible_col
                if self.rotate:
                    possible_row_r = self.mat_size[0] - j + 1
                    possible_col_r = self.mat_size[1] - i + 1
                    unique_lines += possible_row_r * possible_col_r

        print('Possible lines: ', unique_lines)

    def init_model(self, cnn_size, rnn_size, model_type='cnn_gru', alternative=False, threshold=None):
        print(self.unique_lines())
        self.alternative = alternative
        if threshold is not None:
            self.alternative = True

        self.model_type = model_type
        checkpoint_filepath = 'standard.weights.h5'
        model_checkpoint_callback = ks.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                                 monitor='loss', mode='min', save_best_only=True)

        model_early_stopp_callback = ks.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.001,
                                                                restore_best_weights=True, start_from_epoch=20)

        parameters = self.mat_size, cnn_size, rnn_size, self.fades_per_mat

        model = build_model(model_type, parameters)
        optimizer = ks.optimizers.Adam()
        if alternative:
            metrics = [IoU_Maker(n, threshold) for n in range(1, 10)]
            model.compile(optimizer=optimizer, loss=custom_weighted_loss,
                          metrics=[metrics, Precision(name='precision'), Recall(name='recall')])
        else:

            model.compile(optimizer=optimizer, loss=custom_weighted_loss,
                          metrics=[BinaryIoU(name='IoU'), Precision(name='precision'), Recall(name='recall')])

        return model, [model_checkpoint_callback, model_early_stopp_callback]

    def init_generator(self, model, batch_size, num_batch):
        return DataGenerator(self, model, batch_size, num_batch, self.alternative)

    def load_model(self, model, weights_shape='auto'):

        if weights_shape == 'auto':
            if path.exists(f'{self.mat_size}{self.model_type}triangle.weights.h5') and self.shape == 'triangle':
                try:
                    model.load_weights(f'{self.mat_size}{self.model_type}triangle.weights.h5')
                    print('Loaded triangle weights')
                except ValueError:
                    print('Could not load triangle weights')

            elif path.exists(f'{self.mat_size}{self.model_type}line.weights.h5') and self.shape == 'line':
                try:
                    model.load_weights(f'{self.mat_size}{self.model_type}line.weights.h5')
                    print('Loaded line weights')
                except ValueError:
                    print('Could not load line weights')
            else:
                print('Did not find any weights')

        elif weights_shape == 'triangle':
            model.load_weights(f'{self.mat_size}{self.model_type}triangle.weights.h5')
            print('Loaded triangle weights')
        elif weights_shape == 'line':
            model.load_weights(f'{self.mat_size}{self.model_type}line.weights.h5')
            print('Loaded line weights')

    def save_model(self, model, weights_shape='auto'):

        if weights_shape == 'auto':

            if self.shape == 'triangle':
                model.save_weights(f'{self.mat_size}{self.model_type}triangle.weights.h5')
                print('Saved triangle weights')

            elif self.shape == 'line':
                model.save_weights(f'{self.mat_size}{self.model_type}line.weights.h5')
                print('Saved line weights')
            else:
                print(f'error {self.shape} not recognized')

        elif weights_shape == 'triangle':
            model.save_weights(f'{self.mat_size}{self.model_type}triangle.weights.h5')
            print('Saved triangle weights')

        elif weights_shape == 'line':
            model.save_weights(f'{self.mat_size}{self.model_type}line.weights.h5')
            print('Saved line weights')

    def randomize_kernel(self, strength_range):
        strength = np.random.uniform(low=strength_range[0], high=strength_range[1])
        self.kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * strength ** 2)) * np.exp(
                -((x - (self.kernel_size - 1) / 2) ** 2 +
                  (y - (self.kernel_size - 1) / 2) ** 2) / (2 * strength ** 2)),
            (self.kernel_size, self.kernel_size))
        self.kernel = self.kernel / np.sum(self.kernel)

    def eval(self, model, batch):
        input_matrix, true_matrix, predicted_line_pos_mat = self.generate_pred_data(model, batch, False)
        scores = custom_iou(true_matrix, predicted_line_pos_mat)
        self.scores.append(scores)
        return scores

    def clear_score(self):
        self.scores = []

    def plot_scores(self, all_f1_scores):
        if self.alternative:
            return
        scores = np.transpose(np.array(all_f1_scores))
        plt.figure(figsize=(10, 6))

        for i, iou_scores in enumerate(scores):
            plt.plot(iou_scores, label=f"Frame {i}")

        plt.title(f"IoU Scores for {self.model_type} on {self.mat_size} grid")
        plt.xlabel('epoch')
        plt.ylabel('IoU Score')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(scores[-1])

    def plot_training_history(self, training_history_object, list_of_metrics=None):
        """
        Input:
            training_history_object:: Object returned by model.fit() function in keras
            list_of_metrics        :: A list of metrics to be plotted. Use if you only
                                      want to plot a subset of the total set of metrics
                                      in the training history object. By default, it will
                                      plot all of them in individual subplots.
        """
        history_dict = training_history_object.history

        iou_s = []
        preps = []
        other = []
        if list_of_metrics is None:
            list_of_metrics = [key for key in list(history_dict.keys())]
            for metric in list_of_metrics:
                if 'IoU' in metric:
                    iou_s.append(metric)
                elif 'precision' in metric or 'recall' in metric:
                    preps.append(metric)
                else:
                    other.append(metric)

        train_hist_df = pd.DataFrame(history_dict)
        train_keys = other
        nr_plots = len(other)

        if len(iou_s) > 0:
            nr_plots += 1
        if len(preps) > 0:
            nr_plots += 1

        fig, ax = plt.subplots(1, nr_plots, figsize=(5 * nr_plots, 6))

        plt_nr = 0
        done = False

        for i in range(len(other)):
            ax[plt_nr].plot(np.array(train_hist_df[train_keys[plt_nr]]), label=train_keys[plt_nr])
            ax[plt_nr].set_ylim(0, 1)
            ax[plt_nr].set_xlabel('Epoch')
            ax[plt_nr].set_title(train_keys[plt_nr])
            ax[plt_nr].grid('on')
            ax[plt_nr].legend()
            plt_nr += 1

        for k in range(len(preps)):
            done = True
            ax[plt_nr].plot(np.array(train_hist_df[preps[k]]), label=preps[k])
            ax[plt_nr].set_ylim(0, 1)
            ax[plt_nr].set_xlabel('Epoch')
            ax[plt_nr].set_title('Precision and Recall')
            ax[plt_nr].grid('on')
            ax[plt_nr].legend()
        if done:
            plt_nr += 1
            done = False

        for k in range(len(iou_s)):
            ax[plt_nr].plot(np.array(train_hist_df[iou_s[k]]), label=f'Frame {k + 1}')
            ax[plt_nr].set_ylim(0, 1)
            ax[plt_nr].set_xlabel('Epoch')
            ax[plt_nr].set_title('IoU')
            ax[plt_nr].grid('on')
            ax[plt_nr].legend()
        if done:
            plt_nr += 1

        title = f'Metrics from the '

        if not self.rotate:
            title += 'non rotated, '
        if not self.new_background:
            title += 'static background, '

        title += f'{self.shape}, {self.model_type} model on {self.mat_size} matrix'
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()

    def after_training_metrics(self, model, hist=None, epochs=0, movies_to_plot=0, frames_to_show=1000,
                               movies_to_show=0, interval=500):
        if self.alternative is not True:
            self.plot_scores(self.scores)

        if movies_to_plot > 0:
            self.display_frames(model, num_frames=frames_to_show, num_to_pred=movies_to_plot)

        if movies_to_show > 0:
            self.ani = self.plot_matrices(model, num_to_pred=movies_to_show, interval=interval)
            plt.show()

        if hist is not None and epochs != 0:
            self.plot_training_history(hist)


def matrix_maker(mat_size, kernel, line_size=(1, 2), num_per_mat=3, new_background=False):
    rows, cols = mat_size

    # smooth
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
            smooth_matrix = sp.ndimage.convolve(np.random.rand(rows, cols), kernel)

        matrix_with_line = np.ones((rows, cols))
        matrix_with_line[line_start_position[0]:line_start_position[0] + line_size[0],
        line_start_position[1]:line_start_position[1] + line_size[1]] \
            = alfa[i]

        '''if alfa[i] < 0.5:
            matrix_with_line = np.ones((rows, cols))
            matrix_with_line[line_start_position[0]:line_start_position[0] + line_size[0],
            line_start_position[1]:line_start_position[1] + line_size[1]] = alfa[i]'''

        matrix_line_fade.append(smooth_matrix * matrix_with_line)

        if alfa[i] == 1:
            line_pos_mat.append(np.zeros((rows, cols)))

        else:
            matrix_with_line[line_start_position[0]:line_start_position[0] + line_size[0],
            line_start_position[1]:line_start_position[1] + line_size[1]] = 0

            line_pos_mat.append(np.logical_not(matrix_with_line).astype(int))

    return tf.convert_to_tensor(matrix_line_fade), tf.convert_to_tensor(line_pos_mat), tf.convert_to_tensor(alfa)


def matrix_triangle_maker(mat_size, kernel, num_per_mat=3, new_background=False, alternative=False):
    rows, cols = mat_size

    # smooth
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
            smooth_matrix = sp.ndimage.convolve(np.random.rand(rows, cols), kernel)

        matrix_with_line = np.ones((rows, cols))
        fish = arr * (1 - alfa[i])

        matrix_line_fade.append(smooth_matrix * (matrix_with_line - fish))

        if alfa[i] == 1:
            line_pos_mat.append(np.zeros((rows, cols)))

        else:
            line_pos_mat.append(arr)

    return tf.convert_to_tensor(matrix_line_fade), tf.convert_to_tensor(line_pos_mat), tf.convert_to_tensor(alfa)


def rotater(line):
    if rng.randint(0, 1):
        return line[::-1]
    return line


def plots(matrix, interval=200):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        im = ax.imshow(matrix[frame], interpolation='nearest', aspect='auto')

        return [im]

    animation = FuncAnimation(fig, update, frames=len(matrix), interval=interval, repeat=False, blit=True)
    plt.tight_layout()
    plt.show(block=False)
    plt.show()
    return animation


def full_triangle(a, b, c):
    ab = rg.bresenham_line(a, b, endpoint=True)
    for x in set(ab):
        yield from rg.bresenham_line(c, x, endpoint=True)
