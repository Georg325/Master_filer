import os
import time
from ml_funtions import *

import numpy as np
import scipy as sp
import pandas as pd
import random as rd

from tensorflow.keras.metrics import Precision, Recall, BinaryIoU

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from models import build_model

rng = rd.SystemRandom()


class MovieDataHandler:
    def __init__(self, mat_size, fades_per_mat, strength_kernel, size, rotate, new_background, shape,
                 val, val_strength_kernel=None, val_size=None, val_rotate=0,
                 val_new_background=0, val_shape=0, subset=False):

        self.mat_size = mat_size
        self.fades_per_mat = fades_per_mat

        self.size = size
        self.rotate = rotate
        self.new_background = new_background
        self.shape = shape

        self.kernel = set_kernel(strength_kernel)

        self.val = val
        self.val_size = val_size
        self.val_rotate = val_rotate
        self.val_new_background = val_new_background
        self.val_shape = val_shape
        self.subset = subset

        self.val_kernel = set_kernel(val_strength_kernel)

        self.model_type = None
        self.ani = None
        self.possible_pos = None
        self.val_possible_pos = None
        self.line_start()

        self.scores = []

        self.file_name = 'weights'
        self.file_path = self.file_name + '/'
        start_name = f'{self.mat_size}{self.model_type}'
        self.end_name = '.weights.h5'
        self.triangle_path = self.file_path + start_name + 'triangle' + self.end_name
        self.line_path = self.file_path + start_name + 'line' + self.end_name

    @classmethod
    def from_dict(cls, params):
        return cls(**params)

    def generate_pred_data(self, model, num_to_pred, val=False, concatenate=True):
        con_matrix, line_pos_mat, _ = self.create_matrix_in_list(num_to_pred, val)

        input_matrix = np.array(con_matrix[:num_to_pred * self.fades_per_mat])
        true_matrix = line_pos_mat[:num_to_pred * self.fades_per_mat]

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

    def create_matrix_in_list(self, numb_of_time_series, val=False):
        list_matrix = []
        list_pos_mat = []
        list_alfa = []

        if val:
            size, rotate, new_background, shape, kernel, possible_pos = \
                (self.val_size, self.val_rotate, self.val_new_background, self.val_shape, self.val_kernel,
                 self.val_possible_pos)
        else:
            size, rotate, new_background, shape, kernel, possible_pos = \
                (self.size, self.rotate, self.new_background, self.shape, self.kernel, self.possible_pos)

        for k in range(0, numb_of_time_series):
            if shape == 'triangle':
                sfb = matrix_triangle_maker(self.mat_size, kernel,
                                            self.fades_per_mat,
                                            new_background=new_background)

            elif shape == 'face':
                sfb = matrix_triangle_maker(self.mat_size, kernel,
                                            self.fades_per_mat,
                                            new_background=new_background, alternative=True)

            elif shape == 'line':
                if rotate:  # makes the line
                    line_size = rotater(size)

                else:  # does not rotate the line
                    line_size = size

                sfb = matrix_maker(self.mat_size, kernel, possible_pos, line_size,
                                   self.fades_per_mat,
                                   new_background=new_background)

            else:
                print(f'{self.shape} is invalid')
                print('Try triangle, face or line')
                return

            mat, pos, alf = sfb
            list_matrix.append(mat)
            list_pos_mat.append(pos)
            list_alfa.append(alf)

        return tf.convert_to_tensor(list_matrix), list_pos_mat, list_alfa

    def line_start(self):
        rows, cols = self.mat_size
        pos_size = rows - self.size[0], cols - self.size[1]
        val_pos_size = rows - self.val_size[0], cols - self.val_size[1]
        possible_line_pos = np.arange(pos_size[0]), np.arange(pos_size[1])
        val_possible_line_pos = np.arange(val_pos_size[0]), np.arange(val_pos_size[1])

        if self.subset and self.val:

            np.random.shuffle(possible_line_pos[0])
            np.random.shuffle(possible_line_pos[1])

            self.possible_pos = (possible_line_pos[0][pos_size[0] // 2:],
                                 possible_line_pos[1][pos_size[1] // 2:])

            self.val_possible_pos = (val_possible_line_pos[0][:val_pos_size[0] // 2],
                                     val_possible_line_pos[1][:val_pos_size[1] // 2])

        else:
            self.possible_pos = (possible_line_pos[0], possible_line_pos[1])
            self.val_possible_pos = val_possible_line_pos

    def plot_matrices(self, model, num_to_pred, interval=500, val=False):
        input_matrix, true_matrix, predicted_line_pos_mat = self.generate_pred_data(model, num_to_pred, val)
        plt.clf()
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

        title = f'Prediction with '

        if not self.rotate:
            title += 'non rotated, '
        if not self.new_background:
            title += 'static background, '
        if val:
            title += 'val data, '

        title += f' {self.model_type} model on {self.mat_size} matrix'

        fig.suptitle(title)

        plt.tight_layout()
        plt.show()
        return animation

    def display_frames(self, model, num_frames=1000, num_to_pred=1, val=False):
        for _ in range(num_to_pred):
            num_frames = max(min(self.fades_per_mat, num_frames), 2)
            input_matrix, true_matrix, predicted_line_pos_mat = self.generate_pred_data(model, 1, val)
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

            plt.tight_layout(pad=0.15)
            plt.show(block=False)

            title = f'Movie from the '

            if not self.rotate:
                title += 'non rotated, '
            if not self.new_background:
                title += 'static background, '
            if val:
                title += 'val data, '
            title += f'{self.model_type} model'

            fig.suptitle(title)

    def unique_lines(self):
        unique_lines = 0
        possible_row = self.mat_size[0] - self.size[0] + 1
        possible_col = self.mat_size[1] - self.size[1] + 1
        unique_lines += possible_row * possible_col
        if self.rotate:
            possible_row_r = self.mat_size[0] - self.size[1] + 1
            possible_col_r = self.mat_size[1] - self.size[0] + 1
            unique_lines += possible_row_r * possible_col_r

        print('Possible lines: ', unique_lines)
        if self.val:
            unique__lines = 0
            possible_row = self.mat_size[0] - self.val_size[0] + 1
            possible_col = self.mat_size[1] - self.val_size[1] + 1
            unique__lines += possible_row * possible_col
            if self.val_rotate:
                possible_row_r = self.mat_size[0] - self.val_size[1] + 1
                possible_col_r = self.mat_size[1] - self.val_size[0] + 1
                unique_lines += possible_row_r * possible_col_r
            print('Possible val lines: ', unique__lines)

    def init_model(self, model_type='cnn_rnn', iou_s=True, info=False, early_stopping=False, cnn_rnn_scale=(1, 1)):
        self.model_type = model_type
        checkpoint_filepath = self.file_path + 'standard' + self.end_name

        callbacks = [ks.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                  monitor='loss', mode='min', save_best_only=True)]

        if early_stopping:
            callbacks.append(ks.callbacks.EarlyStopping(monitor='loss', patience=8, min_delta=0.0001,
                                                        restore_best_weights=True, start_from_epoch=20))

        parameters = self.mat_size, cnn_rnn_scale[0], cnn_rnn_scale[1], self.fades_per_mat

        model = build_model(model_type, parameters)
        optimizer = ks.optimizers.Adam()
        if iou_s:
            metrics = [IoUMaker(n) for n in range(1, self.fades_per_mat)]
            model.compile(optimizer=optimizer, loss=custom_weighted_loss,
                          metrics=[metrics, Precision(name='precision'), Recall(name='recall')])
        else:
            model.compile(optimizer=optimizer, loss=custom_weighted_loss,
                          metrics=[BinaryIoU(name='IoU'), Precision(name='precision'), Recall(name='recall')])

        if info:
            model.summary()

        return model, callbacks

    def init_generator(self, batch_size, num_batch):
        if self.val:
            return DataGenerator(self, batch_size, num_batch), DataGenerator(self, batch_size, num_batch, self.val)
        else:
            return DataGenerator(self, batch_size, num_batch), None

    def load_model(self, model, weights_shape='auto'):
        if weights_shape == 'none':
            return
        elif weights_shape == 'auto':

            if os.path.exists(self.triangle_path) and self.shape == 'triangle':
                try:
                    model.load_weights(self.triangle_path)
                    print('Loaded triangle weights')
                except ValueError:
                    print('Could not load triangle weights')

            elif os.path.exists(self.line_path) and self.shape == 'line':
                try:
                    model.load_weights(self.line_path)
                    print('Loaded line weights')
                except ValueError:
                    print('Could not load line weights')
            elif os.path.exists('standard' + self.end_name):
                try:
                    model.load_weights(f'standard' + self.end_name)
                    print('Loaded callback weights')
                except ValueError:
                    print('Could not load callback weights')
            else:
                print('Did not find any weights')
        elif weights_shape == 'triangle':
            model.load_weights(self.triangle_path)
            print('Loaded triangle weights')
        elif weights_shape == 'line':
            model.load_weights(self.line_path)
            print('Loaded line weights')
        else:
            if os.path.exists(f'{weights_shape}' + self.end_name):
                try:
                    model.load_weights(f'{weights_shape}' + self.end_name)
                    print(f'Loaded {weights_shape}' + self.end_name)
                except ValueError:
                    print('Could not load custom weights')
            else:
                print(f'Did not find weights with name {weights_shape}' + self.end_name)

    def save_model(self, model, weights_shape='auto', epochs=0):
        if epochs == 0 or weights_shape == 'none':
            return
        elif weights_shape == 'auto':

            if self.shape == 'triangle':
                make_folder(self.file_name)
                model.save_weights(self.triangle_path)
                print('Saved triangle weights')

            elif self.shape == 'line':
                make_folder(self.file_name)
                model.save_weights(self.line_path)
                print('Saved line weights')
            else:
                print(f'error {self.shape} not recognized')

        elif weights_shape == 'triangle':
            make_folder(self.file_name)
            model.save_weights(self.triangle_path)
            print('Saved triangle weights')

        elif weights_shape == 'line':
            make_folder(self.file_name)
            model.save_weights(self.line_path)
            print('Saved line weights')

        else:
            make_folder(self.file_name)
            model.save_weights(f'{self.file_path}{weights_shape}' + self.end_name)
            print(f'Saved custom weights with name {weights_shape}' + self.end_name)

    def plot_training_history(self, training_history_object, show, name_note, end_name, train_time=None):
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

        val_iou_s = []
        val_preps = []
        val_other = []

        list_of_metrics = [key for key in list(history_dict.keys())]

        for metric in list_of_metrics:
            if 'val_IoU' in metric:
                val_iou_s.append(metric)
            elif 'val_precision' in metric or 'val_recall' in metric:
                val_preps.append(metric)
            elif 'val_' in metric:
                val_other.append(metric)
            elif 'IoU' in metric:
                iou_s.append(metric)
            elif 'precision' in metric or 'recall' in metric:
                preps.append(metric)
            else:
                other.append(metric)

        train_hist_df = pd.DataFrame(history_dict)

        make_folder('csv_files')

        if train_time is None:
            train_hist_df.tail(1).round(4).to_csv(
                f'csv_files/{self.model_type}' + end_name + '.csv')
        else:

            trn = train_hist_df.tail(1)
            trn.insert(1, 'Train_time', train_time)
            trn.round(4).to_csv(f'csv_files/{self.model_type}' + end_name + '.csv')

        train_keys = other
        nr_plots = len(other)

        if len(iou_s) > 0:
            nr_plots += 1
        if len(preps) > 0:
            nr_plots += 1
            if self.val:
                nr_plots += 1

        plt_nr = 0

        if self.val:
            fig, ax = plt.subplots(2, nr_plots // 2, figsize=(2 * nr_plots, 8))

            for i in range(len(other)):
                ax[0, plt_nr].plot(np.array(train_hist_df[train_keys[plt_nr]]), label=train_keys[plt_nr])
                ax[0, plt_nr].plot(np.array(train_hist_df[val_other[plt_nr]]), label=val_other[plt_nr])
                ax[0, plt_nr].set_ylim(0, 1)
                ax[0, plt_nr].set_xlabel('Epoch')
                ax[0, plt_nr].set_title(train_keys[plt_nr])
                ax[0, plt_nr].grid('on')
                ax[0, plt_nr].legend()
            plt_nr += 1

            for k in range(len(preps)):
                ax[0, plt_nr].plot(np.array(train_hist_df[preps[k]]), label=preps[k])
                ax[0, plt_nr].plot(np.array(train_hist_df[val_preps[k]]), label=val_preps[k])
                ax[0, plt_nr].set_ylim(0, 1)
                ax[0, plt_nr].set_xlabel('Epoch')
                ax[0, plt_nr].set_title('Precision and Recall')
                ax[0, plt_nr].grid('on')
                ax[0, plt_nr].legend()

            plt_nr = 0
            for k in range(len(iou_s)):
                ax[1, plt_nr].plot(np.array(train_hist_df[iou_s[k]]), label=f'Frame {k + 1}')
                ax[1, plt_nr].set_ylim(0, 1)
                ax[1, plt_nr].set_xlabel('Epoch')
                ax[1, plt_nr].set_title('IoU')
                ax[1, plt_nr].grid('on')
                ax[1, plt_nr].legend()
            plt_nr += 1
            for k in range(len(iou_s)):
                ax[1, plt_nr].plot(np.array(train_hist_df[val_iou_s[k]]), label=f'Frame {k + 1}')
                ax[1, plt_nr].set_ylim(0, 1)
                ax[1, plt_nr].set_xlabel('Epoch')
                ax[1, plt_nr].set_title('val_IoU')
                ax[1, plt_nr].grid('on')
                ax[1, plt_nr].legend()
            plt_nr += 1
        else:
            fig, ax = plt.subplots(1, nr_plots, figsize=(4 * nr_plots, 5))

            for i in range(len(other)):
                ax[plt_nr].plot(np.array(train_hist_df[train_keys[plt_nr]]), label=train_keys[plt_nr])
                if self.val:
                    ax[plt_nr].plot(np.array(train_hist_df[val_other[plt_nr]]), label=val_other[plt_nr])
                ax[plt_nr].set_ylim(0, 1)
                ax[plt_nr].set_xlabel('Epoch')
                ax[plt_nr].set_title(train_keys[plt_nr])
                ax[plt_nr].grid('on')
                ax[plt_nr].legend()
            plt_nr += 1

            for k in range(len(preps)):
                ax[plt_nr].plot(np.array(train_hist_df[preps[k]]), label=preps[k])
                if self.val:
                    ax[plt_nr].plot(np.array(train_hist_df[val_preps[k]]), label=val_preps[k])
                ax[plt_nr].set_ylim(0, 1)
                ax[plt_nr].set_xlabel('Epoch')
                ax[plt_nr].set_title('Precision and Recall')
                ax[plt_nr].grid('on')
                ax[plt_nr].legend()
            plt_nr += 1

            for k in range(len(iou_s)):
                ax[plt_nr].plot(np.array(train_hist_df[iou_s[k]]), label=f'Frame {k + 1}')
                ax[plt_nr].set_ylim(0, 1)
                ax[plt_nr].set_xlabel('Epoch')
                ax[plt_nr].set_title('IoU')
                ax[plt_nr].grid('on')
                ax[plt_nr].legend()

        if show:
            fig.suptitle(f'Metrics from the {self.model_type} model on {self.mat_size} matrix' + end_name)
            fig.tight_layout()
            plt.show()
        else:
            make_folder(name_note)
            fig.suptitle(f'Metrics from the {self.model_type} model on {self.mat_size} matrix' + end_name)
            fig.tight_layout()
            # Save the plot inside the folder
            file_path = os.path.join(name_note, f'{self.model_type} on {self.mat_size}' + end_name)
            plt.savefig(file_path)

    def after_training_metrics(self, model, hist=None, epochs=0, movies_to_plot=0, frames_to_show=1000,
                               movies_to_show=0, with_val=False, both=False, interval=500, plot=True,
                               name_note='test', train_time=None):

        end_name = f'_e{epochs}_{self.size}'

        if self.val:
            end_name += f'_v{self.val_size}'

        end_name += f'_r{str(self.rotate)[0]}_b{str(self.new_background)[0]}'

        if self.val:
            end_name += f'_rv{str(self.val_rotate)[0]}_bv{str(self.val_new_background)[0]}_sub{str(self.subset)[0]}'

        if movies_to_plot > 0:
            if both:
                self.display_frames(model, num_frames=frames_to_show, num_to_pred=movies_to_plot, val=True)
                self.display_frames(model, num_frames=frames_to_show, num_to_pred=movies_to_plot, val=False)
            else:
                self.display_frames(model, num_frames=frames_to_show, num_to_pred=movies_to_plot, val=with_val)

        if movies_to_show > 0:
            if both:
                self.ani = self.plot_matrices(model, num_to_pred=movies_to_show, interval=interval, val=True)
                self.ani = self.plot_matrices(model, num_to_pred=movies_to_show, interval=interval, val=False)
            else:
                self.ani = self.plot_matrices(model, num_to_pred=movies_to_show, interval=interval, val=with_val)

        if hist is not None and epochs != 0:
            self.plot_training_history(hist, plot, name_note, end_name, train_time=train_time)


def matrix_maker(mat_size, kernel, possible_pos, line_size=(1, 2), num_per_mat=3, new_background=False):
    rows, cols = mat_size

    # smooth
    smooth_matrix = sp.ndimage.convolve(np.random.rand(rows, cols), kernel)

    # line_start
    line_start_position = np.random.choice(possible_pos[0]), np.random.choice(possible_pos[1])
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
    import raster_geometry as rg
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


def set_kernel(str_ker):
    strength, kernel_size = str_ker
    center = (kernel_size - 1) / 2  # Calculate the center of the kernel

    kernel = np.fromfunction(
        lambda x, y: np.exp(
            -((x - center) ** 2 +
              (y - center) ** 2)
            / (2 * strength ** 2)),
        (kernel_size, kernel_size))

    return kernel / np.sum(kernel)


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


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
    import raster_geometry as rg
    ab = rg.bresenham_line(a, b, endpoint=True)
    for x in set(ab):
        yield from rg.bresenham_line(c, x, endpoint=True)


def train_multiple(matrix_params, model_types, train_param, val_params, run=False, name_note='test'):
    if run:
        batch_size, batch_num, epochs = train_param

        for f, val_param in enumerate(val_params):

            new_dict = {**matrix_params, **val_param}
            data_handler = MovieDataHandler(**new_dict)

            for model_type in model_types:
                model, callbacks = data_handler.init_model(model_type, iou_s=True, info=False, early_stopping=False)

                generator, val_gen = data_handler.init_generator(batch_size, batch_num)
                print('Training model: ', model_type)
                start = time.time()
                hist = model.fit(generator, validation_data=val_gen, epochs=epochs)
                train_time = time.time() - start

                data_handler.after_training_metrics(model, hist=hist, epochs=epochs,
                                                    plot=False, name_note=name_note, train_time=train_time)


if __name__ == '__main__':
    matrix__params = {
        'mat_size': (10, 10),
        'fades_per_mat': 10,

        'strength_kernel': (1, 3),
        'size': (6, 2),
        'rotate': True,
        'new_background': True,
        'shape': 'line',  # 'line', 'triangle', 'face'

        'val': True,

        'val_strength_kernel': (1, 3),
        'val_size': (6, 2),
        'val_rotate': True,
        'val_new_background': True,
        'val_shape': 'line',  # 'line', 'triangle', 'face'
        'subset': False,
    }

    val__param = [{'val_size': (3, 4)}]
    # [{'rotate': False, 'val_size': (2, 6), 'val_rotate': False}]  # {'val_size': (3, 4), 'subset': True},

    # 'dense', 'cnn', 'cnn-lstm',
    # 'res', 'cnn-res', 'deep-res', 'res-dense', 'brain'
    # 'rnn', 'cnn-rnn',
    # 'unet', 'unet-rnn'
    model__types = ['cnn-lstm']

    train__param = [
        500,  # batch_size =
        20,  # batch_num =
        50,  # epochs =
    ]

    train_multiple(matrix__params, model__types, train__param, val__param, run=True, name_note='big_test')
    # combine_csv_files()
