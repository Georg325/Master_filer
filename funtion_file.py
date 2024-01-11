import numpy as np
import time
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from keras.models import Sequential
from keras.layers import Dense, Flatten


class MatrixMaker:
    def __init__(self, rows, cols=None, kernel_size=(1, 1), line_size=(1, 2), num_per_mat=10):
        self.rows = rows
        self.cols = cols or rows
        self.kernel_size = kernel_size
        self.line_size = line_size
        self.num_per_mat = num_per_mat

        self.smooth_matrix = self.create_smoothed_matrix()
        self.line_start_position = self.create_line()
        self.alfa = self.create_alfa()
        self.matrix_fade = self.create_matrix_line_fade()
        self.line_pos_mat = self.create_line_pos_mat()

    def create_smoothed_matrix(self):
        kernel = np.ones(shape=self.kernel_size, dtype=float) / (self.kernel_size[0] * self.kernel_size[1])
        return sp.ndimage.convolve(np.random.rand(self.rows, self.cols), kernel)

    def create_line(self):
        return (np.random.randint(low=0, high=self.rows - self.line_size[0] + 1),
                np.random.randint(low=0, high=self.cols - self.line_size[1] + 1))

    def create_alfa(self):
        return np.linspace(1, 0, num=self.num_per_mat)

    def create_matrix_with_line(self, alfa):
        matrix = np.ones((self.rows, self.cols))
        matrix[self.line_start_position[0]:self.line_start_position[0] + self.line_size[0],
        self.line_start_position[1]:self.line_start_position[1] + self.line_size[1]] = alfa
        return matrix

    def create_matrix_line_fade(self):
        matrix_line_fade = []
        for i in range(self.num_per_mat):
            line = self.create_matrix_with_line(self.alfa[i])
            matrix_line_fade.append(self.smooth_matrix * line)

        return matrix_line_fade

    def create_line_pos_mat(self):
        return np.logical_not(self.create_matrix_with_line(0)).astype(int)


class MatrixLister:
    def __init__(self, row_len, col_len, kernel_size, max_line_size, num_of_mat, num_per_mat, num_neuron):
        self.row_len = row_len
        self.col_len = col_len
        self.kernel_size = kernel_size
        self.max_line_size = max_line_size
        self.num_of_mat = num_of_mat
        self.num_per_mat = num_per_mat

        self.matrix_list = self.listing_matrix()
        self.con_matrix, self.con_alfa = self.concatenate_matrices()

        self.neural_network = NeuralNetwork(input_size=row_len * col_len, num_neuron=num_neuron)

    def listing_matrix(self):
        line_sizes = [(np.random.randint(1, self.max_line_size[0] + 1),
                       np.random.randint(1, self.max_line_size[1] + 1))
                      for _ in range(self.num_of_mat)]

        return [MatrixMaker(self.row_len, self.col_len, self.kernel_size, line_sizes[i], self.num_per_mat)
                for i in range(self.num_of_mat)]

    def concatenate_matrices(self):
        concatenated_matrices = []
        con_alfa = []

        for matrix in self.matrix_list:
            concatenated_matrices += matrix.create_matrix_line_fade()
            con_alfa += list(matrix.alfa)

        return concatenated_matrices, con_alfa

    def con_line_pos_mat(self):
        con_line_pos_mat = []
        for matrix in self.matrix_list:
            for alfa in matrix.alfa:
                if alfa != 1:
                    con_line_pos_mat += [matrix.line_pos_mat]
                else:
                    con_line_pos_mat += [np.zeros((self.row_len, self.col_len))]
        return con_line_pos_mat

    def train_neural_network(self, num_epochs=10, batch_size=64):

        # Assuming you have input_data and output_data for training
        input_data = np.array(self.con_matrix).reshape(len(self.con_matrix), -1)
        output_data = np.array(self.con_line_pos_mat()).reshape(len(self.con_line_pos_mat()), -1)

        self.neural_network.train(input_data, output_data, num_epochs, batch_size)

    def plot(self, interval=200):
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            im = ax.imshow(self.con_matrix[frame], interpolation='nearest', aspect='auto', vmin=0, vmax=1)

            return [im]

        animation = FuncAnimation(fig, update, frames=len(self.con_matrix), interval=interval, repeat=False, blit=True)
        plt.tight_layout()
        plt.show(block=False)
        plt.show()
        return animation

    def testprint(self):
        return

    def plot_other(self, other, interval=200):
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            im = ax.imshow(other[frame], interpolation='nearest', aspect='auto', vmin=0, vmax=1)

            return [im]

        animation = FuncAnimation(fig, update, frames=len(other), interval=interval, repeat=False, blit=True)
        plt.tight_layout()
        plt.show(block=False)
        plt.show()
        return animation


class NeuralNetwork:
    def __init__(self, input_size, num_neuron):
        self.model = self.build_model(input_size, num_neuron)

    def build_model(self, input_size, num_neuron):
        model = Sequential()
        model.add(Flatten(input_shape=(input_size, 1)))
        model.add(Dense(num_neuron, input_shape=(input_size,), activation='relu'))
        model.add(Dense(input_size, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, input_data, output_data, epochs, batch_size):
        self.model.fit(input_data, output_data, epochs=epochs, batch_size=batch_size)

    def predict(self, input_matrix):
        input_data = input_matrix.reshape(1, -1)
        predicted_output = self.model.predict(input_data)
        predicted_line_pos_mat = (predicted_output > 0.5).astype(np.int8).reshape(input_matrix.shape)
        return predicted_line_pos_mat


row_len = 6
col_len = 6
kernel_size = (2, 2)
max_line_size = (3, 3)
num_of_mat = 500
numb_of_picture = 5
num_of_neurons = 9

matrix_lister = MatrixLister(row_len, col_len, kernel_size, max_line_size, num_of_mat, numb_of_picture, num_of_neurons)
matrix_lister.plot()
batch_size = 64
epochs = 10

start = time.time()
matrix_lister.train_neural_network(batch_size=batch_size, num_epochs=epochs)
print(time.time() - start)

# Assuming you have a trained matrix_lister and a trained neural network
input_matrix = matrix_lister.matrix_list[0].create_matrix_line_fade()[-1]  # Example input matrix
predicted_line_pos_mat = matrix_lister.neural_network.predict(input_matrix)

# Print or use the predicted_line_pos_mat as needed
print("Predicted Line Position Matrix:")
print(predicted_line_pos_mat)
