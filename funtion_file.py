import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class MatrixMaker:
    def __init__(self, rows, cols=None, kernel_size=(1, 1), line_size=(1, 2), num_per_mat=10):
        self.rows = rows
        self.cols = cols
        self.kernel_size = kernel_size
        self.line_size = line_size
        self.num_per_mat = num_per_mat

        if self.cols is None:
            self.cols = rows

        self.matrix = np.random.rand(self.rows, self.cols)

        self.smooth_matrix = self.smoothed_matrix()
        self.line_start_position = self.line_maker()
        self.alfa = self.alfa_maker()
        self.matrix_fade = self.create_matrix_line_fade()

    def smoothed_matrix(self):
        kernel = np.ones(shape=self.kernel_size, dtype=float) / (self.kernel_size[0] * self.kernel_size[1])
        return sp.ndimage.convolve(self.matrix, kernel)

    def line_maker(self):
        return (np.random.randint(low=0, high=self.rows - self.line_size[0] + 1),
                np.random.randint(low=0, high=self.cols - self.line_size[1] + 1))

    def alfa_maker(self):
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

    def line_positions(self):
        pos_lis = []

        for row in range(self.line_size[0]):

            for col in range(self.line_size[1]):
                pos_lis.append((self.line_start_position[0] + row, self.line_start_position[1] + col))

        return pos_lis


class MatrixLister:
    def __init__(self, row_len, col_len, kernel_size, line_size, num_of_mat, num_per_mat):
        self.row_len = row_len
        self.col_len = col_len
        self.kernel_size = kernel_size
        self.line_size = line_size
        self.num_of_mat = num_of_mat
        self.num_per_mat = num_per_mat

        self.matrix_list = self.listing_matrix()
        self.con_matrix, self.con_alfa = self.concatenate_matrices()

    def listing_matrix(self):
        return [MatrixMaker(self.row_len, self.col_len, self.kernel_size, self.line_size, self.num_per_mat)
                for _ in range(self.num_of_mat)]

    def concatenate_matrices(self):
        concatenated_matrices = []
        con_alfa = []

        for matrix in self.matrix_list:
            concatenated_matrices += matrix.create_matrix_line_fade()
            con_alfa += list(matrix.alfa)

        return concatenated_matrices, con_alfa

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
        for matrix in self.matrix_list:
            print(matrix.line_positions())


# Example usage:
row_len = 3
col_len = 3
kernel_size = (2, 2)
line_size = (2, 2)
num_of_mat = 19
numb_of_picture = 5

matrix_lister = MatrixLister(row_len, col_len, kernel_size, line_size, num_of_mat, numb_of_picture)
alfa_lis = matrix_lister.con_alfa
matrix_lister.plot()
