import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class MatrixMaker:
    def __init__(self, rows, cols=None, k=1, line_size=(1, 2)):
        self.rows = rows
        self.cols = cols
        if self.cols is None:
            self.cols = rows

        self.line_size = line_size

        self.matrix = np.random.rand(self.rows, self.cols)
        kernel = np.ones((k, k), dtype=float) / k ** 2
        self.smooth_matrix = sp.ndimage.convolve(self.matrix, kernel)
        self.line_position = (np.random.randint(low=0, high=self.rows - self.line_size[0] + 1),
                              np.random.randint(low=0, high=self.cols - self.line_size[1] + 1))

    def create_matrix_with_line(self, alfa):
        matrix = np.ones((self.rows, self.cols))
        matrix[self.line_position[0]:self.line_position[0] + self.line_size[0],
        self.line_position[1]:self.line_position[1] + self.line_size[1]] = alfa
        return matrix

    def create_matrix_list(self, num_per_mat):
        matrix_list = []
        alfa = np.linspace(1, 0, num=num_per_mat)
        for i in range(num_per_mat):
            line = self.create_matrix_with_line(alfa[i])
            matrix_list.append(self.smooth_matrix * line)

        return matrix_list


def print_matrix(matrix):
    """
    Nicely prints a 2D matrix.

    Parameters:
    - matrix: The matrix to be printed
    """
    for row in matrix:
        print(" ".join(f"{value:.2f}" for value in row))


def plot_matrix_movie(matrix_list, interval=200):
    """
    Plots a movie of a list of matrices.

    Parameters:
    - matrix_list: List of 2D matrices to be displayed in the movie
    - interval: Time delay between frames in milliseconds
    - cmap: Colormap for displaying the matrices

    Returns:
    - An animation object
    """
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        im = ax.imshow(matrix_list[frame], interpolation='nearest', aspect='auto', vmin=0, vmax=1)

        return [im]

    animation = FuncAnimation(fig, update, frames=len(matrix_list), interval=interval, repeat=False, blit=True)
    plt.tight_layout()
    plt.show(block=False)
    plt.show()
    return animation


def concatenate_matrices(matrix_lists):
    concatenated_matrices = []
    for matrix_list in matrix_lists:
        concatenated_matrices += matrix_list
    return concatenated_matrices


# Example usage:
row_len = 128  # number of rows
col_len = 156
kernel_size = 3
line_size = (2, 5)
num_of_mat = 5
numb_of_picture = 14

matrix_maker_list = [MatrixMaker(row_len, col_len, k=kernel_size, line_size=line_size) for _ in range(num_of_mat)]
matrix_list = [matrix_maker_list[i].create_matrix_list(numb_of_picture) for i in range(num_of_mat)]
plot_matrix_movie(concatenate_matrices(matrix_list))
plt.show()
