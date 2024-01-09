import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class MatrixMaker:
    def __init__(self, rows, cols=None, k=1, line_size=(1,2)):
        self.rows = rows
        self.cols = cols
        if self.cols is None:
            self.cols = rows

        self.line_size = line_size

        self.matrix = np.random.rand(self.rows, self.cols)
        kernel = np.ones((k, k), dtype=float) / k ** 2
        self.smooth_matrix = sp.ndimage.convolve(self.matrix, kernel)

    def create_matrix_with_line(self, alfa):
        matrix = np.ones((self.rows, self.cols))
        line_position = (np.random.randint(low=0, high=self.rows - self.line_size[0] + 1),
                         np.random.randint(low=0, high=self.cols - self.line_size[1] + 1))

        matrix[line_position[0]:line_position[0] + self.line_size[0],
               line_position[1]:line_position[1] + self.line_size[1]] = alfa
        return matrix

    def create_matrix_list(self, num_per_mat):
        matrix_list = []
        alfa = np.linspace(0, 1, num=num_per_mat)

        for i in range(num_per_mat):
            line = self.create_matrix_with_line(alfa)
            matrix_list.append(self.smooth_matrix*line)

        return matrix_list


def print_matrix(matrix):
    """
    Nicely prints a 2D matrix.

    Parameters:
    - matrix: The matrix to be printed
    """
    for row in matrix:
        print(" ".join(f"{value:.2f}" for value in row))


def plot_matrix_movie(matrix_list, interval=500):
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
        ax.imshow(matrix_list[frame], interpolation='nearest', aspect='auto')
        ax.set_title(f"Frame {frame + 1}/{len(matrix_list)}")

    animation = FuncAnimation(fig, update, frames=len(matrix_list), interval=interval, repeat=True)
    plt.tight_layout()
    plt.show(block=False)
    plt.show()
    return animation


# Example usage:
row_len = 4  # number of rows
col_len = 6
kernel_size = 3
line_size = (1, 2)

matrixmaker_list = [MatrixMaker(row_len, col_len, k=kernel_size, line_size=line_size) for _ in range(5)]
matrix_list = [matrixmaker_list[i].create_matrix_list(10) for i in range(5)]
plot_matrix_movie(matrix_list[0])
plt.show()
