import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_random_matrix(n, m=None, k=1):
    """
    Creates a matrix of random values between 0 and 1 of size n x m.

    Parameters:
    - n: Number of rows
    - m: Number of columns

    Returns:
    - A numpy array representing the random matrix
    """
    if m is None:
        m = n

    random_matrix = np.random.rand(n, m)
    kernel = np.ones((k, k), dtype=float) / k ** 2
    smoothed = sp.ndimage.convolve(random_matrix, kernel)
    return smoothed


def print_matrix(matrix):
    """
    Nicely prints a 2D matrix.

    Parameters:
    - matrix: The matrix to be printed
    """
    for row in matrix:
        print(" ".join(f"{value:.2f}" for value in row))


def plot_matrix_movie(matrix_list, interval=200, cmap='viridis'):
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
        ax.imshow(matrix_list[frame], cmap=cmap, interpolation=None, aspect='auto')
        ax.set_title(f"Frame {frame + 1}/{len(matrix_list)}")

    animation = FuncAnimation(fig, update, frames=len(matrix_list), interval=interval, repeat=True)

    plt.show(block=False)

    return animation


# Example usage:
n = 2  # number of rows
random_matrix = create_random_matrix(n, k=2)
print_matrix(random_matrix)

random_animation = plot_matrix_movie([create_random_matrix(n, k=2) for _ in range(5)])
