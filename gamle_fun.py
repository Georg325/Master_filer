
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
