import numpy as np
import scipy as sp


def matrix_maker(rows, cols=None, kernel_size=(2, 2), line_size=(1, 2), num_per_mat=3):
    cols = cols or rows

    # smooth
    kernel = np.ones(shape=kernel_size, dtype=float) / np.prod(kernel_size)
    smooth_matrix = sp.ndimage.convolve(np.random.rand(rows, cols), kernel)

    # line_start
    line_start_position = (np.random.randint(low=0, high=rows - line_size[0] + 1),
                           np.random.randint(low=0, high=cols - line_size[1] + 1))

    # alfa
    alfa = np.linspace(1, 0, num=num_per_mat)

    # answer mat
    line_pos_mat = np.zeros((rows, cols))

    # matrix_fade
    matrix_line_fade = []
    for i in range(num_per_mat):
        matrix_with_line = np.ones((rows, cols))
        matrix_with_line[line_start_position[0]:line_start_position[0] + line_size[0],
                         line_start_position[1]:line_start_position[1] + line_size[1]] = alfa[i]
        if alfa[i] == 0:
            line_pos_mat = np.array(np.logical_not(matrix_with_line).astype(int), dtype='float16')
        matrix_line_fade.append(smooth_matrix * matrix_with_line)

    return np.array(matrix_line_fade), np.array(line_pos_mat), np.array(alfa)


if __name__ == '__main__':
    matrix_fade, line_pos_mat_d, alfa_d = matrix_maker(4, 5, (2, 2), (1, 3), 4)

    print(alfa_d)
    print()
    print(line_pos_mat_d)
    print(matrix_fade.shape)

    for mat in matrix_fade:
        print(mat)
        print()
