import numpy as np
import scipy as sp


def matrix_maker(rows, cols=None, kernel_size=(1, 1), line_size=(1, 2), num_per_mat=10):
    cols = cols or rows
    # Create smooth matrix
    kernel = np.ones(shape=kernel_size, dtype=float) / (kernel_size[0] * kernel_size[1])
    smooth_matrix = sp.ndimage.convolve(np.random.rand(rows, cols), kernel)

    # Create line start position
    line_start_position = (np.random.randint(low=0, high=rows - line_size[0] + 1),
                           np.random.randint(low=0, high=cols - line_size[1] + 1))

    # Create alfa array
    alfa = np.linspace(1, 0, num=num_per_mat)

    # Create matrix line fade
    matrix_fade_lis = []
    for i in range(num_per_mat):
        line = np.ones((rows, cols))
        line[line_start_position[0]:line_start_position[0] + line_size[0],
             line_start_position[1]:line_start_position[1] + line_size[1]] = alfa[i]
        matrix_fade_lis.append(smooth_matrix * line)

    # Create line position matrix
    line_pos_mat = np.array(np.logical_not(np.ones((rows, cols))).astype(int), dtype='float16')
    return alfa, matrix_fade_lis, line_pos_mat


# Example usage
alfa, matrix_fade_lis, line_pos_mat = matrix_maker(rows=3, kernel_size=(2, 2), line_size=(1, 2), num_per_mat=3)
print(alfa)
print()
for k in matrix_fade_lis:
    print(k)
    print()
print(line_pos_mat)

