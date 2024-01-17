import tensorflow as tf

def matrix_maker(rows, cols=None, kernel_size=(1, 1), line_size=(1, 2), num_per_mat=10):
    cols = cols or rows

    # Create smooth matrix directly in TensorFlow
    smooth_matrix = tf.random.uniform((rows, cols), dtype=tf.float16)

    # Create line start position
    line_start_position = (
        tf.random.uniform((), maxval=rows - line_size[0] + 1, dtype=tf.int16),
        tf.random.uniform((), maxval=cols - line_size[1] + 1, dtype=tf.int16)
    )

    # Create alfa array directly in TensorFlow
    alfa = tf.linspace(1.0, 0.0, num=num_per_mat)

    # Create matrix line fade directly in TensorFlow
    matrix_fade_lis = []
    for i in range(num_per_mat):
        line = tf.ones((rows, cols), dtype=tf.float16)
        line[
            line_start_position[0]:line_start_position[0] + line_size[0],
            line_start_position[1]:line_start_position[1] + line_size[1]
        ] = alfa[i]
        matrix_fade = smooth_matrix * line
        matrix_fade_lis.append(matrix_fade)

    # Create line position matrix directly in TensorFlow
    line_pos_mat_tensor = tf.ones((rows, cols), dtype=tf.float16)

    return alfa, matrix_fade_lis, line_pos_mat_tensor

# Example usage
alfa_tensor, matrix_fade_tensor_list, line_pos_mat_tensor = matrix_maker(rows=5, cols=5, kernel_size=(2, 2), line_size=(2, 2), num_per_mat=3)

print("Alfa Tensor:")
print(alfa_tensor)

print("\nMatrix Fade Tensors:")
for matrix_fade_tensor in matrix_fade_tensor_list:
    print(matrix_fade_tensor)

print("\nLine Position Matrix Tensor:")
print(line_pos_mat_tensor)


if __name__ == '__main__':
    # Example usage
    alfa, matrix_fade_lis, line_pos_mat = matrix_maker(rows=3, kernel_size=(2, 2), line_size=(1, 2), num_per_mat=3)
    print(alfa)
    print()
    for k in matrix_fade_lis:
        print(k)
        print()
    print(line_pos_mat)

