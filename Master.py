import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import time
from funtion_file import *

matrix_params = {
    'mat_size': (5, 5),
    'strength_kernel': (5, 3),
    'min_max_line_size': [(1, 3), (1, 3)],
    'rotate': True,
    'fades_per_mat': 10,
    'new_background': True,
    'shape': 'line',  # 'line', 'triangle', 'face'
}

matrix_lister = MatrixLister(**matrix_params)

model, callbacks = matrix_lister.init_model(32, 128, 'auto')  # auto, line, triangle, none

batch_size = 50
batch_num = 10
epochs = 0

generator = matrix_lister.init_generator(batch_size, batch_num)

start = time.time()
hist = model.fit(generator, epochs=epochs, callbacks=callbacks)
print(time.time() - start)

matrix_lister.save_model(model, 'none')  # auto, line, triangle, none

matrix_lister.display_frames(model, 16)

#ani = matrix_lister.plot_matrices(model, 50, interval=500)
#plt.show()

#plot_training_history(hist, with_val=False)
