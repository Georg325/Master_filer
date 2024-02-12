import time

import matplotlib.pyplot as plt

from funtion_file import *
#%%
matrix_params = {
    'mat_size': (4, 4),
    'strength_kernel': (5, 3),
    'min_max_line_size': [(1, 3), (1, 3)],
    'rotate': True,
    'fades_per_mat': 10,
    'new_background': False,
    'shape': 'line',  # 'line', 'triangle', 'face'
}

matrix_lister = MatrixLister(**matrix_params)
#%%
model, callbacks = matrix_lister.init_model(32, 64, 'cnn_gru')

matrix_lister.load_model(model, 'none')
#%%
batch_size = 500
batch_num = 10
epochs = 50

generator = matrix_lister.init_generator(model, batch_size, batch_num)


start = time.time()
hist = model.fit(generator, epochs=epochs)
print(time.time() - start)

#%%
matrix_lister.save_model(model, 'auto')  # auto, line, triangle, none

#%%

plot_scores(matrix_lister.scores)
