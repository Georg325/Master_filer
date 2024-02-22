import time

from funtion_file import *
#%%
matrix_params = {
    'mat_size': (6, 6),
    'strength_kernel': (1, 3),
    'min_max_line_size': [(4, 1), (4, 1)],
    'rotate': False,
    'fades_per_mat': 10,
    'new_background': False,
    'shape': 'line',  # 'line', 'triangle', 'face'
}

model_type = 'cnn'  # cnn_rnn, cnn, res, dense?, rnn, cnn_lstm, unet, cnn_res

matrix_lister = MatrixLister(**matrix_params)
#%%
model, callbacks = matrix_lister.init_model(32, 64, model_type, threshold=0.5)

matrix_lister.load_model(model, 'auto')  # auto, line, triangle, none

#%%
batch_size = 250
batch_num = 3
epochs = 2

generator = matrix_lister.init_generator(model, batch_size, batch_num)

start = time.time()
hist = model.fit(generator, epochs=epochs)
print(time.time() - start)

#%%
matrix_lister.save_model(model, 'none')  # auto, line, triangle, none

#%%
matrix_lister.after_training_metrics(model, hist=hist, epochs=epochs, movies_to_plot=2, movies_to_show=20)
