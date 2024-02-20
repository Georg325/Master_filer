import time

from funtion_file import *
#%%
matrix_params = {
    'mat_size': (6, 6),
    'strength_kernel': (1, 3),
    'min_max_line_size': [(1, 4), (1, 4)],
    'rotate': True,
    'fades_per_mat': 10,
    'new_background': True,
    'shape': 'line',  # 'line', 'triangle', 'face'
}

model_type = 'cnn'  # cnn_rnn, cnn, res, dense

matrix_lister = MatrixLister(**matrix_params)
#%%
model, callbacks = matrix_lister.init_model(32, 64, model_type, True)

matrix_lister.load_model(model, 'none')  # auto, line, triangle, none

#%%
batch_size = 500
batch_num = 10
epochs = 50

generator = matrix_lister.init_generator(model, batch_size, batch_num)

start = time.time()
hist = model.fit(generator, epochs=epochs)
print(time.time() - start)

#%%
matrix_lister.save_model(model, 'none')  # auto, line, triangle, none

#%%

matrix_lister.plot_scores(matrix_lister.scores)

matrix_lister.display_frames(model, 16, 0)
#%%
#ani = matrix_lister.plot_matrices(model, 50, interval=500)
#plt.show()

#%%
plot_training_history(hist, model_type)
