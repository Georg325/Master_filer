import time

from funtion_file import *
#%%
matrix_params = {
    'mat_size': (4, 4),
    'strength_kernel': (5, 3),
    'min_max_line_size': [(1, 3), (1, 3)],
    'rotate': True,
    'fades_per_mat': 10,
    'new_background': True,
    'shape': 'line',  # 'line', 'triangle', 'face'
}

matrix_lister = MatrixLister(**matrix_params)
#%%
model, callbacks = matrix_lister.init_model(32, 64, 'cnn_gru')  # cnn_gru, cnn, res

matrix_lister.load_model(model, 'none')

#%%
batch_size = 100
batch_num = 20
epochs = 200

generator = matrix_lister.init_generator(model, batch_size, batch_num)

model.summary()

start = time.time()
hist = model.fit(generator, epochs=epochs)
print(time.time() - start)

#%%
matrix_lister.save_model(model, 'auto')  # auto, line, triangle, none

#%%

matrix_lister.plot_scores(matrix_lister.scores)

matrix_lister.display_frames(model, 16, 3)
#%%
ani = matrix_lister.plot_matrices(model, 50, interval=500)
plt.show()

#%%
# plot_training_history(hist, with_val=False)
