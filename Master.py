import time

from funtion_file import *
#%%
matrix_params = {
    'mat_size': (6, 6),
    'fades_per_mat': 10,

    'strength_kernel': (1, 3),
    'size': [(4, 1), (4, 1)],
    'rotate': False,
    'new_background': True,
    'shape': 'line',  # 'line', 'triangle', 'face'

    'val': True,

    'val_strength_kernel': (1, 3),
    'val_size': [(4, 1), (4, 1)],
    'val_rotate': True,
    'val_new_background': True,
    'val_shape': 'line',  # 'line', 'triangle', 'face'
}
# dense, cnn, cnn_lstm, res, cnn_res, rnn, cnn_rnn, unet, unet_rnn, res_dense
model_type = 'cnn_rnn'

matrix_lister = MovieDataHandler(**matrix_params)
#%%
model, callbacks = matrix_lister.init_model(32, 64, model_type, threshold=0.5)

matrix_lister.load_model(model, 'none')  # auto, line, triangle, none

#%%
model.summary()

batch_size = 300
batch_num = 15
epochs = 50

generator, val_gen = matrix_lister.init_generator(batch_size, batch_num)

start = time.time()
hist = model.fit(generator, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
print(time.time() - start)

#%%
matrix_lister.save_model(model, 'auto', epochs)  # auto, line, triangle, none

#%%
matrix_lister.after_training_metrics(model, hist=hist, epochs=epochs, movies_to_plot=0, movies_to_show=0)
