import time

from funtion_file import *
#%%
matrix_params = {
    'mat_size': (6, 6),
    'fades_per_mat': 10,

    'strength_kernel': (1, 3),
    'size': [(4, 1), (4, 1)],
    'rotate': True,
    'new_background': False,
    'shape': 'line',  # 'line', 'triangle', 'face'

    'val': True,

    'val_strength_kernel': (1, 3),
    'val_size': [(3, 3), (3, 3)],
    'val_rotate': True,
    'val_new_background': False,
    'val_shape': 'line',  # 'line', 'triangle', 'face'
}
# dense, cnn, cnn_lstm, res, cnn_res, rnn, cnn_rnn, unet, unet_rnn, res_dense
model_type = 'dense'

datahandler = MovieDataHandler(**matrix_params)
#%%
model, callbacks = datahandler.init_model(32, 64, model_type, threshold=0.5)

datahandler.load_model(model, 'none')  # auto, line, triangle, none

#%%
model.summary()

batch_size = 300
batch_num = 30
epochs = 0

generator, val_gen = datahandler.init_generator(batch_size, batch_num)

start = time.time()
hist = model.fit(generator, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
print(f'{time.time() - start:.2f} s')

#%%
datahandler.save_model(model, 'none', epochs)  # auto, line, triangle, none

#%%
datahandler.after_training_metrics(model, hist=hist, epochs=epochs, movies_to_plot=0, movies_to_show=0, both=True)
