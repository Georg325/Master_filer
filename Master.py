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
    'val_size': [(2, 2), (2, 2)],
    'val_rotate': True,
    'val_new_background': False,
    'val_shape': 'line',  # 'line', 'triangle', 'face'
}
# dense, cnn, cnn_lstm, res, cnn_res, rnn, cnn_rnn, unet, unet_rnn, res_dense, brain
model_type = 'brain'

data_handler = MovieDataHandler(**matrix_params)
#%%
model, callbacks = data_handler.init_model(model_type, iou_s=True, info=True)

data_handler.load_model(model, 'none')  # auto, line, triangle, none, custom

#%%
batch_size = 250
batch_num = 15
epochs = 60

generator, val_gen = data_handler.init_generator(batch_size, batch_num)

start = time.time()
hist = model.fit(generator, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
print(f'Training tok {time.time() - start:.2f} s')

#%%
data_handler.save_model(model, 'none', epochs)  # auto, line, triangle, none, custom

#%%
data_handler.after_training_metrics(model, hist=hist, epochs=epochs,
                                    movies_to_plot=0, movies_to_show=0,
                                    both=True, show=True)
