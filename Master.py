from function_file import *
from master_prints import train_time_print
import matplotlib.pyplot as plt

# %%
matrix_params = {
    'mat_size': (10, 10),
    'fades_per_mat': 10,

    'strength_kernel': (2, 3),
    'size': (6, 2),
    'rotate': True,
    'new_background': True,
    'shape': 'line',  # 'line', 'triangle', 'face'

    'val': True,

    'val_strength_kernel': (2, 3),
    'val_size': (4, 3),
    'val_rotate': True,
    'val_new_background': True,
    'val_shape': 'line',  # 'line', 'triangle', 'face'
    'subset': False,
}
# dense,
# cnn, cnn-lstm,
# res, cnn-res, deep-res
# rnn, cnn-rnn,
# unet, unet-rnn,
# brain, cnn-brain,
model_type = 'cnn'

data_handler = MovieDataHandler(**matrix_params)
# %%
model, callbacks = data_handler.init_model(model_type, iou_s=True, info=True)

data_handler.load_model(model, model_type)  # auto, line, triangle, none, custom

# %%
batch_size = 500
batch_num = 20
epochs = 50

generator, val_gen = data_handler.init_generator(batch_size, batch_num)

start = time.time()
hist = model.fit(generator, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
train_time_print(start)

wihseat = model.get_layer('time_distributed').get_weights()[0]

wihseat = np.transpose(np.array(wihseat), [2, 3, 0, 1])
# plots(wihseat[0], interval=500)


# %%
data_handler.save_model(model, model_type, epochs)  # auto, line, triangle, none, custom

# %%
data_handler.after_training_metrics(model, hist=hist, epochs=epochs, movies_to_plot=3, movies_to_show=0, both=False,
                                    plot=True)
