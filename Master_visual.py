import os

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import time
from funtion_file import *
from tensorflow.keras.models import Model
from visual import *

matrix_params = {
    'mat_size': (5, 5),
    'strength_kernel': (.005, 3),
    'min_max_line_size': [(1, 3), (1, 3)],
    'rotate': True,
    'fades_per_mat': 20,
    'new_background': True,
    'shape': 'line',  # 'line', 'triangle', 'face'
}

matrix_lister = MatrixLister(**matrix_params)

model, callbacks = matrix_lister.init_model(64, 32, 'auto')  # auto, line, triangle, none

batch_size = 50
batch_num = 3
epochs = 5

generator = matrix_lister.init_generator(batch_size, batch_num)


start = time.time()
hist = model.fit(generator, epochs=epochs, callbacks=callbacks)
print(time.time() - start)


plot_training_history(hist, with_val=False)

matrix_lister.display_frames(model, 16, 3)



'''
matrix_lister.save_model(model, 'line')  # auto, line, triangle, none
ani = matrix_lister.plot_matrices(model, 50, interval=500)
plt.show()
weights = model.get_weights()

weight = np.array(weights[0])

fisk = weight[:, :, 0, :]

print(fisk.shape)

fom = np.transpose(fisk, (2, 0, 1),)

plots(fom, interval=500)


layer = model.get_layer(name='time_distributed')
feature_extractor = ks.Model(inputs=model.inputs, outputs=layer.output)

loss, img = visualize_filter(2, feature_extractor)
print(img.shape)

plt.imshow(img[0, -1, :, :, 0])
plt.show()

plots(img[0, :, :, :, 0])
plt.show()




'''