import numpy as np
import tensorflow as tf
import keras as ks

from abc import ABC



class DataGenerator(ks.utils.Sequence, ABC):
    """Generates data for Keras"""

    def __init__(self, mat_obj, model, batch_size=500, num_batch=2):
        """Initialization"""
        super().__init__()
        self.mat_obj = mat_obj
        self.model = model
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.mat_obj.clear_score()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.num_batch

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data
        x, y, _ = self.mat_obj.create_matrix_in_list(self.batch_size)
        # self.mat_obj.randomize_kernel((0.7, 0.4))
        return np.expand_dims(x, -1), np.expand_dims(y, -1)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.mat_obj.eval(self.model, self.batch_size)
        return


def custom_weighted_loss(y_true, y_pred, scaling=1.5):
    """
    Custom loss function with emphasis on errors for values that are 1 in y_true using mean squared error.
    Weighting factor is dependent on the fill factor of y_true.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels

    Returns:
    - Weighted mean squared error loss
    """
    # Calculate fill factor (proportion of non-zero values)
    fill_factor = tf.keras.backend.mean(tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32))

    # Calculate squared errors
    squared_errors = tf.math.square(tf.cast(y_true, dtype=tf.float32) - y_pred)

    # Calculate weight factor based on fill factor
    weight_factor = scaling * fill_factor

    # Apply weights to positive class
    weighted_squared_errors = (tf.cast(y_true, dtype=tf.float32) * (weight_factor * squared_errors)
                               + tf.cast((1 - y_true), dtype=tf.float32) * squared_errors)

    # Calculate mean loss over all elements
    loss = tf.keras.backend.mean(weighted_squared_errors)

    return loss


def f1_score(y_true, y_pred):
    """
    Custom F1 score calculation.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels

    Returns:
    - F1 score
    """
    true_positives = tf.reduce_sum(tf.cast(y_true * tf.round(y_pred), tf.float32))
    predicted_positives = tf.reduce_sum(tf.round(y_pred))
    actual_positives = tf.reduce_sum(y_true)

    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    return f1


def custom_accuracy(y_true, y_pred):
    """
    Custom F1 score evaluation function for each timestep.

    Parameters:
    - y_true: True labels with shape [batch_size, time, row, column]
    - y_pred: Predicted labels with shape [batch_size, time, row, column]

    Returns:
    - F1 score for each timestep
    """
    f1_per_timestep = []

    # Iterate over timesteps
    for t in range(10):
        # Extract labels for the current timestep
        y_true_timestep = tf.cast(y_true[:, t, :, :], tf.float32)

        # Extract predictions for the current timestep
        y_pred_timestep = tf.cast(y_pred[:, t, :, :], tf.float32)

        # Calculate F1 score for the current timestep
        f1_timestep = f1_score(y_true_timestep, y_pred_timestep)

        # Append F1 score for the current timestep to the list
        f1_per_timestep.append(f1_timestep)

    return np.array(f1_per_timestep)


def custom_iou(y_true, y_pred):
    """
    Custom IoU (Intersection over Union) evaluation function for each timestep.

    Parameters:
    - y_true: True labels with shape [batch_size, time, row, column]
    - y_pred: Predicted labels with shape [batch_size, time, row, column]

    Returns:
    - IoU for each timestep
    """
    iou_per_timestep = []

    # Iterate over timesteps
    for t in range(y_true.shape[1]):
        # Extract labels for the current timestep
        y_true_timestep = tf.cast(y_true[:, t, :, :], tf.float32)

        # Extract predictions for the current timestep
        y_pred_timestep = tf.cast(y_pred[:, t, :, :], tf.float32)

        # Calculate Intersection and Union for the current timestep
        intersection = tf.reduce_sum(tf.math.multiply(y_true_timestep, y_pred_timestep))
        union = tf.reduce_sum(tf.math.add(y_true_timestep, y_pred_timestep))

        # Calculate IoU for the current timestep
        iou_timestep = intersection / (union - intersection)

        # Append IoU for the current timestep to the list
        iou_per_timestep.append(iou_timestep)

    return np.array(iou_per_timestep)




