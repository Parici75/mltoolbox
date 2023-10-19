# type: ignore
from __future__ import annotations

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger(__name__)


def to_keras_categorical(categorical_array: Sequence) -> np.ndarray:
    """Converts a possibly discontinuous categorical array into a One-hot-encoded matrix."""
    unique_values = np.unique(categorical_array)
    mapping = dict(zip(unique_values, range(len(unique_values)), strict=True))
    integer_array = np.array([mapping[i] for i in categorical_array])

    return to_categorical(integer_array)


def plot_loss(model_trained, performance_metrics: list[str] | None) -> plt.Figure:
    """Plots the training and validation loss as well as optional performance metrics."""

    loss = model_trained.history["loss"]
    val_loss = model_trained.history["val_loss"]

    fig = plt.figure()
    plt.plot(model_trained.epoch, loss, "b", label="Training loss")
    plt.plot(model_trained.epoch, val_loss, "r", label="Validation loss")
    if performance_metrics is not None:
        for metrics in performance_metrics:
            plt.plot(
                model_trained.epoch,
                model_trained.history[metrics],
                "bo",
                label=f"Training {metrics}",
            )
            plt.plot(
                model_trained.epoch,
                model_trained.history[f"val_{metrics}"],
                "ro",
                label=f"Validation {metrics}",
            )
    plt.title("Training and validation loss")
    plt.legend()

    return fig


# Keras callback to print updated LR at the end of each epoch
class SGDLearningRateTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None) -> None:
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1.0 / (1.0 + optimizer.decay * optimizer.iterations)))
        logger.info(f"\nLR: {lr:.6f}\n")


# Keras callback to store gradient norm at the end of each epoch
class GradientNorm(keras.callbacks.Callback):
    def on_train_begin(self, logs=None) -> None:
        self.gradient_norm = []

    def on_epoch_end(self, epoch, logs=None) -> None:
        with tf.GradientTape() as tape:
            loss_fct = tf.keras.losses.get(self.model.loss)
            y_pred = self.model.predict(self.validation_data[0])
            loss = loss_fct(y_pred, self.validation_data[0])
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.gradient_norm.append(K.sqrt(sum([K.sum(K.square(g)) for g in grads])))


# Keras callback to store losses and lr at the end of each epoch
class LossHistory(keras.callbacks.Callback):
    def __init__(self, lr_function=None) -> None:
        self.lr_function = lr_function

    def on_train_begin(self, logs=None) -> None:
        self.losses = []
        self.lr = []

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.losses.append(logs.get("loss"))
        if self.lr_function:
            self.lr.append(self.lr_function(len(self.losses)))
        else:
            self.lr.append(logs.get("lr"))


# Keras callbacks to calculate the metrics at the end of the epoch
class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs=None) -> None:
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None) -> None:
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        logger.info(
            f"\n— val_f1: {_val_f1:.2f} — val_precision: {_val_precision:.2f} — val_recall"
            f" {_val_recall:.2f}"
        )
