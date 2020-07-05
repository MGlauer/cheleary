import tensorflow as tf
from registry import Registerable
import os

_MODELS = {}


class Model(Registerable):
    _REGISTRY = _MODELS

    def build(self, **kwargs):
        raise NotImplementedError


class LSTMClassifierModel(Model):
    _ID = "lstm_classifier"

    def build(self, input_size=300, output_size=500, learning_rate=0.001):
        loss = tf.keras.losses.BinaryCrossentropy()

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Embedding(
                input_size, 100, input_shape=(None,), name="inputs"
            )
        )
        forward = tf.keras.layers.LSTM(
            1000,
            activation=tf.keras.activations.tanh,
            recurrent_activation=tf.keras.activations.sigmoid,
            name="forward",
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
        )
        model.add(forward)
        model.add(
            tf.keras.layers.Dense(
                10000,
                use_bias=True,
                name="outputs",
            )
        )
        model.add(
            tf.keras.layers.Dense(
                output_size,
                use_bias=True,
                name="outputs",
            )
        )
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=loss,
            metrics=["mae", "acc", "binary_crossentropy"],
        )
        self.model = model
        return model
