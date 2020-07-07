import tensorflow as tf
from src.cheleary.registry import Registerable
from tensorflow.python.ops import math_ops

_MODELS = {}


class Model(Registerable):
    _REGISTRY = _MODELS

    def build(self, **kwargs):
        raise NotImplementedError


@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
class SparseLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = math_ops.cast(y_true, y_pred.dtype)
        # count ones
        ones = tf.math.reduce_sum(y_true[:])
        # count zeros
        zeros = tf.math.reduce_sum(1 - y_true)
        # weight ones with the number of zeros and vice versa
        weights = y_true*zeros + (1-y_true)*ones
        squares = tf.square(y_true - y_pred)
        return tf.reduce_mean(weights * squares)


tf.keras.losses.SparseLoss = SparseLoss


class LSTMClassifierModel(Model):
    _ID = "lstm_classifier"

    def build(self, input_size=300, output_size=500, learning_rate=0.001):
        loss = SparseLoss(name="sparse_loss") #tf.keras.losses.BinaryCrossentropy()

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Embedding(
                input_size, 100, input_shape=(None,), name="inputs"
            )
        )
        forward = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            1000,
            activation=tf.keras.activations.tanh,
            recurrent_activation=tf.keras.activations.sigmoid,
            name="forward",
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
        ))
        model.add(forward)
        model.add(tf.keras.layers.Dense(output_size, use_bias=True, name="outputs",activation=tf.keras.activations.sigmoid))
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=loss,
            metrics=["mae", "acc", "binary_crossentropy"],
        )
        self.model = model
        return model
