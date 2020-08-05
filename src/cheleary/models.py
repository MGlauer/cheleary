import tensorflow as tf
import tensorflow_addons as tfa
from cheleary.registry import Registerable
from tensorflow.python.ops import math_ops

_MODELS = {}


@tf.keras.utils.register_keras_serializable(package="Custom", name=None)
class SparseLoss(tf.keras.losses.Loss):
    def __init__(self, loss: tf.losses.Loss, **kwargs):
        super(SparseLoss, self).__init__(**kwargs)
        self._internal_loss = loss

    def call(self, y_true, y_pred):
        y_true = math_ops.cast(y_true, y_pred.dtype)
        # count ones
        ones = tf.math.reduce_sum(y_true[:])
        # count zeros
        zeros = tf.math.reduce_sum(1 - y_true)
        # weight ones with the number of zeros and vice versa
        weights = y_true * zeros + (1 - y_true) * ones
        squares = self._internal_loss(y_true, y_pred)
        return tf.reduce_mean(weights * squares)


tf.keras.losses.SparseLoss = SparseLoss


class Model(Registerable):
    _REGISTRY = _MODELS

    def __init__(self, loss=None, optimizer=None):
        self.loss = loss or SparseLoss(name="sparse_loss")
        self.optimizer = optimizer or tf.keras.optimizers.Adamax

    def create_model(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError

    def build(self, learning_rate=0.001, **kwargs):
        model = self.create_model(**kwargs)
        model.compile(
            optimizer=self.optimizer(learning_rate=learning_rate),
            loss=self.loss,
            metrics=[
                "mae",
                "mse",
                "acc",
                "binary_crossentropy",
                tf.metrics.Precision,
                tf.metrics.Recall,
                tfa.metrics.F1Score,
            ],
        )
        self.model = model
        return model

    @classmethod
    def _doc(cls):
        self = cls()
        model = self.build()
        s = []
        model.summary(print_fn=lambda s0: s.append(s0))
        return "\n".join(s)


class LSTMClassifierModel(Model):
    _ID = "lstm_classifier"

    def create_model(self, input_size=300, output_size=500):

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
                10000, use_bias=True, activation=tf.keras.activations.tanh,
            )
        )
        model.add(
            tf.keras.layers.Dense(
                output_size,
                use_bias=True,
                name="outputs",
                activation=tf.keras.activations.sigmoid,
            )
        )
        return model


class BiLSTMClassifierModel(Model):
    _ID = "bi_lstm_classifier"

    def create_model(self, input_size=300, output_size=500):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Embedding(
                input_size, 100, input_shape=(None,), name="inputs"
            )
        )
        forward = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                1000,
                activation=tf.keras.activations.tanh,
                recurrent_activation=tf.keras.activations.sigmoid,
                name="forward",
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
            )
        )
        model.add(forward)
        model.add(
            tf.keras.layers.Dense(
                output_size,
                use_bias=True,
                name="outputs",
                activation=tf.keras.activations.sigmoid,
            )
        )
        return model


class BiLSTMClassifierSpreadModel(Model):
    _ID = "bi_lstm_classifier_spread"

    def create_model(self, input_size=300, output_size=500):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Embedding(
                input_size, 100, input_shape=(None,), name="inputs"
            )
        )
        forward = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                1000,
                activation=tf.keras.activations.tanh,
                recurrent_activation=tf.keras.activations.sigmoid,
                name="forward",
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
            )
        )
        model.add(forward)
        model.add(
            tf.keras.layers.Dense(
                10000,
                use_bias=True,
                name="spread",
                activation=tf.keras.activations.sigmoid,
            )
        )
        model.add(
            tf.keras.layers.Dense(
                output_size,
                use_bias=True,
                name="outputs",
                activation=tf.keras.activations.sigmoid,
            )
        )
        return model
