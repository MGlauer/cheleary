import tensorflow as tf
from tensorflow.python.ops import math_ops
from cheleary.registry import Registerable

_LOSSES = {}


class CustomLoss(tf.keras.losses.Loss):
    _REGISTRY = _LOSSES


@tf.keras.utils.register_keras_serializable(package="Custom", name=None)
class SparseLoss(CustomLoss):
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
