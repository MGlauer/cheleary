import tensorflow as tf
from tensorflow.python.ops import math_ops
from cheleary.registry import Registerable

_LOSSES = {}


class CustomLoss(tf.keras.losses.Loss):
    _REGISTRY = _LOSSES

    @classmethod
    def get(cls, identifier):
        if identifier in cls._REGISTRY:
            return cls._REGISTRY[identifier]
        else:
            return tf.losses.get(identifier)

    def call(self, y_true, y_pred):
        raise NotImplementedError


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

    def get_config(self):
        d = super(SparseLoss, self).get_config()
        if isinstance(self._internal_loss, tf.keras.losses.Loss):
            d["interal"] = self._internal_loss.get_config()
        elif hasattr(self._internal_loss, "_keras_api_names"):
            d["interal"] = self._internal_loss._keras_api_names[0]
        else:
            d["interal"] = repr(self._internal_loss)
        return d

    def from_config(cls, config):
        internal = tf.keras.losses.Loss.from_config(config.pop("internal"))
        config["loss"] = internal
        return super(SparseLoss, cls).from_config(config)


tf.keras.losses.SparseLoss = SparseLoss
