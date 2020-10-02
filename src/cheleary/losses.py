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
    def __init__(self, loss: tf.losses.Loss = None, **kwargs):
        super(SparseLoss, self).__init__(**kwargs)
        self._internal_loss = loss or tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        y_true = math_ops.cast(y_true, y_pred.dtype)
        # count ones
        ones = tf.math.reduce_sum(y_true[:])
        # count zeros
        zeros = tf.math.reduce_sum(1 - y_true)

        # weight ones with the number of zeros and vice versa
        weights = y_true * zeros + (1 - y_true) * ones
        return weights * self._internal_loss(y_true, y_pred)

    def get_config(self):
        d = super(SparseLoss, self).get_config()
        if isinstance(self._internal_loss, tf.keras.losses.Loss):
            d["internal"] = self._internal_loss.get_config()
        return d

    @classmethod
    def from_config(cls, config):
        if "internal" in config:
            d = config.pop("internal")
            internal = tf.keras.losses.deserialize(d["name"])
            config["loss"] = internal
        return super(SparseLoss, cls).from_config(config)


tf.keras.losses.SparseLoss = SparseLoss
