import tensorflow as tf

# import tensorflow_addons as tfa
from cheleary.registry import Registerable
from cheleary.losses import SparseLoss


_MODELS = {}


class Model(Registerable):
    _REGISTRY = _MODELS

    def __init__(self):
        self._loss = SparseLoss(name="sparse_loss")
        self._optimizer = tf.keras.optimizers.Adam
        self.learning_rate = 0.0001

    def create_model(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError

    def build(self, **kwargs):
        model = self.create_model(**kwargs)
        model.compile(
            optimizer=self._optimizer(learning_rate=self.learning_rate),
            loss=self._loss,
            metrics=[
                tf.metrics.MeanSquaredError(),
                tf.metrics.BinaryAccuracy(threshold=0.2),
                tf.metrics.BinaryCrossentropy(),
                tf.metrics.Precision(thresholds=0.2),
                tf.metrics.Recall(thresholds=0.2),
                # tfa.metrics.F1Score(
                #    threshold=0.2, num_classes=kwargs.get("output_size", 500)
                # ),
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


class NPClassifierModel(Model):
    _ID = "np_classifier"

    def __init__(self):
        super(NPClassifierModel, self).__init__()
        self._optimizer = tf.keras.optimizers.Adam
        self._loss = tf.keras.losses.BinaryCrossentropy()
        self.learning_rate = 0.00001

    def create_model(self, input_size=300, output_size=500):
        input_f = tf.keras.layers.Input(shape=(2048 + 4096,))

        X = tf.keras.layers.Dense(6144, activation="relu")(input_f)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dense(3072, activation="relu")(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dense(1536, activation="relu")(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dense(1536, activation="relu")(X)
        X = tf.keras.layers.Dropout(0.2)(X)
        output = tf.keras.layers.Dense(output_size, activation="sigmoid")(X)
        model = tf.keras.Model(inputs=[input_f], outputs=output)

        return model


class ConvModel(Model):
    _ID = "conv"

    def create_model(self, input_size=1024, output_size=500):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_size,)))
        model.add(tf.keras.layers.Reshape(target_shape=(input_size, 1)))
        model.add(tf.keras.layers.Conv1D(5, 5))
        model.add(tf.keras.layers.MaxPool1D())
        model.add(tf.keras.layers.Conv1D(10, 5))
        model.add(tf.keras.layers.MaxPool1D())
        model.add(tf.keras.layers.Conv1D(20, 5))
        model.add(tf.keras.layers.MaxPool1D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(500, activation=tf.keras.activations.sigmoid))
        return model


class LSTMClassifierModel(Model):
    _ID = "lstm_classifier"

    def __init__(self):
        super().__init__()
        self.learning_rate = 0.0005
        self._loss = tf.keras.losses.BinaryCrossentropy()

    def create_model(self, input_size=300, output_size=500):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Embedding(
                input_size, 100, input_shape=(None,), name="inputs"
            )
        )
        forward = tf.keras.layers.LSTM(
            100,
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
                1000, use_bias=True, activation=tf.keras.activations.relu,
            )
        )
        model.add(tf.keras.layers.Dropout(0.2))
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

    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001

    def create_model(self, input_size=300, output_size=500):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.InputLayer(input_shape=(None,), ragged=True))

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
        model.add(tf.keras.layers.Dropout(rate=0.1))
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
                100,
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
                1000,
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
