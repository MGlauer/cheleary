import os
import tensorflow as tf
import tensorflow_addons as tfa
from cheleary.dataprocessor import DataProcessor
from cheleary.models import Model
from cheleary.encode import Encoder
from cheleary.losses import CustomLoss, SparseLoss
import numpy as np
import json


class LearningTask:
    def __init__(
        self,
        identifier,
        dataprocessor: DataProcessor,
        model: Model = None,
        batch_size=1,
        split=0.7,
        prev_epochs=None,
        load_model=False,
    ):

        self.identifier = identifier

        self.dataprocessor = dataprocessor

        self.batch_size = batch_size

        self.split = split

        self.last_epoch = 0
        if load_model:
            last = max(os.listdir(os.path.join(self._model_root, "checkpoints")))
            self.last_epoch = int(last)
            print(f"Load model", last)
            self.model = tf.keras.models.load_model(
                os.path.join(self._model_root, "checkpoints", last),
                custom_objects={
                    "SparseLoss": SparseLoss,
                    "F1Score": tfa.metrics.F1Score,
                },
                compile=True,
            )
        else:
            print(f"Build new model")
            self.model = model.build()
            self.save_config()

        if prev_epochs is None:
            self._prev_epochs = []
        else:
            self._prev_epochs = prev_epochs

    def create_model(self):
        raise NotImplementedError

    def train_model(self, data, test_data=None, epochs=1):
        self._prev_epochs.append(epochs)
        self.model.summary()
        cp_name = "{epoch:03d}"
        os.makedirs(os.path.join(self._model_root, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self._model_root, "best"), exist_ok=True)
        log_callback = tf.keras.callbacks.CSVLogger(
            os.path.join(self._model_root, "training.log"), append=True
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self._model_root, "checkpoints", cp_name),
            save_freq="epoch",
            period=25,
        )
        checkpoint_callback_best = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self._model_root, "best", cp_name),
            save_best_only=True,
            save_freq="epoch",
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=25, restore_best_weights=True
        )
        self.model.fit(
            data,
            epochs=self.last_epoch + epochs,
            shuffle=True,
            callbacks=[
                log_callback,
                checkpoint_callback,
                checkpoint_callback_best,
                early_stop,
            ],
            verbose=2,
            batch_size=1,
            steps_per_epoch=self.dataprocessor.length,
            validation_data=test_data,
            initial_epoch=self.last_epoch,
        )
        if self.last_epoch + epochs % 25:
            print("Create end-of-training checkpoint")
            self.model.save(
                os.path.join(
                    self._model_root, "checkpoints", f"{self.last_epoch+epochs:03d}"
                )
            )

    @property
    def _model_root(self):
        return os.path.join(".tasks", self.identifier)

    @property
    def _train_log_path(self):
        return os.path.join(self._model_root, "training.log")

    def save_config(self):
        os.makedirs(self._model_root)
        with open(os.path.join(".tasks", self.identifier, "config.json"), "w") as f:
            json.dump(self.config, f)

    def test_model(self, training_data, path=None):
        mse_total = 0
        counter = 0
        self.model.summary()
        if not path:
            path = os.path.join(self._version_root, "test.csv")
        with open(path, "w") as fout:
            for (x_real_batch, x_encoded), y_batch in training_data:
                y_pred_batch = self.model.predict(x_encoded)
                for x_real, y_real, y_pred in zip(x_real_batch, y_batch, y_pred_batch):
                    fout.write(str(x_real) + "\n")
                    fout.write(",".join(map(str, y_pred[:])) + "\n")
                    fout.write(",".join(map(str, y_real)) + "\n")

    def run(self, epochs=1):
        dataset = self.dataprocessor.load_data(kind="train", loop=True)
        # Drop unencoded data from dataset
        dataset = ((x[1], y) for x, y in dataset)
        x_test, y_test = tuple(zip(*self.dataprocessor.load_data(kind="test")))
        x_test = [x[1] for x in x_test]
        # x_test = [x for (x, _) in self.dataprocessor.load_data(kind="test")]
        # y_test = [y for (_, y) in self.dataprocessor.load_data(kind="test")]
        print("Start training")
        self.train_model(dataset, test_data=(x_test, y_test), epochs=epochs)
        print("Stop training")

    def test(self, path=None):
        dataset = self.dataprocessor.load_data(kind="test")
        print("Start testing")
        self.test_model(dataset, path=path)

    @property
    def config(self):
        return dict(
            identifier=self.identifier,
            data_path=self.dataprocessor.data_path,
            input_encoder=self.dataprocessor.input_encoder._ID,
            model=self.model.to_json(),
            output_encoder=self.dataprocessor.output_encoder._ID,
        )

    def __repr__(self):
        return json.dumps(self.config)


def load_task(identifier):
    with open(os.path.join(".tasks", identifier, "config.json")) as fin:
        config = json.load(fin)
    return load_from_strings(**config, load_model=True)


def load_from_strings(
    identifier, data_path, input_encoder, model, output_encoder, load_model=False,
):
    m = tf.keras.models.model_from_json(model)
    m.compile()
    ie = Encoder.get(input_encoder)()
    oe = Encoder.get(output_encoder)()
    dp = DataProcessor(data_path=data_path, input_encoder=ie, output_encoder=oe,)
    return LearningTask(identifier=identifier, dataprocessor=dp, load_model=load_model,)
