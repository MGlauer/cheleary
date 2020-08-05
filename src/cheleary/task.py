import os
import tensorflow as tf
from cheleary.dataprocessor import DataProcessor
from cheleary.models import Model
from cheleary.encode import Encoder
import numpy as np
import json


class LearningTask:
    def __init__(
        self,
        identifier,
        dataprocessor: DataProcessor,
        model_container: Model,
        batch_size=1,
        split=0.7,
        version=0,
        prev_epochs=None,
        load_model=False,
    ):

        self.identifier = identifier

        self.dataprocessor = dataprocessor

        self.model_container = model_container

        self.batch_size = batch_size

        self.split = split

        self.version = version

        if load_model:
            print(f"Load model {self._model_path}")
            self.model = tf.keras.models.load_model(self._model_path)
        else:
            print(f"Build new model")
            self.model = model_container.build()

        if prev_epochs is None:
            self._prev_epochs = []
        else:
            self._prev_epochs = prev_epochs

    def create_model(self):
        raise NotImplementedError

    def train_model(self, data, test_data=None, epochs=1):
        self._prev_epochs.append(epochs)
        self.model.summary()
        os.makedirs(self._version_root, exist_ok=True)
        log_callback = tf.keras.callbacks.CSVLogger(self._train_log_path)
        self.model.fit(
            data,
            epochs=epochs,
            shuffle=True,
            callbacks=[log_callback],
            verbose=2,
            batch_size=1,
            steps_per_epoch=self.dataprocessor.length,
            validation_data=test_data,
        )

    @property
    def _version_root(self):
        return os.path.join(".tasks", self.identifier, f"v{self.version}")

    @property
    def _train_log_path(self):
        return os.path.join(self._version_root, "training.log")

    @property
    def _model_path(self):
        return os.path.join(self._version_root, "model")

    def save(self):
        print("Save model")
        self.model.save(self._model_path)
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
        self.version += 1
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
            version=self.version,
            data_path=self.dataprocessor.data_path,
            input_encoder=self.dataprocessor.input_encoder._ID,
            model=self.model_container._ID,
            output_encoder=self.dataprocessor.output_encoder._ID,
            epochs=self._prev_epochs,
            loss=self.model.loss.get_config(),
            optimizer=self.model.optimizer.name,
        )

    def __repr__(self):
        return json.dumps(self.config)


def load_task(identifier):
    with open(os.path.join(".tasks", identifier, "config.json")) as fin:
        config = json.load(fin)
    return load_from_strings(**config, load_model=True)


def load_from_strings(
    identifier,
    data_path,
    input_encoder,
    model,
    output_encoder,
    version=0,
    epochs=None,
    load_model=False,
    loss_dict=None,
    optimizer=None,
):
    ie = Encoder.get(input_encoder)()
    loss = tf.keras.losses.Loss.from_config(loss_dict)
    optimizer = tf.keras.optimizers.get(optimizer)
    model_container = Model.get(model)(loss=loss, optimizer=optimizer)
    oe = Encoder.get(output_encoder)()
    dp = DataProcessor(data_path=data_path, input_encoder=ie, output_encoder=oe,)
    return LearningTask(
        identifier=identifier,
        dataprocessor=dp,
        model_container=model_container,
        version=version,
        prev_epochs=epochs,
        load_model=load_model,
    )
