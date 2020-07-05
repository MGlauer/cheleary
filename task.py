import os
import tensorflow as tf
from dataprocessor import DataProcessor
from models import Model
from encode import Encoder
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
        version=1,
        load_model=False,
    ):

        self.identifier = identifier

        self.dataprocessor = dataprocessor

        self.model_container = model_container


        self.batch_size = batch_size

        self.split = split

        self.version = version

        if load_model:
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.model = model_container.build()

    def create_model(self):
        raise NotImplementedError

    def train_model(self, data, save_model=True, epochs=1):
        self.model.summary()
        self.model.fit(data, epochs=epochs, shuffle=True, verbose=2, steps_per_epoch=self.dataprocessor.length)

    @property
    def model_path(self):
        path = os.path.join(".tasks", self.identifier)
        return os.path.join(path, "model", f"v{self.version}")

    def save(self):
        print("Save model")
        self.model.save(self.model_path)
        with open(os.path.join(".tasks", self.identifier, "config.json"), "w") as f:
            json.dump(self.config, f)

    def test_model(self, training_data):
        mse_total = 0
        counter = 0
        self.model.summary()
        with open(".log/tests.csv", "w") as fout:
            for x_batch, y_batch in training_data:
                y_pred_batch = self.model.predict(x_batch)
                for y_real, y_pred in zip(y_batch, y_pred_batch):
                    fout.write(",".join(map(str, y_pred[:])) + "\n")
                    fout.write(",".join(map(str, y_real)) + "\n")
                    counter += 1
                    mse = np.dot((y_pred[:] - y_real), (y_pred[:] - y_real)) / len(
                        y_real
                    )
                    mse_total += mse
                    print(mse)
        print(mse_total / counter)

    def run(self, epochs=1):
        dataset = self.dataprocessor.load_data(kind="train", loop=True)
        print("Start training")
        self.train_model(dataset, epochs=epochs)
        print("Stop training")
        self.version += 1

    def test(self):
        dataset = self.dataprocessor.load_data(kind="test")
        print("Start training")
        self.test_model(dataset,)

    @property
    def config(self):
        return dict(
            identifier=self.identifier,
            version=self.version,
            data_path=self.dataprocessor.data_path,
            input_encoder=self.dataprocessor.input_encoder._ID,
            model=self.model_container._ID,
            output_encoder=self.dataprocessor.output_encoder._ID,
        )

    def __repr__(self):
        return json.dumps(self.config)


def load_task(identifier):
    with open(os.path.join(".tasks", identifier, "config.json")) as fin:
        config = json.load(fin)
    return load_from_strings(**config)


def load_from_strings(identifier, data_path, input_encoder, model, output_encoder, version=1):
    ie = Encoder.get(input_encoder)()
    model_container = Model.get(model)()
    model_container = Model.get(model)()
    oe = Encoder.get(output_encoder)()
    dp = DataProcessor(data_path=data_path, input_encoder=ie, output_encoder=oe,)
    return LearningTask(identifier=identifier, dataprocessor=dp, model_container=model_container, version=version)
