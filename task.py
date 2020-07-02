import os
import pickle
from random import shuffle
import tensorflow as tf
from encode import Encoder
import numpy as np
from time import time

tf.config.set_soft_device_placement(True)

_TASKS = dict()


def get_task(ID):
    return _TASKS[ID]


def register(cls):
    _TASKS[cls.ID] = cls


class LearningTask:
    ID = None

    def __init__(self, input_encoder:Encoder = None, output_encoder:Encoder = None, model=None, model_path=None, data_path=None, batch_size=1, split=0.7):
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder
        self.batch_size = batch_size



        if model_path is None:
            self.model_path = f'models/{self.ID}/{time()}'
        else:
            self.model_path = model_path
        self.split = split
        if data_path is None:
            self.data_path = f'data/{self.ID}/split_{split*100}'
        else:
            self.data_path = data_path
        if model is None:
            self.load_model()
        else:
            self.model = model

        self.steps_per_epoch = None

        self.training_ratio = 0.7

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def input_datatype(self):
        return tf.int32

    @property
    def output_datatype(self):
        return tf.int32

    def create_model(self):
        raise NotImplementedError

    def generate_data(self, kind="train", loop=False):
        raise NotImplementedError

    def load_data(self, kind="train"):
        in_batch = []
        out_batch = []
        for x, y in self.generate_data(kind=kind, loop=(kind=="train")):
            in_batch.append(self.input_encoder.run(x))
            out_batch.append(self.output_encoder.run(y))
            if len(in_batch) >= self.batch_size:
                yield np.asarray(in_batch), np.asarray(out_batch)
                in_batch = []
                out_batch = []

    def train_model(self, training_data, save_model=True,
                    epochs=1):
        self.model.summary()
        self.model.fit(training_data, epochs=epochs, steps_per_epoch=self.steps_per_epoch)

        if save_model:
            self.model.save(self.model_path)

    def test_model(self, training_data):
        mse_total = 0
        counter = 0
        self.model.summary()
        with open(".log/tests.csv","w") as fout:
            for x_batch, y_batch in training_data:
                y_pred_batch = self.model.predict(x_batch)
                for y_real, y_pred in zip(y_batch, y_pred_batch):
                    fout.write(",".join(map(str,y_pred[:]))+"\n")
                    fout.write(",".join(map(str,y_real))+"\n")
                    counter += 1
                    mse = np.dot((y_pred[:] - y_real), (y_pred[:] - y_real))/len(y_real)
                    mse_total += mse
                    print(mse)
        print(mse_total/counter)



    def load_model(self):

        if os.path.exists(self.model_path):
            print("Load existing model")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print("No model found - create a new one")
            self.model = self.create_model()

    def run(self, epochs=1):
        dataset = self.load_data(kind="train")
        print("Start training")
        self.train_model(
            dataset,
            epochs=epochs
        )

    def test(self):
        dataset = self.load_data(kind="test")
        print("Start training")
        self.test_model(
            dataset,
        )

