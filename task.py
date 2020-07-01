from config import EPOCHS
import os
import pickle
from random import shuffle
import tensorflow as tf
from encode import Encoder
import numpy as np
from time import time

_TASKS = dict()


def get_task(ID):
    return _TASKS[ID]


class LearningTask:
    ID = None

    def __init__(self, input_encoder:Encoder = None, output_encoder:Encoder = None, model=None, model_path=None, data_path=None, batch_size=1, split=0.7):
        assert self.ID is not None, "This class does not have an ID"
        _TASKS[self.ID] = self.__class__

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

    def generate_data(self):
        raise NotImplementedError

    def load_data(self):
        in_batch = []
        out_batch = []
        for x, y in self.generate_data():
            in_batch.append(self.input_encoder.run(x))
            out_batch.append(self.output_encoder.run(y))
            if len(in_batch) >= self.batch_size:
                yield np.asarray(in_batch), np.asarray(out_batch)
                in_batch = []
                out_batch = []

    def train_model(self, training_data, test_data=None, save_model=True,
                    epochs=EPOCHS):
        self.model.summary()
        self.model.fit(training_data, epochs=epochs, steps_per_epoch=self.steps_per_epoch)

        if save_model:
            self.model.save(self.model_path)

        if test_data:
            y_pred = self.model.predict(test_data[0])
            y = test_data[1]

            for y1, y2 in zip(y_pred, y):
                print(y1, y2)

    def train_model(self, training_data):
        self.model.summary()
        for x_batch, y_batch in training_data:
            for x, y in zip(x_batch, y_batch):
                y_pred = self.model.predict(x)
                print(y_pred, y)


    def load_model(self):

        if os.path.exists(self.model_path):
            print("Load existing model")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print("No model found - create a new one")
            self.model = self.create_model()

    def run(self):
        dataset = self.load_data()
        print("Start training")
        self.train_model(
            dataset,
            test_data=None
        )

    def test(self):
        dataset = self.load_data()
        print("Start training")
        self.test_model(
            dataset,
            test_data=None
        )
