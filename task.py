import os
import pickle
from config import LOCAL_SIZE_RESTRICTION
from random import shuffle
import tensorflow as tf
from encode import Encoder
import numpy as np
from datetime import datetime

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
            self.model_path = f'store/{self.ID}/models/{datetime.now().strftime("%Y%m%d")}'
        else:
            self.model_path = model_path
        self.split = split
        if data_path is None:
            self.data_path = f'store/data/split_{split*100}'
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

    def generate_data(self, kind="train"):
        raise NotImplementedError

    def load_data(self, kind="train", loop=False, cached=True):
        if not os.path.exists(os.path.join(self.data_path, f"{kind}.pkl")):
            print("No cached data found. Create new cache.")
            os.makedirs(self.data_path, exist_ok=True)
            data = list(self.generate_data())[:LOCAL_SIZE_RESTRICTION]
            shuffle(data)
            train_amount = int(len(data) * self.training_ratio)
            test_amount = int(len(data) * (1 - self.training_ratio) / 2)
            eval_amount = len(data) - (self.steps_per_epoch + test_amount)
            for _kind, l, r in [("train", 0,train_amount),
                         ("test", train_amount, train_amount+test_amount),
                         ("eval",train_amount+test_amount,-1)]:
                with open(os.path.join(self.data_path, f"{_kind}.pkl"), "wb") as pkl:
                    pickle.dump([(np.asarray([self.input_encoder.run(x)]), np.asarray([self.output_encoder.run(y)])) for x,y in data[l:r]], pkl)
        with open(os.path.join(self.data_path, f"{kind}.pkl"), "rb") as pkl:
            data = pickle.load(pkl)
            while True:
                for x, y in data:
                    yield x, y
                if not loop:
                    break


    def train_model(self, data, save_model=True,
                    epochs=1):
        self.model.summary()
        self.model.fit(data, epochs=epochs, shuffle=True, steps_per_epoch=self.steps_per_epoch)

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
        dataset = self.load_data(kind="train", loop=True)
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

