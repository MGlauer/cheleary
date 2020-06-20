from .config import EPOCHS
import os
import pickle
from random import shuffle
import tensorflow as tf


class LearningTask:
    ID = 'classifier'

    def __init__(self, model=None, model_path=None, data_path=None, split=0.7):
        if model_path is None:
            self.model_path = f'models/{self.ID}'
        else:
            self.model_path = model_path
        if data_path is None:
            self.data_path = f'data/{self.ID}'
        else:
            self.data_path = data_path
        self.split = split
        if model is None:
            self.model = self.create_model()
        else:
            self.model = model

    def create_model(self):
        raise NotImplementedError

    def generate_data(self):
        raise NotImplementedError

    def load_data(self):
        if self.data_path is not None and os.path.exists(self.data_path + "train.p") and os.path.exists(
                self.data_path + "test.p") and os.path.exists(self.data_path + "eval.p"):
            print("Load existing data dump")
            train = pickle.load(open(self.data_path + "train.p", "rb"))
            test = pickle.load(open(self.data_path + "test.p", "rb"))
            # eval = pickle.load(open(self.data_path + "eval.p", "rb"))
            print("done")
        else:
            print("No data dump found! Create new data dump")
            data = list(self.generate_data())
            shuffle(data)
            xs, ys = zip(*data)
            x = tf.ragged.constant(xs)
            y = tf.convert_to_tensor(ys)
            size = x.shape[0]
            train_size = int(size * self.split)
            test_size = int(size * (1 - self.split) / 2)
            train = x[:train_size], y[:train_size]
            test = x[train_size:(train_size + test_size)], y[train_size:(
                        train_size + test_size)]
            evalu = x[(train_size + test_size):], y[
                                                  (train_size + test_size):]

            if self.data_path is not None:
                if not os.path.exists(self.data_path):
                    os.mkdir(self.data_path)
                pickle.dump(train, open(self.data_path + "train.p", "wb"))
                pickle.dump(test, open(self.data_path + "test.p", "wb"))
                pickle.dump(evalu, open(self.data_path + "eval.p", "wb"))
            print("done")
        return train, test

    def train_model(self, training_data, test_data=None, save_model=None,
                    epochs=EPOCHS):
        print("Data: ", training_data[0].shape[0])

        self.model.fit(*training_data, epochs=epochs, use_multiprocessing=True)
        self.model.summary()
        if save_model:
            self.model.save(save_model)

        if test_data:
            y_pred = self.model.predict(test_data[0])
            y = test_data[1]

            for y1, y2 in zip(y_pred, y):
                print(y1, y2)