from config import EPOCHS
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
        self.split = split
        if data_path is None:
            self.data_path = f'data/{self.ID}/split_{split*100}'
        else:
            self.data_path = data_path
        if model is None:
            self.model = self.create_model()
        else:
            self.model = model

    def create_model(self):
        raise NotImplementedError

    def generate_data(self):
        raise NotImplementedError

    def load_data(self):

        train_path = os.path.join(self.data_path, "train.p")
        test_path = os.path.join(self.data_path, "test.p")
        eval_path = os.path.join(self.data_path, "eval.p")
        if self.data_path is not None and os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(eval_path):
            print("Load existing data dump")
            train = pickle.load(open(train_path, "rb"))
            test = pickle.load(open(test_path, "rb"))
            # eval = pickle.load(open(eval_path, "rb"))
            print("done")
        else:
            print("No data dump found! Create new data dump")
            #return tf.data.Dataset.from_generator(self.generate_data,
            #                                      output_types=(dict(inputs=tf.int32), dict(outputs=tf.int32)),
            #                                      output_shapes=(dict(inputs=(None, None, 190)), dict(outputs=(None, 1024))))
            data = list(self.generate_data())
            shuffle(data)
            xs, ys = zip(*data)
            x = tf.ragged.constant(xs)
            y = tf.ragged.constant(ys) #tf.convert_to_tensor(ys)
            size = len(x)
            train_size = int(size * self.split)
            test_size = int(size * (1 - self.split) / 2)
            train = x[:train_size], y[:train_size]
            test = x[train_size:(train_size + test_size)], y[train_size:(
                       train_size + test_size)]
            evalu = x[(train_size + test_size):], y[
                                                 (train_size + test_size):]

            if self.data_path is not None:
               if not os.path.exists(self.data_path):
                   os.makedirs(self.data_path, exist_ok=True)
               pickle.dump(train, open(train_path, "wb"))
               pickle.dump(test, open(test_path, "wb"))
               pickle.dump(evalu, open(eval_path, "wb"))
            print("done")
        return train, test

    def train_model(self, training_data, test_data=None, save_model=None,
                    epochs=EPOCHS):
        #print("Data: ", len(training_data[0]))
        self.model.summary()
        self.model.fit(training_data, epochs=epochs, use_multiprocessing=True, shuffle=False)

        if save_model:
            self.model.save(save_model)

        if test_data:
            y_pred = self.model.predict(test_data[0])
            y = test_data[1]

            for y1, y2 in zip(y_pred, y):
                print(y1, y2)

    def run(self):
        dataset= self.load_data()
        self.train_model(
            dataset,
            test_data=None
        )
