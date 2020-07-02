import tensorflow as tf
import pickle
import multiprocessing as mp
from task import LearningTask, register
import encode
from config import LOCAL_SIZE_RESTRICTION
import numpy as np
import random

class Classifier(LearningTask):
    ID = 'classifier'

    def __init__(self, **kwargs):
        super(Classifier, self).__init__( **kwargs)
        with open('data/chemdata500x100x1024.pkl', 'rb') as output:
            # pickle.dump(chemdata,output)
            chemdata = pickle.load(output)
            if LOCAL_SIZE_RESTRICTION != -1:
                stream = list(chemdata.iterrows())[:LOCAL_SIZE_RESTRICTION]
            else:
                stream = chemdata.iterrows()

            self.total_data = sum(1 for _ in stream)
            self.steps_per_epoch = int(self.total_data*self.training_ratio)
            self.test_amount = int(self.total_data*(1-self.training_ratio)/2)
            self.eval_amount = self.total_data - (self.steps_per_epoch + self.test_amount)

    @property
    def input_shape(self):
        return (None,  None)

    @property
    def output_shape(self):
        return (None,1024)

    @property
    def input_datatype(self):
        return tf.int32

    @property
    def output_datatype(self):
        return tf.int32

    def create_model(self):
        tf.executing_eagerly()
        loss = tf.keras.losses.BinaryCrossentropy()

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(300, 100, input_shape=(None,), name="inputs"))
        forward = tf.keras.layers.LSTM(1000, activation=tf.keras.activations.tanh, recurrent_activation=tf.keras.activations.sigmoid, name="forward", recurrent_dropout=0, unroll=False, use_bias=True)
        model.add(forward)
        model.add(tf.keras.layers.Dense(500, activation=tf.keras.activations.sigmoid, use_bias=True, name="outputs"))
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5), loss=loss, metrics=["mae", "acc", "binary_crossentropy"])
        print(model.losses)
        return model

    def generate_data(self, kind="train", loop=False):
        with open('data/chemdata500x100x1024.pkl', 'rb') as output:
            # pickle.dump(chemdata,output)
            chemdata = pickle.load(output)

        #with mp.Pool(mp.cpu_count() - 2) as pool:
        if LOCAL_SIZE_RESTRICTION != -1:
            stream = (x for x in list(chemdata.iterrows())[:LOCAL_SIZE_RESTRICTION])
        else:
            stream = chemdata.iterrows()
        for i in range(self.steps_per_epoch):
            result = next(stream)
            if kind=="train":
                yield result[1][2], result[1][0]
        if kind in ("test", "eval"):
            for i in range(self.test_amount):
                result = next(stream)
                if kind == "test":
                    yield result[1][2], result[1][0]
        if kind == "eval":
            for i in range(self.test_amount):
                result = next(stream)
                if kind == "eval":
                    yield result[1][2], result[1][0]


register(Classifier)
