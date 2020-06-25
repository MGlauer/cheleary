import tensorflow as tf
import pickle
import multiprocessing as mp
from task import LearningTask
import encode
from config import LOCAL_SIZE_RESTRICTION
import numpy as np

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

            self.steps_per_epoch = int(sum(1 for _ in stream)*self.training_ratio)

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
        model.add(tf.keras.layers.Embedding(300,50, input_shape=(None,), name="inputs"))
        forward = tf.keras.layers.LSTM(1000, activation=tf.keras.activations.tanh, recurrent_activation=tf.keras.activations.sigmoid, name="forward", recurrent_dropout=0, unroll=False, use_bias=True)
        model.add(forward)
        model.add(tf.keras.layers.Dense(1024, activation=tf.keras.activations.sigmoid, use_bias=True, name="outputs"))
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, clipnorm=1.0), loss=loss, metrics=["mae", "acc", "binary_crossentropy"])
        print(model.losses)
        return model

    def generate_data(self, kind="train"):
        with open('data/chemdata500x100x1024.pkl', 'rb') as output:
            # pickle.dump(chemdata,output)
            chemdata = pickle.load(output)

        #with mp.Pool(mp.cpu_count() - 2) as pool:
        while True:
            if LOCAL_SIZE_RESTRICTION != -1:
                stream = (x for x in list(chemdata.iterrows())[:LOCAL_SIZE_RESTRICTION])
            else:
                stream = chemdata.iterrows()
            for i in range(self.steps_per_epoch):
                result = next(stream)
                if kind=="train":
                    yield result[1][2], result[1][3:]


task = Classifier(input_encoder=encode.CharacterOrdEncoder(), output_encoder=encode.IntEncoder())
task.run()
