import tensorflow as tf
import pickle
import multiprocessing as mp
from task import LearningTask
from encode import encode_smiles, input_lenth
from config import LOCAL_SIZE_RESTRICTION
import numpy as np

class Classifier(LearningTask):
    ID = 'classifier'

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
        loss = tf.keras.losses.MSE  # BinaryCrossentropy()

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(300,20, input_shape=(None,), name="inputs"))
        #model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(190)))#
        forward = tf.keras.layers.LSTM(100, activation=tf.keras.activations.tanh, recurrent_activation=tf.keras.activations.tanh, name="forward", use_bias=True, bias_initializer="ones")
        #backward = tf.keras.layers.LSTM(100, activation=tf.keras.activations.tanh, input_shape=input_shape, recurrent_activation=tf.keras.activations.tanh, name="backward", go_backwards=True)
        #model.add(tf.keras.layers.Bidirectional(forward, backward_layer=backward))
        model.add(forward)
        #model.add(tf.keras.layers.Dense(10000, activation="relu"))
        #model.add(tf.keras.layers.Dense(5000, activation="tanh"))
        model.add(tf.keras.layers.Dense(1024, activation=tf.keras.activations.sigmoid, use_bias=True, bias_initializer="ones", name="outputs"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0), loss=loss, metrics=["mae", "acc", "binary_crossentropy"])
        print(model.losses)
        return model

    def generate_data(self):
        with open('data/chemdata500x100x1024.pkl', 'rb') as output:
            # pickle.dump(chemdata,output)
            chemdata = pickle.load(output)

        #with mp.Pool(mp.cpu_count() - 2) as pool:
            if LOCAL_SIZE_RESTRICTION != -1:
                stream = list(chemdata.iterrows())[:LOCAL_SIZE_RESTRICTION]
            else:
                stream = chemdata.iterrows()
            for result in stream:
                smiles = [ord(s) for s in result[1][2]]
                labels = [int(l) for l in result[1][3:]]
                yield (dict(inputs=np.asarray([smiles, smiles])), dict(outputs=np.asarray([labels, labels])))

task = Classifier()
task.run()