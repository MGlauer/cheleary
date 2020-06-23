from itertools import chain
import pickle
from pysmiles.read_smiles import _tokenize, TokenType
import pandas
import numpy as np
import re
import multiprocessing as mp


import tensorflow as tf
import tensorflow.keras.backend as K
  # elements + bonds + ringnumbers
from task import LearningTask
from encode import input_lenth, encode_smiles, atom_chars

def handle_data_line(line):
    smiles = line.split(" ")[0]
    es = encode_smiles(smiles)
    return es, es


class Autoencoder(LearningTask):
    ID="Autoencoder"

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.batch_size = 2
        with open('data/molecules.smi', 'r') as inp:
            # pickle.dump(chemdata,output)
            lines = inp.readlines()
            self.steps_per_epoch = len(lines)

    @property
    def input_shape(self):
        return tf.TensorShape([None, input_lenth])

    @property
    def output_shape(self):
        return tf.TensorShape([None, input_lenth])

    @property
    def input_datatype(self):
        return tf.bool

    @property
    def output_datatype(self):
        return tf.bool

    def create_model(self):
        inp = tf.keras.layers.Input(shape=(None, input_lenth), ragged=True, name="inputs")

        lstm_inp = tf.keras.layers.LSTM(1000, activation='relu', return_sequences=True, name="forward")(inp)

        #reps = tf.keras.layers.RepeatVector(tf.TensorShape([None, input_lenth])[1])(lstm_inp)

        lstm_out = tf.keras.layers.LSTM(1000, activation='relu', return_sequences=True, name="backwards")(lstm_inp)

        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_lenth), input_shape=(None, input_lenth), name="outputs")(lstm_out)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss="binary_crossentropy",
      metrics=["accuracy"])

        return model

    def generate_data(self):
        with open('data/molecules.smi', 'r') as inp:
            x = []
            y = []
            for result in map(handle_data_line, inp.readlines()):
                x.append(result[0])
                y.append(result[1])
                if len(x) >= self.batch_size:
                    yield dict(inputs=x), \
                          dict(outputs=y)
                    x = []
                    y = []





task = Autoencoder()
task.run()