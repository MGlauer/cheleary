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

tf.compat.v1.disable_eager_execution()


def handle_data_line(line):
    smiles = line.split(" ")[0]
    es = encode_smiles(smiles)
    return es, es


class Autoencoder(LearningTask):
    ID="Autoencoder"

    def create_model(self):
        model = tf.keras.Sequential()
        inp = tf.keras.layers.Input(shape=(None, input_lenth))

        lstm_inp = tf.keras.layers.LSTM(1000, activation='relu', input_shape=(None, 190), name="forward")(inp)

        reps = tf.keras.layers.RepeatVector(K.shape(inp)[1])(lstm_inp)

        lstm_out = tf.keras.layers.LSTM(1000, activation='relu', return_sequences=True, name="backwards")(reps)

        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_lenth))(lstm_out)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss="binary_crossentropy",
      metrics=["accuracy"])

        return model

    def generate_data(self):
        with open('data/molecules.smi', 'r') as inp:
            # pickle.dump(chemdata,output)

            with mp.Pool(mp.cpu_count() - 2) as pool:
                for result in pool.imap_unordered(handle_data_line, inp.readlines()):
                    yield result

        print(atom_chars)




task = Autoencoder()
task.run()