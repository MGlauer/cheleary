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

from encode import input_lenth, encode_smiles

tf.compat.v1.disable_eager_execution()





def lstm_model(input_shape):
    model = tf.keras.Sequential()
    inp = tf.keras.layers.Input(shape=(None, input_lenth))

    lstm_inp = tf.keras.layers.LSTM(1000, activation='relu', input_shape=input_shape, name="forward")(inp)

    reps = tf.keras.layers.RepeatVector(K.shape(inp)[1])(lstm_inp)

    lstm_out = tf.keras.layers.LSTM(1000, activation='relu', return_sequences=True, name="backwards")(reps)

    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_lenth))(lstm_out)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss="binary_crossentropy",
  metrics=["accuracy"])

    return model

def cnn_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1000, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(input_shape[0]))
    model.compile(optimizer='adam', loss='mse')
    return model


def generate():
    with open('data/chemdata500x100x1024.pkl', 'rb') as output:
        # pickle.dump(chemdata,output)
        chemdata = pickle.load(output)

    with mp.Pool(mp.cpu_count() - 2) as pool:
        last_len = None
        L = []
        for result in sorted([result for result in pool.imap(encode_smiles, chemdata.iterrows())], key=len):
            if len(result) == last_len:
                L.append(result)
            else:
                if last_len is not None:
                    yield L
                L = [result]
                last_len = len(result)
        yield L



"""
def create_ragged_tensor(i):
    L = []
    index = [0]
    for row in i:
        L += row
        index.append(index[-1]+len(row))
    return L, index
"""
#ds = tf.keras.preprocessing.sequence.pad_sequences(list(generate()))

#ds = tf.data.Dataset.from_generator(generate, output_types=(tf.int8), output_shapes=(None, None, input_lenth))

#ds = ds.reshape((len(ds), input_lenth*max_len))
model = lstm_model((None, input_lenth))
#model = cnn_model((max_len* input_lenth,))

tf.keras.utils.plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# fit model



inps = list(generate())


#data = tf.RaggedTensor.from_row_splits(*create_ragged_tensor(generate()))

for epoch in range(300):
    print("Epoch", epoch)
    for molecules in inps:
        inp = np.asarray(molecules)
        model.fit(inp, inp, use_multiprocessing=True)
model.save("out")

# demonstrate recreation
#yhat = model.predict(chemdata["encoding"], verbose=0)