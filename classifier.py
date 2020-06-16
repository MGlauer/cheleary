import tensorflow as tf
import pickle
import multiprocessing as mp
import numpy as np
from encode import encode_smiles, input_lenth
import os
from random import shuffle
def create_model(input_shape):
    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Input(shape=(None, input_lenth), ragged=True))
    model.add(tf.keras.layers.LSTM(1024, activation='relu', recurrent_activation='relu', input_shape=input_shape, name="forward"))
    #model.add(tf.keras.layers.Dense(10000, activation="relu"))
    #model.add(tf.keras.layers.Dense(5000, activation="tanh"))
    model.add(tf.keras.layers.Dense(1024, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def generate():
    with open('data/chemdata500x100x1024.pkl', 'rb') as output:
        # pickle.dump(chemdata,output)
        chemdata = pickle.load(output)

    with mp.Pool(mp.cpu_count() - 2) as pool:
        for result in pool.imap(encode_smiles, chemdata.iterrows()):
            yield result

model = create_model((None, input_lenth))
#model = cnn_model((max_len* input_lenth,))

tf.keras.utils.plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# fit model
#L = generate()

split = 0.7
prefix = f"data/split_{str(split).replace('.', '_')}/"
if os.path.exists(prefix+"train.p") and os.path.exists(prefix+"test.p") and os.path.exists(prefix+"eval.p"):
    train = pickle.load(open(prefix+"train.p", "rb"))
    test = pickle.load(open(prefix + "test.p", "rb"))
    # eval = pickle.load(open(prefix + "eval.p", "rb"))
else:
    print("No data dump found! Create new data dump")
    data = list(generate())
    shuffle(data)
    xs, ys = zip(*data)
    x = tf.ragged.constant(xs)
    y = tf.convert_to_tensor(ys)
    size = x.shape[0]
    train_size = int(size * split)
    test_size = int(size * (1-split) / 2)
    train = x[:train_size], y[:train_size]
    test = x[train_size:(train_size+test_size)], y[train_size:(train_size+test_size)]
    evalu = x[(train_size+test_size):], y[(train_size+test_size):]
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    pickle.dump(train, open(prefix + "train.p", "wb"))
    pickle.dump(test, open(prefix + "test.p", "wb"))
    pickle.dump(evalu, open(prefix + "eval.p", "wb"))
    print("done")

model.fit(*train, epochs=300, use_multiprocessing=True)
model.save("out")
