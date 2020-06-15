import tensorflow as tf
import pickle
import multiprocessing as mp
import numpy as np
from encode import encode_smiles, input_lenth

def create_model(input_shape):
    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Input(shape=(None, input_lenth), ragged=True))
    model.add(tf.keras.layers.LSTM(1024, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, input_shape=input_shape, name="forward"))
    #model.add(tf.keras.layers.Dense(10000, activation="relu"))
    #model.add(tf.keras.layers.Dense(5000, activation="tanh"))
    #model.add(tf.keras.layers.Dense(1024, activation="tanh"))
    model.compile(optimizer='adam', loss="mse", metrics=["accuracy"])
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
xs, ys = zip(*generate())
x = tf.ragged.constant(xs)
y = tf.convert_to_tensor(ys)
#ds = tf.data.Dataset.from_tensor_slices((x, y))


model.fit(x, y, epochs=300, use_multiprocessing=True)
model.save("out")
