import tensorflow as tf
import pickle
import multiprocessing as mp
import numpy as np
from encode import encode_smiles, input_lenth
import os
from random import shuffle

loss =tf.keras.losses.MSE # BinaryCrossentropy()

LOCAL_SIZE_RESTRICTION = int(os.environ.get("CHEBI_SIZE_CON", -1))
EPOCHS = int(os.environ.get("EPOCHS", 300))

def create_model(input_shape):
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(None, input_lenth), ragged=True))
    forward = tf.keras.layers.LSTM(100, activation=tf.keras.activations.tanh, input_shape=input_shape, recurrent_activation=tf.keras.activations.tanh, name="forward", use_bias=True, bias_initializer="ones")
    #backward = tf.keras.layers.LSTM(100, activation=tf.keras.activations.tanh, input_shape=input_shape, recurrent_activation=tf.keras.activations.tanh, name="backward", go_backwards=True)
    #model.add(tf.keras.layers.Bidirectional(forward, backward_layer=backward))
    model.add(forward)
    #model.add(tf.keras.layers.Dense(10000, activation="relu"))
    #model.add(tf.keras.layers.Dense(5000, activation="tanh"))
    model.add(tf.keras.layers.Dense(1024, activation=tf.keras.activations.sigmoid, use_bias=True, bias_initializer="ones"))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01, clipnorm=1.0), loss=loss, metrics=["mae", "acc", "binary_crossentropy"])
    return model

def generate():
    with open('data/chemdata500x100x1024.pkl', 'rb') as output:
        # pickle.dump(chemdata,output)
        chemdata = pickle.load(output)

    with mp.Pool(mp.cpu_count() - 2) as pool:
        if LOCAL_SIZE_RESTRICTION != -1:
            stream = list(chemdata.iterrows())[:LOCAL_SIZE_RESTRICTION]
        else:
            stream = chemdata.iterrows()
        for result in pool.imap(encode_smiles, stream):
            yield result

model = create_model((None, input_lenth))
#model = cnn_model((max_len* input_lenth,))

tf.keras.utils.plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# fit model
#L = generate()

split = 0.7
prefix = f"data/split_{str(split).replace('.', '_')}/"


def load_data():
    if os.path.exists(prefix+"train.p") and os.path.exists(prefix+"test.p") and os.path.exists(prefix+"eval.p"):
        print("Load existing data dump")
        train = pickle.load(open(prefix+"train.p", "rb"))
        test = pickle.load(open(prefix + "test.p", "rb"))
        # eval = pickle.load(open(prefix + "eval.p", "rb"))
        print("done")
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

    return train, test

train, test = load_data()

print("Data: ", train[0].shape[0])

model.fit(*train, epochs=EPOCHS, use_multiprocessing=True)
model.summary()
model.save("out")


y_pred = model.predict(test[0])
y = test[1]
print("Loss:", loss(y, y_pred))

for y1, y2 in zip(y_pred, y):
    print(y1, y2)

