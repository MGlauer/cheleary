import tensorflow as tf
import pickle
import multiprocessing as mp
from .task import LearningTask
from encode import encode_smiles, input_lenth
from config import LOCAL_SIZE_RESTRICTION



class Classifier(LearningTask):
    ID = 'classifier'

    def create_model(self):
        input_shape = (None, input_lenth)
        loss = tf.keras.losses.MSE  # BinaryCrossentropy()

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

    def generate_data(self):
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

task = Classifier()