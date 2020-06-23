from config import EPOCHS
import os
import pickle
from random import shuffle
import tensorflow as tf
from encode import input_lenth

class LearningTask:
    ID = 'classifier'

    def __init__(self, model=None, model_path=None, data_path=None, split=0.7):
        if model_path is None:
            self.model_path = f'models/{self.ID}'
        else:
            self.model_path = model_path
        self.split = split
        if data_path is None:
            self.data_path = f'data/{self.ID}/split_{split*100}'
        else:
            self.data_path = data_path
        if model is None:
            self.model = self.create_model()
        else:
            self.model = model

        self.steps_per_epoch = None

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def input_datatype(self):
        return tf.int32

    @property
    def output_datatype(self):
        return tf.int32

    def create_model(self):
        raise NotImplementedError

    def generate_data(self):
        raise NotImplementedError

    def load_data(self):

        train_path = os.path.join(self.data_path, "train.p")
        test_path = os.path.join(self.data_path, "test.p")
        eval_path = os.path.join(self.data_path, "eval.p")
        if self.data_path is not None and os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(eval_path):
            print("Load existing data dump")
            train = pickle.load(open(train_path, "rb"))
            test = pickle.load(open(test_path, "rb"))
            # eval = pickle.load(open(eval_path, "rb"))
            print("done")
        else:
            print("No data dump found! Create new data dump")
            return self.generate_data()


    def train_model(self, training_data, test_data=None, save_model=None,
                    epochs=EPOCHS):
        #print("Data: ", len(training_data[0]))
        self.model.summary()
        self.model.fit(training_data, epochs=epochs, shuffle=False, steps_per_epoch=self.steps_per_epoch)

        if save_model:
            self.model.save(save_model)

        if test_data:
            y_pred = self.model.predict(test_data[0])
            y = test_data[1]

            for y1, y2 in zip(y_pred, y):
                print(y1, y2)

    def run(self):
        #dataset= self.load_data()
        dataset = tf.data.Dataset.from_generator(self.load_data, output_shapes=(dict(inputs=self.input_shape), dict(outputs=self.output_shape)), output_types=(dict(inputs=self.input_datatype), dict(outputs=self.output_datatype)))
        x = dataset.take(1)
        print("Start training")
        self.train_model(
            dataset,
            test_data=None
        )
