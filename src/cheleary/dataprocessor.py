import os
from random import shuffle
import pickle
from cheleary.config import LOCAL_SIZE_RESTRICTION
from cheleary.encode import Encoder
import tensorflow as tf
import numpy as np

_DPS = {}


class DataProcessor:
    def __init__(
        self,
        raw_data_path=None,
        data_path=None,
        split=0.7,
        input_encoder: Encoder = None,
        output_encoder: Encoder = None,
    ):
        self.split = split
        self.raw_data_path = raw_data_path or ".data/splits"
        self.data_path = data_path or f".data/cache/{input_encoder._ID}/"
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder
        self.length = int(sum(1 for _ in self.load_data(kind="train")))

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def input_datatype(self):
        raise NotImplementedError

    @property
    def output_datatype(self):
        raise NotImplementedError

    def encode_row(self, row):
        return (
            row[0],
            row[1],
            self.input_encoder.run(row),
            self.output_encoder.run(row),
        )

    def load_data(self, kind="train", loop=False, cached=True):
        if not os.path.exists(os.path.join(self.data_path, f"{kind}.pkl")):
            os.makedirs(self.data_path)
            for _kind in ["train", "test", "eval"]:
                with open(
                    os.path.join(self.raw_data_path, f"{_kind}.pkl"), "rb"
                ) as output:
                    chemdata = pickle.load(output)
                with open(os.path.join(self.data_path, f"{_kind}.pkl"), "wb") as pkl:
                    features = chemdata.apply(self.input_encoder.run, axis=1)
                    labels = chemdata.apply(self.output_encoder.run, axis=1)
                    if len(set(map(len, features))) > 1:
                        input_tensor = tf.ragged.constant(features)
                    else:
                        input_tensor = tf.convert_to_tensor(features.tolist())
                    pickle.dump(
                        (
                            chemdata["MOLECULEID"],
                            chemdata["SMILES"],
                            input_tensor,
                            tf.convert_to_tensor(labels.tolist()),
                        ),
                        pkl,
                    )
        with open(os.path.join(self.data_path, f"{kind}.pkl"), "rb") as pkl:
            print("Use data cached at", self.data_path)
            return pickle.load(pkl)
