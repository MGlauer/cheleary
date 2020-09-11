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
        if data_path is None:
            if raw_data_path is None:
                raise ValueError(
                    "A data processor needs a raw data path or a data path"
                )
            self.data_path = f".data/cache/{input_encoder._ID}/"
        else:
            self.data_path = data_path
        self.raw_data_path = raw_data_path
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
                with open(self.raw_data_path, "rb") as output:
                    chemdata = pickle.load(output)[:LOCAL_SIZE_RESTRICTION]
                with open(os.path.join(self.data_path, f"{_kind}.pkl"), "wb") as pkl:
                    features = chemdata.apply(self.input_encoder.run, axis=1)
                    labels = chemdata.apply(self.output_encoder.run, axis=1)

                    pickle.dump(
                        (
                            chemdata["MOLECULEID"],
                            chemdata["SMILES"],
                            tf.ragged.constant(features),
                            tf.convert_to_tensor(labels.tolist()),
                        ),
                        pkl,
                    )
        with open(os.path.join(self.data_path, f"{kind}.pkl"), "rb") as pkl:
            print("Use data cached at", self.data_path)
            return pickle.load(pkl)
