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
        dataset,
        split=0.7,
        input_encoder: Encoder = None,
        output_encoder: Encoder = None,
    ):
        self.split = split
        self.dataset = dataset
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder
        self.length = int(sum(1 for _ in self.load_data(kind="train")))

    @property
    def raw_data_path(self):
        return f".data/{self.dataset}/raw"

    @property
    def data_path(self):
        return (
            f".data/{self.dataset}/{self.input_encoder._ID}/{self.output_encoder._ID}"
        )

    @property
    def input_shape(self):
        data = self.load_data(kind="train")
        return data[2].shape

    @property
    def output_shape(self):
        return self.load_data(kind="train")[3].shape

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

                    features = chemdata.apply(self.input_encoder.run, axis=1)
                    labels = chemdata.apply(self.output_encoder.run, axis=1)
                    # Filter invalid rows
                    filter = features.notna()
                    features = features[filter]
                    labels = labels[filter]
                    if len(set(map(len, features))) > 1:
                        input_tensor = tf.ragged.constant(features)
                    else:
                        input_tensor = tf.convert_to_tensor(features.tolist())
                    if len(set(map(len, labels))) > 1:
                        label_tensor = tf.ragged.constant(labels)
                    else:
                        label_tensor = tf.convert_to_tensor(labels.tolist())
                    with open(
                        os.path.join(self.data_path, f"{_kind}.pkl"), "wb"
                    ) as pkl:
                        pickle.dump(
                            (
                                chemdata["MOLECULEID"],
                                chemdata["SMILES"],
                                input_tensor,
                                label_tensor,
                            ),
                            pkl,
                        )
        with open(os.path.join(self.data_path, f"{kind}.pkl"), "rb") as pkl:
            print("Use data cached at", self.data_path)
            return pickle.load(pkl)
