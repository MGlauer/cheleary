import os
from random import shuffle
import pickle
from config import LOCAL_SIZE_RESTRICTION
from encode import Encoder
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
            self.data_path = f'.data/cache/{os.path.basename(raw_data_path).split(".")[0]}/split_{int(split*100)}'
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

    def generate_data(self, kind="train", loop=False):
        with open(self.raw_data_path, "rb") as output:
            # pickle.dump(chemdata,output)
            chemdata = pickle.load(output)

        # with mp.Pool(mp.cpu_count() - 2) as pool:
        if LOCAL_SIZE_RESTRICTION != -1:
            stream = (x for x in list(chemdata.iterrows())[:LOCAL_SIZE_RESTRICTION])
        else:
            stream = chemdata.iterrows()
        for result in stream:
            yield result[1][1], result[1][2:]

    def load_data(self, kind="train", loop=False, cached=True):
        if not os.path.exists(os.path.join(self.data_path, f"{kind}.pkl")):
            print("No cached data found. Create new cache.")
            os.makedirs(self.data_path, exist_ok=True)
            data = list(self.generate_data())[:LOCAL_SIZE_RESTRICTION]
            shuffle(data)
            train_amount = int(len(data) * self.split)
            test_amount = int(len(data) * (1 - self.split) / 2)
            eval_amount = len(data) - (train_amount + test_amount)
            for _kind, l, r in [
                ("train", 0, train_amount),
                ("test", train_amount, train_amount + test_amount),
                ("eval", train_amount + test_amount, -1),
            ]:
                with open(os.path.join(self.data_path, f"{_kind}.pkl"), "wb") as pkl:
                    pickle.dump(
                        [
                            (
                                np.asarray([self.input_encoder.run(x)]),
                                np.asarray([self.output_encoder.run(y)]),
                            )
                            for x, y in data[l:r]
                        ],
                        pkl,
                    )
        with open(os.path.join(self.data_path, f"{kind}.pkl"), "rb") as pkl:
            data = pickle.load(pkl)
            while True:
                for x, y in data:
                    yield x, y
                if not loop:
                    break
