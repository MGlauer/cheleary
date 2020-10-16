import os
import tensorflow as tf

from cheleary.dataprocessor import DataProcessor
from cheleary.models import Model
from cheleary.encode import Encoder
from cheleary.losses import SparseLoss
import pandas as pd
import json


class LearningTask:
    def __init__(
        self,
        identifier,
        dataprocessor: DataProcessor,
        model: Model = None,
        batch_size=1,
        split=0.7,
        prev_epochs=None,
        load_model=False,
        load_best=False,
    ):

        self.identifier = identifier

        self.dataprocessor = dataprocessor

        self.batch_size = batch_size

        self.split = split

        self.last_epoch = 0

        self.load_best = load_best

        if load_model:
            folder = "best" if load_best else "checkpoints"
            last = max(os.listdir(os.path.join(self._model_root, folder)))
            self.last_epoch = int(last)
            path = os.path.join(self._model_root, folder, last)
            print(f"Load model", path)
            self.model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "SparseLoss": SparseLoss,
                    # "F1Score": tfa.metrics.F1Score,
                },
                compile=True,
            )
        else:
            print(f"Build new model")
            self.model = model.build(
                self.dataprocessor.input_shape, self.dataprocessor.output_shape
            )
            self.save_config()

        if prev_epochs is None:
            self._prev_epochs = []
        else:
            self._prev_epochs = prev_epochs

    def create_model(self):
        raise NotImplementedError

    def train_model(self, x, y, test_data=None, epochs=1):
        self._prev_epochs.append(epochs)
        self.model.summary()
        cp_name = "{epoch:03d}"
        os.makedirs(os.path.join(self._model_root, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self._model_root, "best"), exist_ok=True)
        log_callback = tf.keras.callbacks.CSVLogger(
            os.path.join(self._model_root, "training.log"), append=True
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self._model_root, "checkpoints", cp_name),
            save_freq="epoch",
            period=25,
        )
        checkpoint_callback_best = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self._model_root, "best", cp_name),
            save_best_only=True,
            save_freq="epoch",
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=25, restore_best_weights=True
        )
        self.model.fit(
            x,
            y,
            epochs=self.last_epoch + epochs,
            shuffle=True,
            callbacks=[
                log_callback,
                checkpoint_callback,
                checkpoint_callback_best,
                # early_stop,
            ],
            verbose=2,
            batch_size=128,
            validation_data=test_data,
            initial_epoch=self.last_epoch,
            use_multiprocessing=True,
        )
        if self.last_epoch + epochs % 25:
            print("Create end-of-training checkpoint")
            self.model.save(
                os.path.join(
                    self._model_root, "checkpoints", f"{self.last_epoch+epochs:03d}"
                )
            )

    @property
    def _model_root(self):
        return os.path.join(".tasks", self.dataprocessor.dataset, self.identifier)

    @property
    def _train_log_path(self):
        return os.path.join(self._model_root, "training.log")

    def save_config(self):
        os.makedirs(self._model_root)
        with open(os.path.join(self._model_root, "config.json"), "w") as f:
            json.dump(self.config, f)

    def test_model(self, training_data, path=None, name="test"):
        self.model.summary()
        path = path or os.path.join(
            self._model_root, name, "best" if self.load_best else "last"
        )
        os.makedirs(path, exist_ok=True)
        ids, smiles, features, labels = training_data
        pred = self.model.predict(features)
        tf.metrics.Precision(thresholds=0.5)(labels, pred)
        metrics = [
            "binary_accuracy",
            "precision",
            "recall",
            "f1",
            "cosine_similarity",
            "area_under_roc",
        ]
        class_eval = pd.DataFrame(
            (
                (cid, *self._evaluate(labels[:, cls], pred[:, cls]))
                for cid, cls in zip(
                    self.dataprocessor.label_headers, range(tf.shape(labels)[1])
                )
            ),
            columns=["id"] + metrics,
        ).sort_values("f1")
        molecule_eval = pd.DataFrame(
            (
                (cid, smile, *self._evaluate(labels[mol, :], pred[mol, :]))
                for cid, smile, mol in zip(ids, smiles, range(tf.shape(labels)[0]))
            ),
            columns=["id", "SMILES"] + metrics,
        ).sort_values("f1")

        fig = px.box(molecule_eval["precision"], x="time", y="total_bill", points="all")
        fig.show()

        with open(os.path.join(path, "classes.csv"), "w") as json_out:
            class_eval.to_csv(json_out)

        with open(os.path.join(path, "molecules.csv"), "w") as json_out:
            molecule_eval.to_csv(json_out)

        with open(os.path.join(path, "test.csv"), "w") as fout:
            for cid, smile, y_real, y_pred in zip(ids, smiles, labels.numpy(), pred):
                fout.write(
                    cid
                    + ","
                    + smile
                    + ","
                    + (",".join(map(lambda x: str(int(x)), y_real)) + "\n")
                )
                fout.write(",," + (",".join(map(str, y_pred)) + "\n"))

    @staticmethod
    def _evaluate(y_true, y_pred):
        p = tf.metrics.Precision(thresholds=0.5)(y_true, y_pred).numpy()
        r = tf.metrics.Recall(thresholds=0.5)(y_true, y_pred).numpy()
        return (
            tf.metrics.BinaryAccuracy(threshold=0.5)(y_true, y_pred).numpy(),
            p,
            r,
            2 * p * r / (p + r) if p + r != 0 else 0,
            tf.metrics.CosineSimilarity()(y_true, y_pred).numpy(),
            tf.metrics.AUC()(y_true, y_pred).numpy(),
        )

    def run(self, epochs=1):
        _, _, x, y = self.dataprocessor.load_data(kind="train", loop=True)
        _, _, x_test, y_test = self.dataprocessor.load_data(kind="test")
        print("Start training")
        self.train_model(x, y, test_data=(x_test, y_test), epochs=epochs)
        print("Stop training")

    def test(self, path=None):
        dataset = self.dataprocessor.load_data(kind="test")
        print("Start testing")
        self.test_model(dataset, path=path)

    def eval(self, path=None):
        dataset = self.dataprocessor.load_data(kind="eval")
        print("Start evaluation")
        self.test_model(dataset, path=path, name="eval")

    @property
    def config(self):
        return dict(
            identifier=self.identifier,
            dataset=self.dataprocessor.dataset,
            input_encoder=self.dataprocessor.input_encoder._ID,
            model=self.model.to_json(),
            output_encoder=self.dataprocessor.output_encoder._ID,
        )

    def __repr__(self):
        return json.dumps(self.config)


def load_task(dataset, identifier, load_best=False):
    with open(os.path.join(".tasks", dataset, identifier, "config.json")) as fin:
        config = json.load(fin)
    return load_from_strings(**config, load_model=True, load_best=load_best)


def load_from_strings(
    identifier,
    dataset,
    input_encoder,
    model,
    output_encoder,
    load_model=False,
    load_best=False,
):
    m = tf.keras.models.model_from_json(model)
    m.compile()
    ie = Encoder.get(input_encoder)()
    oe = Encoder.get(output_encoder)()
    dp = DataProcessor(dataset=dataset, input_encoder=ie, output_encoder=oe,)
    return LearningTask(
        identifier=identifier,
        dataprocessor=dp,
        load_model=load_model,
        load_best=load_best,
    )
