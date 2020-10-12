import click
from cheleary.task import LearningTask, load_task
from cheleary.encode import Encoder
from cheleary.dataprocessor import DataProcessor
from cheleary.models import Model
from cheleary.analysis import analyze as an
from sklearn.model_selection import train_test_split
import pickle
import os

cli = click.Group(
    help="Cheleary is a toolkit to build an easy training environment. It implements different kinds of"
    "encodings and network structures based on `keras` and `tensorflow`. The main focus are learning"
    "tasks around `CHEBI` - an ontology about chemicals"
)


@cli.command(
    "train",
    help="Construct and train a new task. The .tasks folder must not contain "
    "a task with the same `TASK_ID`. If you want to continue the "
    "training of an existing model, use `cheleary continue` instead. "
    "Input data is encoded using the encoder identified by "
    "`INPUT_ENCODER_ID` and the targets by `OUTPUT_ENCODER_ID`.",
)
@click.argument("task_id", required=True)
@click.argument("input_encoder_id", required=True)
@click.argument("output_encoder_id", required=True)
@click.argument(
    "dataset",
    required=True,
    # help="Path to a raw dataset that should be used for this learning task. Cheleary will try to",
)
@click.option("--model", default=None, help="Identifier for the model to use")
@click.option("--epochs", default=1, help="Number of epochs to train")
def train(task_id, input_encoder_id, output_encoder_id, dataset, model, epochs):
    if os.path.exists(os.path.join(".tasks", dataset, task_id)):
        print(
            f"Task '{task_id}' already exists. If you want to continue learning, specify use the `continue` instead of `train`."
        )
        exit(1)
    input_encoder = Encoder.get(input_encoder_id)()
    output_encoder = Encoder.get(output_encoder_id)()
    dp = DataProcessor(
        dataset=dataset, input_encoder=input_encoder, output_encoder=output_encoder,
    )
    t = LearningTask(identifier=task_id, dataprocessor=dp, model=Model.get(model)())
    t.run(epochs=epochs)


@cli.command("continue", help="Load existing task and continue training.")
@click.argument("task_id", required=True)
@click.argument("dataset", required=True)
@click.option("--epochs", default=1)
def cont(task_id, dataset, epochs):
    t = load_task(dataset, task_id)
    t.run(epochs=epochs)


@cli.command("test", help="Load existing task and run tests with cached test data.")
@click.argument("task_id", required=True)
@click.option("--path", default=None)
def test(task_id, path):
    t = load_task(task_id, load_best=True)
    t.test(path)


@cli.command("eval", help="Load existing task and run tests with cached test data.")
@click.argument("task_id", required=True)
@click.option("--path", default=None)
def eval(task_id, path):
    t = load_task(task_id, load_best=True)
    t.eval(path)


@cli.command("analyze")
@click.argument("in-path", required=True)
@click.argument("out-path", required=True)
def analyze(in_path, out_path):
    an(in_path, out_path)


try:
    from chebidblite.learnhelper import ChebiDataPreparer
except ModuleNotFoundError:
    print(
        "`ChebiDataPreparer` could not be loaded. Related commands are not available."
    )
else:

    @cli.command(
        "collect-dl-data",
        help="Command line interface for ChebiDataPreparer.getDataForDeepLearning. Creates a pickled dataset"
        "at `PATH`.",
    )
    @click.argument("path", required=True)
    def collect_dl_data(path):
        # Move this import here because it is not available everywhere
        os.makedirs(os.path.join(".data", "splits"), exist_ok=True)
        dprep = ChebiDataPreparer()
        chemdata = dprep.getDataForDeepLearning(500, 100)
        train, remainder = train_test_split(chemdata, shuffle=True, test_size=0.3)
        test, eval = train_test_split(remainder, shuffle=False, test_size=0.7)
        d = {"test": test, "train": train, "eval": eval}
        for kind in ["train", "test", "eval"]:
            with open(os.path.join(".data", "splits", f"{kind}.pkl"), "wb") as pkl:
                pickle.dump(
                    d[kind], pkl,
                )


def _list_registerables(reg_cls):
    for e, cls in reg_cls.list_identifiers():
        d = cls._doc()
        print(e, "-", d if d else "No description")


@cli.command(help="List all available encoders")
def list_encoders():
    _list_registerables(Encoder)


@cli.command(help="List all available models")
def list_models():
    _list_registerables(Model)
