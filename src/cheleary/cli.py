import click
from cheleary.task import LearningTask, load_task
from cheleary.encode import Encoder
from cheleary.dataprocessor import DataProcessor
from cheleary.models import Model
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
@click.option(
    "--raw-data",
    default=None,
    help="Path to a raw dataset that should be used for this learning task. Cheleary will try to",
)
@click.option("--data", default=None, help="Path to a data dump created by cheleary")
@click.option("--model", default=None, help="Identifier for the model to use")
@click.option("--epochs", default=1, help="Number of epochs to train")
def train(task_id, raw_data, data, input_encoder_id, output_encoder_id, model, epochs):
    if os.path.exists(os.path.join(".tasks", task_id)):
        print(
            f"Task '{task_id}' already exists. If you want to continue learning, specify use the `continue` instead of `train`."
        )
        exit(1)
    input_encoder = Encoder.get(input_encoder_id)()
    output_encoder = Encoder.get(output_encoder_id)()
    dp = DataProcessor(
        raw_data_path=raw_data,
        data_path=data,
        input_encoder=input_encoder,
        output_encoder=output_encoder,
    )
    t = LearningTask(
        identifier=task_id, dataprocessor=dp, model_container=Model.get(model)()
    )
    t.run(epochs=epochs)
    t.save()


@cli.command("continue", help="Load existing task and continue training.")
@click.argument("task_id", required=True)
@click.option("--epochs", default=1)
def cont(task_id, epochs):
    t = load_task(task_id)
    t.run(epochs=epochs)
    t.save()


@cli.command("test", help="Load existing task and run tests with cached test data.")
@click.argument("task_id", required=True)
def test(task_id):
    t = load_task(task_id)
    t.test()


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

        print(path)
        print(os.getcwd())
        dprep = ChebiDataPreparer()
        chemdata = dprep.getDataForDeepLearning(10, 50)
        with open(path, "wb") as outf:
            pickle.dump(chemdata, outf)


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
