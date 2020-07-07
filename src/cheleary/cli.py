import click
from src.cheleary.task import LearningTask, load_task
from src.cheleary.encode import Encoder
from src.cheleary.dataprocessor import DataProcessor
from src.cheleary.models import Model
import pickle
import os

cli = click.Group()


@cli.command("train")
@click.argument("task_id", required=True)
@click.argument("input_encoder_id", required=True)
@click.argument("output_encoder_id", required=True)
@click.option("--raw-data", default=None)
@click.option("--data", default=None)
@click.option("--model", default=None)
@click.option("--epochs", default=1)
def train(task_id, raw_data, data, input_encoder_id, output_encoder_id, model, epochs):
    if os.path.exists(os.path.join("../../.tasks", task_id)):
        print(
            "Task already exists. If you want to continue learning, specify use the `continue` instead of `train`."
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


@cli.command("continue")
@click.argument("task_id", required=True)
@click.option("--epochs", default=1)
def cont(task_id, epochs):
    t = load_task(task_id)
    t.run(epochs=epochs)
    t.save()


@cli.command("test")
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

    @cli.command("collect-dl-data")
    @click.argument("path", required=True)
    def collect_dl_data(path):
        # Move this import here because it is not available everywhere

        print(path)
        print(os.getcwd())
        dprep = ChebiDataPreparer()
        chemdata = dprep.getDataForDeepLearning(10, 50)
        with open(path, "wb") as outf:
            pickle.dump(chemdata, outf)
