import click
from task import get_task
import classifier
from encode import get_encoder
cli = click.Group()

@cli.command("train")
@click.argument("task_id", required=True)
@click.argument("input_encoder_id", required=True)
@click.argument("output_encoder_id", required=True)
@click.option("--model-path", default=None)
@click.option("--epochs", default=1)
def train(task_id, input_encoder_id, output_encoder_id, model_path, epochs):
    task_cls = get_task(task_id)
    input_encoder_cls = get_encoder(input_encoder_id)
    output_encoder_cls = get_encoder(output_encoder_id)
    t = task_cls(input_encoder=input_encoder_cls(), output_encoder=output_encoder_cls(), model_path=model_path)
    t.run(epochs=epochs)

@cli.command("test")
@click.argument("task_id", required=True)
@click.argument("input_encoder_id", required=True)
@click.argument("output_encoder_id", required=True)
@click.option("--model-path", default=None)
def test(task_id, model_path, input_encoder_id, output_encoder_id):
    task_cls = get_task(task_id)
    input_encoder_cls = get_encoder(input_encoder_id)
    output_encoder_cls = get_encoder(output_encoder_id)
    t = task_cls(input_encoder=input_encoder_cls(), output_encoder=output_encoder_cls(), model_path=model_path)
    t.test()