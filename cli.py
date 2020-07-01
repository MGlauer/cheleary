import click
from task import get_task
from encode import get_encoder
cli = click.Group()

@cli.command
@click.argument("task_id", required=True)
@click.argument("input_encoder_id", required=True)
@click.argument("output_encoder_id", required=True)
def train(task_id, input_encoder_id, output_encoder_id):
    task_cls = get_task(task_id)
    input_encoder_cls = get_encoder(input_encoder_id)
    output_encoder_cls = get_encoder(output_encoder_id)
    t = task_cls(input_encoder_cls(), output_encoder_cls())
    t.run()

@cli.command
@click.argument("task_id", required=True)
@click.argument("model_path", required=True)
@click.argument("input_encoder_id", required=True)
@click.argument("output_encoder_id", required=True)
def train(task_id, model_path, input_encoder_id, output_encoder_id):
    task_cls = get_task(task_id)
    input_encoder_cls = get_encoder(input_encoder_id)
    output_encoder_cls = get_encoder(output_encoder_id)
    t = task_cls(input_encoder_cls(), output_encoder_cls(), model_path=model_path)
    t.test()