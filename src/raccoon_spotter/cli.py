import os

"""Command line tools for manipulating a Kedro project.
Intended to be invoked via `kedro`."""
import click
from kedro.framework.cli.utils import CONTEXT_SETTINGS

@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""

@cli.command()
@click.option("--override", type=bool, default=False, is_flag=True)
def authenticate(override=False):
    click.secho((
        "Welcome to the authentication procedure.\n"
        "This protocol will guide you through the creation of your conf/local/credentials.yaml file.\n"
        "The information you are about to enter is sensitive and should not be shared with other users.\n"
    ),fg='yellow')
    aws_access_key_id = click.prompt("aws_access_key_id", type=str)
    aws_secret_access_key = click.prompt("aws_secret_access_key", type=str)
    wandb_access = click.prompt("[wandb_access]", type=str, default="", show_default=False)

    # Check if conf/local directory exists
    if not os.path.isdir("conf/local"):
        raise click.UsageError("conf/local directory does not exist. Make sure you're in the root of the project.")

    if os.path.exists("conf/local/credentials.yaml") and not override:
        raise click.UsageError("conf/local/credentials.yaml already exists. Remove it or specify --override option to replace it.")

    # Prepare the YAML content
    yaml_content = f"""\
aws_access:
    client_kwargs:
        aws_access_key_id: {aws_access_key_id}
        aws_secret_access_key: {aws_secret_access_key}
        region_name: "eu-west-2"
wandb_access: {wandb_access}
"""

    with open("conf/local/credentials.yaml", "w") as file:
        file.write(yaml_content)

    print("Credentials written to conf/local/credentials.yaml")