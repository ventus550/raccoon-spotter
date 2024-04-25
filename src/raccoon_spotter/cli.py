import os
import nb_clean
import pathlib
from typing import cast
import nbformat


"""Command line tools for manipulating a Kedro project.
Intended to be invoked via `kedro`."""
import click
from kedro.framework.cli.utils import CONTEXT_SETTINGS

@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""

@cli.command()
@click.option("--override", type=bool, default=False, is_flag=True, help="Override credentials if present.")
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

@cli.group()
def clean():
    """Kedro cleaning tools."""

@clean.command()
def data():
    click.secho("Deleting local data..." ,fg='yellow')
    def remove_files_except_gitkeep(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                remove_files_except_gitkeep(item_path)
            elif item != ".gitkeep":
                os.remove(item_path)

        # After removing files, check if the directory is empty
        if not os.listdir(directory) and directory != ".":
            os.rmdir(directory)

    try:
        remove_files_except_gitkeep("data")
    except:
        click.secho("Data cleaning procedure has encountered an error." ,fg='red')
    else:
        click.secho("Done." ,fg='yellow')

@clean.command()
def notebooks():

    def read_notebook(filepath: pathlib.Path) -> nbformat.NotebookNode:
        return cast(
            nbformat.NotebookNode,
            nbformat.read(filepath, as_version=nbformat.NO_CONVERT),  # type: ignore[no-untyped-call]
        )

    def sanitize_notebook(filepath: pathlib.Path):
        notebook = read_notebook(filepath)
        notebook = nb_clean.clean_notebook(notebook, remove_empty_cells=True)
        nbformat.write(notebook, filepath)

    def sanitize_directory(dir):
        for path in dir.iterdir():            
            if path.is_file() and path.suffix == ".ipynb":
                try:
                    sanitize_notebook(path)
                except:
                    click.secho(f"Failed to sanitize {path}!", fg="red")
            if path.is_dir():
                sanitize_directory(path)
    
    sanitize_directory(pathlib.Path(__file__).parent.parent.parent / "notebooks")
