# Command line tools for manipulating a Kedro project.
# Intended to be invoked via `kedro`.


import os
import pathlib
import subprocess
from typing import cast

import click
import nb_clean
import nbformat
from kedro.framework.cli.project import (
    ASYNC_ARG_HELP,
    CONF_SOURCE_HELP,
    CONFIG_FILE_HELP,
    FROM_INPUTS_HELP,
    FROM_NODES_HELP,
    LOAD_VERSION_HELP,
    NODE_ARG_HELP,
    PARAMS_ARG_HELP,
    PIPELINE_ARG_HELP,
    RUNNER_ARG_HELP,
    TAG_ARG_HELP,
    TO_NODES_HELP,
    TO_OUTPUTS_HELP,
    project_group,
)
from kedro.framework.cli.utils import (
    CONTEXT_SETTINGS,
    _config_file_callback,
    _split_load_versions,
    _split_params,
    env_option,
    split_node_names,
    split_string,
)
from kedro.framework.session import KedroSession
from kedro.utils import load_obj


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@project_group.command()
@click.option(
    "--from-inputs", type=str, default="", help=FROM_INPUTS_HELP, callback=split_string
)
@click.option(
    "--to-outputs", type=str, default="", help=TO_OUTPUTS_HELP, callback=split_string
)
@click.option(
    "--from-nodes",
    type=str,
    default="",
    help=FROM_NODES_HELP,
    callback=split_node_names,
)
@click.option(
    "--to-nodes", type=str, default="", help=TO_NODES_HELP, callback=split_node_names
)
@click.option(
    "--nodes", "-n", "node_names", type=str, multiple=True, help=NODE_ARG_HELP
)
@click.option(
    "--runner", "-r", type=str, default=None, multiple=False, help=RUNNER_ARG_HELP
)
@click.option("--async", "is_async", is_flag=True, multiple=False, help=ASYNC_ARG_HELP)
@env_option
@click.option("--tags", "-t", type=str, multiple=True, help=TAG_ARG_HELP)
@click.option(
    "--load-versions",
    "-lv",
    type=str,
    multiple=True,
    help=LOAD_VERSION_HELP,
    callback=_split_load_versions,
)
@click.option("--pipeline", "-p", type=str, default=None, help=PIPELINE_ARG_HELP)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help=CONFIG_FILE_HELP,
    callback=_config_file_callback,
)
@click.option(
    "--conf-source",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help=CONF_SOURCE_HELP,
)
@click.option(
    "--params",
    type=click.UNPROCESSED,
    default="",
    help=PARAMS_ARG_HELP,
    callback=_split_params,
)
def run(  # noqa: PLR0913
    tags,
    env,
    runner,
    is_async,
    node_names,
    to_nodes,
    from_nodes,
    from_inputs,
    to_outputs,
    load_versions,
    pipeline,
    config,
    conf_source,
    params,
):
    """Run the pipeline."""

    runner = load_obj(runner or "SequentialRunner", "kedro.runner")
    tags = tuple(tags)
    node_names = tuple(node_names)

    with KedroSession.create(
        env=env, conf_source=conf_source, extra_params=params
    ) as session:
        session.run(
            tags=tags,
            runner=runner(is_async=is_async),
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            from_inputs=from_inputs,
            to_outputs=to_outputs,
            load_versions=load_versions,
            pipeline_name=pipeline,
        )


@cli.command()
@click.option(
    "--override",
    type=bool,
    default=False,
    is_flag=True,
    help="Override credentials if present.",
)
def authenticate(override=False):
    click.secho(
        (
            "Welcome to the authentication procedure.\n"
            "This protocol will guide you through the creation of your conf/local/credentials.yaml file.\n"
            "The information you are about to enter is sensitive and should not be shared with other users.\n"
        ),
        fg="yellow",
    )
    aws_access_key_id = click.prompt("aws_access_key_id", type=str)
    aws_secret_access_key = click.prompt("aws_secret_access_key", type=str)
    wandb_access = click.prompt(
        "[wandb_access]", type=str, default="", show_default=False
    )

    # Check if conf/local directory exists
    if not os.path.isdir("conf/local"):
        raise click.UsageError(
            "conf/local directory does not exist. Make sure you're in the root of the project."
        )

    if os.path.exists("conf/local/credentials.yaml") and not override:
        raise click.UsageError(
            "conf/local/credentials.yaml already exists. Remove it or specify --override option to replace it."
        )

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

    click.secho("Credentials written to conf/local/credentials.yaml", fg="yellow")


@cli.group()
def clean():
    """Kedro cleaning tools."""


@clean.command()
def data():
    click.secho("Deleting local data...", fg="yellow")

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
    except Exception:
        click.secho("Data cleaning procedure has encountered an error.", fg="red")
    else:
        click.secho("Done.", fg="yellow")


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
                except Exception:
                    click.secho(f"Failed to sanitize {path}!", fg="red")
            if path.is_dir():
                sanitize_directory(path)

    sanitize_directory(pathlib.Path(__file__).parent.parent.parent / "notebooks")


@cli.command()
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True, dir_okay=False, resolve_path=False),
    help="Model path to be used for this application instance.",
)
def app(model=None):
    click.secho("Starting application building process.", fg="yellow")

    if model:
        click.secho(f"Using custom model path ({model}).", fg="yellow")
        model = f"--build-arg MODEL_PATH={model}"

    subprocess.run(
        ["sudo", "bash", "-c", f"docker build {model or ''} -t raccoon-spotter ."],
        text=True,
        check=True,
    )

    subprocess.run(
        ["sudo", "bash", "-c", "docker run -it --rm -p 5000:5000 raccoon-spotter"],
        check=False,
    )
