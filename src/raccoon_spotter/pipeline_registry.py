"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    pipelines["etl"] = (
        pipelines["data_loading"]
        + pipelines["data_processing"]
        + pipelines["feature_extraction"]
    )
    pipelines["train"] = pipelines["model_training"]
    return pipelines
