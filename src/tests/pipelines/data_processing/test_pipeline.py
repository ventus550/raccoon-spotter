from kedro.pipeline import Pipeline
from raccoon_spotter.pipelines.data_processing.pipeline import (
    create_pipeline,
)

NUM_INPUT = 4
NUM_OUTPUT = 2


def test_pipeline_creation():
    """Test checking data_processing pipeline creation."""
    pipeline = create_pipeline()

    assert type(pipeline) == Pipeline
    assert len(pipeline.nodes) > 0
    assert len(pipeline.all_inputs()) == NUM_INPUT
    assert len(pipeline.all_outputs()) == NUM_OUTPUT
