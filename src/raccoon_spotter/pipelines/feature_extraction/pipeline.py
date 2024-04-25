from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_test_split


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs="preprocessed_raccoon_data_array",
                outputs="raccoon_features_data_array",
                name="identity",
            ),
            node(
                func=train_test_split,
                inputs=["raccoon_features_data_array", "params:test_size"],
                outputs=[
                    "raccoon_train_features_data_array",
                    "raccoon_test_features_data_array",
                ]
            ),
        ]
    )
