from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["raccoon_features_data_array", "params:split_dataset"],
                outputs=["X_test", "y_test"],
            ),
            node(
                func=evaluate_model,
                inputs={
                    "trained_model": "trained_model",
                    "X_test": "X_test",
                    "y_test": "y_test",
                },
                outputs="metrics",
            ),
        ]
    )
