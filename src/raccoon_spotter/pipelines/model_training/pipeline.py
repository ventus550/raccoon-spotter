from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_model, train_model, upload_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_model,
                inputs=None,
                outputs="pretrained_model",
                name="build_model_node",
            ),
            node(
                func=train_model,
                inputs=["raccoon_features_data_array", "pretrained_model"],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=upload_model,
                inputs=[
                    "trained_model",
                    "params:upload.temporary_save_path",
                    "params:upload.skip",
                ],
                outputs=None,
                name="upload_model_node",
            ),
        ]
    )
