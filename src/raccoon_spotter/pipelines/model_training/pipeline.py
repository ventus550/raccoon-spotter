from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_model, train_model, upload_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_model,
                inputs=None,
                outputs="untrained_model",
            ),
            node(
                func=train_model,
                inputs=[
                    "raccoon_train_features_data_array",
                    "untrained_model",
                    "params:training",
                ],
                outputs="trained_model",
            ),
            node(
                func=upload_model,
                inputs=[
                    "trained_model",
                    "params:upload.temporary_save_path",
                    "params:upload.skip",
                ],
                outputs=None,
            ),
        ]
    )
