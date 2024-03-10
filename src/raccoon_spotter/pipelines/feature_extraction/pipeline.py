from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs=["preprocessed_raccoon_data_array"],
                outputs="raccoon_features_data_array",
                name="identity",
            ),
        ]
    )
