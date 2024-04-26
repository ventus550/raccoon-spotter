from kedro.pipeline import Pipeline, node, pipeline

from .nodes import construct_data_array


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=construct_data_array,
                inputs=["raccoon_images", "raccoon_labels"],
                outputs="raccoon_data_array",
            ),
            node(
                func=print,
                inputs="raccoon_data_array",
                outputs=None,
            ),
        ]
    )
