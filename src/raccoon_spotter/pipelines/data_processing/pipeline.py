from kedro.pipeline import Pipeline, node, pipeline

from .nodes import reshape_image_arrays, resize_image_arrays


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=reshape_image_arrays,
                inputs=["raccoon_data_array"],
                outputs="reshaped_raccoon_data_array",
                name="reshape_data_array_node",
            ),
            node(
                func=resize_image_arrays,
                inputs=["params:resize_image", "reshaped_raccoon_data_array"],
                outputs="resized_raccoon_data_array",
                name="resize_data_array_node",
            ),
        ]
    )
