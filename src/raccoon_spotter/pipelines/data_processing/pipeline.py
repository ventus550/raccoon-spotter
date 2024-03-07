from kedro.pipeline import Pipeline, node, pipeline

from .nodes import reshape_image_arrays, resize_image_arrays


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=reshape_image_arrays,
                inputs=["raccoon_data_array"],
                outputs="raccoon_data_reshaped_array",
                name="reshaped_data_array_node",
            ),
            node(
                func=resize_image_arrays,
                inputs=["raccoon_data_reshaped_array"],
                outputs="raccoon_data_resize_array",
                name="resized_data_array_node",
            ),
        ]
    )
