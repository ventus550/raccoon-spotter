from kedro.pipeline import Pipeline, node, pipeline

from .nodes import add_rgb_channel_to_image_arrays, pad_image_arrays


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=add_rgb_channel_to_image_arrays,
                inputs=["raccoon_data_array"],
                outputs="rgb_raccoon_data_array",
                name="add_rgb_channel",
            ),
            node(
                func=pad_image_arrays,
                inputs=[
                    "rgb_raccoon_data_array",
                    "params:padded_shape",
                    "params:padding",
                ],
                outputs="preprocessed_raccoon_data_array",
                name="apply_padding",
            ),
        ]
    )
