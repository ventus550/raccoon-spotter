import numpy as np
import pytest
from raccoon_spotter.pipelines.data_processing.nodes import (
    add_rgb_channel_to_image_arrays,
    pad_image_arrays,
)

dummy_image_arrays = {
    "x": [np.ones((100, 100)), np.ones((400, 200, 3))],
    "y": np.array([0, 1]),
}


dummy_resize_config = {"target_width": 200, "target_height": 150}

RGB_CHANNELS = 3
num_img = len(dummy_image_arrays["x"])


@pytest.fixture
def test_add_rgb_channel_to_image_arrays():
    reshaped_dict = add_rgb_channel_to_image_arrays(dummy_image_arrays)
    assert isinstance(reshaped_dict, dict)
    assert "x" in reshaped_dict
    assert "y" in reshaped_dict
    assert reshaped_dict["x"].shape[0] == num_img
    assert reshaped_dict["x"][0].shape[2] == RGB_CHANNELS  # grayscale img
    assert reshaped_dict["x"][1].shape[2] == RGB_CHANNELS  # RGB img
    assert np.array_equal(reshaped_dict["y"], dummy_image_arrays["y"])


@pytest.fixture
def test_pad_image_arrays():
    resized_dict = pad_image_arrays(dummy_image_arrays, dummy_image_arrays)
    assert isinstance(resized_dict, dict)
    assert "x" in resized_dict
    assert "y" in resized_dict
    assert resized_dict["x"].shape[0] == num_img
    assert resized_dict["x"][0].shape[:2] == (
        dummy_resize_config["target_height"],
        dummy_resize_config["target_width"],
    )
    assert resized_dict["x"][1].shape[:2] == (
        dummy_resize_config["target_height"],
        dummy_resize_config["target_width"],
    )
    assert np.array_equal(resized_dict["y"], dummy_image_arrays["y"])
