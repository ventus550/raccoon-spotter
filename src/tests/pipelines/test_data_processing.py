import numpy as np
import pytest
from raccoon_spotter.pipelines.data_processing.nodes import (
    add_rgb_channel_to_image_arrays,
    pad_image_arrays,
)

RGB_CHANNELS = 3


@pytest.fixture
def grayscale_image_arrays():
    x = [np.ones((100, 100)), np.ones((200, 150))]
    y = np.array([[10, 20, 50, 70], [30, 40, 80, 90]])
    return {"x": x, "y": y}


@pytest.fixture
def color_image_arrays():
    x = [np.ones((100, 100, 3)), np.ones((200, 150, 3))]
    y = np.array([[10, 20, 50, 70], [30, 40, 80, 90]])
    return {"x": x, "y": y}


class TestImageDataProcessingNodes:
    def test_add_rgb_channel_to_image_arrays_grayscale(self, grayscale_image_arrays):
        result = add_rgb_channel_to_image_arrays(grayscale_image_arrays)
        assert isinstance(result, dict)
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == len(grayscale_image_arrays["x"])
        for img in result["x"]:
            assert img.shape[2] == RGB_CHANNELS
        assert np.array_equal(result["y"], grayscale_image_arrays["y"])

    def test_add_rgb_channel_to_image_arrays_color(self, color_image_arrays):
        result = add_rgb_channel_to_image_arrays(color_image_arrays)
        assert isinstance(result, dict)
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == len(color_image_arrays["x"])
        for img in result["x"]:
            assert img.shape[2] == RGB_CHANNELS

    def test_pad_image_arrays(self, color_image_arrays):
        padded_shape = {"width": 300, "height": 200}
        result = pad_image_arrays(color_image_arrays, padded_shape, padding=True)
        assert isinstance(result, dict)
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == len(color_image_arrays["x"])
        for img in result["x"]:
            assert img.shape[:2] == (200, 300)

    def test_pad_image_arrays_no_padding(self, color_image_arrays):
        padded_shape = {"width": 300, "height": 200}
        result = pad_image_arrays(color_image_arrays, padded_shape, padding=False)
        assert isinstance(result, dict)
        assert "x" in result
        assert "y" in result
        assert np.array_equal(result["y"], color_image_arrays["y"])
