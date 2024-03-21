import numpy as np
import pytest
from raccoon_spotter.pipelines.data_processing.nodes import (
    add_rgb_channel_to_image_arrays,
    pad_image_arrays,
)


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
    @pytest.mark.parametrize(
        "input_data_fixture, expected_shapes",
        [
            ("grayscale_image_arrays", [(100, 100, 3), (200, 150, 3)]),
            ("color_image_arrays", [(100, 100, 3), (200, 150, 3)]),
        ],
    )
    def test_add_rgb_channel_to_image_arrays(
        self, input_data_fixture, expected_shapes, request
    ):
        input_data = request.getfixturevalue(input_data_fixture)
        result = add_rgb_channel_to_image_arrays(input_data)
        assert isinstance(result, dict)
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == len(input_data["x"])
        for img, expected_shape in zip(result["x"], expected_shapes):
            assert img.shape == expected_shape
        assert np.array_equal(result["y"], input_data["y"])

    @pytest.mark.parametrize(
        "padding, expected_shapes",
        [
            (True, [(200, 300), (200, 300)]),
            (False, [(100, 100), (200, 150)]),
        ],
    )
    def test_pad_image_arrays(self, color_image_arrays, padding, expected_shapes):
        color_image_data = color_image_arrays
        padded_shape = {"width": 300, "height": 200}
        result = pad_image_arrays(color_image_data, padded_shape, padding=padding)
        assert isinstance(result, dict)
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == len(color_image_data["x"])
        for img, expected_shape in zip(result["x"], expected_shapes):
            assert img.shape[:2] == expected_shape[:2]
