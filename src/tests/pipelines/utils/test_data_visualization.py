import cv2
import numpy as np
import pytest
from raccoon_spotter.utils.data_visualization import roi


class TestDrawBoundingBox:
    @pytest.mark.parametrize(
        "image_array,bounding_box,expected",
        [
            (
                np.zeros((100, 100, 3), dtype=np.uint8),
                np.array([20, 30, 80, 90]),
                cv2.rectangle(
                    np.zeros((100, 100, 3), dtype=np.uint8),
                    (20, 80),
                    (30, 90),
                    (0, 255, 0),
                    1,
                ),
            ),
            (
                np.zeros((150, 200, 3), dtype=np.uint8),
                np.array([50, 100, 60, 120]),
                cv2.rectangle(
                    np.zeros((150, 200, 3), dtype=np.uint8),
                    (50, 60),
                    (100, 120),
                    (0, 255, 0),
                    1,
                ),
            ),
        ],
    )
    def test_roi(self, image_array, bounding_box, expected):
        xmin, xmax, ymin, ymax = bounding_box
        result = roi(image_array, bounding_box)
        assert np.array_equal(
            result[ymin:ymax, xmin:xmax], expected[ymin:ymax, xmin:xmax]
        ), "Bounding box not drawn correctly"
