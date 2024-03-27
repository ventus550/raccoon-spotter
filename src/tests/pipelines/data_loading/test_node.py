import numpy as np
import pandas as pd
import pytest
from raccoon_spotter.pipelines.data_loading.nodes import (
    construct_data_array,
)


@pytest.fixture
def sample_raccoon_images():
    return {
        "images/image1.jpg": np.zeros((100, 100, 3), dtype=np.uint8),
        "images/image2.jpg": np.ones((200, 200, 3), dtype=np.uint8),
    }


@pytest.fixture
def sample_raccoon_labels():
    return pd.DataFrame(
        {
            "filename": ["image1.jpg", "image2.jpg"],
            "xmin": [10, 20],
            "xmax": [30, 40],
            "ymin": [50, 60],
            "ymax": [70, 80],
        }
    )


class TestImageDataLoadingNodes:
    @pytest.mark.parametrize(
        "raccoon_images,raccoon_labels,expected_output",
        [
            (
                {
                    "images/image1.jpg": np.zeros((100, 100, 3), dtype=np.uint8),
                    "images/image2.jpg": np.ones((200, 200, 3), dtype=np.uint8),
                },
                pd.DataFrame(
                    {
                        "filename": ["image1.jpg", "image2.jpg"],
                        "xmin": [10, 20],
                        "xmax": [30, 40],
                        "ymin": [50, 60],
                        "ymax": [70, 80],
                    }
                ),
                {
                    "x": [
                        np.zeros((100, 100, 3), dtype=np.uint8).tolist(),
                        np.ones((200, 200, 3), dtype=np.uint8).tolist(),
                    ],
                    "y": [[10, 30, 50, 70], [20, 40, 60, 80]],
                },
            )
        ],
    )
    def test_construct_data_array(
        self, raccoon_images, raccoon_labels, expected_output
    ):
        result = construct_data_array(raccoon_images, raccoon_labels)
        assert result["y"].tolist() == expected_output["y"]
        for i in range(len(result["x"])):
            assert np.array_equal(result["x"][i], expected_output["x"][i])
