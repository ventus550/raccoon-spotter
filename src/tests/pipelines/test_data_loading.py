import numpy as np
import pandas as pd
import pytest
from raccoon_spotter.pipelines.data_loading.nodes import construct_data_array


@pytest.fixture
def dummy_images():
    return {
        "images/raccoon-17.jpg": np.zeros([32, 32, 3]),
        "images/raccoon-18.jpg": np.zeros([64, 64, 3]),
    }


@pytest.fixture
def dummy_labels():
    return pd.DataFrame(
        {
            "filename": ["raccoon-17.jpg"],
            "width": [259],
            "height": [194],
            "class": ["raccoon"],
            "xmin": [95],
            "ymin": [60],
            "xmax": [167],
            "ymax": [118],
        }
    )


class TestDataLoadingNodes:
    def test_construct_data_array(self, dummy_images, dummy_labels):
        data = construct_data_array(dummy_images, dummy_labels)
        assert len(data) == 2  # noqa: PLR2004
        assert data["x"][0].shape == (32, 32, 3)
        assert data["y"].shape == (1, 4)
