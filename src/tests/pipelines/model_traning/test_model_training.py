import numpy
import pytest
from raccoon_spotter.pipelines.model_training.nodes import (
    build_model,
)
from raccoon_spotter.utils.configs import configs

numpy.random.seed(configs["seed"])


@pytest.fixture
def dummy_images():
    # return numpy.array([
    #     numpy.random.normal(size=(300, 600, 3)),
    #     numpy.random.normal(size=(100, 200, 3)),
    # ], dtype=object)
    return numpy.random.normal(size=(2, 600, 600, 3))


@pytest.fixture
def dummy_labels():
    return numpy.random.normal(size=(2, 4))


class TestDataLoadingNodes:
    def test_train_model(self, dummy_images, dummy_labels):
        model = build_model()
        model.fit(dummy_images, dummy_labels, epochs=1, batch_size=1)
        assert tuple(model(dummy_images).shape) == (2, 4)
