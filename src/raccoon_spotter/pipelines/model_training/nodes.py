import logging

import numpy as np
from keras import Model

from raccoon_spotter.models.architectures import simple_regressor


def build_model() -> Model:
    model = simple_regressor.build_model()
    model.compile(loss="mse", optimizer="sgd", metrics=["cosine_similarity"])
    model.summary(print_fn=lambda x, **kwargs: logging.getLogger(__name__).info(x))
    return model


def train_model(training_data_arrays: np.ndarray, model: Model) -> Model:
    X, Y = training_data_arrays.values()
    model.fit(X, Y, epochs=1)
    return model
