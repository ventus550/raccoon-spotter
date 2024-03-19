import logging

import numpy as np
from keras import Model

from raccoon_spotter.models.architectures import simple_regressor
from raccoon_spotter.utils.wandb import Client


def build_model() -> Model:
    model = simple_regressor.build_model()
    model.compile(loss="mse", optimizer="sgd", metrics=["cosine_similarity"])
    model.summary(print_fn=lambda x, **kwargs: logging.getLogger(__name__).info(x))
    return model


def train_model(training_data_arrays: np.ndarray, model: Model) -> Model:
    X, Y = training_data_arrays.values()
    callbacks = None
    if (wandb := Client()).enabled:
        Logs, _ = wandb.init_from_keras_model(model)
        callbacks = [Logs(log_freq=1)]
    model.fit(X, Y, epochs=10, callbacks=callbacks)
    return model
