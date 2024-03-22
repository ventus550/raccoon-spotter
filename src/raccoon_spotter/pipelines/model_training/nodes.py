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
    wandb = Client.from_keras_model(model)
    model.fit(X, Y, epochs=2, callbacks=[wandb.keras.WandbMetricsLogger(log_freq=1)])
    return model


def upload_model(model: Model, temporary_save_path: str, skip: bool):
    if not skip:
        model.save(temporary_save_path)
        Client().save(temporary_save_path)
