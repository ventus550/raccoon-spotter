import logging

import numpy as np
from keras import Model

from raccoon_spotter.models.architectures import simple_regressor
from raccoon_spotter.utils.wandb import Client


def build_model() -> Model:
    model = simple_regressor.build_model()
    model.compile(loss="mse", optimizer="adam", metrics=["cosine_similarity"])
    model.summary(print_fn=lambda x, **kwargs: logging.getLogger(__name__).info(x))
    return model


def train_model(
    training_data_arrays: np.ndarray, model: Model, training_args: dict
) -> Model:
    X, Y = training_data_arrays.values()
    wandb = Client.from_keras_model(model)
    wandb.msg(f"Logging training data to {wandb.run.get_url() or 'local'}.")
    log_freq = training_args.pop("log_freq")
    model.fit(
        X, Y, callbacks=[wandb.keras.WandbMetricsLogger(log_freq)], **training_args
    )
    return model


def upload_model(model: Model, temporary_save_path: str, skip: bool):
    wandb = Client()
    if not skip and wandb.online:
        model.save(temporary_save_path)
        wandb.link_model(path=temporary_save_path, registered_model_name=model.name)
        wandb.msg("Model uploaded successfully.")
    else:
        wandb.msg("Skipping model upload.")
