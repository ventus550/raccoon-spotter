import logging

import numpy as np
from keras import Model
import matplotlib.pyplot as plt

from raccoon_spotter.models.architectures import simple_regressor
from raccoon_spotter.utils.wandb import Client
from raccoon_spotter.utils.data_visualization import draw_bounding_box

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
    wandb.log(f"Logging training data to {wandb.run.get_url() or 'local'}.")
    log_freq = training_args.pop("log_freq")
    model.fit(
        X, Y, callbacks=[wandb.keras.WandbMetricsLogger(log_freq)], **training_args
    )
    return model

def sample_model(training_data_arrays: np.ndarray, model: Model):
    X, Y = training_data_arrays.values()
    P = model.predict(X).astype(int)
    fig, axs = plt.subplots(6, 2, figsize=(12, 24))
    for i, ax in enumerate(axs):
        ax[0].imshow(draw_bounding_box(X[i*2], Y[i*2]))
        ax[1].imshow(draw_bounding_box(X[i*2], P[i*2]))
    return fig

def upload_model(model: Model, temporary_save_path: str, skip: bool):
    wandb = Client()
    if not skip and wandb.online:
        model.save(temporary_save_path)
        wandb.link_model(path=temporary_save_path, registered_model_name=model.name)
        wandb.log("Model uploaded successfully.")
    else:
        wandb.log("Skipping model upload.")
