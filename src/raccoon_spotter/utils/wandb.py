import json
import logging

from keras import Model

import wandb
from raccoon_spotter.utils.configs import configs, credentials
from wandb.keras import (
    WandbMetricsLogger,
    WandbModelCheckpoint,
)


class Client:
    def __init__(self):
        self.enabled = True
        try:
            wandb.login(key=credentials["wandb_access"])
        except KeyError:
            self.enabled = False
            logging.getLogger("WeightsAndBiasesClient").warning(
                "Failed to establish connection. Missing access key."
            )
        self.enabled &= configs["wandb"]

    def __getattr__(self, attr):
        if self.enabled:
            # If enabled, return the corresponding wandb function
            return getattr(self.wandb, attr) or self
        else:
            # If disabled, return a dummy function
            return lambda *args, **kwargs: None

    def init_from_keras_model(self, model: Model):
        wandb.init(name=model.name, config=json.loads(model.to_json()))
        return WandbMetricsLogger, WandbModelCheckpoint
