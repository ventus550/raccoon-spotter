import json
import logging
import os

from keras import Model

import wandb
from raccoon_spotter.utils.configs import configs, loader
from wandb import errors


class Client:
    def __init__(self, name=None, config={}, **kwargs):
        wandb_configs = configs["wandb"]
        self.mode = wandb_configs.pop("mode", "offline")
        for key, value in wandb_configs.items():
            os.environ[f"WANDB_{key.upper()}"] = str(value).lower()

        if self.mode == "online":
            try:
                wandb.login(key=loader["credentials"]["wandb_access"], verify=True)
            except (KeyError, errors.AuthenticationError) as error:
                self.mode = "offline"
                self.logger.warning(f"Failed to establish wandb connection ({error}).")
        self.run = wandb.init(name=name, config=config, mode=self.mode, **kwargs)

    def __getattr__(self, attr):
        return getattr(wandb, attr) or self

    @property
    def logger(self):
        return logging.getLogger(__name__)  # problems with getLogger("wandb")

    @property
    def online(self):
        return self.mode == "online"

    @staticmethod
    def from_keras_model(model: Model):
        return Client(name=model.name, config=json.loads(model.to_json()))
