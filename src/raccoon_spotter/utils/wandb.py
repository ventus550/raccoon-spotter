import json
import logging
import os

from keras import Model

import wandb
from raccoon_spotter.utils.configs import configs, loader
from wandb import errors


class Client:
    def __init__(self, name=None, config={}, **kwargs):
        wandb_configs = configs["wandb"].copy()
        self.mode = wandb_configs.pop("mode", "offline")
        self.logger = logging.getLogger(__name__)
        for key, value in wandb_configs.items():
            os.environ[f"WANDB_{key.upper()}"] = str(value).lower()

        if self.mode == "online":
            try:
                wandb.login(key=loader["credentials"]["wandb_access"], verify=True)
            except (KeyError, errors.AuthenticationError):
                self.mode = "offline"
                self.log(
                    "Failed to establish wandb connection ({error}).",
                    level=logging.WARNING,
                )
        self.run = wandb.init(name=name, config=config, mode=self.mode, **kwargs)

    def __getattr__(self, attr):
        return getattr(wandb, attr) or self

    def msg(self, msg: str, level=logging.INFO):
        self.logger.log(level, f"[yellow]{msg}", extra={"markup": True})

    def log(self, data: dict, **kwargs):
        self.run.log(data, **kwargs)

    @property
    def online(self):
        return self.mode == "online"

    @staticmethod
    def from_keras_model(model: Model):
        return Client(name=model.name, config=json.loads(model.to_json()))
