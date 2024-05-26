import logging
import random

import keras
import matplotlib as mpl
import numpy as np
from kedro.framework.hooks import hook_impl

from .utils.configs import configs


class MatplotlibSettingsHook:
    @hook_impl
    def before_node_run(self):
        mpl.style.use("cyberpunk")
        rc = {
            "axes.grid": False,
            "axes.spines.left": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.top": False,
            "xtick.bottom": False,
            "xtick.labelbottom": False,
            "ytick.labelleft": False,
            "ytick.left": False,
        }
        mpl.rcParams.update(rc)


class SeedPipelineHook:
    @property
    def _logger(self):
        return logging.getLogger(__name__)

    @hook_impl
    def before_pipeline_run(self) -> None:
        import tensorflow

        seed = configs["seed"]

        # Set seed for Python's random module
        random.seed(seed)

        # Set seed for numpy
        np.random.seed(seed)

        # Set seed for torch
        # torch.manual_seed(42)
        # torch.cuda.manual_seed_all(42)

        # Set seed for keras
        keras.utils.set_random_seed(seed)
        tensorflow.random.set_seed(seed)

        self._logger.info(f"Set seed = {seed}")
