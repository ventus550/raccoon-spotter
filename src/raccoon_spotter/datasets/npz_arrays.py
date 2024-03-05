from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import numpy as np
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class NPZArrayDataset(AbstractDataset):
    def __init__(self, filepath: str):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> Dict[str, np.ndarray]:
        load_path = get_filepath_str(self._filepath, self._protocol)
        return np.load(load_path, allow_pickle=True)

    def _save(self, data: Dict[str, np.array]) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        np.savez(save_path, **data)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)
