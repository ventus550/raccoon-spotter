from typing import Any, Dict

import numpy as np

from .abstract_filesystem import AbstractFileSystemDataset


class NPZArrayDataset(AbstractFileSystemDataset[np.ndarray, np.ndarray]):
    def __init__(self, filepath: str, credentials: dict = {}):
        super().__init__(filepath, fsspec_kwargs=credentials)

    def _load(self) -> Dict[str, np.ndarray]:
        load_path = self._get_filepath_str()
        return np.load(load_path, allow_pickle=True)

    def _save(self, data: Dict[str, np.array]) -> None:
        save_path = self._get_filepath_str()
        np.savez(save_path, **data)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)
