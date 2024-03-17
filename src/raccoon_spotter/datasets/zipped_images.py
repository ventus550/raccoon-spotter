from typing import Any, Dict
from zipfile import ZipFile

import numpy as np
from PIL import Image

from .abstract_filesystem import AbstractFileSystemDataset


class ZippedImagesDataset(AbstractFileSystemDataset[np.ndarray, np.ndarray]):
    def __init__(self, filepath: str, credentials: dict, extension: str = ".jpg"):
        super().__init__(filepath, fsspec_kwargs=credentials)
        self._ext = extension

    def _load(self) -> Dict[str, np.array]:
        load_path = self._get_filepath_str()
        with self._fs.open(load_path, mode="rb") as file:
            with ZipFile(file, "r") as zipped:
                return {
                    impath: np.asarray(Image.open(zipped.open(impath)))
                    for impath in zipped.namelist()
                    if impath.endswith(self._ext)
                }

    def _save(self, data: Dict[str, np.array]) -> None:
        raise NotImplementedError("Saving not supported for ZippedImagesDataset")

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)
