from pathlib import PurePosixPath
from typing import Any, Dict
from zipfile import ZipFile

import fsspec
import numpy as np
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path
from PIL import Image


class ZippedImagesDataset(AbstractDataset[np.ndarray, np.ndarray]):
    def __init__(self, filepath: str, credentials: dict, extension: str = ".jpg"):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol, **credentials.copy())
        self._ext = extension

    def _load(self) -> Dict[str, np.array]:
        load_path = get_filepath_str(self._filepath, self._protocol)
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
