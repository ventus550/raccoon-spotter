from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import keras
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class KerasModelDataset(AbstractDataset[keras.Model, keras.Model]):
    def __init__(self, filepath: str, custom_components: Dict[str, Any] = {}):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._custom_components = custom_components
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> keras.Model:
        load_path = get_filepath_str(self._filepath, self._protocol)
        return keras.models.load_model(
            load_path, custom_objects=self._custom_components
        )

    def _save(self, model: keras.Model) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        model.save(save_path)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            custom_components=self._custom_components,
        )
