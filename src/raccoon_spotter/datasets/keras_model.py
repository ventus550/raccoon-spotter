from typing import Any, Dict

import keras

from .abstract_filesystem import AbstractFileSystemDataset


class KerasModelDataset(AbstractFileSystemDataset[keras.Model, keras.Model]):
    def __init__(self, filepath: str, custom_components: Dict[str, Any] = {}):
        super().__init__(filepath)
        self._custom_components = custom_components

    def _load(self) -> keras.Model:
        load_path = self._get_filepath_str()
        return keras.models.load_model(
            load_path, custom_objects=self._custom_components
        )

    def _save(self, model: keras.Model) -> None:
        save_path = self._get_filepath_str()
        model.save(save_path)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            custom_components=self._custom_components,
        )
