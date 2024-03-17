import abc
from pathlib import PurePosixPath
from typing import Any, Dict, Generic

import fsspec
from kedro.io import AbstractDataset
from kedro.io.core import _DI, _DO, get_filepath_str, get_protocol_and_path


class AbstractFileSystemDataset(AbstractDataset, abc.ABC, Generic[_DI, _DO]):
    def __init__(self, filepath: str, fsspec_kwargs: dict = {}):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol, **fsspec_kwargs.copy())

    def _get_filepath_str(self):
        return get_filepath_str(self._filepath, self._protocol)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
        )
