from stylistic_retrieval.utils.io import read_yaml

from pathlib import Path
from typing import Any


class Config:
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        self._config = read_yaml(self.file_path)

    @property
    def as_dict(self) -> dict[str, Any]:
        return self._config

    def __getitem__(self, key) -> dict[str, Any]:
        return self._config[key]
