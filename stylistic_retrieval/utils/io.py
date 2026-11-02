import json
from pathlib import Path
from typing import Any

import yaml


def read_json(file_path: Path) -> Any:
    if not file_path.is_file():
        raise FileNotFoundError(f"The file '{file_path.as_posix()}' was not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_txt(file_path: Path) -> Any:
    if not file_path.is_file():
        raise FileNotFoundError(f"The file '{file_path.as_posix()}' was not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_yaml(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"The file '{file_path.as_posix()}' was not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
