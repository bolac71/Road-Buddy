from __future__ import annotations
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_base_pack(name: str) -> dict:
    return _read_yaml(BASE_DIR / "base" / f"{name}.yaml")


def load_task_pack(dataset_type: str) -> dict:
    return _read_yaml(BASE_DIR / "task_packs" / f"{dataset_type}.yaml")
