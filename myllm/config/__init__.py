from __future__ import annotations

"""Configuration utilities (Hydra/TOML/Pydantic dataclasses).
Currently contains only stubs – will be extended with real schemas.
"""

from pathlib import Path
from typing import Any, Dict

import yaml
from .schema import Config  # noqa: E402

__all__ = ["load"]


def load(path: str | Path) -> Config:  # type: ignore[override]
    """Load YAML/TOML file and return validated Config."""
    data = _load_raw(path)
    return Config.from_dict(data)


def _load_raw(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text())
    raise ValueError(f"Unsupported config extension: {path.suffix}") 