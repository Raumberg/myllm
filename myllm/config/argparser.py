from __future__ import annotations

"""Smart argument parser that merges YAML config files with CLI overrides.

Inspired by HuggingFace HfArgumentParser but simplified. Accepts:
  $ myllm train --config configs/my.yaml --training.lr 1e-5 data.name=/path
CLI overrides use dot-paths (nested) or attr=value.
"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Type, TypeVar

import yaml

from .schema import Config

T = TypeVar("T")


def _set_nested(cfg: Dict[str, Any], path: List[str], value: Any):  # noqa: D401
    cur = cfg
    for p in path[:-1]:
        cur = cur.setdefault(p, {})
    cur[path[-1]] = value


class SmartParser(ArgumentParser):
    """Parser that returns Config dataclass and leftover CLI arguments."""

    def __init__(self):
        super().__init__(add_help=True)
        self.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
        self.add_argument("overrides", nargs="*", help="Dot-list CLI overrides (key=value)")

    def parse(self) -> Tuple[Config, Namespace]:  # noqa: D401
        args, unknown = self.parse_known_args()
        cfg_dict = yaml.safe_load(args.config.read_text())

        # Apply overrides
        if args.overrides:
            for ov in args.overrides:
                if "=" not in ov:
                    raise ValueError(f"Override must be key=value, got: {ov}")
                key, val = ov.split("=", 1)
                path = key.split(".")
                _set_nested(cfg_dict, path, _infer_type(val))

        return Config.from_dict(cfg_dict), args  # type: ignore[arg-type]

    # Helper for Typer integration
    @staticmethod
    def load_from_file(config_path: Path, overrides: List[str] | None = None) -> Config:  # noqa: D401
        """Load YAML file and apply CLI-style overrides without touching sys.argv."""
        cfg_dict = yaml.safe_load(config_path.read_text())
        overrides = overrides or []
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"Override must be key=value, got: {ov}")
            key, val = ov.split("=", 1)
            _set_nested(cfg_dict, key.split("."), _infer_type(val))
        return Config.from_dict(cfg_dict)


def _infer_type(v: str):  # noqa: D401
    if v.lower() in {"true", "false"}:  # bool
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v  # string 