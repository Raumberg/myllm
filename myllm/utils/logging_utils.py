from __future__ import annotations

"""Utility helpers to configure global logging / warnings based on YAML config."""

from typing import Sequence
import logging
import os
import warnings

from myllm.config.schema import LoggingCfg

__all__ = ["apply_logging_cfg"]


_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
}


def _silence_modules(mod_names: Sequence[str]):  # noqa: D401
    """Set ERROR level for target modules and all their children."""
    registry = logging.root.manager.loggerDict
    for base in mod_names:
        logging.getLogger(base).setLevel(logging.ERROR)
        for logger_name in registry:
            if logger_name.startswith(f"{base}."):
                logging.getLogger(logger_name).setLevel(logging.ERROR)


def apply_logging_cfg(cfg: LoggingCfg) -> None:  # noqa: D401
    """Apply verbosity/suppression settings globally.

    Should be called **early** (before heavy imports) to take effect.
    """

    lvl = _LEVEL_MAP.get(cfg.level.lower(), logging.INFO)
    logging.getLogger().setLevel(lvl)

    # basic config only if root handlers empty
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    _silence_modules(cfg.suppress)

    # HuggingFace env vars respect VERBOSITY
    if "transformers" in cfg.suppress:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    if "datasets" in cfg.suppress:
        os.environ.setdefault("DATASETS_VERBOSITY", "error")

    # Warnings filter – honour only patterns provided in config
    for pat in cfg.warnings_ignore:
        warnings.filterwarnings("ignore", message=pat)

    # TQDM global disable (used by transformers, datasets, accelerate)
    if cfg.disable_tqdm:
        os.environ["DISABLE_TQDM"] = "1" 