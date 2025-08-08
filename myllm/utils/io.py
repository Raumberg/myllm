import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import enum

from accelerate import PartialState


logger = logging.getLogger(__name__)


def _to_dict_recursive(obj: Any) -> Any:
    """Recursively convert an object to a dict, handling common types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if is_dataclass(obj):
        return _to_dict_recursive(asdict(obj))
    if hasattr(obj, "to_dict"):
        return _to_dict_recursive(obj.to_dict())
    if hasattr(obj, "model_dump"):
        return _to_dict_recursive(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): _to_dict_recursive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_dict_recursive(i) for i in obj]
    if hasattr(obj, "__dict__"):
        return {
            k: _to_dict_recursive(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }

    try:
        return str(obj)
    except Exception:
        return f"<unserializable type: {type(obj).__name__}>"

def get_run_dir(directory: str | Path) -> Path:
    """Create and return a run directory with timestamp for logging.
    
    Args:
        directory: Base directory where the run directory will be created
        
    Returns:
        Path to the created run directory
    """
    if PartialState().is_main_process:
        directory = Path(directory)
        timestamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        run_dir = directory / ".run" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created run directory: %s", run_dir)
        return run_dir
    return None


class ConfigDumper:
    def __init__(self, output_dir: Path):
        # timestamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        # self.run_dir = output_dir / ".run" / timestamp
        self.run_dir = output_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Dumping run configs to %s", self.run_dir)

    def dump(self, config_obj: Any, name: str):
        """Dumps a config object to a JSON file."""
        if config_obj is None:
            return

        if not name.endswith(".json"):
            name = f"{name.lower()}.json"

        filepath = self.run_dir / name

        try:
            config_dict = _to_dict_recursive(config_obj)
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.warning(
                "Could not dump config '%s': %s", name, e, exc_info=True
            )
            # As a fallback, try to dump a plain text representation
            try:
                with filepath.with_suffix(".txt").open(
                    "w", encoding="utf-8"
                ) as f:
                    f.write(str(config_obj))
            except Exception as fallback_e:
                logger.error(
                    "Fallback dump for '%s' also failed: %s", name, fallback_e
                ) 