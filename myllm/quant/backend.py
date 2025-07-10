from typing import Any

__all__ = ["FP8Backend"]

class _TEBackend:
    """Singleton-like class to handle the optional Transformer Engine dependency."""

    def __init__(self):
        self.te: Any = None
        self.DelayedScaling: Any = object
        self.Format: Any = object
        self.is_available: bool = False
        self.is_fp8_supported: bool = False

        try:
            import transformer_engine.pytorch as te_module
            from transformer_engine.common.recipe import (
                DelayedScaling as ds_class,
                Format as fmt_class,
            )
            import torch
        except (ImportError, FileNotFoundError):
            return  # Dependencies are not met, stay with defaults

        # If imports are successful, populate the attributes
        self.te = te_module
        self.DelayedScaling = ds_class
        self.Format = fmt_class
        self.is_available = True

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
            self.is_fp8_supported = True

# Instantiate the backend handler, hiding the import logic
FP8Backend = _TEBackend()