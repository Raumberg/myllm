import importlib
import os
from types import ModuleType
from typing import Any


class LazyImporter(ModuleType):
    """
    This class allows for the lazy importation of a module. It is useful when dealing with optional dependencies.
    Instead of performing a standard import, which would raise an ImportError if the module is not found,
    this class creates a proxy object. The actual import is delayed until an attribute of the module is accessed.
    If the module is not installed, an ImportError is raised at that point, but not before.

    This is inspired from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py

    Usage::
        >>> # In a constants.py file
        >>> from myllm.utils.lazy_importer import LazyImporter
        >>>
        >>> _optional_dependencies = {
        ...     "peft": (False, "peft"),
        ... }
        >>> peft = LazyImporter("peft")
        >>>
        >>> # In a downstream file
        >>> from .constants import peft
        >>>
        >>> # The following line will only raise an error if peft is not installed when it's accessed.
        >>> peft.LlamaForCausalLM
    """

    def __init__(self, module_name: str):
        super().__init__(module_name)
        self._module = None
        self._module_name = module_name
        self._class_to_module = {}
        if module_name == "torch":
            self._class_to_module.update(
                {
                    "Tensor": "torch",
                    "device": "torch",
                    "dtype": "torch",
                }
            )

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._module_name)

        return getattr(self._module, name)

    def __dir__(self):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)

        return dir(self._module)

    def __reduce__(self):
        return (self.__class__, (self._module_name,)) 


_optional_dependencies = {
    "peft": (False, "peft"),
    "bitsandbytes": (False, "bitsandbytes"),
    "optimum": (False, "optimum"),
    "auto_gptq": (False, "auto_gptq"),
    "flash_attn": (False, "flash-attn"),
    "deepspeed": (False, "deepspeed"),
}

peft = LazyImporter("peft")
bitsandbytes = LazyImporter("bitsandbytes")
optimum = LazyImporter("optimum")
auto_gptq = LazyImporter("auto_gptq")
flash_attn = LazyImporter("flash_attn")
deepspeed = LazyImporter("deepspeed") 