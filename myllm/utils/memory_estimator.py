"""Utilities for estimating model memory usage, adapted from `accelerate.commands.estimate`."""

import torch
from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from accelerate import init_empty_weights
from accelerate.utils import is_timm_available, is_transformers_available


# Lazy imports for performance
if is_transformers_available():
    import transformers
    from transformers import AutoConfig, AutoModel

if is_timm_available():
    import timm


def verify_on_hub(repo: str, token: str = None):
    "Verifies that the model is on the hub and returns the model info."
    try:
        return model_info(repo, token=token)
    except (OSError, GatedRepoError):
        return "gated"
    except RepositoryNotFoundError:
        return "repo"


def check_has_model(error):
    """
    Checks what library spawned `error` when a model is not found
    """
    if is_timm_available() and isinstance(error, RuntimeError) and "Unknown model" in error.args[0]:
        return "timm"
    elif (
        is_transformers_available()
        and isinstance(error, OSError)
        and "does not appear to have a file named" in error.args[0]
    ):
        return "transformers"
    else:
        return "unknown"


def create_empty_model(model_name: str, library_name: str, trust_remote_code: bool = False, access_token: str = None):
    """
    Creates an empty model in full precision from its parent library on the `Hub` to calculate the overall memory
    consumption.
    """
    model_info_obj = verify_on_hub(model_name, access_token)
    if model_info_obj == "gated":
        raise GatedRepoError(
            f"Repo for model `{model_name}` is gated. You must be authenticated to access it. Please run `huggingface-cli login`."
        )
    elif model_info_obj == "repo":
        raise RepositoryNotFoundError(
            f"Repo for model `{model_name}` does not exist on the Hub. If you are trying to access a private repo,"
            " make sure you are authenticated via `huggingface-cli login` and have access."
        )
    
    if library_name is None:
        library_name = getattr(model_info_obj, "library_name", False)
        if not library_name:
            raise ValueError(
                f"Model `{model_name}` does not have any library metadata on the Hub, please manually pass in a `--library_name` to use (such as `transformers`)"
            )

    if library_name == "transformers":
        if not is_transformers_available():
            raise ImportError(
                f"To check `{model_name}`, `transformers` must be installed. Please install it via `pip install transformers`"
            )
        print(f"\nLoading pretrained config for `{model_name}` from `transformers`...\n")
        if model_info_obj.config is None:
            raise RuntimeError(f"Tried to load `{model_name}` with `transformers` but it does not have any metadata.")

        auto_map = model_info_obj.config.get("auto_map", False)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code, token=access_token)
        with init_empty_weights():
            constructor = AutoModel
            if isinstance(auto_map, dict):
                value = None
                for key in auto_map.keys():
                    if key.startswith("AutoModelFor"):
                        value = key
                        break
                if value is not None:
                    constructor = getattr(transformers, value)
            model = constructor.from_config(config, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)
    
    elif library_name == "timm":
        if not is_timm_available():
            raise ImportError(
                f"To check `{model_name}`, `timm` must be installed. Please install it via `pip install timm`"
            )
        print(f"Loading pretrained config for `{model_name}` from `timm`...")
        with init_empty_weights():
            model = timm.create_model(model_name, pretrained=False)
    else:
        raise ValueError(
            f"Library `{library_name}` is not supported yet, please open an issue on GitHub for us to add support."
        )
    return model


def rich_table(headers: list, rows: list, title: str):
    from rich.table import Table
    from rich.box import HEAVY_HEAD

    table = Table(title=title, box=HEAVY_HEAD, show_header=True, header_style="bold magenta")
    for header in headers:
        table.add_column(header, justify="center")
    
    for row in rows:
        table.add_row(*[str(item) for item in row])
        
    return table

def estimate_training_usage(size_in_bytes: int):
    """
    Given a model's size in bytes, returns a dict of estimated training usage.
    Assumes AdamW optimizer (model states + 2 optimizer states per parameter).
    - Model weights: 1x
    - Gradients: 1x
    - Optimizer states: 2x
    Total for Adam: 4x model size in full precision (e.g. float32)
    """
    return {"float32": size_in_bytes * 4, "float16": size_in_bytes * 2} 