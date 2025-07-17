from typing import Any, Dict, Tuple

import torch
from rich.table import Table
from transformers import PreTrainedModel, PretrainedConfig


def model_summary(model: PreTrainedModel, max_depth: int = 1) -> Table:
    """
    Generate a summary of a pre-trained model, traversing it layer by layer.

    Args:
        model (PreTrainedModel): The model to summarize.
        max_depth (int): The maximum depth to traverse the model's modules.

    Returns:
        Table: A rich Table object containing the model summary.
    """

    def get_layer_type(layer: torch.nn.Module) -> str:
        return layer.__class__.__name__

    def count_parameters(module: torch.nn.Module) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        return trainable, non_trainable

    table = Table(
        title=f"Model Summary: {model.__class__.__name__} (Max Depth: {max_depth})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Layer (type)", justify="left", style="cyan", no_wrap=True)
    table.add_column("Output Shape", justify="right", style="magenta")
    table.add_column("Params (Trainable)", justify="right", style="green")
    table.add_column("Params (Frozen)", justify="right", style="red")
    table.add_column("Config", justify="left", style="yellow")

    def get_config_str(layer: torch.nn.Module) -> str:
        """Create a string representation of the layer's config."""
        config_info = {}
        if hasattr(layer, "config") and isinstance(layer.config, PretrainedConfig):
            config_info = layer.config.to_diff_dict()
        # Special case for activation functions inside MLP layers
        if hasattr(layer, "act_fn"):
            config_info["activation"] = layer.act_fn.__class__.__name__
        elif hasattr(layer, "activation"): # for other model types
             config_info["activation"] = layer.activation.__class__.__name__
        
        if not config_info:
            return ""
        
        # Pretty print the config dict
        return "\n".join([f"{k}: {v}" for k, v in config_info.items()])

    def process_module(module: torch.nn.Module, parent_name: str = "", depth: int = 0):
        if depth > max_depth:
            return

        children_to_process = list(module.named_children())

        # If no children, or at max depth, process the module itself if it's not the root
        if parent_name and (not children_to_process or depth == max_depth):
            return

        for name, layer in children_to_process:
            layer_name = f"{parent_name}.{name}" if parent_name else name
            layer_type = get_layer_type(layer)
            trainable, non_trainable = count_parameters(layer)

            # Attempt to get output shape
            output_shape = "N/A"
            try:
                # A more robust way to create a dummy input
                if "Embedding" in layer_type:
                    input_ids = torch.zeros((1, 1), dtype=torch.long, device=model.device)
                    output = layer(input_ids)
                else:
                    if hasattr(model.config, "hidden_size"):
                        hidden_size = model.config.hidden_size
                        dummy_input = torch.zeros(
                            (1, 1, hidden_size),
                            dtype=next(model.parameters()).dtype,
                            device=model.device,
                        )
                        output = layer(dummy_input)
                    else: # Fallback for models without standard hidden_size
                        output_shape = "N/A"

                if isinstance(output, tuple):
                    output = output[0] # Usually the hidden states are the first element
                
                if hasattr(output, 'shape'):
                    output_shape = str(tuple(output.shape))

            except Exception:
                output_shape = "N/A"
            
            config_str = get_config_str(layer)

            table.add_row(
                "  " * depth + f"{name} ({layer_type})",
                output_shape,
                f"{trainable:,}" if trainable else "0",
                f"{non_trainable:,}" if non_trainable else "0",
                config_str
            )
            
            # Recurse into children
            process_module(layer, parent_name=layer_name, depth=depth + 1)

    # Start processing from the root model, but don't add the root to the table itself
    process_module(model, parent_name="", depth=0)

    # Calculate totals at the end
    total_params_trainable, total_params_non_trainable = count_parameters(model)
    table.add_row(
        "\n[bold]Total",
        "",
        f"[bold green]{total_params_trainable:,}",
        f"[bold red]{total_params_non_trainable:,}",
        ""
    )

    return table
