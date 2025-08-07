from torch import nn

def patch_model(model: nn.Module, with_rms_norm: bool = True, with_swiglu: bool = True):
    """
    Patch the model with all available fused kernels.
    """
    if with_rms_norm:
        _patch_rms_norm_internal(model)
    if with_swiglu:
        _patch_swiglu_internal(model)


def _patch_rms_norm_internal(model: nn.Module):
    """
    Recursively replace all instances of a specified layer with a new layer.
    
    Args:
        model (nn.Module): The model to patch.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        from .fused_rms_norm import FusedRMSNorm
    except ImportError:
        print("LlamaRMSNorm not found. Are you sure you have transformers installed?")
        return

    for name, module in model.named_children():
        if isinstance(module, LlamaRMSNorm):
            # Replace the layer
            new_layer = FusedRMSNorm(module.weight.shape[0], module.variance_epsilon)
            setattr(model, name, new_layer)
        else:
            # Recurse
            _patch_rms_norm_internal(module)


def _patch_swiglu_internal(model: nn.Module):
    """
    Recursively replaces LlamaMLP with FusedSwiGLU.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaMLP
        from .fused_swiglu import FusedSwiGLU
    except ImportError:
        print("LlamaMLP not found. Are you sure you have transformers installed?")
        return

    for name, module in model.named_children():
        if isinstance(module, LlamaMLP):
            new_mlp = FusedSwiGLU(
                in_features=module.config.hidden_size,
                hidden_features=module.config.intermediate_size,
                bias=False, # LlamaMLP doesn't have bias
            )
            setattr(model, name, new_mlp)
        else:
            _patch_swiglu_internal(module) 