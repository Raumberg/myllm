import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

from .interface import FusedDynamicTanh

from typing import Optional

def getFusedDynTanh(model, copy_params: bool = False):
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            if len(module.normalized_shape) != 1:
                raise ValueError("Only 1D LayerNorm is supported.")
            
            hidden_size = module.normalized_shape[0]
            new_module = FusedDynamicTanh(hidden_size)
            
            if copy_params:
                with torch.no_grad():
                    new_module.weight.copy_(module.weight)
                    new_module.bias.copy_(module.bias)
            
            setattr(model, name, new_module)
        else:
            getFusedDynTanh(module, copy_params)
    
    return model

def FusedDynTanhPatch(
    model: nn.Module,
    copy_params: bool = True,
    verbose: bool = False,
    dtype: Optional[torch.dtype] = None
) -> nn.Module:

    def _convert_module(module: nn.Module) -> nn.Module:
        nonlocal dtype
        if isinstance(module, (nn.LayerNorm, LayerNorm2d)):
            # params
            normalized_shape = module.normalized_shape
            if isinstance(normalized_shape, tuple):
                normalized_shape = normalized_shape[0]
            
            channels_last = not isinstance(module, LayerNorm2d)
            new_dtype = dtype or module.weight.dtype if hasattr(module, 'weight') else torch.float32
            
            # module
            new_module = FusedDynamicTanh(
                normalized_shape=normalized_shape,
                channels_last=channels_last,
                dtype=new_dtype
            )
            
            # param copy
            if copy_params and module.elementwise_affine:
                with torch.no_grad():
                    new_module.weight.copy_(module.weight)
                    new_module.bias.copy_(module.bias)
                    
                    # alpha scale
                    if hasattr(module, 'eps'):
                        new_module.alpha.data.fill_(1.0 / (module.eps**0.5))
            elif copy_params:
                new_module.weight.requires_grad_(False)
                new_module.bias.requires_grad_(False)
            
            if verbose:
                print(f"Replaced {module.__class__.__name__} with {new_module}")
                
            return new_module
        return module
    
    # LayerNorm replacement
    for name, child in model.named_children():
        new_child = _convert_module(child)
        if new_child is not child:
            model.add_module(name, new_child)
        else:
            FusedDynTanhPatch(child, copy_params, verbose, dtype)
    
    return model