import torch
import torch.nn as nn

from .interface import FusedDynamicTanh

from typing import Optional

def FusedDynTanhPatch(
    model: nn.Module,
    copy_params: bool = True,
    verbose: bool = False,
    dtype: Optional[torch.dtype] = None
) -> nn.Module:

    def _convert_module(name: str, module: nn.Module) -> nn.Module:
        nonlocal dtype
        # if isinstance(module, (nn.LayerNorm, LayerNorm2d)):
        if any([key in name.lower() for key in ["layernorm", "norm"]]):
            # params
            if not hasattr(module, "weight"):
                return module

            # normalized_shape = module.weight.shape # size(0)
            normalized_shape = 3584
            
            channels_last = False     # not isinstance(module, LayerNorm2d)
            new_dtype = dtype or module.weight.dtype if hasattr(module, 'weight') else torch.float32
            
            # module
            new_module = FusedDynamicTanh(
                normalized_shape=normalized_shape,
                channels_last=channels_last,
                dtype=new_dtype
            )
            
            # param copy
            if copy_params:
                with torch.no_grad():
                    new_module.weight.copy_(module.weight)
                    # new_module.bias.copy_(module.bias)
                    
                    # alpha scale
                    # if hasattr(module, 'eps'):
                    #     new_module.alpha.data.fill_(1.0 / (module.eps**0.5))
            # elif copy_params:
            #     new_module.weight.requires_grad_(False)
            #     new_module.bias.requires_grad_(False)
            
            if verbose:
                print(f"Replaced {module.__class__.__name__} with {new_module}")
                
            return new_module
        return module
    
    # LayerNorm replacement
    for name, child in model.named_children():
        new_child = _convert_module(name, child)
        if new_child is not child:
            # model.add_module(name, new_child)
            model._modules[name] = new_child
        else:
            FusedDynTanhPatch(child, copy_params, verbose, dtype)
    
    return model