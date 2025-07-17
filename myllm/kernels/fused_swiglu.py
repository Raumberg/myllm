import torch
import torch.nn as nn
import torch.nn.functional as F

from . import load_swiglu_kernel

class FusedSwiGLU(nn.Module):
    """
    Fused SwiGLU module that uses a custom CUDA kernel for the forward pass.
    It is designed to replace the MLP layer in models like Llama.
    """
    def __init__(self, in_features, hidden_features, bias=False):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features, bias=bias)
        self.w_up = nn.Linear(in_features, hidden_features, bias=bias)
        self.w_down = nn.Linear(hidden_features, in_features, bias=bias)
        
        # Load the custom kernel
        self.kernel = load_swiglu_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get gate and up projections
        gate_proj = self.w_gate(x)
        up_proj = self.w_up(x)
        
        # Apply fused SwiGLU
        activated_proj = self.kernel.forward(gate_proj, up_proj)
        
        # Down projection
        return self.w_down(activated_proj)

def replace_swiglu(model: nn.Module):
    """
    Recursively replaces LlamaMLP with FusedSwiGLU.
    """
    from transformers.models.llama.modeling_llama import LlamaMLP

    for name, module in model.named_children():
        if isinstance(module, LlamaMLP):
            new_mlp = FusedSwiGLU(
                in_features=module.config.hidden_size,
                hidden_features=module.config.intermediate_size,
                bias=False, # LlamaMLP doesn't have bias
            )
            setattr(model, name, new_mlp)
        else:
            replace_swiglu(module) 