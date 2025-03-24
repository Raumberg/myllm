import torch
import torch.nn as nn
import triton
from typing import Optional
from .fusion import DynTanhKerr

class FusedDynamicTanh(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        channels_last: bool = True,
        alpha_init: float = 0.5,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        self.dtype = dtype
        
        # params
        self.alpha = nn.Parameter(torch.full((1,), alpha_init))
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=dtype))
        
        # param init
    #     self._init_parameters()

    # def _init_parameters(self):
    #     nn.init.ones_(self.weight)
    #     nn.init.zeros_(self.bias)
    #     nn.init.constant_(self.alpha, 0.5)
        
    #     # alpha limit
    #     with torch.no_grad():
    #         self.alpha.clamp_(min=0.1, max=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], \
            f"Unsupported dtype: {x.dtype}"
        
        original_shape = x.shape
        x = x.reshape(-1, self.normalized_shape)
        output = torch.empty_like(x)
        
        n_rows, n_cols = x.shape
        dtype_str = 'float16' if x.dtype == torch.float16 else \
                   'bfloat16' if x.dtype == torch.bfloat16 else 'float32'
        
        # blocksize autoinit
        block_size = triton.next_power_of_2(self.normalized_shape)
        block_size = min(block_size, 1024)
        
        grid = (n_rows,)
        DynTanhKerr[grid](
            x, self.alpha, self.weight, self.bias, output,
            HIDDEN_SIZE=self.normalized_shape,
            DTYPE=dtype_str,
            BLOCK_SIZE=block_size
        )
        
        # channels_last ensureform
        output = output.view(original_shape)
        if not self.channels_last and output.dim() == 4:
            output = output.permute(0, 3, 1, 2)  # NHWC -> NCHW
            
        return output

    def __repr__(self):
        return (
            f"FusedDynamicTanh(normalized_shape={self.normalized_shape}, "
            f"alpha={self.alpha.item():.2f}, "
            f"channels_last={self.channels_last}, "
            f"dtype={self.dtype})"
        )