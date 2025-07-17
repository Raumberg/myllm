import torch
import torch.nn as nn
from torch.autograd import Function

from . import load_rms_norm_kernel


class RMSNormFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, eps):
        # Allocate output tensors
        output = torch.empty_like(input)
        
        # Determine accumulator type for rstd
        if input.dtype == torch.double:
            rstd_dtype = torch.double
        else:
            rstd_dtype = torch.float32
        rstd = torch.empty(input.size(0), dtype=rstd_dtype, device=input.device)

        load_rms_norm_kernel().forward(output, rstd, input, weight, eps)

        # Save tensors for backward pass
        ctx.save_for_backward(input, weight, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, rstd = ctx.saved_tensors

        # Allocate gradient tensors
        grad_input = torch.empty_like(input)
        # grad_weight is now accumulated in a separate kernel, so it should
        # match the weight's dtype.
        grad_weight = torch.zeros_like(weight)

        # Call the backward kernel
        load_rms_norm_kernel().backward(grad_input, grad_weight, grad_output,
                                        input, weight, rstd)

        return grad_input, grad_weight.to(weight.dtype), None


class FusedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        return RMSNormFunction.apply(hidden_states, self.weight, self.eps) 