import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

# Mark all tests in this file as 'kernel'
pytestmark = pytest.mark.kernel

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_rms_norm_compilation():
    """
    Test if the FusedRMSNorm CUDA kernel can be compiled and loaded successfully.
    """
    try:
        from myllm.kerr import load_rms_norm_kernel
        kernel = load_rms_norm_kernel()
        assert kernel is not None, "Kernel module should not be None"
        assert hasattr(kernel, "forward"), "Kernel module should have a 'forward' function"
    except Exception as e:
        pytest.fail(f"Failed to compile or load FusedRMSNorm kernel: {e}")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_rms_norm_forward_and_backward():
    """
    Test the forward and backward pass of the custom FusedRMSNorm kernel against a reference implementation.
    """
    from myllm.kerr.fused_rms_norm import FusedRMSNorm
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    hidden_size = 128
    batch_size = 4
    seq_len = 512
    eps = 1e-5

    # Use double for more precise gradient checking
    x = torch.randn(
        batch_size, seq_len, hidden_size, device="cuda", dtype=torch.double, requires_grad=True
    )
    # Create a separate clone for the torch implementation to avoid in-place modification issues
    x_torch = x.clone().detach().requires_grad_(True)
    
    # Custom kernel
    fused_norm = FusedRMSNorm(hidden_size, eps=eps).to("cuda", dtype=torch.double)
    fused_output = fused_norm(x)

    # Reference LlamaRMSNorm
    torch_norm = LlamaRMSNorm(hidden_size, eps=eps).to("cuda", dtype=torch.double)
    torch_norm.weight.data.copy_(fused_norm.weight.data)
    torch_output = torch_norm(x_torch)

    # 1. Test Forward Pass
    assert torch.allclose(fused_output, torch_output, atol=1e-5, rtol=1e-5), \
        "Forward pass output does not match the reference implementation."

    # 2. Test Backward Pass
    grad_output = torch.randn_like(fused_output)
    fused_output.backward(gradient=grad_output)
    torch_output.backward(gradient=grad_output.clone()) # Use clone to be safe

    # Check weight gradients
    assert torch.allclose(fused_norm.weight.grad, torch_norm.weight.grad, atol=1e-5, rtol=1e-5), \
        "Weight gradients do not match."

    # Check input gradients
    assert torch.allclose(x.grad, x_torch.grad, atol=1e-5, rtol=1e-5), \
        "Input gradients do not match."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_rms_norm_gradcheck():
    """
    Perform a gradcheck to numerically verify the correctness of the backward pass.
    """
    from myllm.kerr.fused_rms_norm import RMSNormFunction

    hidden_size = 16
    batch_size = 2
    seq_len = 8
    eps = 1e-6

    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.double, requires_grad=True)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(RMSNormFunction.apply, (x, weight, eps)), \
        "Gradcheck failed for FusedRMSNorm"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_swiglu_compilation():
    """
    Test if the FusedSwiGLU CUDA kernel can be compiled and loaded successfully.
    """
    try:
        from myllm.kerr import load_swiglu_kernel
        kernel = load_swiglu_kernel()
        assert kernel is not None, "Kernel module should not be None"
        assert hasattr(kernel, "forward"), "Kernel module should have a 'forward' function"
    except Exception as e:
        pytest.fail(f"Failed to compile or load FusedSwiGLU kernel: {e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_swiglu_forward_pass():
    """
    Test the forward pass of the custom FusedSwiGLU kernel against the PyTorch equivalent.
    """
    from myllm.kerr.fused_swiglu import FusedSwiGLU

    in_features = 128
    hidden_features = 256
    batch_size = 4
    seq_len = 16

    # Create dummy input
    x = torch.randn(batch_size, seq_len, in_features, device="cuda", dtype=torch.float16)
    
    # Custom kernel
    fused_swiglu = FusedSwiGLU(in_features, hidden_features).to("cuda", dtype=torch.float16)
    fused_output = fused_swiglu(x)

    # PyTorch equivalent for comparison
    class TorchSwiGLU(nn.Module):
        def __init__(self, in_features, hidden_features):
            super().__init__()
            self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
            self.w_up = nn.Linear(in_features, hidden_features, bias=False)
            self.w_down = nn.Linear(hidden_features, in_features, bias=False)

            # Copy weights from the fused module
            self.w_gate.weight = fused_swiglu.w_gate.weight
            self.w_up.weight = fused_swiglu.w_up.weight
            self.w_down.weight = fused_swiglu.w_down.weight

        def forward(self, x):
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    torch_swiglu = TorchSwiGLU(in_features, hidden_features).to("cuda", dtype=torch.float16)
    torch_output = torch_swiglu(x)

    # Compare the outputs
    assert torch.allclose(fused_output, torch_output, atol=1e-3, rtol=1e-3), \
        "Fused SwiGLU output does not match PyTorch implementation" 