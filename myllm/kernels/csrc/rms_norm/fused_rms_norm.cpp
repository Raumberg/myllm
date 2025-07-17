#include "fused_rms_norm_kernel.cuh"
#include <torch/extension.h>

void forward_cu(torch::Tensor &output, torch::Tensor &rstd,
                const torch::Tensor &input, const torch::Tensor &weight,
                float eps) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

  rms_norm_forward(output, rstd, input, weight, eps);
}

void backward_cu(torch::Tensor &grad_input, torch::Tensor &grad_weight,
                 const torch::Tensor &grad_output, const torch::Tensor &input,
                 const torch::Tensor &weight, const torch::Tensor &rstd) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
  TORCH_CHECK(rstd.is_cuda(), "rstd must be a CUDA tensor");
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(grad_weight.is_contiguous(), "grad_weight must be contiguous");

  rms_norm_backward(grad_input, grad_weight, grad_output, input, weight, rstd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_cu, "RMSNorm forward (CUDA)");
  m.def("backward", &backward_cu, "RMSNorm backward (CUDA)");
} 