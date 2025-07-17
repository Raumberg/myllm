#include <torch/extension.h>

namespace myllm {
namespace kernels {

// Forward declaration
void swiglu_forward(
    torch::Tensor& out,
    const torch::Tensor& gate,
    const torch::Tensor& up
);

} // namespace kernels
} // namespace myllm

torch::Tensor swiglu_forward_py(
    const torch::Tensor& gate,
    const torch::Tensor& up
) {
    TORCH_CHECK(gate.is_cuda(), "Gate tensor must be on a CUDA device");
    TORCH_CHECK(up.is_cuda(), "Up tensor must be on a CUDA device");
    TORCH_CHECK(gate.is_contiguous(), "Gate tensor must be contiguous");
    TORCH_CHECK(up.is_contiguous(), "Up tensor must be contiguous");
    TORCH_CHECK(gate.sizes() == up.sizes(), "Gate and Up tensors must have the same shape");

    auto out = torch::empty_like(gate);
    myllm::kernels::swiglu_forward(out, gate, up);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swiglu_forward_py, "SwiGLU Forward Pass (CUDA)");
} 