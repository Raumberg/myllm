#include <torch/extension.h>
#include "geglu_kerr.cuh"
#include <c10/cuda/CUDAException.h>

// ================ Exact Forward ================ //
template <typename T>
void dispatch_geglu_exact_forward(const torch::Tensor& gate, const torch::Tensor& up, torch::Tensor& h) {
    const int64_t n_elements = gate.numel();
    constexpr int BLOCK_SIZE = 1024;
    const int grid_size = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    geglu_exact_forward_kernel<T><<<grid_size, BLOCK_SIZE>>>(
        reinterpret_cast<const T*>(gate.data_ptr()),
        reinterpret_cast<const T*>(up.data_ptr()),
        reinterpret_cast<T*>(h.data_ptr()),
        n_elements
    );
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());
}

// ================ Exact Backward ================ //
template <typename T>
void dispatch_geglu_exact_backward(
    const torch::Tensor& grad_h,
    const torch::Tensor& gate,
    const torch::Tensor& up,
    torch::Tensor& grad_gate,
    torch::Tensor& grad_up
) {
    const int64_t n_elements = gate.numel();
    constexpr int BLOCK_SIZE = 1024;
    const int grid_size = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    geglu_exact_backward_kernel<T><<<grid_size, BLOCK_SIZE>>>(
        reinterpret_cast<const T*>(grad_h.data_ptr()),
        reinterpret_cast<const T*>(gate.data_ptr()),
        reinterpret_cast<const T*>(up.data_ptr()),
        reinterpret_cast<T*>(grad_gate.data_ptr()),
        reinterpret_cast<T*>(grad_up.data_ptr()),
        n_elements
    );
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());
}

// ================ Approx Forward ================ //
template <typename T>
void dispatch_geglu_approx_forward(const torch::Tensor& gate, const torch::Tensor& up, torch::Tensor& h) {
    const int64_t n_elements = gate.numel();
    constexpr int BLOCK_SIZE = 1024;
    const int grid_size = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    geglu_approx_forward_kernel<T><<<grid_size, BLOCK_SIZE>>>(
        reinterpret_cast<const T*>(gate.data_ptr()),
        reinterpret_cast<const T*>(up.data_ptr()),
        reinterpret_cast<T*>(h.data_ptr()),
        n_elements
    );
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());
}

// ================ Approx Backward ================ //
template <typename T>
void dispatch_geglu_approx_backward(
    const torch::Tensor& grad_h,
    const torch::Tensor& gate,
    const torch::Tensor& up,
    torch::Tensor& grad_gate,
    torch::Tensor& grad_up
) {
    const int64_t n_elements = gate.numel();
    constexpr int BLOCK_SIZE = 1024;
    const int grid_size = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    geglu_approx_backward_kernel<T><<<grid_size, BLOCK_SIZE>>>(
        reinterpret_cast<const T*>(grad_h.data_ptr()),
        reinterpret_cast<const T*>(gate.data_ptr()),
        reinterpret_cast<const T*>(up.data_ptr()),
        reinterpret_cast<T*>(grad_gate.data_ptr()),
        reinterpret_cast<T*>(grad_up.data_ptr()),
        n_elements
    );
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());
}

// ================ Python Bindings ================ //
torch::Tensor geglu_exact_forward(torch::Tensor gate, torch::Tensor up) {
    auto h = torch::empty_like(gate);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        gate.scalar_type(), "geglu_exact_forward", [&] {
            dispatch_geglu_exact_forward<scalar_t>(gate, up, h);
    });
    return h;
}

std::tuple<torch::Tensor, torch::Tensor> geglu_exact_backward(
    torch::Tensor grad_h, torch::Tensor gate, torch::Tensor up
) {
    auto grad_gate = torch::empty_like(gate);
    auto grad_up = torch::empty_like(up);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        gate.scalar_type(), "geglu_exact_backward", [&] {
            dispatch_geglu_exact_backward<scalar_t>(grad_h, gate, up, grad_gate, grad_up);
    });
    return {grad_gate, grad_up};
}

torch::Tensor geglu_approx_forward(torch::Tensor gate, torch::Tensor up) {
    auto h = torch::empty_like(gate);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        gate.scalar_type(), "geglu_approx_forward", [&] {
            dispatch_geglu_approx_forward<scalar_t>(gate, up, h);
    });
    return h;
}

std::tuple<torch::Tensor, torch::Tensor> geglu_approx_backward(
    torch::Tensor grad_h, torch::Tensor gate, torch::Tensor up
) {
    auto grad_gate = torch::empty_like(gate);
    auto grad_up = torch::empty_like(up);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        gate.scalar_type(), at::ScalarType::Half, at::ScalarType::BFloat16,
        "geglu_approx_backward", [&] {
            dispatch_geglu_approx_backward<scalar_t>(grad_h, gate, up, grad_gate, grad_up);
    });
    return {grad_gate, grad_up};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exact_forward", &geglu_exact_forward, "Exact GEGLU forward");
    m.def("exact_backward", &geglu_exact_backward, "Exact GEGLU backward");
    m.def("approx_forward", &geglu_approx_forward, "Approx GEGLU forward");
    m.def("approx_backward", &geglu_approx_backward, "Approx GEGLU backward");
}