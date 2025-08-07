#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cmath>

namespace myllm {
namespace kernels {

template <typename T>
__device__ __forceinline__ float sigmoid_approx(float x) {
    // A common and fast approximation for sigmoid
    return 1.0f / (1.0f + expf(-x));
}

template <typename T>
__global__ void swiglu_forward_kernel(
    T* __restrict__ out,
    const T* __restrict__ gate,
    const T* __restrict__ up,
    int n_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        // Load gate and up-projection
        const float gate_val_fp32 = static_cast<float>(gate[idx]);
        const T up_val = up[idx];

        // Compute SiLU activation in fp32 for stability
        const float silu_val_fp32 = gate_val_fp32 * sigmoid_approx<float>(gate_val_fp32);

        // Multiply with the up-projection and cast back to original type
        out[idx] = static_cast<T>(silu_val_fp32) * up_val;
    }
}

void swiglu_forward(
    torch::Tensor& out,
    const torch::Tensor& gate,
    const torch::Tensor& up
) {
    const int n_elements = gate.numel();
    
    // Heuristic for block and grid size
    const int block_size = 1024;
    const int grid_size = (n_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gate.scalar_type(), "swiglu_forward", [&] {
        swiglu_forward_kernel<scalar_t><<<grid_size, block_size>>>(
            out.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            up.data_ptr<scalar_t>(),
            n_elements
        );
    });
}

} // namespace kernels
} // namespace myllm 