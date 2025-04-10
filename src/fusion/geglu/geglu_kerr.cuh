#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper functions for type conversion
template <typename T>
__device__ float type_to_float(T val);

template <>
__device__ float type_to_float<half>(half val) {
    return __half2float(val);
}

template <>
__device__ float type_to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template <typename T>
__device__ T float_to_type(float val);

template <>
__device__ half float_to_type<half>(float val) {
    return __float2half_rn(val);
}

template <>
__device__ __nv_bfloat16 float_to_type<__nv_bfloat16>(float val) {
    return __float2bfloat16_rn(val);
}

// Exact Forward Kernel
template <typename T>
__global__ void geglu_exact_forward_kernel(const T* gate, const T* up, T* h, int n_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    const float e = type_to_float(gate[idx]);
    const float g_val = type_to_float(up[idx]);

    const float gelu_e = 0.5f * e * (1.0f + erff(e * 0.7071067811865475f));
    h[idx] = float_to_type<T>(gelu_e * g_val);
}

// Explicit instantiations for supported types
template __global__ void geglu_exact_forward_kernel<half>(const half*, const half*, half*, int);
template __global__ void geglu_exact_forward_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);

// Exact Backward Kernel
template <typename T>
__global__ void geglu_exact_backward_kernel(
    const T* grad_h, const T* gate, const T* up, T* grad_gate, T* grad_up, int n_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    const float dh = type_to_float(grad_h[idx]);
    const float e = type_to_float(gate[idx]);
    const float g = type_to_float(up[idx]);

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float df_de = 0.5f * (1.0f + erff(e * 0.7071067811865475f)) + 
                      sqrt_2_over_pi * e * expf(-0.5f * e * e);

    const float gelu_e = 0.5f * e * (1.0f + erff(e * 0.7071067811865475f));
    grad_up[idx] = float_to_type<T>(dh * gelu_e);
    grad_gate[idx] = float_to_type<T>(dh * g * df_de);
}

template __global__ void geglu_exact_backward_kernel<half>(const half*, const half*, const half*, half*, half*, int);
template __global__ void geglu_exact_backward_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

// Approximate Forward Kernel
template <typename T>
__global__ void geglu_approx_forward_kernel(const T* gate, const T* up, T* h, int n_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    const float e = type_to_float(gate[idx]);
    const float g_val = type_to_float(up[idx]);

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float a = sqrt_2_over_pi * e;
    const float b = a * 0.044715f * e * e;
    const float tanh_val = tanhf(a + b);
    const float gelu_approx = 0.5f * e * (1.0f + tanh_val);

    h[idx] = float_to_type<T>(gelu_approx * g_val);
}

template __global__ void geglu_approx_forward_kernel<half>(const half*, const half*, half*, int);
template __global__ void geglu_approx_forward_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);

// Approximate Backward Kernel
template <typename T>
__global__ void geglu_approx_backward_kernel(
    const T* grad_h, const T* gate, const T* up, T* grad_gate, T* grad_up, int n_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    const float dh = type_to_float(grad_h[idx]);
    const float e = type_to_float(gate[idx]);
    const float g = type_to_float(up[idx]);

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float a = sqrt_2_over_pi * e;
    const float b = a * 0.044715f * e * e;
    const float tanh_val = tanhf(a + b);
    
    // FIXED: Renamed 'T' to 't_val' to avoid template parameter conflict
    const float t_val = 1.0f + tanh_val;
    const float T2 = 0.5f * t_val;
    const float a_plus_3b = a + 3.0f * b;
    const float Q2 = -T2 * (t_val - 2.0f) * a_plus_3b;
    const float df_de = T2 + Q2;

    grad_up[idx] = float_to_type<T>(dh * T2 * e);
    grad_gate[idx] = float_to_type<T>(dh * g * df_de);
}

template __global__ void geglu_exact_forward_kernel<half>(
    const half*, const half*, half*, int
);
template __global__ void geglu_exact_forward_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int
);

template __global__ void geglu_exact_backward_kernel<half>(
    const half*, const half*, const half*, half*, half*, int
);
template __global__ void geglu_exact_backward_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int
);
