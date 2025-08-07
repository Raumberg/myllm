#include "fused_rms_norm_kernel.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

constexpr int WARP_SIZE = 32;

template <typename T>
__device__ T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <typename T>
__device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Now, the first warp aggregates the results from shared memory.
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : (T)0.0;
  if (wid == 0)
    val = warpReduceSum(val);
  
  // The final sum is in the first thread of the first warp.
  // We need to broadcast it to all threads.
  if (wid == 0 && lane == 0) {
      shared[0] = val;
  }
  __syncthreads();
  return shared[0];
}

template <typename T, typename T_ACC>
__global__ void rms_norm_forward_kernel(T *__restrict__ output,
                                        T_ACC *__restrict__ rstd,
                                        const T *__restrict__ input,
                                        const T *__restrict__ weight,
                                        const int hidden_size,
                                        const float eps) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int row_offset = bid * hidden_size;

  T_ACC sum = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    const T_ACC val = static_cast<T_ACC>(input[row_offset + i]);
    sum += val * val;
  }

  sum = blockReduceSum<T_ACC>(sum);

  if (tid == 0) {
    if (sizeof(T_ACC) == sizeof(double)) {
        rstd[bid] = rsqrt(sum / hidden_size + eps);
    } else {
        rstd[bid] = rsqrtf(sum / hidden_size + eps);
    }
  }
  __syncthreads();

  const T_ACC inv_rms_val = rstd[bid];
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    output[row_offset + i] =
        static_cast<T>(static_cast<T_ACC>(input[row_offset + i]) * inv_rms_val *
                       static_cast<T_ACC>(weight[i]));
  }
}

template <typename T, typename T_ACC>
__global__ void rms_norm_backward_kernel(
    T *__restrict__ grad_input, T *__restrict__ grad_weight_part,
    const T *__restrict__ grad_output, const T *__restrict__ input,
    const T *__restrict__ weight, const T_ACC *__restrict__ rstd,
    const int hidden_size) {
  const int tid = threadIdx.x;
  const int row_idx = blockIdx.x;
  const int row_offset = row_idx * hidden_size;

  const T_ACC rstd_val = rstd[row_idx];

  T_ACC dot_grad_out_input_weighted = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    dot_grad_out_input_weighted += static_cast<T_ACC>(grad_output[row_offset + i]) *
                                   static_cast<T_ACC>(input[row_offset + i]) *
                                   static_cast<T_ACC>(weight[i]);
  }
  dot_grad_out_input_weighted = blockReduceSum<T_ACC>(dot_grad_out_input_weighted);

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    const T_ACC grad_out_val = static_cast<T_ACC>(grad_output[row_offset + i]);
    const T_ACC input_val = static_cast<T_ACC>(input[row_offset + i]);
    const T_ACC weight_val = static_cast<T_ACC>(weight[i]);

    const T_ACC grad_weight_val = grad_out_val * input_val * rstd_val;
    grad_weight_part[row_idx * hidden_size + i] = static_cast<T>(grad_weight_val);

    T_ACC grad_input_val = rstd_val * (grad_out_val * weight_val - 
                            (1.0f / hidden_size) * dot_grad_out_input_weighted * input_val * rstd_val * rstd_val);
    
    grad_input[row_offset + i] = static_cast<T>(grad_input_val);
  }
}

template <typename T_in, typename T_out, typename T_ACC>
__global__ void sum_grad_weight_kernel(T_out *__restrict__ grad_weight,
                                       const T_in *__restrict__ grad_weight_part,
                                       const int batch_size,
                                       const int hidden_size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        T_ACC sum = 0.0;
        for (int j = 0; j < batch_size; ++j) {
            sum += static_cast<T_ACC>(grad_weight_part[j * hidden_size + i]);
        }
        grad_weight[i] = static_cast<T_out>(sum);
    }
}

void rms_norm_forward(torch::Tensor &output, torch::Tensor &rstd,
                      const torch::Tensor &input, const torch::Tensor &weight,
                      float eps) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const auto batch_size = input.size(0);
  const auto hidden_size = input.size(1);
  const int block_size = 1024;

  if (input.scalar_type() == torch::kFloat16) {
    rms_norm_forward_kernel<at::Half, float>
        <<<batch_size, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            output.data_ptr<at::Half>(), rstd.data_ptr<float>(),
            input.data_ptr<at::Half>(), weight.data_ptr<at::Half>(),
            hidden_size, eps);
  } else if (input.scalar_type() == torch::kBFloat16) {
    rms_norm_forward_kernel<at::BFloat16, float>
        <<<batch_size, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            output.data_ptr<at::BFloat16>(), rstd.data_ptr<float>(),
            input.data_ptr<at::BFloat16>(), weight.data_ptr<at::BFloat16>(),
            hidden_size, eps);
  } else if (input.scalar_type() == torch::kFloat32) {
    rms_norm_forward_kernel<float, float>
        <<<batch_size, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            output.data_ptr<float>(), rstd.data_ptr<float>(),
            input.data_ptr<float>(), weight.data_ptr<float>(), hidden_size,
            eps);
  } else if (input.scalar_type() == torch::kDouble) {
    rms_norm_forward_kernel<double, double>
        <<<batch_size, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            output.data_ptr<double>(), rstd.data_ptr<double>(),
            input.data_ptr<double>(), weight.data_ptr<double>(), hidden_size,
            eps);
  } else {
    TORCH_CHECK(false, "Unsupported dtype for forward");
  }
}

void rms_norm_backward(torch::Tensor &grad_input, torch::Tensor &grad_weight,
                       const torch::Tensor &grad_output,
                       const torch::Tensor &input, const torch::Tensor &weight,
                       const torch::Tensor &rstd) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const auto batch_size = input.size(0);
  const auto hidden_size = input.size(1);
  const int block_size = 1024;
  const int grad_weight_block_size = 256;

  auto grad_weight_part = torch::empty({batch_size, hidden_size}, input.options());

  if (input.scalar_type() == torch::kFloat16) {
    rms_norm_backward_kernel<at::Half, float>
        <<<batch_size, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            grad_input.data_ptr<at::Half>(), grad_weight_part.data_ptr<at::Half>(),
            grad_output.data_ptr<at::Half>(), input.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(), rstd.data_ptr<float>(), hidden_size);
  } else if (input.scalar_type() == torch::kBFloat16) {
    rms_norm_backward_kernel<at::BFloat16, float>
        <<<batch_size, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            grad_input.data_ptr<at::BFloat16>(), grad_weight_part.data_ptr<at::BFloat16>(),
            grad_output.data_ptr<at::BFloat16>(),
            input.data_ptr<at::BFloat16>(), weight.data_ptr<at::BFloat16>(),
            rstd.data_ptr<float>(), hidden_size);
  } else if (input.scalar_type() == torch::kFloat32) {
    rms_norm_backward_kernel<float, float>
        <<<batch_size, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            grad_input.data_ptr<float>(), grad_weight_part.data_ptr<float>(),
            grad_output.data_ptr<float>(), input.data_ptr<float>(),
            weight.data_ptr<float>(), rstd.data_ptr<float>(), hidden_size);
  } else if (input.scalar_type() == torch::kDouble) {
    rms_norm_backward_kernel<double, double>
        <<<batch_size, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            grad_input.data_ptr<double>(), grad_weight_part.data_ptr<double>(),
            grad_output.data_ptr<double>(), input.data_ptr<double>(),
            weight.data_ptr<double>(), rstd.data_ptr<double>(), hidden_size);
  } else {
    TORCH_CHECK(false, "Unsupported dtype for backward");
  }

  // Launch the summation kernel for grad_weight
  const int grad_weight_grid_size =
      (hidden_size + grad_weight_block_size - 1) / grad_weight_block_size;

  if (input.scalar_type() == torch::kFloat16) {
    sum_grad_weight_kernel<at::Half, at::Half, float>
        <<<grad_weight_grid_size, grad_weight_block_size, 0,
           c10::cuda::getCurrentCUDAStream()>>>(
            grad_weight.data_ptr<at::Half>(),
            grad_weight_part.data_ptr<at::Half>(), batch_size, hidden_size);
  } else if (input.scalar_type() == torch::kBFloat16) {
    sum_grad_weight_kernel<at::BFloat16, at::BFloat16, float>
        <<<grad_weight_grid_size, grad_weight_block_size, 0,
           c10::cuda::getCurrentCUDAStream()>>>(
            grad_weight.data_ptr<at::BFloat16>(),
            grad_weight_part.data_ptr<at::BFloat16>(), batch_size,
            hidden_size);
  } else if (input.scalar_type() == torch::kFloat32) {
    sum_grad_weight_kernel<float, float, float>
        <<<grad_weight_grid_size, grad_weight_block_size, 0,
           c10::cuda::getCurrentCUDAStream()>>>(
            grad_weight.data_ptr<float>(), grad_weight_part.data_ptr<float>(),
            batch_size, hidden_size);
  } else if (input.scalar_type() == torch::kDouble) {
    sum_grad_weight_kernel<double, double, double>
        <<<grad_weight_grid_size, grad_weight_block_size, 0,
           c10::cuda::getCurrentCUDAStream()>>>(
            grad_weight.data_ptr<double>(),
            grad_weight_part.data_ptr<double>(), batch_size, hidden_size);
  }
} 