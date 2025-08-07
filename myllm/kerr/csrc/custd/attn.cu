// taken from https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_utils.cuh

// Q*K^T operation.
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
  using A_vec = typename FloatVec<Vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = vllm::fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += VLLM_SHFL_XOR_SYNC(qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template <typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};