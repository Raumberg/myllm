template <typename T>
__device__ __forceinline__ float sigmoid_approx(float x) {
    return 1.0f / (1.0f + expf(-x));
}