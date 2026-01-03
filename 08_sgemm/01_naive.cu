#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;

// Naive SGEMM kernel: C = A @ B
// - A: M x K matrix (row-major)
// - B: K x N matrix (row-major)
// - C: M x N matrix (row-major)
__global__ void sgemm_naive_kernel(float* __restrict__ c_ptr,
                                   float const* __restrict__ a_ptr,
                                   float const* __restrict__ b_ptr, int const M,
                                   int const N, int const K) {
  int const row = blockDim.y * blockIdx.y + threadIdx.y;
  int const col = blockDim.x * blockIdx.x + threadIdx.x;

  float accum = 0.0f;
  for (int k = 0; k < K; k++) {
    accum += a_ptr[row * K + k] * b_ptr[k * N + col];
  }
  c_ptr[row * N + col] = accum;
}

void launch_sgemm_naive(float* c_ptr, float const* a_ptr, float const* b_ptr,
                        int const M, int const N, int const K) {
  dim3 const block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 const grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  sgemm_naive_kernel<<<grid, block>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}