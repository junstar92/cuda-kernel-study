#include <cuda_runtime.h>

// SGEMM kernel using shared memory tiling
// - Each thread block computes a BLOCK_SIZE x BLOCK_SIZE tile of C
// - A and B are loaded into shared memory in BLOCK_SIZE tiles
// - Reduces global memory accesses to improve memory bandwidth utilization
template <int BLOCK_SIZE = 32>
__global__ void sgemm_smem_naive_kernel(float* __restrict__ c_ptr,
                                        float const* __restrict__ a_ptr,
                                        float const* __restrict__ b_ptr,
                                        int const M, int const N, int const K) {
  int const row = threadIdx.y;
  int const col = threadIdx.x;

  a_ptr += BLOCK_SIZE * blockIdx.y * K;
  b_ptr += BLOCK_SIZE * blockIdx.x;
  c_ptr += BLOCK_SIZE * blockIdx.y * N + BLOCK_SIZE * blockIdx.x;

  __shared__ float a_smem[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float b_smem[BLOCK_SIZE][BLOCK_SIZE];

  float accum = 0.0f;

  for (int tile = 0; tile < K; tile += BLOCK_SIZE) {
    a_smem[row][col] = a_ptr[row * K + col];
    b_smem[row][col] = b_ptr[row * N + col];
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
      accum += a_smem[row][k] * b_smem[k][col];
    }
    __syncthreads();

    a_ptr += BLOCK_SIZE;
    b_ptr += BLOCK_SIZE * N;
  }

  c_ptr[row * N + col] = accum;
}

template <int BLOCK_SIZE>
void launch_sgemm_smem_naive(float* c_ptr, float const* a_ptr,
                             float const* b_ptr, int const M, int const N,
                             int const K) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  sgemm_smem_naive_kernel<<<grid, block>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

template void launch_sgemm_smem_naive<32>(float*, float const*, float const*,
                                          int const, int const, int const);