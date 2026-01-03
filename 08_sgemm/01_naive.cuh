#pragma once
#include <cuda_runtime.h>

#include "utils.cuh"

namespace sgemm {
namespace naive {

constexpr int kBlockSize = 32;

namespace row {

/** Naive SGEMM kernel
 * - threadIdx.y / blockIdx.y select the output row
 * - threadIdx.x / blockIdx.x select the output column
 * - each thread computes one C[row, col] element directly from global memory
 */
__global__ void sgemm_naive_row(float* __restrict__ c_ptr,
                                float const* __restrict__ a_ptr,
                                float const* __restrict__ b_ptr, int const M,
                                int const N, int const K) {
  int const row = blockDim.y * blockIdx.y + threadIdx.y;
  int const col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  float accum = 0.f;
  for (int k = 0; k < K; ++k) {
    accum += a_ptr[row * K + k] * b_ptr[k * N + col];
  }

  c_ptr[row * N + col] = accum;
}

inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  dim3 const block(kBlockSize, kBlockSize);
  dim3 const grid((N + kBlockSize - 1) / kBlockSize,
                  (M + kBlockSize - 1) / kBlockSize);

  sgemm_naive_row<<<grid, block>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace row

namespace col {

/** Naive SGEMM kernel
 * - threadIdx.x / blockIdx.x select the output row
 * - threadIdx.y / blockIdx.y select the output column
 * - arithmetic is identical to sgemm_naive_row; only the thread-to-output
 * mapping changes
 */
__global__ void sgemm_naive_col(float* __restrict__ c_ptr,
                                float const* __restrict__ a_ptr,
                                float const* __restrict__ b_ptr, int const M,
                                int const N, int const K) {
  int const row = blockDim.x * blockIdx.x + threadIdx.x;
  int const col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= M || col >= N) {
    return;
  }

  float accum = 0.f;
  for (int k = 0; k < K; ++k) {
    accum += a_ptr[row * K + k] * b_ptr[k * N + col];
  }

  c_ptr[row * N + col] = accum;
}

inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  dim3 const block(kBlockSize, kBlockSize);
  dim3 const grid(utils::ceil_div(N, kBlockSize),
                  utils::ceil_div(M, kBlockSize));

  sgemm_naive_col<<<grid, block>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace col

}  // namespace naive
}  // namespace sgemm