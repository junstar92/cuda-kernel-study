#pragma once
#include <cuda_runtime.h>

#include "utils.cuh"

namespace sgemm {
namespace smem_tiling {

constexpr int kBlockSize = 32;

/** Shared-memory tiled SGEMM kernel
 * - one thread block computes one kBlockSize x kBlockSize output tile
 * - threadIdx.y / threadIdx.x map to one element inside that tile
 * - A and B sub-tiles are staged through shared memory before the inner-product
 * loop
 * - this assumes clean kBlockSize tiling and does not guard matrix edges
 */
__global__ void sgemm_smem_tiling(float* __restrict__ c_ptr,
                                  float const* __restrict__ a_ptr,
                                  float const* __restrict__ b_ptr, int const M,
                                  int const N, int const K) {
  int const tile_m = blockIdx.x;
  int const tile_n = blockIdx.y;

  int const block_row = tile_m * kBlockSize;
  int const block_col = tile_n * kBlockSize;

  int const tile_row = threadIdx.y;
  int const tile_col = threadIdx.x;

  a_ptr += block_row * K;
  b_ptr += block_col;
  c_ptr += block_row * N + block_col;

  __shared__ float smem_a[kBlockSize][kBlockSize];
  __shared__ float smem_b[kBlockSize][kBlockSize];

  float accum = 0.f;

  int const num_tiles = K / kBlockSize;
  if (num_tiles == 0) {
    return;
  }

#pragma unroll
  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    smem_a[tile_row][tile_col] = a_ptr[tile_row * K + tile_col];
    smem_b[tile_row][tile_col] = b_ptr[tile_row * N + tile_col];
    __syncthreads();

    a_ptr += kBlockSize;
    b_ptr += kBlockSize * N;

    float partial_accum = 0.f;
#pragma unroll
    for (int k = 0; k < kBlockSize; ++k) {
      partial_accum += smem_a[tile_row][k] * smem_b[k][tile_col];
    }
    accum += partial_accum;
    __syncthreads();
  }

  c_ptr[tile_row * N + tile_col] = accum;
}

inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  dim3 const block(kBlockSize, kBlockSize);
  dim3 const grid(utils::ceil_div(N, kBlockSize),
                  utils::ceil_div(M, kBlockSize));

  sgemm_smem_tiling<<<grid, block>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace smem_tiling
}  // namespace sgemm