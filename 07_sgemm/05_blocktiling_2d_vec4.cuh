#pragma once
#include <cuda_runtime.h>

namespace sgemm {
namespace blocktiling_2d_vec4 {

/** 2D Block-tiling SGEMM kernel with vectorized global-memory traffic
 * - one thread block computes a BLOCK_M x BLOCK_N output tile
 * - each thread computes THREAD_M x THREAD_N output tile in registers
 * - stages A and B with float4 loads, so the vector width is 4 floats
 * - writes C with float4 stores
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
          int SMEM_A_PADDING>
__global__ void sgemm_blocktiling_2d_vec4(float* __restrict__ c_ptr,
                                          float const* __restrict__ a_ptr,
                                          float const* __restrict__ b_ptr,
                                          int const M, int const N,
                                          int const K) {
  constexpr int vec_width = 4;
  static_assert(BLOCK_K % vec_width == 0);
  static_assert(BLOCK_N % vec_width == 0);
  static_assert(THREAD_N % vec_width == 0);

  constexpr int num_threads = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
  constexpr int smem_stride_a = BLOCK_M + SMEM_A_PADDING;

  int const tile_m = blockIdx.x;
  int const tile_n = blockIdx.y;

  int const block_row = tile_m * BLOCK_M;
  int const block_col = tile_n * BLOCK_N;

  // move global pointers to this block's output tile origin
  a_ptr += block_row * K;
  b_ptr += block_col;
  c_ptr += block_row * N + block_col;

  // shared memory layout:
  // [ smem_a (BLOCK_K x smem_stride_a) | smem_b (BLOCK_K x BLOCK_N) ]
  extern __shared__ float smem[];
  float* smem_a = smem;
  float* smem_b = smem + BLOCK_K * smem_stride_a;

  // per-thread accumulators for THREAD_M x THREAD_N
  float accum[THREAD_M][THREAD_N] = {0.f};

  // thread mapping inside the block tile
  int const tile_row = threadIdx.x / (BLOCK_N / THREAD_N);
  int const tile_col = threadIdx.x % (BLOCK_N / THREAD_N);

  int const num_tiles = K / BLOCK_K;
  if (num_tiles == 0) {
    return;
  }

#pragma unroll
  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    utils::load_a_threadblock_tile_vec4<num_threads, BLOCK_M, BLOCK_K,
                                        smem_stride_a>(smem_a, a_ptr, K);
    utils::load_b_threadblock_tile_vec4<num_threads, BLOCK_K, BLOCK_N, BLOCK_N>(
        smem_b, b_ptr, N);
    __syncthreads();

    // advance to the next K tile in global memory
    a_ptr += BLOCK_K;
    b_ptr += BLOCK_K * N;

#pragma unroll
    for (int k = 0; k < BLOCK_K; ++k) {
#pragma unroll
      for (int m = 0; m < THREAD_M; ++m) {
        float const elem_a =
            smem_a[smem_stride_a * k + THREAD_M * tile_row + m];
#pragma unroll
        for (int n = 0; n < THREAD_N; ++n) {
          float const elem_b = smem_b[BLOCK_N * k + THREAD_N * tile_col + n];
          accum[m][n] += elem_a * elem_b;
        }
      }
    }
    __syncthreads();
  }

  // store accumulated results back to global memory
#pragma unroll
  for (int m = 0; m < THREAD_M; ++m) {
    int const c_row = THREAD_M * tile_row + m;
    int const c_col_base = THREAD_N * tile_col;
#pragma unroll
    for (int n = 0; n < THREAD_N; n += vec_width) {
      float4 const value = utils::make_float4_from_array(&accum[m][n]);
      utils::store_float4(&c_ptr[c_row * N + c_col_base + n], value);
    }
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
          int SMEM_A_PADDING = 4>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  constexpr int num_threads = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
  constexpr int smem_size_b =
      ((BLOCK_M + SMEM_A_PADDING) * BLOCK_K + BLOCK_K * BLOCK_N) *
      sizeof(float);

  dim3 const block(num_threads);
  dim3 const grid(utils::ceil_div(N, BLOCK_N), utils::ceil_div(M, BLOCK_M));

  sgemm_blocktiling_2d_vec4<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N,
                            SMEM_A_PADDING>
      <<<grid, block, smem_size_b>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace blocktiling_2d_vec4
}  // namespace sgemm