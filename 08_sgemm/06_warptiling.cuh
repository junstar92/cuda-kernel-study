#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

namespace sgemm {
namespace warptiling {

namespace detail {

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K>
struct KernelTraits {
  static constexpr int kThreadblockM = BLOCK_M;
  static constexpr int kThreadblockN = BLOCK_N;
  static constexpr int kThreadblockK = BLOCK_K;
  static constexpr int kWarpM = WARP_M;
  static constexpr int kWarpN = WARP_N;
  static constexpr int kWarpK = WARP_K;

  static constexpr int kWarpCountM = BLOCK_M / WARP_M;
  static constexpr int kWarpCountN = BLOCK_N / WARP_N;
  static constexpr int kNumWarps = kWarpCountM * kWarpCountN;
  static constexpr int kWarpSize = 32;
  static constexpr int kNumThreads = kNumWarps * kWarpSize;
  static constexpr int kElementBitsA = utils::sizeof_bits<float>::value;
  static constexpr int kElementBitsB = utils::sizeof_bits<float>::value;

  static constexpr int kWarpThreadsM = (WARP_M > WARP_N) ? 8 : 4;
  static constexpr int kWarpThreadsN = 32 / kWarpThreadsM;
  static constexpr int kThreadTileM = WARP_M / kWarpThreadsM;
  static constexpr int kThreadTileN = WARP_N / kWarpThreadsN;

  static constexpr int kLaneLayout =
      (kThreadTileM > 4 && kThreadTileN > 4) ? 2 : 1;
  static constexpr int kLaneStride = kWarpThreadsN * kLaneLayout;

  static constexpr int kSmemPaddingA =
      utils::simt_transpose_padding(kWarpSize, kThreadblockK, kElementBitsA);
  static constexpr int kSmemPaddingB = 0;

  static constexpr int kSmemStrideA = kThreadblockM + kSmemPaddingA;
  static constexpr int kSmemNumElemsA = kThreadblockK * kSmemStrideA;
  static constexpr int kSmemStrideB = kThreadblockN + kSmemPaddingB;
  static constexpr int kSmemNumElemsB = kThreadblockK * kSmemStrideB;
  static constexpr int kTotalSmemElems = kSmemNumElemsA + kSmemNumElemsB;
  static constexpr int kSmemBytes = kTotalSmemElems * sizeof(float);

  static constexpr int kLaneMmaShape = 4;
  static constexpr int kThreadblockALoads =
      (kThreadblockM * kThreadblockK) / kNumThreads;
  static constexpr int kThreadblockBLoads =
      (kThreadblockK * kThreadblockN) / kNumThreads;

  static_assert(kNumThreads == 256,
                "This kernel assumes exactly 256 threads per block.");
  static_assert(kThreadTileM == 8 && kThreadTileN == 8,
                "This kernel assumes an 8x8 thread tile.");
  static_assert(kLaneLayout == 2,
                "This kernel assumes RowMajorInterleaved<2> lane layout.");
  static_assert((kThreadblockM * kThreadblockK) % kNumThreads == 0,
                "A tile must be evenly covered by per-thread scalar loads.");
  static_assert((kThreadblockK * kThreadblockN) % kNumThreads == 0,
                "B tile must be evenly covered by per-thread scalar loads.");
};

}  // namespace detail

template <typename Traits>
__device__ __forceinline__ void load_a_thread_fragment(
    float (&tb_frag_a)[Traits::kThreadblockALoads], float const* a_ptr, int lda,
    int tile_k) {
#pragma unroll
  for (int iter = 0; iter < Traits::kThreadblockALoads; ++iter) {
    int const elem_idx = threadIdx.x + iter * Traits::kNumThreads;
    int const row = elem_idx / Traits::kThreadblockK;
    int const col = elem_idx % Traits::kThreadblockK;

    float const* src = a_ptr + row * lda + tile_k + col;
    tb_frag_a[iter] = *src;
  }
}

template <typename Traits>
__device__ __forceinline__ void load_b_thread_fragment(
    float (&tb_frag_b)[Traits::kThreadblockBLoads], float const* b_ptr, int ldb,
    int tile_k) {
#pragma unroll
  for (int iter = 0; iter < Traits::kThreadblockBLoads; ++iter) {
    int const elem_idx = threadIdx.x + iter * Traits::kNumThreads;
    int const row = elem_idx / Traits::kThreadblockN;
    int const col = elem_idx % Traits::kThreadblockN;

    float const* src = b_ptr + (tile_k + row) * ldb + col;
    tb_frag_b[iter] = *src;
  }
}

template <typename Traits>
__device__ __forceinline__ void store_a_thread_fragment_to_smem(
    float* smem_a, float const (&tb_frag_a)[Traits::kThreadblockALoads]) {
#pragma unroll
  for (int iter = 0; iter < Traits::kThreadblockALoads; ++iter) {
    int const elem_idx = threadIdx.x + iter * Traits::kNumThreads;
    int const row = elem_idx / Traits::kThreadblockK;
    int const col = elem_idx % Traits::kThreadblockK;

    smem_a[col * Traits::kSmemStrideA + row] = tb_frag_a[iter];
  }
}

template <typename Traits>
__device__ __forceinline__ void store_b_thread_fragment_to_smem(
    float* smem_b, float const (&tb_frag_b)[Traits::kThreadblockBLoads]) {
#pragma unroll
  for (int iter = 0; iter < Traits::kThreadblockBLoads; ++iter) {
    int const elem_idx = threadIdx.x + iter * Traits::kNumThreads;
    int const row = elem_idx / Traits::kThreadblockN;
    int const col = elem_idx % Traits::kThreadblockN;

    smem_b[row * Traits::kSmemStrideB + col] = tb_frag_b[iter];
  }
}

template <typename Traits>
__device__ __forceinline__ void warp_mma(float (&accum)[64],
                                         float const* smem_a_ptr,
                                         float const* smem_b_ptr, int warp_m,
                                         int warp_n, int thread_m,
                                         int thread_n) {
  float const* warp_a_ptr = smem_a_ptr + warp_m * Traits::kWarpM;
  float const* warp_b_ptr = smem_b_ptr + warp_n * Traits::kWarpN;

#pragma unroll
  for (int k = 0; k < Traits::kThreadblockK; ++k) {
    float const* a_src_0 = warp_a_ptr + k * Traits::kSmemStrideA +
                           thread_m * Traits::kLaneMmaShape;
    float const* a_src_1 =
        warp_a_ptr + k * Traits::kSmemStrideA +
        (thread_m + Traits::kWarpThreadsM) * Traits::kLaneMmaShape;
    float const* b_src_0 = warp_b_ptr + k * Traits::kSmemStrideB +
                           thread_n * Traits::kLaneMmaShape;
    float const* b_src_1 =
        warp_b_ptr + k * Traits::kSmemStrideB +
        (thread_n + Traits::kWarpThreadsN) * Traits::kLaneMmaShape;

    float const a[8] = {a_src_0[0], a_src_0[1], a_src_0[2], a_src_0[3],
                        a_src_1[0], a_src_1[1], a_src_1[2], a_src_1[3]};
    float const b[8] = {b_src_0[0], b_src_0[1], b_src_0[2], b_src_0[3],
                        b_src_1[0], b_src_1[1], b_src_1[2], b_src_1[3]};

    // #pragma unroll
    //     for (int n = 0; n < 8; n += 2) {
    //       bool const reverse_m = (n & 2);

    // #pragma unroll
    //       for (int m_pair = 0; m_pair < 4; ++m_pair) {
    //         int const m = reverse_m ? (6 - 2 * m_pair) : (2 * m_pair);

    //         accum[(m + 0) * 8 + (n + 0)] += a[m + 0] * b[n + 0];
    //         accum[(m + 1) * 8 + (n + 0)] += a[m + 1] * b[n + 0];
    //         accum[(m + 1) * 8 + (n + 1)] += a[m + 1] * b[n + 1];
    //         accum[(m + 0) * 8 + (n + 1)] += a[m + 0] * b[n + 1];
    //       }
    //     }

#pragma unroll
    for (int m = 0; m < Traits::kThreadTileM; ++m) {
#pragma unroll
      for (int n = 0; n < Traits::kThreadTileN; ++n) {
        accum[m * Traits::kThreadTileN + n] += a[m] * b[n];
      }
    }
  }
}

template <typename Traits>
__device__ __forceinline__ void store_accum_to_gmem(float* c_ptr,
                                                    float const (&accum)[64],
                                                    int warp_m, int warp_n,
                                                    int thread_m, int thread_n,
                                                    int ldc) {
  constexpr int kNumRowGroups = Traits::kThreadTileM / Traits::kLaneMmaShape;
  constexpr int kNumColGroups = Traits::kThreadTileN / Traits::kLaneMmaShape;
  constexpr int kRowGroupStride = Traits::kWarpThreadsM * Traits::kLaneMmaShape;
  constexpr int kColGroupStride = Traits::kWarpThreadsN * Traits::kLaneMmaShape;

  int const row_base =
      warp_m * Traits::kWarpM + thread_m * Traits::kLaneMmaShape;
  int const col_base =
      warp_n * Traits::kWarpN + thread_n * Traits::kLaneMmaShape;

#pragma unroll
  for (int row_group = 0; row_group < kNumRowGroups; ++row_group) {
    int const row_group_base = row_base + row_group * kRowGroupStride;

#pragma unroll
    for (int col_group = 0; col_group < kNumColGroups; ++col_group) {
      int const col_group_base = col_base + col_group * kColGroupStride;

#pragma unroll
      for (int m = 0; m < Traits::kLaneMmaShape; ++m) {
        float* dst = c_ptr + (row_group_base + m) * ldc + col_group_base;

#pragma unroll
        for (int n = 0; n < Traits::kLaneMmaShape; ++n) {
          int const accum_row = row_group * Traits::kLaneMmaShape + m;
          int const accum_col = col_group * Traits::kLaneMmaShape + n;
          dst[n] = accum[accum_row * Traits::kThreadTileN + accum_col];
        }
      }
    }
  }
}

template <typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads) void sgemm_warptiling(
    float* __restrict__ c_ptr, float const* __restrict__ a_ptr,
    float const* __restrict__ b_ptr, int const M, int const N, int const K) {
  int const tile_m = blockIdx.x;
  int const tile_n = blockIdx.y;

  int const block_row = tile_m * Traits::kThreadblockM;
  int const block_col = tile_n * Traits::kThreadblockN;

  a_ptr += block_row * K;
  b_ptr += block_col;
  c_ptr += block_row * N + block_col;

  int const tid = threadIdx.x;
  int const warp_id = tid / 32;
  int const lane_id = tid % 32;

  int const warp_m = warp_id % Traits::kWarpCountM;
  int const warp_n = warp_id / Traits::kWarpCountM;

  int const row_major = lane_id / Traits::kLaneStride;
  int const residual = lane_id - row_major * Traits::kLaneStride;
  int const thread_n = residual / Traits::kLaneLayout;
  int const row_minor = residual - thread_n * Traits::kLaneLayout;
  int const thread_m = row_major * Traits::kLaneLayout + row_minor;

  extern __shared__ float smem[];
  float* smem_a = smem;
  float* smem_b = smem + Traits::kSmemNumElemsA;

  float accum[64] = {0.0f};
  float tb_frag_a[Traits::kThreadblockALoads];
  float tb_frag_b[Traits::kThreadblockBLoads];

  int const num_tiles = K / Traits::kThreadblockK;
  if (num_tiles == 0) {
    return;
  }

  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    int const tile_k = tile_idx * Traits::kThreadblockK;

    load_a_thread_fragment<Traits>(tb_frag_a, a_ptr, K, tile_k);
    load_b_thread_fragment<Traits>(tb_frag_b, b_ptr, N, tile_k);

    store_a_thread_fragment_to_smem<Traits>(smem_a, tb_frag_a);
    store_b_thread_fragment_to_smem<Traits>(smem_b, tb_frag_b);

    __syncthreads();

    warp_mma<Traits>(accum, smem_a, smem_b, warp_m, warp_n, thread_m, thread_n);

    __syncthreads();

    // if (tile_idx + 1 < num_tiles) {
    //   int const next_tile_k = tile_k + Traits::kThreadblockK;
    //   load_a_thread_fragment<Traits>(tb_frag_a, a_ptr, K, next_tile_k);
    //   load_b_thread_fragment<Traits>(tb_frag_b, b_ptr, N, next_tile_k);
    // }
  }

  store_accum_to_gmem<Traits>(c_ptr, accum, warp_m, warp_n, thread_m, thread_n,
                              N);
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  using Traits =
      detail::KernelTraits<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K>;

  dim3 const grid(utils::ceil_div(M, BLOCK_M), utils::ceil_div(N, BLOCK_N));
  dim3 const block(Traits::kNumThreads);

  sgemm_warptiling<Traits>
      <<<grid, block, Traits::kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace warptiling
}  // namespace sgemm
