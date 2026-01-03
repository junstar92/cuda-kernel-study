#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

namespace sgemm {
namespace warptiling {

namespace detail {

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K>
struct KernelTraits {
  // Tile sizes at each hierarchy level:
  static constexpr int kThreadblockM = BLOCK_M;
  static constexpr int kThreadblockN = BLOCK_N;
  static constexpr int kThreadblockK = BLOCK_K;
  static constexpr int kWarpM = WARP_M;
  static constexpr int kWarpN = WARP_N;
  static constexpr int kWarpK = WARP_K;

  // How many warp tiles fit inside one CTA tile, and therefore how many
  // hardware threads are needed for one block.
  static constexpr int kWarpCountM = BLOCK_M / WARP_M;
  static constexpr int kWarpCountN = BLOCK_N / WARP_N;
  static constexpr int kNumWarps = kWarpCountM * kWarpCountN;
  static constexpr int kWarpSize = 32;
  static constexpr int kNumThreads = kNumWarps * kWarpSize;
  static constexpr int kElementBitsA = utils::sizeof_bits<float>::value;
  static constexpr int kElementBitsB = utils::sizeof_bits<float>::value;

  // Lay out the 32 lanes of a warp as a kWarpThreadsM x kWarpThreadsN grid.
  // The split is chosen so each thread gets a near-square micro-tile inside the
  // warp tile.
  static constexpr int kWarpThreadsM = (WARP_M > WARP_N) ? 8 : 4;
  static constexpr int kWarpThreadsN = 32 / kWarpThreadsM;
  static constexpr int kThreadTileM = WARP_M / kWarpThreadsM;
  static constexpr int kThreadTileN = WARP_N / kWarpThreadsN;
  static constexpr int kNumAccumulators = kThreadTileM * kThreadTileN;

  // RowMajorInterleaved<2>-style lane mapping. Two neighboring lane ids are
  // paired along M before moving in N, so a logical thread tile is later
  // visited as serveral 4x4 islands.
  static constexpr int kLaneLayout =
      (kThreadTileM > 4 && kThreadTileN > 4) ? 2 : 1;
  static constexpr int kLaneStride = kWarpThreadsN * kLaneLayout;

  // Shared-memory staging:
  // A is stored transposed and padded so warp reads along M avoid bank
  // conflicts. B is kept row-major because the warp already consumes it along
  // N.
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
  static constexpr int kThreadALoadIterations =
      (kThreadblockM * kThreadblockK) / kNumThreads;
  static constexpr int kThreadBLoadIterations =
      (kThreadblockK * kThreadblockN) / kNumThreads;

  // v5 epilogue staging reorders one CTA tile through shared memory before
  // cooperative vectorized global stores.
  static constexpr int kEpilogueElementsPerAccess = 4;
  static constexpr int kEpilogueSmemPadding = 4;
  static constexpr int kEpilogueSmemStride =
      kThreadblockN + kEpilogueSmemPadding;
  static constexpr int kEpilogueSmemNumElems =
      kThreadblockM * kEpilogueSmemStride;
  static constexpr int kEpilogueVectorsPerRow =
      kThreadblockN / kEpilogueElementsPerAccess;
  static constexpr int kEpilogueStoresPerThread =
      (kThreadblockM * kEpilogueVectorsPerRow) / kNumThreads;
  static constexpr int kV5SmemElems =
      utils::max_value(kTotalSmemElems, kEpilogueSmemNumElems);
  static constexpr int kV5SmemBytes = kV5SmemElems * sizeof(float);

  static_assert((kThreadblockM * kThreadblockK) % kNumThreads == 0,
                "A tile must be evenly covered by per-thread scalar loads.");
  static_assert((kThreadblockK * kThreadblockN) % kNumThreads == 0,
                "B tile must be evenly covered by per-thread scalar loads.");
  static_assert(kThreadblockN % kEpilogueElementsPerAccess == 0,
                "v5 epilogue requires float4-aligned CTA tiles.");
  static_assert(
      kEpilogueSmemStride % kEpilogueElementsPerAccess == 0,
      "v5 epilogue shared-memory stride must preserve float4 alignment.");
  static_assert((kThreadblockM * kEpilogueVectorsPerRow) % kNumThreads == 0,
                "CTA tile must map evenly to per-thread epilogue stores.");
};

}  // namespace detail

namespace v0 {

template <typename Traits>
__device__ __forceinline__ void warp_mma(
    float (&accum)[Traits::kNumAccumulators], float const* smem_a_ptr,
    float const* smem_b_ptr, int warp_m, int warp_n, int thread_m,
    int thread_n) {
  float const* warp_a_ptr = smem_a_ptr + warp_m * Traits::kWarpM;
  float const* warp_b_ptr = smem_b_ptr + warp_n * Traits::kWarpN;

  int const advance_offset_a = Traits::kSmemStrideA;
  int offset_a = thread_m * Traits::kThreadTileM;

  int const advance_offset_b = Traits::kSmemStrideB;
  int offset_b = thread_n * Traits::kThreadTileN;

#pragma unroll
  for (int k = 0; k < Traits::kThreadblockK; ++k) {
    float warp_frag_a[Traits::kThreadTileM] = {0.f};
    float warp_frag_b[Traits::kThreadTileN] = {0.f};

#pragma unroll
    for (int m = 0; m < Traits::kThreadTileM; ++m) {
      warp_frag_a[m] = warp_a_ptr[offset_a + m];
    }
#pragma unroll
    for (int n = 0; n < Traits::kThreadTileN; ++n) {
      warp_frag_b[n] = warp_b_ptr[offset_b + n];
    }

#pragma unroll
    for (int m = 0; m < Traits::kThreadTileM; ++m) {
#pragma unroll
      for (int n = 0; n < Traits::kThreadTileN; ++n) {
        accum[m * Traits::kThreadTileN + n] += warp_frag_a[m] * warp_frag_b[n];
      }
    }

    offset_a += advance_offset_a;
    offset_b += advance_offset_b;
  }
}

template <typename Traits>
__device__ __forceinline__ void store_accum_to_gmem(
    float* c_ptr, float const (&accum)[Traits::kNumAccumulators], int warp_m,
    int warp_n, int thread_m, int thread_n, int ldc) {
  int const row_base =
      warp_m * Traits::kWarpM + thread_m * Traits::kThreadTileM;
  int const col_base =
      warp_n * Traits::kWarpN + thread_n * Traits::kThreadTileN;

  for (int m = 0; m < Traits::kThreadTileM; ++m) {
    float* dst = c_ptr + (row_base + m) * ldc + col_base;
    for (int n = 0; n < Traits::kThreadTileN; ++n) {
      dst[n] = accum[m * Traits::kThreadTileN + n];
    }
  }
}

template <typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads) void sgemm_warptiling_v0(
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
  int const thread_m = lane_id / Traits::kWarpThreadsN;
  int const thread_n = lane_id % Traits::kWarpThreadsN;

  extern __shared__ float smem[];
  float* smem_a = smem;
  float* smem_b = smem + Traits::kSmemNumElemsA;

  float accum[Traits::kNumAccumulators] = {0.0f};

  int const num_tiles = K / Traits::kThreadblockK;
  if (num_tiles == 0) {
    return;
  }

  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    int const tile_k = tile_idx * Traits::kThreadblockK;

    utils::load_a_threadblock_tile<Traits::kNumThreads, Traits::kThreadblockM,
                                   Traits::kThreadblockK, Traits::kSmemStrideA>(
        smem_a, a_ptr + tile_k, K);
    utils::load_b_threadblock_tile<Traits::kNumThreads, Traits::kThreadblockK,
                                   Traits::kThreadblockN, Traits::kSmemStrideB>(
        smem_b, b_ptr + tile_k * N, N);

    __syncthreads();

    warp_mma<Traits>(accum, smem_a, smem_b, warp_m, warp_n, thread_m, thread_n);

    __syncthreads();
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

  sgemm_warptiling_v0<Traits>
      <<<grid, block, Traits::kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v0

namespace v1 {

template <typename Traits>
__device__ __forceinline__ void warp_mma(
    float (&accum)[Traits::kNumAccumulators], float const* smem_a_ptr,
    float const* smem_b_ptr, int warp_m, int warp_n, int thread_m,
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
__device__ __forceinline__ void store_accum_to_gmem(
    float* c_ptr, float const (&accum)[Traits::kNumAccumulators], int warp_m,
    int warp_n, int thread_m, int thread_n, int ldc) {
  constexpr int num_row_groups = Traits::kThreadTileM / Traits::kLaneMmaShape;
  constexpr int num_col_groups = Traits::kThreadTileN / Traits::kLaneMmaShape;
  constexpr int row_group_stride =
      Traits::kWarpThreadsM * Traits::kLaneMmaShape;
  constexpr int col_group_stride =
      Traits::kWarpThreadsN * Traits::kLaneMmaShape;

  int const row_base =
      warp_m * Traits::kWarpM + thread_m * Traits::kLaneMmaShape;
  int const col_base =
      warp_n * Traits::kWarpN + thread_n * Traits::kLaneMmaShape;

#pragma unroll
  for (int row_group = 0; row_group < num_row_groups; ++row_group) {
    int const row_group_base = row_base + row_group * row_group_stride;

#pragma unroll
    for (int col_group = 0; col_group < num_col_groups; ++col_group) {
      int const col_group_base = col_base + col_group * col_group_stride;

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
__global__ __launch_bounds__(Traits::kNumThreads) void sgemm_warptiling_v1(
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

  float accum[Traits::kNumAccumulators] = {0.0f};

  int const num_tiles = K / Traits::kThreadblockK;
  if (num_tiles == 0) {
    return;
  }

  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    int const tile_k = tile_idx * Traits::kThreadblockK;

    utils::load_a_threadblock_tile<Traits::kNumThreads, Traits::kThreadblockM,
                                   Traits::kThreadblockK, Traits::kSmemStrideA>(
        smem_a, a_ptr + tile_k, K);
    utils::load_b_threadblock_tile<Traits::kNumThreads, Traits::kThreadblockK,
                                   Traits::kThreadblockN, Traits::kSmemStrideB>(
        smem_b, b_ptr + tile_k * N, N);

    __syncthreads();

    warp_mma<Traits>(accum, smem_a, smem_b, warp_m, warp_n, thread_m, thread_n);

    __syncthreads();
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

  sgemm_warptiling_v1<Traits>
      <<<grid, block, Traits::kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v1

namespace v2 {

// template <typename Traits>
// __device__ __forceinline__ void load_a_thread_fragment(
//     float (&tb_frag_a)[Traits::kThreadALoadIterations], float const* a_ptr,
//     int lda, int tile_k) {
// #pragma unroll
//   for (int iter = 0; iter < Traits::kThreadALoadIterations; ++iter) {
//     int const elem_idx = threadIdx.x + iter * Traits::kNumThreads;
//     int const row = elem_idx / Traits::kThreadblockK;
//     int const col = elem_idx % Traits::kThreadblockK;

//     tb_frag_a[iter] = a_ptr[row * lda + tile_k + col];
//   }
// }

// template <typename Traits>
// __device__ __forceinline__ void load_b_thread_fragment(
//     float (&tb_frag_b)[Traits::kThreadBLoadIterations], float const* b_ptr,
//     int ldb, int tile_k) {
// #pragma unroll
//   for (int iter = 0; iter < Traits::kThreadBLoadIterations; ++iter) {
//     int const elem_idx = threadIdx.x + iter * Traits::kNumThreads;
//     int const row = elem_idx / Traits::kThreadblockN;
//     int const col = elem_idx % Traits::kThreadblockN;

//     tb_frag_b[iter] = b_ptr[(row + tile_k) * ldb + col];
//   }
// }

template <typename Traits>
__device__ __forceinline__ void store_a_thread_fragment_to_smem(
    float* smem_a, float const (&tb_frag_a)[Traits::kThreadALoadIterations]) {
#pragma unroll
  for (int iter = 0; iter < Traits::kThreadALoadIterations; ++iter) {
    int const elem_idx = threadIdx.x + iter * Traits::kNumThreads;
    int const row = elem_idx / Traits::kThreadblockK;
    int const col = elem_idx % Traits::kThreadblockK;

    smem_a[col * Traits::kSmemStrideA + row] = tb_frag_a[iter];
  }
}

template <typename Traits>
__device__ __forceinline__ void store_b_thread_fragment_to_smem(
    float* smem_b, float const (&tb_frag_b)[Traits::kThreadBLoadIterations]) {
#pragma unroll
  for (int iter = 0; iter < Traits::kThreadBLoadIterations; ++iter) {
    int const elem_idx = threadIdx.x + iter * Traits::kNumThreads;
    int const row = elem_idx / Traits::kThreadblockN;
    int const col = elem_idx % Traits::kThreadblockN;

    smem_b[row * Traits::kSmemStrideB + col] = tb_frag_b[iter];
  }
}

template <typename Traits>
__device__ __forceinline__ void load_a_thread_fragment(
    float (&tb_frag_a)[Traits::kThreadALoadIterations], float const* a_ptr,
    int lda, int tile_k) {
  constexpr int row_advance_per_iter =
      Traits::kNumThreads / Traits::kThreadblockK;
  constexpr int col_advance_per_iter =
      Traits::kNumThreads % Traits::kThreadblockK;

  int const advance_offset = row_advance_per_iter * lda + col_advance_per_iter;
  int const row = threadIdx.x / Traits::kThreadblockK;
  int const col = threadIdx.x % Traits::kThreadblockK;
  int offset = row * lda + tile_k + col;

#pragma unroll
  for (int iter = 0; iter < Traits::kThreadALoadIterations; ++iter) {
    tb_frag_a[iter] = a_ptr[offset];
    offset += advance_offset;
  }
}

template <typename Traits>
__device__ __forceinline__ void load_b_thread_fragment(
    float (&tb_frag_b)[Traits::kThreadBLoadIterations], float const* b_ptr,
    int ldb, int tile_k) {
  constexpr int row_advance_per_iter =
      Traits::kNumThreads / Traits::kThreadblockN;
  constexpr int col_advance_per_iter =
      Traits::kNumThreads % Traits::kThreadblockN;

  int const advance_offset = row_advance_per_iter * ldb + col_advance_per_iter;
  int const row = threadIdx.x / Traits::kThreadblockN;
  int const col = threadIdx.x % Traits::kThreadblockN;
  int offset = (tile_k + row) * ldb + col;

#pragma unroll
  for (int iter = 0; iter < Traits::kThreadBLoadIterations; ++iter) {
    tb_frag_b[iter] = b_ptr[offset];
    offset += advance_offset;
  }
}

// template <typename Traits>
// __device__ __forceinline__ void store_a_thread_fragment_to_smem(
//     float* smem_a, float const (&tb_frag_a)[Traits::kThreadALoadIterations])
//     {
//   constexpr int row_advance_per_iter =
//       Traits::kNumThreads / Traits::kThreadblockK;
//   constexpr int col_advance_per_iter =
//       Traits::kNumThreads % Traits::kThreadblockK;

//   constexpr int advance_offset =
//       col_advance_per_iter * Traits::kSmemStrideA + row_advance_per_iter;
//   int const row = threadIdx.x / Traits::kThreadblockK;
//   int const col = threadIdx.x % Traits::kThreadblockK;
//   int offset = col * Traits::kSmemStrideA + row;

// #pragma unroll
//   for (int iter = 0; iter < Traits::kThreadALoadIterations; ++iter) {
//     smem_a[offset] = tb_frag_a[iter];
//     offset += advance_offset;
//   }
// }

// template <typename Traits>
// __device__ __forceinline__ void store_b_thread_fragment_to_smem(
//     float* smem_b, float const (&tb_frag_b)[Traits::kThreadBLoadIterations])
//     {
//   constexpr int row_advance_per_iter =
//       Traits::kNumThreads / Traits::kThreadblockN;
//   constexpr int col_advance_per_iter =
//       Traits::kNumThreads % Traits::kThreadblockN;

//   constexpr int advance_offset =
//       row_advance_per_iter * Traits::kSmemStrideB + col_advance_per_iter;
//   int const row = threadIdx.x / Traits::kThreadblockN;
//   int const col = threadIdx.x % Traits::kThreadblockN;
//   int offset = row * Traits::kSmemStrideB + col;

// #pragma unroll
//   for (int iter = 0; iter < Traits::kThreadBLoadIterations; ++iter) {
//     smem_b[offset] = tb_frag_b[iter];
//     offset += advance_offset;
//   }
// }

template <typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads) void sgemm_warptiling_v2(
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

  float accum[Traits::kNumAccumulators] = {0.0f};
  float tb_frag_a[Traits::kThreadALoadIterations];
  float tb_frag_b[Traits::kThreadBLoadIterations];

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

    v1::warp_mma<Traits>(accum, smem_a, smem_b, warp_m, warp_n, thread_m,
                         thread_n);

    __syncthreads();
  }

  v1::store_accum_to_gmem<Traits>(c_ptr, accum, warp_m, warp_n, thread_m,
                                  thread_n, N);
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  using Traits =
      detail::KernelTraits<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K>;

  dim3 const grid(utils::ceil_div(M, BLOCK_M), utils::ceil_div(N, BLOCK_N));
  dim3 const block(Traits::kNumThreads);

  sgemm_warptiling_v2<Traits>
      <<<grid, block, Traits::kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v2

namespace v3 {

template <typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads) void sgemm_warptiling_v3(
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

  float accum[Traits::kNumAccumulators] = {0.0f};
  float tb_frag_a[Traits::kThreadALoadIterations];
  float tb_frag_b[Traits::kThreadBLoadIterations];

  int const num_tiles = K / Traits::kThreadblockK;
  if (num_tiles == 0) {
    return;
  }

  v2::load_a_thread_fragment<Traits>(tb_frag_a, a_ptr, K, 0);
  v2::load_b_thread_fragment<Traits>(tb_frag_b, b_ptr, N, 0);

  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    int const tile_k = tile_idx * Traits::kThreadblockK;

    v2::store_a_thread_fragment_to_smem<Traits>(smem_a, tb_frag_a);
    v2::store_b_thread_fragment_to_smem<Traits>(smem_b, tb_frag_b);

    __syncthreads();

    v1::warp_mma<Traits>(accum, smem_a, smem_b, warp_m, warp_n, thread_m,
                         thread_n);

    if (tile_idx + 1 < num_tiles) {
      int const next_tile_k = tile_k + Traits::kThreadblockK;
      v2::load_a_thread_fragment<Traits>(tb_frag_a, a_ptr, K, next_tile_k);
      v2::load_b_thread_fragment<Traits>(tb_frag_b, b_ptr, N, next_tile_k);
    }

    __syncthreads();
  }

  v1::store_accum_to_gmem<Traits>(c_ptr, accum, warp_m, warp_n, thread_m,
                                  thread_n, N);
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  using Traits =
      detail::KernelTraits<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K>;

  dim3 const grid(utils::ceil_div(M, BLOCK_M), utils::ceil_div(N, BLOCK_N));
  dim3 const block(Traits::kNumThreads);

  sgemm_warptiling_v3<Traits>
      <<<grid, block, Traits::kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v3

namespace v4 {

template <typename Traits>
__device__ __forceinline__ void load_a_thread_fragment_vec4(
    float4 (&tb_frag_b)[Traits::kThreadBLoadIterations / 4], float const* a_ptr,
    int lda, int tile_k) {
  constexpr int vec_width = 4;
  static_assert(Traits::kThreadblockK % vec_width == 0);
  static_assert(Traits::kThreadALoadIterations % vec_width == 0);

  constexpr int vec_tb_k = Traits::kThreadblockK / vec_width;
  constexpr int frag_vec_iters = Traits::kThreadALoadIterations / vec_width;
  constexpr int row_advance_per_iter = Traits::kNumThreads / vec_tb_k;
  constexpr int col_advance_per_iter = Traits::kNumThreads % vec_tb_k;

  int const vec_lda = lda / vec_width;
  int const vec_advance_offset =
      row_advance_per_iter * vec_lda + col_advance_per_iter;
  int const row = threadIdx.x / vec_tb_k;
  int const col = threadIdx.x % vec_tb_k;
  int vec_offset = row * vec_lda + (tile_k / vec_width) + col;

  float4 const* vec_a_ptr = reinterpret_cast<float4 const*>(a_ptr);
#pragma unroll
  for (int iter = 0; iter < frag_vec_iters; ++iter) {
    tb_frag_b[iter] = vec_a_ptr[vec_offset];
    vec_offset += vec_advance_offset;
  }
}

template <typename Traits>
__device__ __forceinline__ void load_b_thread_fragment_vec4(
    float4 (&tb_frag_b)[Traits::kThreadBLoadIterations / 4], float const* b_ptr,
    int ldb, int tile_k) {
  constexpr int vec_width = 4;
  static_assert(Traits::kThreadblockN % vec_width == 0);
  static_assert(Traits::kThreadBLoadIterations % vec_width == 0);

  constexpr int vec_tb_n = Traits::kThreadblockN / vec_width;
  constexpr int frag_vec_iters = Traits::kThreadBLoadIterations / vec_width;
  constexpr int row_advance_per_iter = Traits::kNumThreads / vec_tb_n;
  constexpr int col_advance_per_iter = Traits::kNumThreads % vec_tb_n;

  int const vec_ldb = ldb / vec_width;
  int const vec_advance_offset =
      row_advance_per_iter * vec_ldb + col_advance_per_iter;
  int const row = threadIdx.x / vec_tb_n;
  int const col = threadIdx.x % vec_tb_n;
  int vec_offset = (tile_k + row) * vec_ldb + col;

  float4 const* vec_b_ptr = reinterpret_cast<float4 const*>(b_ptr);
#pragma unroll
  for (int iter = 0; iter < Traits::kThreadBLoadIterations / vec_width;
       ++iter) {
    tb_frag_b[iter] = vec_b_ptr[vec_offset];
    vec_offset += vec_advance_offset;
  }
}

template <typename Traits>
__device__ __forceinline__ void store_a_thread_fragment_to_smem_vec4(
    float* smem_a,
    float4 const (&tb_frag_a)[Traits::kThreadALoadIterations / 4]) {
  constexpr int vec_width = 4;
  static_assert(Traits::kThreadblockK % vec_width == 0);
  static_assert(Traits::kThreadALoadIterations % vec_width == 0);

  constexpr int vec_tb_k = Traits::kThreadblockK / vec_width;

#pragma unroll
  for (int iter = 0; iter < Traits::kThreadALoadIterations / vec_width;
       ++iter) {
    int const vec_elem_idx = threadIdx.x + iter * Traits::kNumThreads;
    int const row = vec_elem_idx / vec_tb_k;
    int const col = (vec_elem_idx % vec_tb_k) * vec_width;

    float4 const frag = tb_frag_a[iter];

    smem_a[(col + 0) * Traits::kSmemStrideA + row] = frag.x;
    smem_a[(col + 1) * Traits::kSmemStrideA + row] = frag.y;
    smem_a[(col + 2) * Traits::kSmemStrideA + row] = frag.z;
    smem_a[(col + 3) * Traits::kSmemStrideA + row] = frag.w;
  }
}

template <typename Traits>
__device__ __forceinline__ void store_b_thread_fragment_to_smem_vec4(
    float* smem_b,
    float4 const (&tb_frag_b)[Traits::kThreadBLoadIterations / 4]) {
  constexpr int vec_width = 4;
  static_assert(Traits::kThreadblockN % vec_width == 0);
  static_assert(Traits::kThreadBLoadIterations % vec_width == 0);

  constexpr int vec_tb_n = Traits::kThreadblockN / vec_width;
  float4* vec_smem_b = reinterpret_cast<float4*>(smem_b);

#pragma unroll
  for (int iter = 0; iter < Traits::kThreadBLoadIterations / vec_width;
       ++iter) {
    int const vec_elem_idx = threadIdx.x + iter * Traits::kNumThreads;
    int const row = vec_elem_idx / vec_tb_n;
    int const col = vec_elem_idx % vec_tb_n;

    vec_smem_b[row * (Traits::kSmemStrideB / vec_width) + col] =
        tb_frag_b[iter];
  }
}

template <typename Traits>
__device__ __forceinline__ void store_accum_to_gmem_vec4(
    float* c_ptr, float const (&accum)[Traits::kNumAccumulators], int warp_m,
    int warp_n, int thread_m, int thread_n, int ldc) {
  constexpr int vec_width = 4;
  constexpr int num_row_groups = Traits::kThreadTileM / vec_width;
  constexpr int num_col_groups = Traits::kThreadTileN / vec_width;
  constexpr int row_group_stride = Traits::kWarpThreadsM * vec_width;
  constexpr int col_group_stride = Traits::kWarpThreadsN * vec_width;
  static_assert(Traits::kLaneMmaShape == vec_width);
  static_assert(Traits::kThreadTileM % vec_width == 0);
  static_assert(Traits::kThreadTileN % vec_width == 0);

  int const row_base = warp_m * Traits::kWarpM + thread_m * vec_width;
  int const col_base = warp_n * Traits::kWarpN + thread_n * vec_width;

#pragma unroll
  for (int row_group = 0; row_group < num_row_groups; ++row_group) {
    int const row_group_base = row_base + row_group * row_group_stride;

#pragma unroll
    for (int col_group = 0; col_group < num_col_groups; ++col_group) {
      int const col_group_base = col_base + col_group * col_group_stride;

#pragma unroll
      for (int m = 0; m < vec_width; ++m) {
        int const accum_row = row_group * vec_width + m;
        int const accum_col = col_group * vec_width;

        float4 out = utils::make_float4_from_array(
            &accum[accum_row * Traits::kThreadTileN + accum_col]);

        utils::store_float4(c_ptr + (row_group_base + m) * ldc + col_group_base,
                            out);
      }
    }
  }
}

template <typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads) void sgemm_warptiling_v4(
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

  float accum[Traits::kNumAccumulators] = {0.0f};
  constexpr int vec_width = 4;
  float4 tb_frag_a[Traits::kThreadALoadIterations / vec_width];
  float4 tb_frag_b[Traits::kThreadBLoadIterations / vec_width];

  int const num_tiles = K / Traits::kThreadblockK;
  if (num_tiles == 0) {
    return;
  }

  load_a_thread_fragment_vec4<Traits>(tb_frag_a, a_ptr, K, 0);
  load_b_thread_fragment_vec4<Traits>(tb_frag_b, b_ptr, N, 0);

  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    int const tile_k = tile_idx * Traits::kThreadblockK;

    store_a_thread_fragment_to_smem_vec4<Traits>(smem_a, tb_frag_a);
    store_b_thread_fragment_to_smem_vec4<Traits>(smem_b, tb_frag_b);

    __syncthreads();

    v1::warp_mma<Traits>(accum, smem_a, smem_b, warp_m, warp_n, thread_m,
                         thread_n);

    if (tile_idx + 1 < num_tiles) {
      int const next_tile_k = tile_k + Traits::kThreadblockK;
      load_a_thread_fragment_vec4<Traits>(tb_frag_a, a_ptr, K, next_tile_k);
      load_b_thread_fragment_vec4<Traits>(tb_frag_b, b_ptr, N, next_tile_k);
    }

    __syncthreads();
  }

  v1::store_accum_to_gmem<Traits>(c_ptr, accum, warp_m, warp_n, thread_m,
                                  thread_n, N);
  // store_accum_to_gmem_vec4<Traits>(c_ptr, accum, warp_m, warp_n, thread_m,
  //                                  thread_n, N);
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  using Traits =
      detail::KernelTraits<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K>;

  dim3 const grid(utils::ceil_div(M, BLOCK_M), utils::ceil_div(N, BLOCK_N));
  dim3 const block(Traits::kNumThreads);

  sgemm_warptiling_v4<Traits>
      <<<grid, block, Traits::kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v4

namespace v5 {

template <typename Traits>
__device__ __forceinline__ void store_accum_to_epilogue_smem(
    float* smem_epilogue, float const (&accum)[Traits::kNumAccumulators],
    int warp_m, int warp_n, int thread_m, int thread_n) {
  constexpr int vec_width = Traits::kEpilogueElementsPerAccess;
  constexpr int num_row_groups = Traits::kThreadTileM / vec_width;
  constexpr int num_col_groups = Traits::kThreadTileN / vec_width;
  constexpr int row_group_stride = Traits::kWarpThreadsM * vec_width;
  constexpr int col_group_stride = Traits::kWarpThreadsN * vec_width;

  static_assert(Traits::kLaneMmaShape == vec_width);
  static_assert(Traits::kThreadTileM % vec_width == 0);
  static_assert(Traits::kThreadTileN % vec_width == 0);

  int const row_base = warp_m * Traits::kWarpM + thread_m * vec_width;
  int const col_base = warp_n * Traits::kWarpN + thread_n * vec_width;

#pragma unroll
  for (int row_group = 0; row_group < num_row_groups; ++row_group) {
    int const row_group_base = row_base + row_group * row_group_stride;

#pragma unroll
    for (int col_group = 0; col_group < num_col_groups; ++col_group) {
      int const col_group_base = col_base + col_group * col_group_stride;

#pragma unroll
      for (int m = 0; m < vec_width; ++m) {
        int const accum_row = row_group * vec_width + m;
        int const accum_col = col_group * vec_width;
        float4 const fragment = utils::make_float4_from_array(
            &accum[accum_row * Traits::kThreadTileN + accum_col]);
        float* dst = smem_epilogue +
                     (row_group_base + m) * Traits::kEpilogueSmemStride +
                     col_group_base;

        utils::store_float4(dst, fragment);
      }
    }
  }
}

template <typename Traits>
__device__ __forceinline__ void store_epilogue_smem_to_gmem(
    float* c_ptr, float const* smem_epilogue, int ldc) {
  constexpr int vec_width = Traits::kEpilogueElementsPerAccess;

#pragma unroll
  for (int iter = 0; iter < Traits::kEpilogueStoresPerThread; ++iter) {
    int const vec_idx = threadIdx.x + iter * Traits::kNumThreads;
    int const row = vec_idx / Traits::kEpilogueVectorsPerRow;
    int const vec_col = vec_idx % Traits::kEpilogueVectorsPerRow;
    int const col = vec_col * vec_width;
    float4 const value = utils::load_float4(
        smem_epilogue + row * Traits::kEpilogueSmemStride + col);

    utils::store_float4(c_ptr + row * ldc + col, value);
  }
}

// template <typename Traits>
// __device__ __forceinline__ void store_epilogue_smem_to_gmem(
//     float* c_ptr, float const* smem_epilogue, int ldc) {
//   constexpr int vec_width = Traits::kEpilogueElementsPerAccess;
//   constexpr int row_advance_per_iter =
//       Traits::kNumThreads / Traits::kEpilogueVectorsPerRow;
//   constexpr int vec_col_advance_per_iter =
//       Traits::kNumThreads % Traits::kEpilogueVectorsPerRow;

//   int const row = threadIdx.x / Traits::kEpilogueVectorsPerRow;
//   int const vec_col = threadIdx.x % Traits::kEpilogueVectorsPerRow;
//   int const col = vec_col * vec_width;

//   int smem_offset = row * Traits::kEpilogueSmemStride + col;
//   int gmem_offset = row * ldc + col;
//   constexpr int smem_advance =
//       row_advance_per_iter * Traits::kEpilogueSmemStride +
//       vec_col_advance_per_iter * vec_width;
//   int const gmem_advance =
//       row_advance_per_iter * ldc + vec_col_advance_per_iter * vec_width;

// #pragma unroll
//   for (int iter = 0; iter < Traits::kEpilogueStoresPerThread; ++iter) {
//     float4 const value = utils::load_float4(smem_epilogue + smem_offset);

//     utils::store_float4(c_ptr + gmem_offset, value);
//     smem_offset += smem_advance;
//     gmem_offset += gmem_advance;
//   }
// }

template <typename Traits>
__device__ __forceinline__ void epilogue(
    float* c_ptr, float* smem_epilogue,
    float const (&accum)[Traits::kNumAccumulators], int warp_m, int warp_n,
    int thread_m, int thread_n, int ldc) {
  store_accum_to_epilogue_smem<Traits>(smem_epilogue, accum, warp_m, warp_n,
                                       thread_m, thread_n);
  __syncthreads();
  store_epilogue_smem_to_gmem<Traits>(c_ptr, smem_epilogue, ldc);
}

template <typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads) void sgemm_warptiling_v5(
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
  float* smem_epilogue = smem;

  float accum[Traits::kNumAccumulators] = {0.0f};
  constexpr int vec_width = 4;
  float4 tb_frag_a[Traits::kThreadALoadIterations / vec_width];
  float4 tb_frag_b[Traits::kThreadBLoadIterations / vec_width];

  int const num_tiles = K / Traits::kThreadblockK;
  if (num_tiles == 0) {
    return;
  }

  v4::load_a_thread_fragment_vec4<Traits>(tb_frag_a, a_ptr, K, 0);
  v4::load_b_thread_fragment_vec4<Traits>(tb_frag_b, b_ptr, N, 0);

  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    int const tile_k = tile_idx * Traits::kThreadblockK;

    v4::store_a_thread_fragment_to_smem_vec4<Traits>(smem_a, tb_frag_a);
    v4::store_b_thread_fragment_to_smem_vec4<Traits>(smem_b, tb_frag_b);

    __syncthreads();

    v1::warp_mma<Traits>(accum, smem_a, smem_b, warp_m, warp_n, thread_m,
                         thread_n);

    if (tile_idx + 1 < num_tiles) {
      int const next_tile_k = tile_k + Traits::kThreadblockK;
      v4::load_a_thread_fragment_vec4<Traits>(tb_frag_a, a_ptr, K, next_tile_k);
      v4::load_b_thread_fragment_vec4<Traits>(tb_frag_b, b_ptr, N, next_tile_k);
    }

    __syncthreads();
  }

  epilogue<Traits>(c_ptr, smem_epilogue, accum, warp_m, warp_n, thread_m,
                   thread_n, N);
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  using Traits =
      detail::KernelTraits<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K>;

  dim3 const grid(utils::ceil_div(M, BLOCK_M), utils::ceil_div(N, BLOCK_N));
  dim3 const block(Traits::kNumThreads);

  if (Traits::kV5SmemBytes > 48 * 1024) {
    (void)cudaFuncSetAttribute(sgemm_warptiling_v5<Traits>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               Traits::kV5SmemBytes);
  }

  sgemm_warptiling_v5<Traits>
      <<<grid, block, Traits::kV5SmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v5

}  // namespace warptiling
}  // namespace sgemm
