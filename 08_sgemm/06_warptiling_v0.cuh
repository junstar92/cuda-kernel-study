#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

namespace sgemm {
namespace warptiling {
namespace v0 {

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM,
          int WarpN, int WarpK>
struct KernelTraits {
  // Tile sizes at each hierarchy level:
  static constexpr int kThreadblockM = ThreadblockM;
  static constexpr int kThreadblockN = ThreadblockN;
  static constexpr int kThreadblockK = ThreadblockK;
  static constexpr int kWarpM = WarpM;
  static constexpr int kWarpN = WarpN;
  static constexpr int kWarpK = WarpK;

  // How many warp tiles fit inside one CTA tile, and therefore how many
  // hardware threads are needed for one block.
  static constexpr int kWarpCountM = kThreadblockM / kWarpM;
  static constexpr int kWarpCountN = kThreadblockN / kWarpN;
  static constexpr int kWarpCount = kWarpCountM * kWarpCountN;
  static constexpr int kWarpSize = 32;
  static constexpr int kThreads = kWarpCount * kWarpSize;

  // Lay out the 32 lanes of a warp as a kWarpThreadsM x kWarpThreadsN grid.
  // The split is chosen so each thread gets a near-square micro-tile inside the
  // warp tile.
  static constexpr int kWarpThreadsM = (kWarpM > kWarpN) ? 8 : 4;
  static constexpr int kWarpThreadsN = kWarpSize / kWarpThreadsM;
  static constexpr int kThreadTileM = kWarpM / kWarpThreadsM;
  static constexpr int kThreadTileN = kWarpN / kWarpThreadsN;

  static constexpr int kWarpGemmIterations = kThreadblockK;

  using ThreadMapA =
      utils::PitchLinearStripminedThreadMap<kThreadblockK, kThreadblockM,
                                            kThreads>;
  using ThreadMapB =
      utils::PitchLinearStripminedThreadMap<kThreadblockN, kThreadblockK,
                                            kThreads>;

  static constexpr int kThreadblockALoadsPerThread =
      (kThreadblockM * kThreadblockK) / kThreads;
  static constexpr int kThreadblockBLoadsPerThread =
      (kThreadblockK * kThreadblockN) / kThreads;

  // Shared-memory staging:
  // A is stored transposed and padded so warp reads along M avoid bank
  // conflicts. B is kept row-major because the warp already consumes it along
  // N.
  static constexpr int kSmemPadA = 32 / kThreadblockK;
  static constexpr int kSmemPadB = 0;
  static constexpr int kSmemStrideA = kThreadblockM + kSmemPadA;
  static constexpr int kSmemStrideB = kThreadblockN + kSmemPadB;
  static constexpr int kSmemStageElemsA = kThreadblockK * kSmemStrideA;
  static constexpr int kSmemStageElemsB = kThreadblockK * kSmemStrideB;
  static constexpr int kSmemElements = kSmemStageElemsA + kSmemStageElemsB;
  static constexpr int kSmemBytes = kSmemElements * sizeof(float);

  using SmemThreadMapA = utils::TransposePitchLinearThreadMapSimt<ThreadMapA>;
  using SmemThreadMapB =
      utils::PitchLinearStripminedThreadMap<kThreadblockN, kThreadblockK,
                                            kThreads>;

  static constexpr int kAccumulatorElements = kThreadTileM * kThreadTileN;
  static constexpr int kAccumulatorAccessElements = kThreadTileN;
  static constexpr int kAccumulatorAccessesPerIter =
      kThreadTileN / kAccumulatorAccessElements;

  using ThreadblockTileIteratorA =
      utils::ThreadblockTileIterator<kThreadblockK, kThreadblockM, ThreadMapA,
                                     0>;
  using ThreadblockTileIteratorB =
      utils::ThreadblockTileIterator<kThreadblockN, kThreadblockK, ThreadMapB,
                                     1>;
  using SmemTileIteratorA =
      utils::SmemTileIterator<kSmemStrideA, kThreadblockK, SmemThreadMapA, 1>;
  using SmemTileIteratorB =
      utils::SmemTileIterator<kThreadblockN, kThreadblockK, SmemThreadMapB, 1>;

  static_assert((kThreadblockM * kThreadblockK) % kThreads == 0,
                "A tile must be evenly covered by per-thread scalar loads.");
  static_assert((kThreadblockK * kThreadblockN) % kThreads == 0,
                "B tile must be evenly covered by per-thread scalar loads.");
};

template <typename Traits>
__DEVICE__ void warp_mma(float (&accum)[Traits::kAccumulatorElements],
                         float const* smem_a_ptr, float const* smem_b_ptr,
                         int const warp_tile_m_idx, int const warp_tile_n_idx,
                         int const thread_tile_m_idx,
                         int const thread_tile_n_idx) {
  float const* warp_a_ptr = smem_a_ptr + warp_tile_m_idx * Traits::kWarpM +
                            thread_tile_m_idx * Traits::kThreadTileM;
  float const* warp_b_ptr = smem_b_ptr + warp_tile_n_idx * Traits::kWarpN +
                            thread_tile_n_idx * Traits::kThreadTileN;

  float warp_frag_a[Traits::kThreadTileM] = {0.f};
  float warp_frag_b[Traits::kThreadTileN] = {0.f};
#pragma unroll
  for (int k = 0; k < Traits::kWarpGemmIterations; ++k) {
#pragma unroll
    for (int m = 0; m < Traits::kThreadTileM; ++m) {
      warp_frag_a[m] = warp_a_ptr[m];
    }
#pragma unroll
    for (int n = 0; n < Traits::kThreadTileN; ++n) {
      warp_frag_b[n] = warp_b_ptr[n];
    }

#pragma unroll
    for (int m = 0; m < Traits::kThreadTileM; ++m) {
      for (int n = 0; n < Traits::kThreadTileN; ++n) {
        accum[m * Traits::kThreadTileN + n] += warp_frag_a[m] * warp_frag_b[n];
      }
    }

    warp_a_ptr += Traits::kSmemStrideA;
    warp_b_ptr += Traits::kSmemStrideB;
  }
}

template <typename Traits>
__DEVICE__ void store_accum_to_global(
    float* gmem_ptr, float const (&accum)[Traits::kAccumulatorElements],
    int const warp_tile_m_idx, int const warp_tile_n_idx,
    int const thread_tile_m_idx, int const thread_tile_n_idx,
    int const dst_stride) {
  int const row_base = warp_tile_m_idx * Traits::kWarpM +
                       thread_tile_m_idx * Traits::kThreadTileM;
  int const col_base = warp_tile_n_idx * Traits::kWarpN +
                       thread_tile_n_idx * Traits::kThreadTileN;
#pragma unroll
  for (int m = 0; m < Traits::kThreadTileM; ++m) {
    float* dst = gmem_ptr + (row_base + m) * dst_stride + col_base;
#pragma unroll
    for (int n = 0; n < Traits::kThreadTileN; ++n) {
      dst[n] = accum[m * Traits::kThreadTileN + n];
    }
  }
}

template <typename Traits>
__global__ void sgemm_warptiling(float* __restrict__ c_ptr,
                                 float const* __restrict__ a_ptr,
                                 float const* __restrict__ b_ptr, int const M,
                                 int const N, int const K) {
  int const cta_m_idx = blockIdx.y;
  int const cta_n_idx = blockIdx.x;
  int const cta_row = cta_m_idx * Traits::kThreadblockM;
  int const cta_col = cta_n_idx * Traits::kThreadblockN;
  // thread mapping inside the block tile
  int const tid = threadIdx.x;
  int const warp_id = tid / 32;
  int const lane_id = tid % 32;

  int const warp_tile_m_idx = warp_id % Traits::kWarpCountM;
  int const warp_tile_n_idx = warp_id / Traits::kWarpCountM;
  int const thread_tile_m_idx = lane_id / Traits::kWarpThreadsN;
  int const thread_tile_n_idx = lane_id % Traits::kWarpThreadsN;

  c_ptr += cta_row * N + cta_col;

  extern __shared__ float smem[];
  float* smem_a = smem;
  float* smem_b = smem + Traits::kSmemStageElemsA;

  auto threadblock_tile_iterator_a =
      typename Traits::ThreadblockTileIteratorA(a_ptr, K, tid);
  auto threadblock_tile_iterator_b =
      typename Traits::ThreadblockTileIteratorB(b_ptr, N, tid);
  threadblock_tile_iterator_a.add_tile_offset(cta_m_idx, 0);
  threadblock_tile_iterator_b.add_tile_offset(0, cta_n_idx);

  auto smem_tile_iterator_a =
      typename Traits::SmemTileIteratorA(smem_a, Traits::kSmemStrideA, tid);
  auto smem_tile_iterator_b =
      typename Traits::SmemTileIteratorB(smem_b, Traits::kThreadblockN, tid);

  float accum[Traits::kAccumulatorElements] = {0.f};
  typename Traits::ThreadblockTileIteratorA::Fragment frag_a;
  typename Traits::ThreadblockTileIteratorB::Fragment frag_b;

  int gemm_k_iterations = K / Traits::kThreadblockK;

#pragma unroll
  for (; gemm_k_iterations > 0; --gemm_k_iterations) {
    threadblock_tile_iterator_a.load(frag_a);
    threadblock_tile_iterator_b.load(frag_b);

    ++threadblock_tile_iterator_a;
    ++threadblock_tile_iterator_b;

    smem_tile_iterator_a.store(frag_a);
    smem_tile_iterator_b.store(frag_b);

    __syncthreads();

    warp_mma<Traits>(accum, smem_a, smem_b, warp_tile_m_idx, warp_tile_n_idx,
                     thread_tile_m_idx, thread_tile_n_idx);

    __syncthreads();
  }

  // store accumulated results back to global memory
  store_accum_to_global<Traits>(c_ptr, accum, warp_tile_m_idx, warp_tile_n_idx,
                                thread_tile_m_idx, thread_tile_n_idx, N);
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM,
          int WarpN, int WarpK>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  using Traits = KernelTraits<ThreadblockM, ThreadblockN, ThreadblockK, WarpM,
                              WarpN, WarpK>;

  dim3 const block(Traits::kThreads);
  dim3 const grid(utils::ceil_div(N, ThreadblockN),
                  utils::ceil_div(M, ThreadblockM));

  sgemm_warptiling<Traits>
      <<<grid, block, Traits::kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v0
}  // namespace warptiling
}  // namespace sgemm
