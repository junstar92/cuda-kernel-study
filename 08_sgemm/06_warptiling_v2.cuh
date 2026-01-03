#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

namespace sgemm {
namespace warptiling {
namespace v2 {

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM,
          int WarpN, int WarpK, int Stages>
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

  // RowMajorInterleaved<2>-style lane mapping. Two neighboring lane ids are
  // paired along M before moving in N, so a logical thread tile is later
  // visited as serveral 4x4 islands.
  static constexpr int kLaneLayout =
      (kThreadTileM > 4 && kThreadTileN > 4) ? 2 : 1;
  static constexpr int kLaneMmaM = (kThreadTileM < 4) ? kThreadTileM : 4;
  static constexpr int kLaneMmaN = (kThreadTileN < 4) ? kThreadTileN : 4;
  static constexpr int kLaneStride = kWarpThreadsN * kLaneLayout;

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

  static constexpr int kStages = Stages;
  static_assert(kStages == 2);

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
  static constexpr int kSmemBytes = kStages * kSmemElements * sizeof(float);

  using SmemThreadMapA = utils::TransposePitchLinearThreadMapSimt<ThreadMapA>;
  using SmemThreadMapB =
      utils::PitchLinearStripminedThreadMap<kThreadblockN, kThreadblockK,
                                            kThreads>;

  static constexpr int kAccumulatorElements = kThreadTileM * kThreadTileN;
  static constexpr int kAccumulatorAccessElements = kLaneMmaN;
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
  using WarpTileIteratorA =
      utils::WarpTileIteratorA<kWarpM, kWarpThreadsM, kThreadTileM, kLaneMmaM,
                               kSmemStrideA, kSmemStageElemsA>;
  using WarpTileIteratorB =
      utils::WarpTileIteratorB<kWarpN, kWarpThreadsN, kThreadTileN, kLaneMmaN,
                               kSmemStrideB, kSmemStageElemsB>;

  static_assert((kThreadblockM * kThreadblockK) % kThreads == 0,
                "A tile must be evenly covered by per-thread scalar loads.");
  static_assert((kThreadblockK * kThreadblockN) % kThreads == 0,
                "B tile must be evenly covered by per-thread scalar loads.");
  static_assert(kThreadTileM % kLaneMmaM == 0,
                "Thread tile M must be divisible by lane MMA M.");
  static_assert(kThreadTileN % kLaneMmaN == 0,
                "Thread tile N must be divisible by lane MMA N.");
};

template <typename Traits>
__DEVICE__ void store_fragments_to_stage(
    float* smem_a, float* smem_b, int const stage, int const tid,
    typename Traits::ThreadblockTileIteratorA::Fragment& frag_a,
    typename Traits::ThreadblockTileIteratorB::Fragment& frag_b) {
  auto smem_tile_iterator_a = typename Traits::SmemTileIteratorA(
      smem_a + stage * Traits::kSmemStageElemsA, Traits::kSmemStrideA, tid);
  smem_tile_iterator_a.store(frag_a);

  auto smem_tile_iterator_b = typename Traits::SmemTileIteratorB(
      smem_b + stage * Traits::kSmemStageElemsB, Traits::kThreadblockN, tid);
  smem_tile_iterator_b.store(frag_b);
}

template <typename Traits>
__DEVICE__ void warp_mma(
    float (&accum)[Traits::kAccumulatorElements],
    typename Traits::WarpTileIteratorA::Fragment const& warp_frag_a,
    typename Traits::WarpTileIteratorB::Fragment const& warp_frag_b) {
#pragma unroll
  for (int m = 0; m < Traits::kThreadTileM; ++m) {
    for (int n = 0; n < Traits::kThreadTileN; ++n) {
      accum[m * Traits::kThreadTileN + n] += warp_frag_a[m] * warp_frag_b[n];
    }
  }
}

template <typename Traits>
__DEVICE__ void store_accum_to_global(
    float* gmem_ptr, float const (&accum)[Traits::kAccumulatorElements],
    int const warp_tile_m_idx, int const warp_tile_n_idx, int const lane_m_idx,
    int const lane_n_idx, int const dst_stride) {
  constexpr int kIterationsRow = Traits::kThreadTileM / Traits::kLaneMmaM;
  constexpr int kIterationsColumn = Traits::kThreadTileN / Traits::kLaneMmaN;

  constexpr int kRowAdvance = Traits::kWarpThreadsM * Traits::kLaneMmaM;
  constexpr int kColumnAdvance = Traits::kWarpThreadsN * Traits::kLaneMmaN;

  int row_base =
      warp_tile_m_idx * Traits::kWarpM + lane_m_idx * Traits::kLaneMmaM;
  int col_base =
      warp_tile_n_idx * Traits::kWarpN + lane_n_idx * Traits::kLaneMmaN;

#pragma unroll
  for (int row = 0; row < kIterationsRow; ++row) {
#pragma unroll
    for (int column = 0; column < kIterationsColumn; ++column) {
#pragma unroll
      for (int m = 0; m < Traits::kLaneMmaM; ++m) {
        float* dst = gmem_ptr +
                     (row_base + row * kRowAdvance + m) * dst_stride +
                     (col_base + column * kColumnAdvance);

#pragma unorll
        for (int n = 0; n < Traits::kLaneMmaN; ++n) {
          int accum_m = row * Traits::kLaneMmaM + m;
          int accum_n = column * Traits::kLaneMmaN + n;
          dst[n] = accum[accum_m * Traits::kThreadTileN + accum_n];
        }
      }
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

  int const row_major = lane_id / Traits::kLaneStride;
  int const residual = lane_id - row_major * Traits::kLaneStride;
  int const lane_n_idx = residual / Traits::kLaneLayout;
  int const row_minor = residual - lane_n_idx * Traits::kLaneLayout;
  int const lane_m_idx = row_major * Traits::kLaneLayout + row_minor;

  c_ptr += cta_row * N + cta_col;

  extern __shared__ float smem[];
  float* smem_a = smem;
  float* smem_b = smem + Traits::kStages * Traits::kSmemStageElemsA;

  auto threadblock_tile_iterator_a =
      typename Traits::ThreadblockTileIteratorA(a_ptr, K, tid);
  auto threadblock_tile_iterator_b =
      typename Traits::ThreadblockTileIteratorB(b_ptr, N, tid);
  threadblock_tile_iterator_a.add_tile_offset(cta_m_idx, 0);
  threadblock_tile_iterator_b.add_tile_offset(0, cta_n_idx);

  float accum[Traits::kAccumulatorElements] = {0.f};
  typename Traits::ThreadblockTileIteratorA::Fragment frag_a;
  typename Traits::ThreadblockTileIteratorB::Fragment frag_b;

  int gemm_k_iterations = K / Traits::kThreadblockK;

  threadblock_tile_iterator_a.load(frag_a);
  threadblock_tile_iterator_b.load(frag_b);
  ++threadblock_tile_iterator_a;
  ++threadblock_tile_iterator_b;

  store_fragments_to_stage<Traits>(smem_a, smem_b, 0, tid, frag_a, frag_b);

  __syncthreads();

  int read_stage = 0;
#pragma unroll
  for (; gemm_k_iterations > 0; --gemm_k_iterations) {
    bool const has_next = gemm_k_iterations > 1;
    if (has_next) {
      threadblock_tile_iterator_a.load(frag_a);
      threadblock_tile_iterator_b.load(frag_b);
      ++threadblock_tile_iterator_a;
      ++threadblock_tile_iterator_b;
    }

    auto warp_tile_iterator_a = typename Traits::WarpTileIteratorA(
        smem_a, read_stage, warp_tile_m_idx, lane_m_idx);
    auto warp_tile_iterator_b = typename Traits::WarpTileIteratorB(
        smem_b, read_stage, warp_tile_n_idx, lane_n_idx);
    typename Traits::WarpTileIteratorA::Fragment warp_frag_a;
    typename Traits::WarpTileIteratorB::Fragment warp_frag_b;

#pragma unroll
    for (int warp_mma_k = 0; warp_mma_k < Traits::kWarpGemmIterations;
         ++warp_mma_k) {
      warp_tile_iterator_a.load(warp_frag_a);
      warp_tile_iterator_b.load(warp_frag_b);

      warp_mma<Traits>(accum, warp_frag_a, warp_frag_b);

      ++warp_tile_iterator_a;
      ++warp_tile_iterator_b;
    }

    int const write_stage = read_stage ^ 1;
    if (has_next) {
      store_fragments_to_stage<Traits>(smem_a, smem_b, write_stage, tid, frag_a,
                                       frag_b);
    }

    __syncthreads();

    read_stage = write_stage;
  }

  // store accumulated results back to global memory
  store_accum_to_global<Traits>(c_ptr, accum, warp_tile_m_idx, warp_tile_n_idx,
                                lane_m_idx, lane_n_idx, N);
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM,
          int WarpN, int WarpK>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  using Traits = KernelTraits<ThreadblockM, ThreadblockN, ThreadblockK, WarpM,
                              WarpN, WarpK, 2>;

  dim3 const block(Traits::kThreads);
  dim3 const grid(utils::ceil_div(N, ThreadblockN),
                  utils::ceil_div(M, ThreadblockM));

  sgemm_warptiling<Traits>
      <<<grid, block, Traits::kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v2
}  // namespace warptiling
}  // namespace sgemm
