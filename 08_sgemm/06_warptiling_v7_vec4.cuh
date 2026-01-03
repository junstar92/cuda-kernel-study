#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

namespace sgemm {
namespace warptiling {
namespace v7 {
namespace vec4 {

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM,
          int WarpN, int WarpK, int Stages>
struct KernelTraits {
  static constexpr int kVecWidth = 4;

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
                                            kThreads, kVecWidth>;
  using ThreadMapB =
      utils::PitchLinearStripminedThreadMap<kThreadblockN, kThreadblockK,
                                            kThreads, kVecWidth>;

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

  using SmemThreadMapA =
      utils::TransposePitchLinearThreadMapSimtVec4Store<ThreadMapA>;
  using SmemThreadMapB =
      utils::PitchLinearStripminedThreadMap<kThreadblockN, kThreadblockK,
                                            kThreads, kVecWidth>;

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
      utils::SmemTransposeVec4TileIterator<kSmemStrideA, kThreadblockK,
                                           SmemThreadMapA, 1>;
  using SmemTileIteratorB =
      utils::SmemTileIterator<kThreadblockN, kThreadblockK, SmemThreadMapB, 1>;
  using WarpTileIteratorA =
      utils::WarpTileIteratorA<kWarpM, kWarpThreadsM, kThreadTileM, kLaneMmaM,
                               kSmemStrideA, kSmemStageElemsA>;
  using WarpTileIteratorB =
      utils::WarpTileIteratorB<kWarpN, kWarpThreadsN, kThreadTileN, kLaneMmaN,
                               kSmemStrideB, kSmemStageElemsB>;

  static constexpr int kEpilogueRows = kWarpCountM * kWarpThreadsM;
  static constexpr int kEpilogueColumns = kWarpCountN * kWarpN;
  static constexpr int kEpiloguePad = 5;
  static constexpr int kEpilogueStride = kEpilogueColumns + kEpiloguePad;

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
__DEVICE__ void epilogue(float* c_ptr,
                         float const (&accum)[Traits::kAccumulatorElements],
                         float* smem, int const warp_tile_m_idx,
                         int const warp_tile_n_idx, int const lane_m_idx,
                         int const lane_n_idx, int tid, int ldc) {
  utils::AccumulatorFragmentIterator<Traits> accum_iter(accum);
  utils::WarpEpilogueStoreIterator<Traits> warp_store_iter(
      smem, warp_tile_m_idx, warp_tile_n_idx, lane_m_idx, lane_n_idx,
      Traits::kEpilogueStride);
  utils::EpilogueSharedLoadIteratorVec4<Traits> shared_load_iter(
      smem, tid, Traits::kEpilogueStride);
  utils::EpilogueOutputStoreIteratorVec4<Traits> out_iter(c_ptr, tid, ldc);

  typename utils::AccumulatorFragmentIterator<Traits>::Fragment accum_frag;
  typename utils::EpilogueOutputStoreIteratorVec4<Traits>::Fragment output_frag;

#pragma unroll
  for (int iter = 0; iter < Traits::kThreadTileM; ++iter) {
    accum_iter.load(accum_frag);
    ++accum_iter;

    warp_store_iter.store(accum_frag);

    __syncthreads();

    shared_load_iter.load(output_frag);

    out_iter.store(output_frag);
    __syncthreads();
    ++out_iter;
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

  auto smem_tile_iterator_a =
      typename Traits::SmemTileIteratorA(smem_a, Traits::kSmemStrideA, tid);
  auto smem_tile_iterator_b =
      typename Traits::SmemTileIteratorB(smem_b, Traits::kThreadblockN, tid);

  auto warp_tile_iterator_a = typename Traits::WarpTileIteratorA(
      smem_a, 0, warp_tile_m_idx, lane_m_idx);
  auto warp_tile_iterator_b = typename Traits::WarpTileIteratorB(
      smem_b, 0, warp_tile_n_idx, lane_n_idx);

  alignas(Traits::kLaneMmaN *
          sizeof(float)) float accum[Traits::kAccumulatorElements] = {0.f};
  typename Traits::ThreadblockTileIteratorA::Fragment tb_frag_a;
  typename Traits::ThreadblockTileIteratorB::Fragment tb_frag_b;
  typename Traits::WarpTileIteratorA::Fragment warp_frag_a[2];
  typename Traits::WarpTileIteratorB::Fragment warp_frag_b[2];

  int gemm_k_iterations = K / Traits::kThreadblockK;
  int smem_write_stage_idx = 0;

  // prologue
  threadblock_tile_iterator_a.load(tb_frag_a, true);
  threadblock_tile_iterator_b.load(tb_frag_b, true);
  ++threadblock_tile_iterator_a;
  ++threadblock_tile_iterator_b;

  smem_tile_iterator_a.store(tb_frag_a);
  smem_tile_iterator_b.store(tb_frag_b);

  // advance write stage
  ++smem_tile_iterator_a;
  ++smem_tile_iterator_b;
  smem_write_stage_idx ^= 1;

  __syncthreads();

  // load fragments from shared memory
  warp_tile_iterator_a.load(warp_frag_a[0]);
  warp_tile_iterator_b.load(warp_frag_b[0]);

  ++warp_tile_iterator_a;
  ++warp_tile_iterator_b;

#pragma unroll
  for (; gemm_k_iterations > 0; --gemm_k_iterations) {
#pragma unroll
    for (int warp_mma_k = 0; warp_mma_k < Traits::kWarpGemmIterations;
         ++warp_mma_k) {
      if (warp_mma_k == Traits::kWarpGemmIterations - 1) {
        // write fragments to shared memory
        smem_tile_iterator_a.store(tb_frag_a);
        smem_tile_iterator_b.store(tb_frag_b);

        __syncthreads();

        // advance smem read and write stages
        ++smem_tile_iterator_a;
        ++smem_tile_iterator_b;

        if (smem_write_stage_idx == 1) {
          smem_tile_iterator_a.add_stage_offset(-Traits::kStages);
          smem_tile_iterator_b.add_stage_offset(-Traits::kStages);
        } else {
          warp_tile_iterator_a.add_tile_offset(-Traits::kStages *
                                               Traits::kWarpGemmIterations);
          warp_tile_iterator_b.add_tile_offset(-Traits::kStages *
                                               Traits::kWarpGemmIterations);
        }

        smem_write_stage_idx ^= 1;
      }

      warp_tile_iterator_a.load(warp_frag_a[(warp_mma_k + 1) % 2]);
      warp_tile_iterator_b.load(warp_frag_b[(warp_mma_k + 1) % 2]);

      ++warp_tile_iterator_a;
      ++warp_tile_iterator_b;

      if (warp_mma_k == 0) {
        // load fragments from global
        threadblock_tile_iterator_a.load(tb_frag_a, gemm_k_iterations > 1);
        threadblock_tile_iterator_b.load(tb_frag_b, gemm_k_iterations > 1);

        ++threadblock_tile_iterator_a;
        ++threadblock_tile_iterator_b;
      }

      warp_mma<Traits>(accum, warp_frag_a[warp_mma_k % 2],
                       warp_frag_b[warp_mma_k % 2]);
    }
  }

  // store accumulated results back to global memory
  epilogue<Traits>(c_ptr, accum, smem, warp_tile_m_idx, warp_tile_n_idx,
                   lane_m_idx, lane_n_idx, tid, N);
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

}  // namespace vec4
}  // namespace v7
}  // namespace warptiling
}  // namespace sgemm
