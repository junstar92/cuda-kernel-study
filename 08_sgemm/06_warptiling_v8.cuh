#pragma once

#include <cuda_runtime.h>

#include "06_warptiling_v7.cuh"
#include "utils.cuh"

namespace sgemm {
namespace warptiling {
namespace v8 {

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

  alignas(Traits::kLaneMmaN *
          sizeof(float)) float accum[Traits::kAccumulatorElements] = {0.f};
  typename Traits::ThreadblockTileIteratorA::Fragment tb_frag_a;
  typename Traits::ThreadblockTileIteratorB::Fragment tb_frag_b;
  typename Traits::WarpTileIteratorA::Fragment warp_frag_a[2];
  typename Traits::WarpTileIteratorB::Fragment warp_frag_b[2];

  int gemm_k_iterations = K / Traits::kThreadblockK;
  int read_stage = 0;
  int write_stage = 1;

  auto make_smem_store_iterator_a = [&](int stage_idx) {
    return typename Traits::SmemTileIteratorA(
        smem_a + stage_idx * Traits::kSmemStageElemsA, Traits::kSmemStrideA,
        tid);
  };
  auto make_smem_store_iterator_b = [&](int stage_idx) {
    return typename Traits::SmemTileIteratorB(
        smem_b + stage_idx * Traits::kSmemStageElemsB, Traits::kThreadblockN,
        tid);
  };
  auto make_warp_tile_iterator_a = [&](int stage_idx) {
    return typename Traits::WarpTileIteratorA(
        smem_a + stage_idx * Traits::kSmemStageElemsA, 0, warp_tile_m_idx,
        lane_m_idx);
  };
  auto make_warp_tile_iterator_b = [&](int stage_idx) {
    return typename Traits::WarpTileIteratorB(
        smem_b + stage_idx * Traits::kSmemStageElemsB, 0, warp_tile_n_idx,
        lane_n_idx);
  };

  // prologue
  threadblock_tile_iterator_a.load(tb_frag_a, true);
  threadblock_tile_iterator_b.load(tb_frag_b, true);

  ++threadblock_tile_iterator_a;
  ++threadblock_tile_iterator_b;

  {
    auto smem_store_a = make_smem_store_iterator_a(read_stage);
    auto smem_store_b = make_smem_store_iterator_b(read_stage);

    smem_store_a.store(tb_frag_a);
    smem_store_b.store(tb_frag_b);
  }

  __syncthreads();

#pragma unroll
  for (; gemm_k_iterations > 0; --gemm_k_iterations) {
    bool const has_next = gemm_k_iterations > 1;

    auto warp_tile_iterator_a = make_warp_tile_iterator_a(read_stage);
    auto warp_tile_iterator_b = make_warp_tile_iterator_b(read_stage);

    warp_tile_iterator_a.load(warp_frag_a[0]);
    warp_tile_iterator_b.load(warp_frag_b[0]);

    ++warp_tile_iterator_a;
    ++warp_tile_iterator_b;

#pragma unroll
    for (int warp_mma_k = 0; warp_mma_k < Traits::kWarpGemmIterations;
         ++warp_mma_k) {
      warp_tile_iterator_a.load(warp_frag_a[(warp_mma_k + 1) % 2]);
      warp_tile_iterator_b.load(warp_frag_b[(warp_mma_k + 1) % 2]);

      ++warp_tile_iterator_a;
      ++warp_tile_iterator_b;

      if (warp_mma_k == 0) {
        threadblock_tile_iterator_a.load(tb_frag_a, has_next);
        threadblock_tile_iterator_b.load(tb_frag_b, has_next);

        ++threadblock_tile_iterator_a;
        ++threadblock_tile_iterator_b;
      }

      v7::warp_mma<Traits>(accum, warp_frag_a[warp_mma_k % 2],
                           warp_frag_b[warp_mma_k % 2]);
    }

    auto smem_store_a = make_smem_store_iterator_a(write_stage);
    auto smem_store_b = make_smem_store_iterator_b(write_stage);

    smem_store_a.store(tb_frag_a);
    smem_store_b.store(tb_frag_b);

    __syncthreads();

    read_stage ^= 1;
    write_stage ^= 1;
  }

  // store accumulated results back to global memory
  v7::epilogue<Traits>(c_ptr, accum, smem, warp_tile_m_idx, warp_tile_n_idx,
                       lane_m_idx, lane_n_idx, tid, N);
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM,
          int WarpN, int WarpK>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  using Traits = v7::KernelTraits<ThreadblockM, ThreadblockN, ThreadblockK,
                                  WarpM, WarpN, WarpK, 2>;

  dim3 const block(Traits::kThreads);
  dim3 const grid(utils::ceil_div(N, ThreadblockN),
                  utils::ceil_div(M, ThreadblockM));

  sgemm_warptiling<Traits>
      <<<grid, block, Traits::kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v8
}  // namespace warptiling
}  // namespace sgemm
