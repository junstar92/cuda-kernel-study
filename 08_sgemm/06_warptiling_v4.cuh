#pragma once

#include <cuda_runtime.h>

#include "06_warptiling_v2.cuh"
#include "utils.cuh"

namespace sgemm {
namespace warptiling {
namespace v4 {

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

  threadblock_tile_iterator_a.load(frag_a, true);
  threadblock_tile_iterator_b.load(frag_b, true);
  ++threadblock_tile_iterator_a;
  ++threadblock_tile_iterator_b;

  v2::store_fragments_to_stage<Traits>(smem_a, smem_b, 0, tid, frag_a, frag_b);

  __syncthreads();

  int read_stage = 0;
#pragma unroll
  for (; gemm_k_iterations > 0; --gemm_k_iterations) {
    threadblock_tile_iterator_a.load(frag_a, gemm_k_iterations > 1);
    threadblock_tile_iterator_b.load(frag_b, gemm_k_iterations > 1);
    ++threadblock_tile_iterator_a;
    ++threadblock_tile_iterator_b;

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

      v2::warp_mma<Traits>(accum, warp_frag_a, warp_frag_b);

      ++warp_tile_iterator_a;
      ++warp_tile_iterator_b;
    }

    int const write_stage = read_stage ^ 1;
    v2::store_fragments_to_stage<Traits>(smem_a, smem_b, write_stage, tid,
                                         frag_a, frag_b);

    __syncthreads();

    read_stage = write_stage;
  }

  // store accumulated results back to global memory
  v2::store_accum_to_global<Traits>(c_ptr, accum, warp_tile_m_idx,
                                    warp_tile_n_idx, lane_m_idx, lane_n_idx, N);
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM,
          int WarpN, int WarpK>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  using Traits = v2::KernelTraits<ThreadblockM, ThreadblockN, ThreadblockK,
                                  WarpM, WarpN, WarpK, 2>;

  dim3 const block(Traits::kThreads);
  dim3 const grid(utils::ceil_div(N, ThreadblockN),
                  utils::ceil_div(M, ThreadblockM));

  sgemm_warptiling<Traits>
      <<<grid, block, Traits::kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v4
}  // namespace warptiling
}  // namespace sgemm
