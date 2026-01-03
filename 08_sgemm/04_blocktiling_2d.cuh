#pragma once
#include <cuda_runtime.h>

#include "utils.cuh"

namespace sgemm {
namespace blocktiling_2d {

/** 2D Block-tiling SGEMM kernel
 * - one thread block computes a ThreadblockM x ThreadblockN output tile
 * - each thread computes ThreadM x ThreadN output tile in registers
 * - A is staged into shared memory with transpose and padding to reduce bank
 * conflicts
 * - B is staged into shared memory in row-major order
 */
template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int ThreadN, int SmemPaddingA>
__global__ void sgemm_blocktiling_2d(float* __restrict__ c_ptr,
                                     float const* __restrict__ a_ptr,
                                     float const* __restrict__ b_ptr,
                                     int const M, int const N, int const K) {
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / (ThreadM * ThreadN);
  constexpr int kSmemStrideA = ThreadblockM + SmemPaddingA;

  using ThreadMapA =
      utils::PitchLinearStripminedThreadMap<ThreadblockK, ThreadblockM,
                                            kThreads>;
  using ThreadMapB =
      utils::PitchLinearStripminedThreadMap<ThreadblockN, ThreadblockK,
                                            kThreads>;
  using ThreadblockTileIteratorA =
      utils::ThreadblockTileIterator<ThreadblockK, ThreadblockM, ThreadMapA, 0>;
  using ThreadblockTileIteratorB =
      utils::ThreadblockTileIterator<ThreadblockN, ThreadblockK, ThreadMapB, 1>;

  using SmemThreadMapA = utils::TransposePitchLinearThreadMapSimt<ThreadMapA>;
  using SmemThreadMapB =
      utils::PitchLinearStripminedThreadMap<ThreadblockN, ThreadblockK,
                                            kThreads>;
  using SmemTileIteratorA =
      utils::SmemTileIterator<kSmemStrideA, ThreadblockK, SmemThreadMapA, 1>;
  using SmemTileIteratorB =
      utils::SmemTileIterator<ThreadblockN, ThreadblockK, SmemThreadMapB, 1>;

  int const cta_m_idx = blockIdx.y;
  int const cta_n_idx = blockIdx.x;
  int const cta_row = cta_m_idx * ThreadblockM;
  int const cta_col = cta_n_idx * ThreadblockN;
  // thread mapping inside the block tile
  int const thread_tile_m_idx = threadIdx.x / (ThreadblockN / ThreadN);
  int const thread_tile_n_idx = threadIdx.x % (ThreadblockN / ThreadN);

  c_ptr += cta_row * N + cta_col;

  // shared memory layout:
  // [ smem_a (ThreadblockK x kSmemStrideA) | smem_b (ThreadblockK x
  // ThreadblockN) ]
  extern __shared__ float smem[];
  float* smem_a = smem;
  float* smem_b = smem + ThreadblockK * kSmemStrideA;

  auto threadblock_tile_iterator_a =
      ThreadblockTileIteratorA(a_ptr, K, threadIdx.x);
  auto threadblock_tile_iterator_b =
      ThreadblockTileIteratorB(b_ptr, N, threadIdx.x);
  threadblock_tile_iterator_a.add_tile_offset(cta_m_idx, 0);
  threadblock_tile_iterator_b.add_tile_offset(0, cta_n_idx);

  auto smem_tile_iterator_a =
      SmemTileIteratorA(smem_a, kSmemStrideA, threadIdx.x);
  auto smem_tile_iterator_b =
      SmemTileIteratorB(smem_b, ThreadblockN, threadIdx.x);

  // per-thread accumulators for ThreadM outpus along M
  float accum[ThreadM][ThreadN] = {0.f};
  typename ThreadblockTileIteratorA::Fragment frag_a;
  typename ThreadblockTileIteratorB::Fragment frag_b;

  int const num_tiles = K / ThreadblockK;
  if (num_tiles == 0) {
    return;
  }

#pragma unroll
  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    threadblock_tile_iterator_a.load(frag_a);
    threadblock_tile_iterator_b.load(frag_b);
    smem_tile_iterator_a.store(frag_a);
    smem_tile_iterator_b.store(frag_b);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < ThreadblockK; ++k) {
#pragma unroll
      for (int m = 0; m < ThreadM; ++m) {
        float const elem_a =
            smem_a[kSmemStrideA * k + ThreadM * thread_tile_m_idx + m];
#pragma unroll
        for (int n = 0; n < ThreadN; ++n) {
          float const elem_b =
              smem_b[ThreadblockN * k + ThreadN * thread_tile_n_idx + n];
          accum[m][n] += elem_a * elem_b;
        }
      }
    }
    __syncthreads();

    ++threadblock_tile_iterator_a;
    ++threadblock_tile_iterator_b;
  }

  // store accumulated results back to global memory
#pragma unroll
  for (int m = 0; m < ThreadM; ++m) {
#pragma unroll
    for (int n = 0; n < ThreadN; ++n) {
      c_ptr[(ThreadM * thread_tile_m_idx + m) * N +
            ThreadN * thread_tile_n_idx + n] = accum[m][n];
    }
  }
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int ThreadN, int SmemPaddingA = 4>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / (ThreadM * ThreadN);
  constexpr int kSmemBytes = ((ThreadblockM + SmemPaddingA) * ThreadblockK +
                              ThreadblockK * ThreadblockN) *
                             sizeof(float);

  dim3 const block(kThreads);
  dim3 const grid(utils::ceil_div(N, ThreadblockN),
                  utils::ceil_div(M, ThreadblockM));

  sgemm_blocktiling_2d<ThreadblockM, ThreadblockN, ThreadblockK, ThreadM,
                       ThreadN, SmemPaddingA>
      <<<grid, block, kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace blocktiling_2d
}  // namespace sgemm