#pragma once
#include <cuda_runtime.h>

#include "utils.cuh"

namespace sgemm {
namespace blocktiling_2d_vec4 {

namespace v0 {

template <int Threads, int ThreadblockM, int ThreadblockK, int SmemStrideA>
__DEVICE__ void load_a_threadblock_tile_scalar(float* smem_a,
                                               float const* a_ptr, int lda) {
  constexpr int kIterations = (ThreadblockM * ThreadblockK) / Threads;
  constexpr int kRowAdvance = Threads / ThreadblockK;
  constexpr int kColAdvance = Threads % ThreadblockK;

  int const src_advance = kRowAdvance * lda + kColAdvance;
  int const dst_advance = kColAdvance * SmemStrideA + kRowAdvance;
  int const base_row = threadIdx.x / ThreadblockK;
  int const base_col = threadIdx.x % ThreadblockK;
  int src_offset = base_row * lda + base_col;
  int dst_offset = base_col * SmemStrideA + base_row;

#pragma unroll
  for (int iter = 0; iter < kIterations; ++iter) {
    smem_a[dst_offset] = a_ptr[src_offset];
    src_offset += src_advance;
    dst_offset += dst_advance;
  }
}

/** 2D Block-tiling SGEMM kernel with vectorized B traffic
 * - one thread block computes a ThreadblockM x ThreadblockN output tile
 * - each thread computes ThreadM x ThreadN output tile in registers
 * - A is staged into shared memory with a local scalar transpose helper
 * - B is staged with vec4 iterators
 * - writes C with float4 stores
 */
template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int ThreadN, int SmemPaddingA>
__global__ void sgemm_blocktiling_2d_vec4(float* __restrict__ c_ptr,
                                          float const* __restrict__ a_ptr,
                                          float const* __restrict__ b_ptr,
                                          int const M, int const N,
                                          int const K) {
  constexpr int kVecWidth = 4;
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / (ThreadM * ThreadN);
  constexpr int kSmemStrideA = ThreadblockM + SmemPaddingA;

  static_assert(ThreadblockN % kVecWidth == 0);
  static_assert(ThreadN % kVecWidth == 0);

  using ThreadMapB =
      utils::PitchLinearStripminedThreadMap<ThreadblockN, ThreadblockK,
                                            kThreads, kVecWidth>;
  using ThreadblockTileIteratorB =
      utils::ThreadblockTileIterator<ThreadblockN, ThreadblockK, ThreadMapB, 1>;
  using SmemTileIteratorB =
      utils::SmemTileIterator<ThreadblockN, ThreadblockK, ThreadMapB, 0>;

  int const cta_m_idx = blockIdx.y;
  int const cta_n_idx = blockIdx.x;
  int const cta_row = cta_m_idx * ThreadblockM;
  int const cta_col = cta_n_idx * ThreadblockN;
  int const thread_tile_m_idx = threadIdx.x / (ThreadblockN / ThreadN);
  int const thread_tile_n_idx = threadIdx.x % (ThreadblockN / ThreadN);

  float const* threadblock_ptr_a = a_ptr + cta_row * K;
  c_ptr += cta_row * N + cta_col;

  extern __shared__ float smem[];
  float* smem_a = smem;
  float* smem_b = smem + ThreadblockK * kSmemStrideA;

  auto threadblock_tile_iterator_b =
      ThreadblockTileIteratorB(b_ptr, N, threadIdx.x);
  threadblock_tile_iterator_b.add_tile_offset(0, cta_n_idx);

  auto smem_tile_iterator_b =
      SmemTileIteratorB(smem_b, ThreadblockN, threadIdx.x);

  float accum[ThreadM][ThreadN] = {0.f};
  typename ThreadblockTileIteratorB::Fragment frag_b;

  int const num_tiles = K / ThreadblockK;
  if (num_tiles == 0) {
    return;
  }

#pragma unroll
  for (int k_tile_idx = 0; k_tile_idx < num_tiles; ++k_tile_idx) {
    load_a_threadblock_tile_scalar<kThreads, ThreadblockM, ThreadblockK,
                                   kSmemStrideA>(smem_a, threadblock_ptr_a, K);
    threadblock_tile_iterator_b.load(frag_b);
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

    threadblock_ptr_a += ThreadblockK;
    ++threadblock_tile_iterator_b;
  }

#pragma unroll
  for (int m = 0; m < ThreadM; ++m) {
    int const c_row = ThreadM * thread_tile_m_idx + m;
    int const c_col_base = ThreadN * thread_tile_n_idx;
#pragma unroll
    for (int n = 0; n < ThreadN; n += kVecWidth) {
      float4 const value = utils::make_float4_from_array(&accum[m][n]);
      utils::store_float4(&c_ptr[c_row * N + c_col_base + n], value);
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

  sgemm_blocktiling_2d_vec4<ThreadblockM, ThreadblockN, ThreadblockK, ThreadM,
                            ThreadN, SmemPaddingA>
      <<<grid, block, kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v0

namespace v1 {

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int ThreadN, int SmemPaddingA>
__global__ void sgemm_blocktiling_2d_vec4(float* __restrict__ c_ptr,
                                          float const* __restrict__ a_ptr,
                                          float const* __restrict__ b_ptr,
                                          int const M, int const N,
                                          int const K) {
  constexpr int kVecWidth = 4;
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / (ThreadM * ThreadN);
  constexpr int kSmemStrideA = ThreadblockM + SmemPaddingA;

  static_assert(ThreadblockN % kVecWidth == 0);
  static_assert(ThreadN % kVecWidth == 0);

  using ThreadMapA =
      utils::PitchLinearStripminedThreadMap<ThreadblockK, ThreadblockM,
                                            kThreads, kVecWidth>;
  using ThreadMapB =
      utils::PitchLinearStripminedThreadMap<ThreadblockN, ThreadblockK,
                                            kThreads, kVecWidth>;
  using ThreadblockTileIteratorA =
      utils::ThreadblockTileIterator<ThreadblockK, ThreadblockM, ThreadMapA, 0>;
  using ThreadblockTileIteratorB =
      utils::ThreadblockTileIterator<ThreadblockN, ThreadblockK, ThreadMapB, 1>;
  using SmemThreadMapA =
      utils::TransposePitchLinearThreadMapSimtVec4Store<ThreadMapA>;
  using SmemTileIteratorA =
      utils::SmemTransposeVec4TileIterator<ThreadblockM, ThreadblockK,
                                           SmemThreadMapA, 1>;
  using SmemTileIteratorB =
      utils::SmemTileIterator<ThreadblockN, ThreadblockK, ThreadMapB, 1>;

  int const cta_m_idx = blockIdx.y;
  int const cta_n_idx = blockIdx.x;
  int const cta_row = cta_m_idx * ThreadblockM;
  int const cta_col = cta_n_idx * ThreadblockN;
  int const thread_tile_m_idx = threadIdx.x / (ThreadblockN / ThreadN);
  int const thread_tile_n_idx = threadIdx.x % (ThreadblockN / ThreadN);

  c_ptr += cta_row * N + cta_col;

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

  float accum[ThreadM][ThreadN] = {0.f};
  typename ThreadblockTileIteratorA::Fragment frag_a;
  typename ThreadblockTileIteratorB::Fragment frag_b;

  int const num_tiles = K / ThreadblockK;
  if (num_tiles == 0) {
    return;
  }

#pragma unroll
  for (int k_tile_idx = 0; k_tile_idx < num_tiles; ++k_tile_idx) {
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

#pragma unroll
  for (int m = 0; m < ThreadM; ++m) {
    int const c_row = ThreadM * thread_tile_m_idx + m;
    int const c_col_base = ThreadN * thread_tile_n_idx;
#pragma unroll
    for (int n = 0; n < ThreadN; n += kVecWidth) {
      float4 const value = utils::make_float4_from_array(&accum[m][n]);
      utils::store_float4(&c_ptr[c_row * N + c_col_base + n], value);
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

  sgemm_blocktiling_2d_vec4<ThreadblockM, ThreadblockN, ThreadblockK, ThreadM,
                            ThreadN, SmemPaddingA>
      <<<grid, block, kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v1

}  // namespace blocktiling_2d_vec4
}  // namespace sgemm
