#pragma once
#include <cuda_runtime.h>

#include "utils.cuh"

namespace sgemm {
namespace blocktiling_1d {

namespace v0 {

// Load an A tile from global memory to shared memory with transpose
template <int Threads, int ThreadblockM, int ThreadblockK, int SmemStride>
__forceinline__ __device__ void load_a_threadblock_tile(float* tb_tile_a,
                                                        float const* a_ptr,
                                                        int lda) {
  constexpr int num_iters = (ThreadblockM * ThreadblockK) / Threads;
  constexpr int row_advance_per_iter = Threads / ThreadblockK;
  constexpr int col_advance_per_iter = Threads % ThreadblockK;

  int const src_advance_offset =
      row_advance_per_iter * lda + col_advance_per_iter;
  int const dst_advance_offset =
      col_advance_per_iter * SmemStride + row_advance_per_iter;
  int const base_row = threadIdx.x / ThreadblockK;
  int const base_col = threadIdx.x % ThreadblockK;
  int src_offset = base_row * lda + base_col;
  int dst_offset = base_col * SmemStride + base_row;

#pragma unroll
  for (int iter = 0; iter < num_iters; ++iter) {
    tb_tile_a[dst_offset] = a_ptr[src_offset];
    src_offset += src_advance_offset;
    dst_offset += dst_advance_offset;
  }
}

// Load a B tile from global memory to shared memory without transpose
template <int Threads, int ThreadblockK, int ThreadblockN, int SmemStride>
__forceinline__ __device__ void load_b_threadblock_tile(float* tb_tile_b,
                                                        float const* b_ptr,
                                                        int ldb) {
  constexpr int num_iters = (ThreadblockK * ThreadblockN) / Threads;
  constexpr int row_advance_per_iter = Threads / ThreadblockN;
  constexpr int col_advance_per_iter = Threads % ThreadblockN;

  int const src_advance_offset =
      row_advance_per_iter * ldb + col_advance_per_iter;
  int const dst_advance_offset =
      row_advance_per_iter * SmemStride + col_advance_per_iter;
  int const base_row = threadIdx.x / ThreadblockN;
  int const base_col = threadIdx.x % ThreadblockN;
  int src_offset = base_row * ldb + base_col;
  int dst_offset = base_row * SmemStride + base_col;

#pragma unroll
  for (int iter = 0; iter < num_iters; ++iter) {
    tb_tile_b[dst_offset] = b_ptr[src_offset];
    src_offset += src_advance_offset;
    dst_offset += dst_advance_offset;
  }
}

/** 1D Block-tiling SGEMM kernel
 * - one thread block computes a ThreadblockM x ThreadblockN output tile
 * - each thread computes ThreadM results along the M dimension
 * - A is staged into shared memory with transpose and padding to reduce bank
 * conflicts
 * - B is staged into shared memory in row-major order
 */
template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int SmemPaddingA>
__global__ void kernel(float* __restrict__ c_ptr,
                       float const* __restrict__ a_ptr,
                       float const* __restrict__ b_ptr, int const M,
                       int const N, int const K) {
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / ThreadM;
  constexpr int kSmemStrideA = ThreadblockM + SmemPaddingA;

  int const cta_m_idx = blockIdx.y;
  int const cta_n_idx = blockIdx.x;

  int const cta_row = cta_m_idx * ThreadblockM;
  int const cta_col = cta_n_idx * ThreadblockN;

  // move global pointers to this block's output tile origin
  a_ptr += cta_row * K;
  b_ptr += cta_col;
  c_ptr += cta_row * N + cta_col;

  // shared memory layout:
  // [ smem_a (ThreadblockK x kSmemStrideA) | smem_b (ThreadblockK x
  // ThreadblockN) ]
  extern __shared__ float smem[];
  float* smem_a = smem;
  float* smem_b = smem + ThreadblockK * kSmemStrideA;

  // per-thread accumulators for ThreadM outpus along M
  float accum[ThreadM] = {0.f};

  // thread mapping inside the block tile
  int const thread_tile_m_idx = threadIdx.x / ThreadblockN;
  int const thread_tile_n_idx = threadIdx.x % ThreadblockN;

  int const num_tiles = K / ThreadblockK;
  if (num_tiles == 0) {
    return;
  }

#pragma unroll
  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    load_a_threadblock_tile<kThreads, ThreadblockM, ThreadblockK, kSmemStrideA>(
        smem_a, a_ptr, K);
    load_b_threadblock_tile<kThreads, ThreadblockK, ThreadblockN, ThreadblockN>(
        smem_b, b_ptr, N);
    __syncthreads();

    // advance to the next K tile in global memory
    a_ptr += ThreadblockK;
    b_ptr += ThreadblockK * N;

#pragma unroll
    for (int k = 0; k < ThreadblockK; ++k) {
      float const elem_b = smem_b[ThreadblockN * k + thread_tile_n_idx];

#pragma unroll
      for (int thread_m_idx = 0; thread_m_idx < ThreadM; ++thread_m_idx) {
        float const elem_a = smem_a[kSmemStrideA * k +
                                    ThreadM * thread_tile_m_idx + thread_m_idx];
        accum[thread_m_idx] += elem_a * elem_b;
      }
    }
    __syncthreads();
  }

  // store accumulated results back to global memory
#pragma unroll
  for (int m = 0; m < ThreadM; ++m) {
    c_ptr[(ThreadM * thread_tile_m_idx + m) * N + thread_tile_n_idx] = accum[m];
  }
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int SmemPaddingA = 4>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / ThreadM;
  constexpr int kSmemBytes = ((ThreadblockM + SmemPaddingA) * ThreadblockK +
                              ThreadblockK * ThreadblockN) *
                             sizeof(float);

  dim3 const block(kThreads);
  dim3 const grid(utils::ceil_div(N, ThreadblockN),
                  utils::ceil_div(M, ThreadblockM));

  kernel<ThreadblockM, ThreadblockN, ThreadblockK, ThreadM, SmemPaddingA>
      <<<grid, block, kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v0

namespace v1 {

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int SmemPaddingA>
__global__ void kernel(float* __restrict__ c_ptr,
                       float const* __restrict__ a_ptr,
                       float const* __restrict__ b_ptr, int const M,
                       int const N, int const K) {
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / ThreadM;
  constexpr int kSmemStrideA = ThreadblockM + SmemPaddingA;

  using ThreadMapA =
      utils::PitchLinearStripminedThreadMap<ThreadblockK, ThreadblockM,
                                            kThreads>;
  using ThreadMapB =
      utils::PitchLinearStripminedThreadMap<ThreadblockN, ThreadblockK,
                                            kThreads>;
  using ThreadblockTileIteratorA =
      utils::ThreadblockTileIteratorV0<ThreadblockK, ThreadblockM, ThreadMapA,
                                       0>;
  using ThreadblockTileIteratorB =
      utils::ThreadblockTileIteratorV0<ThreadblockN, ThreadblockK, ThreadMapB,
                                       1>;

  using SmemThreadMapA = utils::TransposePitchLinearThreadMapSimt<ThreadMapA>;
  using SmemThreadMapB =
      utils::PitchLinearStripminedThreadMap<ThreadblockN, ThreadblockK,
                                            kThreads>;
  using SmemTileIteratorA =
      utils::SmemTileIteratorV0<kSmemStrideA, ThreadblockK, SmemThreadMapA, 1>;
  using SmemTileIteratorB =
      utils::SmemTileIteratorV0<ThreadblockN, ThreadblockK, SmemThreadMapB, 1>;

  int const cta_m_idx = blockIdx.y;
  int const cta_n_idx = blockIdx.x;
  int const cta_row = cta_m_idx * ThreadblockM;
  int const cta_col = cta_n_idx * ThreadblockN;
  // thread mapping inside the block tile
  int const thread_tile_m_idx = threadIdx.x / ThreadblockN;
  int const thread_tile_n_idx = threadIdx.x % ThreadblockN;

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
  float accum[ThreadM] = {0.f};
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
      float const elem_b = smem_b[ThreadblockN * k + thread_tile_n_idx];

#pragma unroll
      for (int thread_m_idx = 0; thread_m_idx < ThreadM; ++thread_m_idx) {
        float const elem_a = smem_a[kSmemStrideA * k +
                                    ThreadM * thread_tile_m_idx + thread_m_idx];
        accum[thread_m_idx] += elem_a * elem_b;
      }
    }
    __syncthreads();

    ++threadblock_tile_iterator_a;
    ++threadblock_tile_iterator_b;
  }

  // store accumulated results back to global memory
#pragma unroll
  for (int m = 0; m < ThreadM; ++m) {
    c_ptr[(ThreadM * thread_tile_m_idx + m) * N + thread_tile_n_idx] = accum[m];
  }
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int SmemPaddingA = 4>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / ThreadM;
  constexpr int kSmemBytes = ((ThreadblockM + SmemPaddingA) * ThreadblockK +
                              ThreadblockK * ThreadblockN) *
                             sizeof(float);

  dim3 const block(kThreads);
  dim3 const grid(utils::ceil_div(N, ThreadblockN),
                  utils::ceil_div(M, ThreadblockM));

  kernel<ThreadblockM, ThreadblockN, ThreadblockK, ThreadM, SmemPaddingA>
      <<<grid, block, kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v1

namespace v2 {

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int SmemPaddingA>
__global__ void kernel(float* __restrict__ c_ptr,
                       float const* __restrict__ a_ptr,
                       float const* __restrict__ b_ptr, int const M,
                       int const N, int const K) {
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / ThreadM;
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
      utils::SmemTileIterator<ThreadblockN, ThreadblockK, SmemThreadMapB, 0>;

  int const cta_m_idx = blockIdx.y;
  int const cta_n_idx = blockIdx.x;
  int const cta_row = cta_m_idx * ThreadblockM;
  int const cta_col = cta_n_idx * ThreadblockN;
  // thread mapping inside the block tile
  int const thread_tile_m_idx = threadIdx.x / ThreadblockN;
  int const thread_tile_n_idx = threadIdx.x % ThreadblockN;

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
  float accum[ThreadM] = {0.f};
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
      float const elem_b = smem_b[ThreadblockN * k + thread_tile_n_idx];

#pragma unroll
      for (int thread_m_idx = 0; thread_m_idx < ThreadM; ++thread_m_idx) {
        float const elem_a = smem_a[kSmemStrideA * k +
                                    ThreadM * thread_tile_m_idx + thread_m_idx];
        accum[thread_m_idx] += elem_a * elem_b;
      }
    }
    __syncthreads();

    ++threadblock_tile_iterator_a;
    ++threadblock_tile_iterator_b;
  }

  // store accumulated results back to global memory
#pragma unroll
  for (int m = 0; m < ThreadM; ++m) {
    c_ptr[(ThreadM * thread_tile_m_idx + m) * N + thread_tile_n_idx] = accum[m];
  }
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int SmemPaddingA = 4>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / ThreadM;
  constexpr int kSmemBytes = ((ThreadblockM + SmemPaddingA) * ThreadblockK +
                              ThreadblockK * ThreadblockN) *
                             sizeof(float);

  dim3 const block(kThreads);
  dim3 const grid(utils::ceil_div(N, ThreadblockN),
                  utils::ceil_div(M, ThreadblockM));

  kernel<ThreadblockM, ThreadblockN, ThreadblockK, ThreadM, SmemPaddingA>
      <<<grid, block, kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v2

namespace v3 {

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int SmemPaddingA>
__global__ void kernel(float* __restrict__ c_ptr,
                       float const* __restrict__ a_ptr,
                       float const* __restrict__ b_ptr, int const M,
                       int const N, int const K) {
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / ThreadM;
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
      utils::SmemTileIterator<ThreadblockN, ThreadblockK, SmemThreadMapB, 0>;

  int const cta_m_idx = blockIdx.y;
  int const cta_n_idx = blockIdx.x;
  int const cta_row = cta_m_idx * ThreadblockM;
  int const cta_col = cta_n_idx * ThreadblockN;
  // thread mapping inside the block tile
  int const thread_tile_m_idx = threadIdx.x / ThreadblockN;
  int const thread_tile_n_idx = threadIdx.x % ThreadblockN;

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
  float accum[ThreadM] = {0.f};
  typename ThreadblockTileIteratorA::Fragment frag_a;
  typename ThreadblockTileIteratorB::Fragment frag_b;

  int const num_tiles = K / ThreadblockK;
  if (num_tiles == 0) {
    return;
  }

  threadblock_tile_iterator_a.load(frag_a);
  threadblock_tile_iterator_b.load(frag_b);

  ++threadblock_tile_iterator_a;
  ++threadblock_tile_iterator_b;

#pragma unroll
  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    smem_tile_iterator_a.store(frag_a);
    smem_tile_iterator_b.store(frag_b);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < ThreadblockK; ++k) {
      float const elem_b = smem_b[ThreadblockN * k + thread_tile_n_idx];

#pragma unroll
      for (int thread_m_idx = 0; thread_m_idx < ThreadM; ++thread_m_idx) {
        float const elem_a = smem_a[kSmemStrideA * k +
                                    ThreadM * thread_tile_m_idx + thread_m_idx];
        accum[thread_m_idx] += elem_a * elem_b;
      }
    }
    __syncthreads();

    threadblock_tile_iterator_a.load(frag_a);
    threadblock_tile_iterator_b.load(frag_b);
    ++threadblock_tile_iterator_a;
    ++threadblock_tile_iterator_b;
  }

  // store accumulated results back to global memory
#pragma unroll
  for (int m = 0; m < ThreadM; ++m) {
    c_ptr[(ThreadM * thread_tile_m_idx + m) * N + thread_tile_n_idx] = accum[m];
  }
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int ThreadM,
          int SmemPaddingA = 4>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  constexpr int kThreads = (ThreadblockM * ThreadblockN) / ThreadM;
  constexpr int kSmemBytes = ((ThreadblockM + SmemPaddingA) * ThreadblockK +
                              ThreadblockK * ThreadblockN) *
                             sizeof(float);

  dim3 const block(kThreads);
  dim3 const grid(utils::ceil_div(N, ThreadblockN),
                  utils::ceil_div(M, ThreadblockM));

  kernel<ThreadblockM, ThreadblockN, ThreadblockK, ThreadM, SmemPaddingA>
      <<<grid, block, kSmemBytes>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

}  // namespace v3

}  // namespace blocktiling_1d
}  // namespace sgemm