#include <cuda_runtime.h>

// Load a tile from global memory to shared memory
// - Distributes work evenly across threads (ELEMENTS_PER_THREAD each)
// - Supports optional transpose for bank-conflict-free access patterns
template <int NUM_ROWS, int NUM_COLS, int DST_STRIDE, int NUM_THREADS,
          bool TRANSPOSE = false>
__device__ __forceinline__ void load_gmem_to_smem(float* dst, float const* src,
                                                  int ld) {
  constexpr int TOTAL_ELEMENTS = NUM_ROWS * NUM_COLS;
  constexpr int ELEMENTS_PER_THREAD = TOTAL_ELEMENTS / NUM_THREADS;
  static_assert(TOTAL_ELEMENTS % NUM_THREADS == 0);

  int const tid = threadIdx.x;

#pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
    int const idx = tid + i * NUM_THREADS;
    int const row = idx / NUM_COLS;
    int const col = idx % NUM_COLS;

    int const src_offset = row * ld + col;
    int const dst_offset =
        TRANSPOSE ? (col * DST_STRIDE + row) : (row * DST_STRIDE + col);

    dst[dst_offset] = src[src_offset];
  }
}

// 2D block tiling SGEMM kernel
// - Each thread computes THREAD_M x THREAD_N elements
// - A is transposed in shared memory to enable coalesced access
// - Padding added to A's shared memory to avoid bank conflicts
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
          int A_SMEM_PADDING = 4>
__global__ void sgemm_2d_blocktiling_kernel(float* __restrict__ c_ptr,
                                            float const* __restrict__ a_ptr,
                                            float const* __restrict__ b_ptr,
                                            int const M, int const N,
                                            int const K) {
  constexpr int NUM_THREADS = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
  constexpr int A_SMEM_STRIDE = BLOCK_M + A_SMEM_PADDING;

  extern __shared__ float smem[];
  float* a_smem = smem;
  float* b_smem = smem + A_SMEM_STRIDE * BLOCK_K;

  a_ptr += BLOCK_M * blockIdx.y * K;
  b_ptr += BLOCK_N * blockIdx.x;
  c_ptr += BLOCK_M * blockIdx.y * N + BLOCK_N * blockIdx.x;

  int const thread_row = threadIdx.x / (BLOCK_N / THREAD_N);
  int const thread_col = threadIdx.x % (BLOCK_N / THREAD_N);

  float accum[THREAD_M][THREAD_N] = {0.f};

  for (int tile = 0; tile < K; tile += BLOCK_K) {
    load_gmem_to_smem<BLOCK_M, BLOCK_K, A_SMEM_STRIDE, NUM_THREADS, true>(
        a_smem, a_ptr, K);
    load_gmem_to_smem<BLOCK_K, BLOCK_N, BLOCK_N, NUM_THREADS, false>(b_smem,
                                                                     b_ptr, N);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_K; k++) {
#pragma unroll
      for (int m = 0; m < THREAD_M; m++) {
        float a_val = a_smem[k * A_SMEM_STRIDE + thread_row * THREAD_M + m];

#pragma unroll
        for (int n = 0; n < THREAD_N; n++) {
          float b_val = b_smem[k * BLOCK_N + thread_col * THREAD_N + n];
          accum[m][n] += a_val * b_val;
        }
      }
    }

    a_ptr += BLOCK_K;
    b_ptr += BLOCK_K * N;
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < THREAD_M; m++) {
#pragma unroll
    for (int n = 0; n < THREAD_N; n++) {
      c_ptr[(thread_row * THREAD_M + m) * N + thread_col * THREAD_N + n] =
          accum[m][n];
    }
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
          int A_SMEM_PADDING = 4>
void launch_sgemm_2d_blocktiling(float* c_ptr, float const* a_ptr,
                                 float const* b_ptr, int const M, int const N,
                                 int const K) {
  constexpr int NUM_THREADS = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
  constexpr int SMEM_SIZE =
      BLOCK_K * (BLOCK_M + A_SMEM_PADDING + BLOCK_N) * sizeof(float);

  dim3 block(NUM_THREADS);
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  sgemm_2d_blocktiling_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N,
                              A_SMEM_PADDING>
      <<<grid, block, SMEM_SIZE>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

template void launch_sgemm_2d_blocktiling<128, 128, 8, 8, 8>(
    float*, float const*, float const*, int const, int const, int const);
