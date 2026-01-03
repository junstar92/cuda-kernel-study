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

// Warp tiling SGEMM kernel
// - Block is divided into warps, each warp computes WARP_M x WARP_N output tile
// - Each thread in a warp computes NUM_THREAD_TILE_M x NUM_THREAD_TILE_N
// elements
// - A is transposed in shared memory for coalesced access
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K, int A_SMEM_PADDING = 4>
__global__ void sgemm_warptiling_kernel(float* __restrict__ c_ptr,
                                        float const* __restrict__ a_ptr,
                                        float const* __restrict__ b_ptr,
                                        int const M, int const N, int const K) {
  static_assert(BLOCK_K == WARP_K);

  constexpr int NUM_WARP_TILE_M = BLOCK_M / WARP_M;
  constexpr int NUM_WARP_TILE_N = BLOCK_N / WARP_N;
  constexpr int NUM_THREADS = NUM_WARP_TILE_M * NUM_WARP_TILE_N * 32;
  constexpr int NUM_WARP_THREADS_M = (WARP_M > WARP_N) ? 8 : 4;
  constexpr int NUM_WARP_THREADS_N = 32 / NUM_WARP_THREADS_M;
  constexpr int NUM_THREAD_TILE_M = WARP_M / NUM_WARP_THREADS_M;
  constexpr int NUM_THREAD_TILE_N = WARP_N / NUM_WARP_THREADS_N;
  constexpr int A_SMEM_STRIDE = BLOCK_M + A_SMEM_PADDING;

  extern __shared__ float smem[];
  float* a_smem = smem;
  float* b_smem = smem + A_SMEM_STRIDE * BLOCK_K;

  a_ptr += BLOCK_M * blockIdx.y * K;
  b_ptr += BLOCK_N * blockIdx.x;
  c_ptr += BLOCK_M * blockIdx.y * N + BLOCK_N * blockIdx.x;

  int const tid = threadIdx.x;
  int const warp_id = tid / 32;
  int const lane_id = tid % 32;

  int const warp_m = warp_id / NUM_WARP_TILE_N;
  int const warp_n = warp_id % NUM_WARP_TILE_N;
  int const thread_m = lane_id / NUM_WARP_THREADS_N;
  int const thread_n = lane_id % NUM_WARP_THREADS_N;

  float a_frag[NUM_THREAD_TILE_M];
  float b_frag[NUM_THREAD_TILE_N];
  float accum[NUM_THREAD_TILE_M][NUM_THREAD_TILE_N] = {0.f};

  float const* a_warp_smem = a_smem + warp_m * WARP_M;
  float const* b_warp_smem = b_smem + warp_n * WARP_N;

  for (int tile = 0; tile < K; tile += BLOCK_K) {
    load_gmem_to_smem<BLOCK_M, BLOCK_K, A_SMEM_STRIDE, NUM_THREADS, true>(
        a_smem, a_ptr, K);
    load_gmem_to_smem<BLOCK_K, BLOCK_N, BLOCK_N, NUM_THREADS, false>(b_smem,
                                                                     b_ptr, N);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < WARP_K; k++) {
#pragma unroll
      for (int m = 0; m < NUM_THREAD_TILE_M; m++) {
        int const m_idx = thread_m * NUM_THREAD_TILE_M + m;
        a_frag[m] = a_warp_smem[k * A_SMEM_STRIDE + m_idx];
      }

#pragma unroll
      for (int n = 0; n < NUM_THREAD_TILE_N; n++) {
        int const n_idx = thread_n * NUM_THREAD_TILE_N + n;
        b_frag[n] = b_warp_smem[k * BLOCK_N + n_idx];
      }

#pragma unroll
      for (int m = 0; m < NUM_THREAD_TILE_M; m++) {
#pragma unroll
        for (int n = 0; n < NUM_THREAD_TILE_N; n++) {
          accum[m][n] += a_frag[m] * b_frag[n];
        }
      }
    }

    a_ptr += BLOCK_K;
    b_ptr += BLOCK_K * N;
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < NUM_THREAD_TILE_M; m++) {
#pragma unroll
    for (int n = 0; n < NUM_THREAD_TILE_N; n++) {
      int const row = warp_m * WARP_M + thread_m * NUM_THREAD_TILE_M + m;
      int const col = warp_n * WARP_N + thread_n * NUM_THREAD_TILE_N + n;
      c_ptr[row * N + col] = accum[m][n];
    }
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K, int A_SMEM_PADDING = 4>
void launch_sgemm_warptiling(float* c_ptr, float const* a_ptr,
                             float const* b_ptr, int const M, int const N,
                             int const K) {
  constexpr int NUM_WARP_TILE_M = BLOCK_M / WARP_M;
  constexpr int NUM_WARP_TILE_N = BLOCK_N / WARP_N;
  constexpr int NUM_THREADS = NUM_WARP_TILE_M * NUM_WARP_TILE_N * 32;
  constexpr int SMEM_SIZE =
      BLOCK_K * (BLOCK_M + A_SMEM_PADDING + BLOCK_N) * sizeof(float);

  dim3 block(NUM_THREADS);
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  sgemm_warptiling_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K,
                          A_SMEM_PADDING>
      <<<grid, block, SMEM_SIZE>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

template void launch_sgemm_warptiling<128, 128, 8, 64, 32, 8>(
    float*, float const*, float const*, int const, int const, int const);

template void launch_sgemm_warptiling<128, 128, 16, 64, 32, 16>(
    float*, float const*, float const*, int const, int const, int const);
