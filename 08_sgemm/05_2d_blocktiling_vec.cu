#include <cuda_runtime.h>

// Vectorized load from global memory to shared memory using float4
// - Each thread loads VECTORS_PER_THREAD float4 elements
// - Supports optional transpose for bank-conflict-free access patterns
template <int NUM_ROWS, int NUM_COLS, int DST_STRIDE, int NUM_THREADS,
          bool TRANSPOSE = false>
__device__ __forceinline__ void load_gmem_to_smem_vec(float* dst,
                                                      float const* src,
                                                      int ld) {
  constexpr int VECTOR_SIZE = 4;
  constexpr int NUM_COLS_VEC = NUM_COLS / VECTOR_SIZE;
  constexpr int TOTAL_VECTORS = NUM_ROWS * NUM_COLS_VEC;
  constexpr int VECTORS_PER_THREAD = TOTAL_VECTORS / NUM_THREADS;

  static_assert(NUM_COLS % VECTOR_SIZE == 0, "NUM_COLS must be divisible by 4");
  static_assert(TOTAL_VECTORS % NUM_THREADS == 0, "Work must divide evenly");

  int const tid = threadIdx.x;
  float4 const* src_vec = reinterpret_cast<float4 const*>(src);

#pragma unroll
  for (int i = 0; i < VECTORS_PER_THREAD; i++) {
    int const idx = tid + i * NUM_THREADS;
    int const row = idx / NUM_COLS_VEC;
    int const col_vec = idx % NUM_COLS_VEC;

    int const src_offset = row * (ld / VECTOR_SIZE) + col_vec;
    float4 data = src_vec[src_offset];

    int const col_base = col_vec * VECTOR_SIZE;

    if constexpr (TRANSPOSE) {
      dst[(col_base + 0) * DST_STRIDE + row] = data.x;
      dst[(col_base + 1) * DST_STRIDE + row] = data.y;
      dst[(col_base + 2) * DST_STRIDE + row] = data.z;
      dst[(col_base + 3) * DST_STRIDE + row] = data.w;
    } else {
      *reinterpret_cast<float4*>(&dst[row * DST_STRIDE + col_base]) = data;
    }
  }
}

// 2D block tiling SGEMM kernel with vectorized memory access
// - Each thread computes THREAD_M x THREAD_N elements
// - Uses float4 for coalesced global memory access
// - Register caching for A and B fragments
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
          int A_SMEM_PADDING = 4>
__global__ void sgemm_2d_blocktiling_vec_kernel(float* __restrict__ c_ptr,
                                                float const* __restrict__ a_ptr,
                                                float const* __restrict__ b_ptr,
                                                int const M, int const N,
                                                int const K) {
  static_assert(BLOCK_K % 4 == 0, "BLOCK_K must be divisible by 4");
  static_assert(BLOCK_N % 4 == 0, "BLOCK_N must be divisible by 4");

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
  float a_reg[THREAD_M];
  float b_reg[THREAD_N];

  for (int tile = 0; tile < K; tile += BLOCK_K) {
    load_gmem_to_smem_vec<BLOCK_M, BLOCK_K, A_SMEM_STRIDE, NUM_THREADS, true>(
        a_smem, a_ptr, K);
    load_gmem_to_smem_vec<BLOCK_K, BLOCK_N, BLOCK_N, NUM_THREADS, false>(
        b_smem, b_ptr, N);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_K; k++) {
#pragma unroll
      for (int m = 0; m < THREAD_M; m++) {
        a_reg[m] = a_smem[k * A_SMEM_STRIDE + thread_row * THREAD_M + m];
      }

#pragma unroll
      for (int n = 0; n < THREAD_N; n++) {
        b_reg[n] = b_smem[k * BLOCK_N + thread_col * THREAD_N + n];
      }

#pragma unroll
      for (int m = 0; m < THREAD_M; m++) {
#pragma unroll
        for (int n = 0; n < THREAD_N; n++) {
          accum[m][n] += a_reg[m] * b_reg[n];
        }
      }
    }

    a_ptr += BLOCK_K;
    b_ptr += BLOCK_K * N;
    __syncthreads();
  }

  // Vectorized store to global memory
#pragma unroll
  for (int m = 0; m < THREAD_M; m++) {
    int const row_idx = thread_row * THREAD_M + m;

    if constexpr (THREAD_N % 4 == 0) {
#pragma unroll
      for (int n = 0; n < THREAD_N; n += 4) {
        float4 out;
        out.x = accum[m][n + 0];
        out.y = accum[m][n + 1];
        out.z = accum[m][n + 2];
        out.w = accum[m][n + 3];

        int const col_idx = thread_col * THREAD_N + n;
        *reinterpret_cast<float4*>(&c_ptr[row_idx * N + col_idx]) = out;
      }
    } else {
#pragma unroll
      for (int n = 0; n < THREAD_N; n++) {
        c_ptr[row_idx * N + thread_col * THREAD_N + n] = accum[m][n];
      }
    }
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
          int A_SMEM_PADDING = 4>
void launch_sgemm_2d_blocktiling_vec(float* c_ptr, float const* a_ptr,
                                     float const* b_ptr, int const M,
                                     int const N, int const K) {
  constexpr int NUM_THREADS = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
  constexpr int SMEM_SIZE =
      BLOCK_K * (BLOCK_M + A_SMEM_PADDING + BLOCK_N) * sizeof(float);

  dim3 block(NUM_THREADS);
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  sgemm_2d_blocktiling_vec_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N,
                                  A_SMEM_PADDING>
      <<<grid, block, SMEM_SIZE>>>(c_ptr, a_ptr, b_ptr, M, N, K);
}

template void launch_sgemm_2d_blocktiling_vec<128, 128, 8, 8, 8>(
    float*, float const*, float const*, int const, int const, int const);
