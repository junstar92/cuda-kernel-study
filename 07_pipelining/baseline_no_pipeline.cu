// ─────────────────────────────────────────────────────────────
//  baseline_no_pipeline.cu
//  Tiled SGEMM without any pipelining.
//  Load and compute are fully serialized within each k-tile.
// ─────────────────────────────────────────────────────────────
#include "common.cuh"

__global__ void sgemm_baseline(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C) {
  const int bx = blockIdx.x;   // block along N
  const int by = blockIdx.y;   // block along M
  const int tx = threadIdx.x;  // 0..BLOCK_DIM_X-1
  const int ty = threadIdx.y;  // 0..BLOCK_DIM_Y-1

  // Shared memory for one tile of A and B
  __shared__ float sA[BLOCK_TILE_M][BLOCK_TILE_K];
  __shared__ float sB[BLOCK_TILE_K][BLOCK_TILE_N];

  // Accumulator in registers
  float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

  // Row/col base for this block
  const int row_base = by * BLOCK_TILE_M;
  const int col_base = bx * BLOCK_TILE_N;

  // How many elements each thread loads for A and B tiles
  // A tile: BLOCK_TILE_M x BLOCK_TILE_K = 128x8 = 1024 elements, 256 threads →
  // 4 each B tile: BLOCK_TILE_K x BLOCK_TILE_N = 8x128 = 1024 elements, 256
  // threads → 4 each
  const int tid = ty * BLOCK_DIM_X + tx;
  const int num_threads = BLOCK_DIM_X * BLOCK_DIM_Y;

  for (int k_tile = 0; k_tile < NUM_TILES_K; k_tile++) {
// ── Phase 1: Load A tile and B tile into shared memory ──
#pragma unroll
    for (int i = 0; i < (BLOCK_TILE_M * BLOCK_TILE_K) / num_threads; i++) {
      int idx = i * num_threads + tid;
      int r = idx / BLOCK_TILE_K;
      int c = idx % BLOCK_TILE_K;
      sA[r][c] = A[(row_base + r) * K + k_tile * BLOCK_TILE_K + c];
    }

#pragma unroll
    for (int i = 0; i < (BLOCK_TILE_K * BLOCK_TILE_N) / num_threads; i++) {
      int idx = i * num_threads + tid;
      int r = idx / BLOCK_TILE_N;
      int c = idx % BLOCK_TILE_N;
      sB[r][c] = B[(k_tile * BLOCK_TILE_K + r) * N + col_base + c];
    }

    __syncthreads();

// ── Phase 2: Compute ──
#pragma unroll
    for (int kk = 0; kk < BLOCK_TILE_K; kk++) {
      float a_frag[THREAD_TILE_M];
      float b_frag[THREAD_TILE_N];

#pragma unroll
      for (int tm = 0; tm < THREAD_TILE_M; tm++) {
        a_frag[tm] = sA[ty * THREAD_TILE_M + tm][kk];
      }
#pragma unroll
      for (int tn = 0; tn < THREAD_TILE_N; tn++) {
        b_frag[tn] = sB[kk][tx * THREAD_TILE_N + tn];
      }

#pragma unroll
      for (int tm = 0; tm < THREAD_TILE_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < THREAD_TILE_N; tn++) {
          acc[tm][tn] += a_frag[tm] * b_frag[tn];
        }
      }
    }

    __syncthreads();  // ensure smem is safe to overwrite
  }

// ── Write results ──
#pragma unroll
  for (int tm = 0; tm < THREAD_TILE_M; tm++) {
#pragma unroll
    for (int tn = 0; tn < THREAD_TILE_N; tn++) {
      int row = row_base + ty * THREAD_TILE_M + tm;
      int col = col_base + tx * THREAD_TILE_N + tn;
      if (row < M && col < N) {
        C[row * N + col] = acc[tm][tn];
      }
    }
  }
}

// ─────────────────────────────────────────────
//  Launch wrapper
// ─────────────────────────────────────────────
void launch_baseline(const float* d_A, const float* d_B, float* d_C) {
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid(N / BLOCK_TILE_N, M / BLOCK_TILE_M);
  sgemm_baseline<<<grid, block>>>(d_A, d_B, d_C);
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main() {
  printf("=== SGEMM Baseline (No Pipelining) ===\n");
  printf("M=%d, N=%d, K=%d\n", M, N, K);
  printf("Block tile: %dx%dx%d, Thread tile: %dx%d\n\n", BLOCK_TILE_M,
         BLOCK_TILE_N, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N);

  // Host alloc
  float* h_A = (float*)malloc(M * K * sizeof(float));
  float* h_B = (float*)malloc(K * N * sizeof(float));
  float* h_C = (float*)malloc(M * N * sizeof(float));
  float* h_C_ref = (float*)malloc(M * N * sizeof(float));

  srand(42);
  init_matrix(h_A, M, K);
  init_matrix(h_B, K, N);

  // CPU reference
  printf("Computing CPU reference...\n");
  cpu_gemm_ref(h_A, h_B, h_C_ref, M, N, K);
  printf("Done.\n\n");

  // Device alloc
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  benchmark_kernel("Baseline (No Pipeline)", launch_baseline, d_A, d_B, d_C,
                   h_C, h_C_ref);

  // Cleanup
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return 0;
}
