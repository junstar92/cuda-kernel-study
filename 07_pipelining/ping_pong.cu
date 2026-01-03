// ─────────────────────────────────────────────────────────────
//  ping_pong.cu
//  Tiled SGEMM with ping-pong (double buffering) pipelining.
//  Two shared memory buffers alternate: one loads while the
//  other is used for compute.
//
//  Data path: Global ──LDG──▶ Register ──STS──▶ Shared Memory
// ─────────────────────────────────────────────────────────────
#include "common.cuh"

__global__ void sgemm_ping_pong(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // ── Double buffer: 2 slots ──
  __shared__ float sA[2][BLOCK_TILE_M][BLOCK_TILE_K];
  __shared__ float sB[2][BLOCK_TILE_K][BLOCK_TILE_N];

  float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

  const int row_base = by * BLOCK_TILE_M;
  const int col_base = bx * BLOCK_TILE_N;
  const int tid = ty * BLOCK_DIM_X + tx;
  const int num_threads = BLOCK_DIM_X * BLOCK_DIM_Y;

  // ────────────────────────────────────────
  //  Helper: load tile into buffer `buf`
  // ────────────────────────────────────────
  auto load_tile = [&](int buf, int k_tile) {
#pragma unroll
    for (int i = 0; i < (BLOCK_TILE_M * BLOCK_TILE_K) / num_threads; i++) {
      int idx = i * num_threads + tid;
      int r = idx / BLOCK_TILE_K;
      int c = idx % BLOCK_TILE_K;
      sA[buf][r][c] = A[(row_base + r) * K + k_tile * BLOCK_TILE_K + c];
    }
#pragma unroll
    for (int i = 0; i < (BLOCK_TILE_K * BLOCK_TILE_N) / num_threads; i++) {
      int idx = i * num_threads + tid;
      int r = idx / BLOCK_TILE_N;
      int c = idx % BLOCK_TILE_N;
      sB[buf][r][c] = B[(k_tile * BLOCK_TILE_K + r) * N + col_base + c];
    }
  };

  // ────────────────────────────────────────
  //  Helper: compute from buffer `buf`
  // ────────────────────────────────────────
  auto compute_tile = [&](int buf) {
#pragma unroll
    for (int kk = 0; kk < BLOCK_TILE_K; kk++) {
      float a_frag[THREAD_TILE_M];
      float b_frag[THREAD_TILE_N];

#pragma unroll
      for (int tm = 0; tm < THREAD_TILE_M; tm++) {
        a_frag[tm] = sA[buf][ty * THREAD_TILE_M + tm][kk];
      }
#pragma unroll
      for (int tn = 0; tn < THREAD_TILE_N; tn++) {
        b_frag[tn] = sB[buf][kk][tx * THREAD_TILE_N + tn];
      }

#pragma unroll
      for (int tm = 0; tm < THREAD_TILE_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < THREAD_TILE_N; tn++) {
          acc[tm][tn] += a_frag[tm] * b_frag[tn];
        }
      }
    }
  };

  // ══════════════════════════════════════════
  //  Prologue: load first tile into buf[0]
  // ══════════════════════════════════════════
  load_tile(0, 0);
  __syncthreads();

  // ══════════════════════════════════════════
  //  Main Loop: ping-pong between buf[0] and buf[1]
  //
  //  Timeline:
  //    k=1: load→buf[1], compute←buf[0]
  //    k=2: load→buf[0], compute←buf[1]
  //    k=3: load→buf[1], compute←buf[0]
  //    ...
  // ══════════════════════════════════════════
  for (int k_tile = 1; k_tile < NUM_TILES_K; k_tile++) {
    int curr = (k_tile - 1) & 1;  // compute from this buffer
    int next = k_tile & 1;        // load into this buffer

    // Issue LDG for next tile (these become in-flight loads)
    load_tile(next, k_tile);

    // Compute using current buffer (overlaps with LDG latency)
    compute_tile(curr);

    // Barrier: ensures both load and compute are complete
    __syncthreads();
  }

  // ══════════════════════════════════════════
  //  Epilogue: compute the last tile
  // ══════════════════════════════════════════
  compute_tile((NUM_TILES_K - 1) & 1);

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
void launch_ping_pong(const float* d_A, const float* d_B, float* d_C) {
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid(N / BLOCK_TILE_N, M / BLOCK_TILE_M);
  sgemm_ping_pong<<<grid, block>>>(d_A, d_B, d_C);
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main() {
  printf("=== SGEMM Ping-Pong (Double Buffering) ===\n");
  printf("M=%d, N=%d, K=%d\n", M, N, K);
  printf("Block tile: %dx%dx%d, Thread tile: %dx%d\n\n", BLOCK_TILE_M,
         BLOCK_TILE_N, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N);

  float* h_A = (float*)malloc(M * K * sizeof(float));
  float* h_B = (float*)malloc(K * N * sizeof(float));
  float* h_C = (float*)malloc(M * N * sizeof(float));
  float* h_C_ref = (float*)malloc(M * N * sizeof(float));

  srand(42);
  init_matrix(h_A, M, K);
  init_matrix(h_B, K, N);

  printf("Computing CPU reference...\n");
  cpu_gemm_ref(h_A, h_B, h_C_ref, M, N, K);
  printf("Done.\n\n");

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  benchmark_kernel("Ping-Pong (Double Buffering)", launch_ping_pong, d_A, d_B,
                   d_C, h_C, h_C_ref);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return 0;
}
