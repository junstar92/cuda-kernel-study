// ─────────────────────────────────────────────────────────────
//  cp_async_pipeline.cu
//  Tiled SGEMM with cp.async multi-stage software pipelining.
//  Uses hardware async copy (Global → Shared, register bypass)
//  with commit_group / wait_group for multi-stage control.
//
//  Data path: Global ──cp.async──▶ Shared Memory (no register)
//  Requires: SM80 (Ampere) or later
// ─────────────────────────────────────────────────────────────
#include <cuda_pipeline.h>  // __pipeline_memcpy_async, __pipeline_commit, __pipeline_wait_prior

#include "common.cuh"

#ifndef STAGES
#define STAGES 2
#endif

__global__ void sgemm_cp_async(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // ── Multi-stage buffer: STAGES slots ──
  __shared__ float sA[STAGES][BLOCK_TILE_M][BLOCK_TILE_K];
  __shared__ float sB[STAGES][BLOCK_TILE_K][BLOCK_TILE_N];

  float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

  const int row_base = by * BLOCK_TILE_M;
  const int col_base = bx * BLOCK_TILE_N;
  const int tid = ty * BLOCK_DIM_X + tx;
  const int num_threads = BLOCK_DIM_X * BLOCK_DIM_Y;

  // ────────────────────────────────────────
  //  Helper: issue async copy for tile k_tile into buffer `stage`
  //  cp.async bypasses registers — data goes directly Global → Shared
  // ────────────────────────────────────────
  auto async_load_tile = [&](int stage, int k_tile) {
#pragma unroll
    for (int i = 0; i < (BLOCK_TILE_M * BLOCK_TILE_K) / num_threads; i++) {
      int idx = i * num_threads + tid;
      int r = idx / BLOCK_TILE_K;
      int c = idx % BLOCK_TILE_K;
      __pipeline_memcpy_async(
          &sA[stage][r][c], &A[(row_base + r) * K + k_tile * BLOCK_TILE_K + c],
          sizeof(float));
    }
#pragma unroll
    for (int i = 0; i < (BLOCK_TILE_K * BLOCK_TILE_N) / num_threads; i++) {
      int idx = i * num_threads + tid;
      int r = idx / BLOCK_TILE_N;
      int c = idx % BLOCK_TILE_N;
      __pipeline_memcpy_async(
          &sB[stage][r][c], &B[(k_tile * BLOCK_TILE_K + r) * N + col_base + c],
          sizeof(float));
    }
  };

  // ────────────────────────────────────────
  //  Helper: compute from buffer `stage`
  // ────────────────────────────────────────
  auto compute_tile = [&](int stage) {
#pragma unroll
    for (int kk = 0; kk < BLOCK_TILE_K; kk++) {
      float a_frag[THREAD_TILE_M];
      float b_frag[THREAD_TILE_N];

#pragma unroll
      for (int tm = 0; tm < THREAD_TILE_M; tm++) {
        a_frag[tm] = sA[stage][ty * THREAD_TILE_M + tm][kk];
      }
#pragma unroll
      for (int tn = 0; tn < THREAD_TILE_N; tn++) {
        b_frag[tn] = sB[stage][kk][tx * THREAD_TILE_N + tn];
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
//  Prologue: fill the pipeline (issue STAGES async loads)
//
//  After this, the commit queue looks like:
//    [Group 0] [Group 1] [Group 2]
//     oldest                newest
//
//  Stage assignment:
//    s0 ← tile 0 (Group 0)
//    s1 ← tile 1 (Group 1)
//    s2 ← tile 2 (Group 2)
// ══════════════════════════════════════════
#pragma unroll
  for (int s = 0; s < STAGES; s++) {
    if (s < NUM_TILES_K) {
      async_load_tile(s, s);
    }
    __pipeline_commit();
  }

  // ══════════════════════════════════════════
  //  Main Loop
  //
  //  Critical ordering per iteration:
  //    1. WAIT    — oldest in-flight group completes
  //    2. COMPUTE — use the now-ready buffer
  //    3. LOAD    — reuse that buffer for the next tile
  //
  //  This order is essential: we must consume the data
  //  BEFORE overwriting the buffer with a new async load.
  //
  //  Trace (STAGES=3):
  //    k=0: wait G0, compute s0 (tile0), load s0←tile3, commit G3
  //    k=1: wait G1, compute s1 (tile1), load s1←tile4, commit G4
  //    k=2: wait G2, compute s2 (tile2), load s2←tile5, commit G5
  //    k=3: wait G3, compute s0 (tile3), load s0←tile6, commit G6
  //    ...
  // ══════════════════════════════════════════
  for (int k = 0; k < NUM_TILES_K - STAGES; k++) {
    int stage = k % STAGES;

    // 1. Wait for the oldest group to complete
    //    wait_prior(STAGES-1): "keep (STAGES-1) most recent groups in-flight,
    //    ensure everything older is done"
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();

    // 2. Compute from the now-ready buffer
    compute_tile(stage);

    // 3. This buffer is consumed — safe to reuse for next tile
    //    syncthreads ensures all threads finished reading before overwrite
    __syncthreads();
    async_load_tile(stage, k + STAGES);
    __pipeline_commit();
  }

// ══════════════════════════════════════════
//  Epilogue: drain the pipeline
//  No more loads to issue — just wait and compute remaining STAGES tiles
// ══════════════════════════════════════════
#pragma unroll
  for (int s = 0; s < STAGES; s++) {
    int stage = (NUM_TILES_K - STAGES + s) % STAGES;

    __pipeline_wait_prior(STAGES - 1 - s);
    __syncthreads();

    compute_tile(stage);
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
void launch_cp_async(const float* d_A, const float* d_B, float* d_C) {
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid(N / BLOCK_TILE_N, M / BLOCK_TILE_M);
  sgemm_cp_async<<<grid, block>>>(d_A, d_B, d_C);
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main() {
  printf("=== SGEMM cp.async (%d-Stage Pipeline) ===\n", STAGES);
  printf("M=%d, N=%d, K=%d, Stages=%d\n", M, N, K, STAGES);
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

  benchmark_kernel("cp.async (Multi-Stage Pipeline)", launch_cp_async, d_A, d_B,
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
