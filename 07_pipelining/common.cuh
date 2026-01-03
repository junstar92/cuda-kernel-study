#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// ─────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────
constexpr int M = 2048;
constexpr int N = 2048;
constexpr int K = 2048;

constexpr int BLOCK_TILE_M = 128;
constexpr int BLOCK_TILE_N = 128;
constexpr int BLOCK_TILE_K = 8;

// Thread block: each thread loads one element per tile dimension
constexpr int BLOCK_DIM_X = 16;  // threads along N
constexpr int BLOCK_DIM_Y = 16;  // threads along M

// Each thread computes a TM x TN sub-tile
constexpr int THREAD_TILE_M = BLOCK_TILE_M / BLOCK_DIM_Y;  // 8
constexpr int THREAD_TILE_N = BLOCK_TILE_N / BLOCK_DIM_X;  // 8

constexpr int NUM_TILES_K = K / BLOCK_TILE_K;

// ─────────────────────────────────────────────
//  Error checking
// ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

// ─────────────────────────────────────────────
//  Utilities
// ─────────────────────────────────────────────
inline void init_matrix(float* mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    mat[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  }
}

inline void zero_matrix(float* mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    mat[i] = 0.0f;
  }
}

// Naive CPU GEMM for verification (only checks a subset for speed)
inline void cpu_gemm_ref(const float* A, const float* B, float* C_ref, int m,
                         int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int p = 0; p < k; p++) {
        sum += A[i * k + p] * B[p * n + j];
      }
      C_ref[i * n + j] = sum;
    }
  }
}

inline bool verify(const float* C_gpu, const float* C_ref, int m, int n,
                   float tol = 1e-2f) {
  int errors = 0;
  for (int i = 0; i < m * n; i++) {
    float diff = fabsf(C_gpu[i] - C_ref[i]);
    if (diff > tol * fabsf(C_ref[i]) + tol) {
      if (errors < 5) {
        fprintf(stderr, "  Mismatch at [%d]: gpu=%.6f ref=%.6f diff=%.6f\n", i,
                C_gpu[i], C_ref[i], diff);
      }
      errors++;
    }
  }
  if (errors > 0) {
    fprintf(stderr, "  Total mismatches: %d / %d\n", errors, m * n);
  }
  return errors == 0;
}

// ─────────────────────────────────────────────
//  Benchmark harness
// ─────────────────────────────────────────────
struct BenchResult {
  float avg_ms;
  bool correct;
};

template <typename KernelFunc>
BenchResult benchmark_kernel(const char* name, KernelFunc launch_fn,
                             const float* d_A, const float* d_B, float* d_C,
                             float* h_C, const float* h_C_ref, int warmup = 5,
                             int repeat = 20) {
  // Warmup
  for (int i = 0; i < warmup; i++) {
    launch_fn(d_A, d_B, d_C);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed runs
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warm-up
  for (int i = 0; i < 10; i++) {
    launch_fn(d_A, d_B, d_C);
  }

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; i++) {
    launch_fn(d_A, d_B, d_C);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  float avg_ms = total_ms / repeat;

  // Verify
  CUDA_CHECK(
      cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = verify(h_C, h_C_ref, M, N);

  float gflops = 2.0f * M * N * K / (avg_ms * 1e6f);
  printf("[%s]\n", name);
  printf("  Avg time : %.3f ms\n", avg_ms);
  printf("  GFLOPS   : %.1f\n", gflops);
  printf("  Correct  : %s\n", ok ? "YES" : "NO");
  printf("\n");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return {avg_ms, ok};
}
