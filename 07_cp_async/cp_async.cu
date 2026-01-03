// nvcc -O3 -std=c++17 -arch=sm_80 cp_async.cu -o cp_async
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

#define CHECK_CUDA(x)                                               \
  do {                                                              \
    cudaError_t err = (x);                                          \
    if (err != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                             \
      exit(1);                                                      \
    }                                                               \
  } while (0)

constexpr int NUM_STAGES = 2;
constexpr int NUM_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int CHUNK_ELEMS = 6144;       // 24 KB per buffer (float)
constexpr int COMPUTE_ITERATIONS = 64;  // per-element FMA count

// cp.async helper function
template <int bytes>
__device__ __forceinline__ void cp_async(void* smem_dst, void const* gmem_src) {
  unsigned smem_addr =
      static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));

  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
               :
               : "r"(smem_addr), "l"(gmem_src), "n"(bytes));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  // Wait for N outstanding async copy groups
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <>
__device__ __forceinline__ void cp_async_wait<0>() {
  // Wait for all previous cp.async.commit_group operations have committed
  asm volatile("cp.async.wait_all;\n");
}

// -----------------------------------------------------
// Kernel 1) Single-stage
// -----------------------------------------------------
__global__ void single_stage_kernel(float* __restrict__ out,
                                    float const* __restrict__ in, int N) {
  extern __shared__ float smem[];  // [CHUNK_ELEMS]

  int const stride = gridDim.x * CHUNK_ELEMS;
  int base = blockIdx.x * CHUNK_ELEMS;

  float alpha = 1.0001f, beta = 0.9993f;

  while (base < N) {
    // GMEM -> SMEM (all threads collaborate)
    for (int i = threadIdx.x; i < CHUNK_ELEMS; i += blockDim.x) {
      int idx = base + i;
      smem[i] = (idx < N) ? __ldg(in + idx) : 0.f;
    }
    __syncthreads();

    // Compute on SMEM
    for (int i = threadIdx.x; i < CHUNK_ELEMS; i += blockDim.x) {
      float v = smem[i];
#pragma unroll
      for (int it = 0; it < COMPUTE_ITERATIONS; it++) {
        v = fmaf(v, alpha, beta);  // 1 FMA
      }
      smem[i] = v;
    }

    // Write results back to global memory
    for (int i = threadIdx.x; i < CHUNK_ELEMS; i += blockDim.x) {
      int idx = base + i;
      if (idx < N) out[idx] = smem[i];
    }

    base += stride;
  }
}

// -----------------------------------------------------
// Kernel 2) Double-stage (non cp.async)
//   - Uses ping-pong SMEM buffers to preload next chunk
//   - Note: Limited pipelining - all threads do load AND compute sequentially
//   - True overlap requires async load (cp.async) or dedicated producer threads
// -----------------------------------------------------
__global__ void double_stage_wo_cp_async_kernel(float* __restrict__ out,
                                                float const* __restrict__ in,
                                                int N) {
  extern __shared__ float smem[];  // [2 * CHUNK_ELEMS]
  float* smem0 = smem;
  float* smem1 = smem + CHUNK_ELEMS;

  int const stride = gridDim.x * CHUNK_ELEMS;
  int base = blockIdx.x * CHUNK_ELEMS;

  float alpha = 1.0001f, beta = 0.9993f;

  // Preload: load first stage chunk
  if (base < N) {
    for (int i = threadIdx.x; i < CHUNK_ELEMS; i += blockDim.x) {
      int idx = base + i;
      smem0[i] = (idx < N) ? __ldg(in + idx) : 0.f;
    }
  }
  __syncthreads();

  int stage = 0;
  while (base < N) {
    int next_base = base + stride;

    // Preload next chunk into alterante buffer
    if (next_base < N) {
      float* dst = (stage == 0) ? smem1 : smem0;
      for (int i = threadIdx.x; i < CHUNK_ELEMS; i += blockDim.x) {
        int idx = next_base + i;
        dst[i] = (idx < N) ? __ldg(in + idx) : 0.f;
      }
    }

    // Compute on current stage
    float* cur = (stage == 0) ? smem0 : smem1;
    for (int i = threadIdx.x; i < CHUNK_ELEMS; i += blockDim.x) {
      float v = cur[i];
#pragma unroll
      for (int it = 0; it < COMPUTE_ITERATIONS; it++) {
        v = fmaf(v, alpha, beta);
      }
      cur[i] = v;
    }

    // Write results back to global memory
    for (int i = threadIdx.x; i < CHUNK_ELEMS; i += blockDim.x) {
      int idx = base + i;
      if (idx < N) out[idx] = ((stage == 0) ? smem0 : smem1)[i];
    }

    // Move to next stage
    base += stride;
    stage ^= 1;
    __syncthreads();
  }
}

// -----------------------------------------------------
// Kernel 3) Double-stage (cp.async)
//   - Uses cp.async for asynchronous copies
//   - Ping-pong buffering
// -----------------------------------------------------
__global__ void double_stage_cp_async_kernel(float* __restrict__ out,
                                             float const* __restrict__ in,
                                             int N) {
  extern __shared__ float smem[];  // [2 * CHUNK_ELEMS]
  float* smem0 = smem;
  float* smem1 = smem + CHUNK_ELEMS;

  int const stride = gridDim.x * CHUNK_ELEMS;
  int base = blockIdx.x * CHUNK_ELEMS;

  float alpha = 1.0001f, beta = 0.9993f;

  auto async_load_chunk = [&](float* smem_dst, int g_base) {
    for (int i = threadIdx.x; i < CHUNK_ELEMS; i += blockDim.x) {
      int idx = g_base + i;
      if (idx < N) {
        cp_async<4>(smem_dst + i, in + idx);
      }
    }
    cp_async_commit();
  };

  // Preload the first stage
  if (base < N) {
    async_load_chunk(smem0, base);
    cp_async_wait<0>();  // wait for completion of the first group
  }
  __syncthreads();

  int stage = 0;
  while (base < N) {
    int next_base = base + stride;

    // Preload next chunk asynchronously
    if (next_base < N) {
      float* nxt = (stage == 0) ? smem1 : smem0;
      async_load_chunk(nxt, next_base);
    }

    // Compute on current buffer
    float* cur = (stage == 0) ? smem0 : smem1;
    for (int i = threadIdx.x; i < CHUNK_ELEMS; i += blockDim.x) {
      float v = cur[i];
#pragma unroll
      for (int it = 0; it < COMPUTE_ITERATIONS; it++) {
        v = fmaf(v, alpha, beta);
      }
      cur[i] = v;
    }

    // Write results back to global memory
    for (int i = threadIdx.x; i < CHUNK_ELEMS; i += blockDim.x) {
      int idx = base + i;
      if (idx < N) out[idx] = ((stage == 0) ? smem0 : smem1)[i];
    }

    // Move to next stage
    base += stride;
    stage ^= 1;

    // Wait until next chunk is fully loaded
    if (next_base < N) {
      cp_async_wait<0>();
    }
    __syncthreads();
  }
}

// -----------------------------------------------------
// Kernel 4) CUDA Pipeline API with Producer/Consumer
//   - Uses cuda::pipeline for structured async operations
//   - Multi-stage pipeline (e.g., 2 stages)
// -----------------------------------------------------
__global__ void cuda_pipeline_kernel(float* __restrict__ out,
                                     float const* __restrict__ in, int N) {
  extern __shared__ float smem[];  // [NUM_STAGES * CHUNK_ELEMS]

  int const stride = gridDim.x * CHUNK_ELEMS;
  int base = blockIdx.x * CHUNK_ELEMS;

  float alpha = 1.0001f, beta = 0.9993f;

  // Create thread block and thread block tile
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<WARP_SIZE>(block);

  // Create a pipeline
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES>
      shared_state;
  auto pipeline = cuda::make_pipeline(block, &shared_state);
}

// -----------------------------------------------------
// CPU reference implementation
// -----------------------------------------------------
void cpu_reference(float* out, float const* in, int N) {
  float alpha = 1.0001f, beta = 0.9993f;
  for (int i = 0; i < N; i++) {
    float v = in[i];
    for (int it = 0; it < COMPUTE_ITERATIONS; it++) {
      v = fmaf(v, alpha, beta);
    }
    out[i] = v;
  }
}

// -----------------------------------------------------
// Verification helper
// -----------------------------------------------------
bool verify_results(float const* ref, float const* test, int N,
                    char const* kernel_name, float tol = 1e-4f) {
  int errors = 0;
  int max_print = 5;

  for (int i = 0; i < N; i++) {
    float diff = fabsf(ref[i] - test[i]);
    float rel_err = diff / (fabsf(ref[i]) + 1e-8f);

    if (rel_err > tol) {
      if (errors < max_print) {
        printf("  [%s] Mismatch at %d: ref=%.6f, test=%.6f, rel_err=%.6e\n",
               kernel_name, i, ref[i], test[i], rel_err);
      }
      errors++;
    }
  }

  if (errors == 0) {
    printf("✓ [%s] PASSED\n", kernel_name);
    return true;
  } else {
    printf("✗ [%s] FAILED: %d/%d errors\n", kernel_name, errors, N);
    return false;
  }
}

// -----------------------------------------------------
// Benchmark helper
// -----------------------------------------------------
float benchmark_kernel(void (*kernel)(float*, float const*, int), float* d_out,
                       float const* d_in, int N, int smem_bytes, int num_blocks,
                       char const* kernel_name, int warmup = 2,
                       int iterations = 10) {
  // Initialize output buffer to zero before benchmarking
  CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

  // Warmup
  for (int i = 0; i < warmup; i++) {
    kernel<<<num_blocks, NUM_THREADS, smem_bytes>>>(d_out, d_in, N);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timing
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    kernel<<<num_blocks, NUM_THREADS, smem_bytes>>>(d_out, d_in, N);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  float avg_ms = ms / iterations;

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  // Calculate bandwidth and throughput
  size_t bytes_read = N * sizeof(float);
  size_t bytes_write = N * sizeof(float);
  size_t total_bytes = bytes_read + bytes_write;
  float bandwidth_gb = (total_bytes / 1e9f) / (avg_ms / 1e3f);

  size_t total_ops = (size_t)N * COMPUTE_ITERATIONS;           // FMAs
  float gflops = (2.0f * total_ops / 1e9f) / (avg_ms / 1e3f);  // FMA = 2 ops

  printf("  [%s]\n", kernel_name);
  printf("    Time:      %.3f ms\n", avg_ms);
  printf("    Bandwidth: %.2f GB/s\n", bandwidth_gb);
  printf("    Throughput: %.2f GFLOPS\n", gflops);

  return avg_ms;
}

// -----------------------------------------------------
// Main
// -----------------------------------------------------
int main(int argc, char** argv) {
  int N = (argc > 1) ? atoi(argv[1]) : (1 << 24);  // Default: 16M elements

  printf("=================================================\n");
  printf("CUDA Pipelining Kernels Benchmark\n");
  printf("=================================================\n");
  printf("Array size: %d elements (%.2f MB)\n", N, N * sizeof(float) / 1e6f);
  printf("Chunk size: %d elements (%.2f KB)\n", CHUNK_ELEMS,
         CHUNK_ELEMS * sizeof(float) / 1e3f);
  printf("Compute iterations per element: %d\n", COMPUTE_ITERATIONS);
  printf("Block size: %d threads\n", NUM_THREADS);
  printf("Number of stages: %d\n", NUM_STAGES);
  printf("\n");

  // Allocate host memory
  float* h_in = new float[N];
  float* h_out = new float[N];
  float* h_ref = new float[N];

  // Initialize input
  for (int i = 0; i < N; i++) {
    h_in[i] = sinf((float)i * 0.001f) + 1.0f;
  }

  // Compute CPU reference
  printf("Computing CPU reference...\n");
  cpu_reference(h_ref, h_in, N);
  printf("CPU reference done.\n\n");

  // Allocate device memory
  float *d_in, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

  // Calculate grid size
  int num_blocks = (N + CHUNK_ELEMS - 1) / CHUNK_ELEMS;
  num_blocks = std::min(num_blocks, 256);  // Limit for better profiling

  printf("Grid configuration: %d blocks\n\n", num_blocks);

  // =================================================
  // Benchmark Kernel 1: Single-stage
  // =================================================
  printf("-------------------------------------------------\n");
  printf("Kernel 1: Single-stage\n");
  printf("-------------------------------------------------\n");
  int smem1 = CHUNK_ELEMS * sizeof(float);
  float time1 = benchmark_kernel(single_stage_kernel, d_out, d_in, N, smem1,
                                 num_blocks, "Single-stage");

  CHECK_CUDA(
      cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  verify_results(h_ref, h_out, N, "Single-stage");
  printf("\n");

  // =================================================
  // Benchmark Kernel 2: Double-stage (non cp.async)
  // =================================================
  printf("-------------------------------------------------\n");
  printf("Kernel 2: Double-stage (non cp.async)\n");
  printf("-------------------------------------------------\n");
  int smem2 = 2 * CHUNK_ELEMS * sizeof(float);
  float time2 = benchmark_kernel(double_stage_wo_cp_async_kernel, d_out, d_in,
                                 N, smem2, num_blocks, "Double-stage");

  CHECK_CUDA(
      cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  verify_results(h_ref, h_out, N, "Double-stage (non cp.async)");
  printf("\n");

  // =================================================
  // Benchmark Kernel 3: Double-stage (cp.async)
  // =================================================
  printf("-------------------------------------------------\n");
  printf("Kernel 3: Double-stage (cp.async)\n");
  printf("-------------------------------------------------\n");
  int smem3 = 2 * CHUNK_ELEMS * sizeof(float);
  float time3 = benchmark_kernel(double_stage_cp_async_kernel, d_out, d_in, N,
                                 smem3, num_blocks, "Double-stage (cp.async)");

  CHECK_CUDA(
      cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  verify_results(h_ref, h_out, N, "Double-stage (cp.async)");
  printf("\n");

  // =================================================
  // Benchmark Kernel 4: CUDA Pipeline API
  // =================================================
  // printf("-------------------------------------------------\n");
  // printf("Kernel 4: CUDA Pipeline API\n");
  // printf("-------------------------------------------------\n");
  // int smem4 = NUM_STAGES * CHUNK_ELEMS * sizeof(float);
  // float time4 = benchmark_kernel(cuda_pipeline_kernel, d_out, d_in, N, smem4,
  //                                num_blocks, "CUDA Pipeline");

  // CHECK_CUDA(
  //     cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  // verify_results(h_ref, h_out, N, "CUDA Pipeline");
  // printf("\n");

  // =================================================
  // Performance Summary
  // =================================================
  printf("=================================================\n");
  printf("Performance Summary (Speedup vs Kernel 1)\n");
  printf("=================================================\n");
  printf("Kernel 1 (single-stage):                        %.3f ms (1.00x)\n",
         time1);
  printf("Kernel 2 (double-stage w/o cp.async):           %.3f ms (%.2fx)\n",
         time2, time1 / time2);
  printf("Kernel 3 (double-stage w/ cp.async):            %.3f ms (%.2fx)\n",
         time3, time1 / time3);
  // printf("Kernel 4 (CUDA Pipeline):          %.3f ms (%.2fx)\n", time4,
  //        time1 / time4);
  printf("=================================================\n");

  // Cleanup
  delete[] h_in;
  delete[] h_out;
  delete[] h_ref;
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));

  return 0;
}