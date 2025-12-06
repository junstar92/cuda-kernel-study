/*
 * This file provides several shared memory access patterns designed to
 * demonstrate when shared memory bank conflicts occur and when they do not.
 *
 * Each kernel implements a different access pattern (linear, strided,
 * irregular, broadcast, etc.), and you can use NVIDIA Nsight Compute
 * to examine metrics such as:
 *
 *   - l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
 *   - smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st
 *
 * These metrics will allow you to analyze how each pattern results
 * in either conflict-free behavior, 2-way conflicts, or broadcast loads.
 */
#include <cuda_runtime.h>

#include <cstdio>

#define CUDA_CHECK(x)                                      \
  do {                                                     \
    cudaError_t err__ = (x);                               \
    if (err__ != cudaSuccess) {                            \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err__));                   \
      exit(1);                                             \
    }                                                      \
  } while (0)

// =====================================================================
// 1. Linear addressing with a stride of one 32-bit word (no bank conflict)
//    Each thread accesses smem[tid]. Because float is 4 bytes, each
//    thread maps to a distinct bank.
// =====================================================================
__global__ void kernel_stride1(float* out) {
  __shared__ float smem[128];

  int tid = threadIdx.x;

  smem[tid] = (float)tid;
  __syncthreads();

  int idx = tid;  // stride = 1 word
  float v = smem[idx];

  out[tid] = v;  // prevents compiler elimination
}

// =====================================================================
// 2. Linear addressing with a stride of two 32-bit words (2-way conflict)
//    Access pattern: smem[0], smem[2], smem[4], ... smem[62]
//    This produces a 2-way conflict because every two threads map to
//    the same bank.
// =====================================================================
__global__ void kernel_stride2(float* out) {
  __shared__ float smem[128];

  int tid = threadIdx.x;

  smem[tid * 2] = (float)tid;
  __syncthreads();

  int idx = tid * 2;  // stride = 2 words → 2-way conflict
  float v = smem[idx];

  out[tid] = v;
}

// =====================================================================
// 3. Linear addressing with a stride of three 32-bit words (no conflict)
//    Access pattern: smem[0], smem[3], smem[6], ... smem[93]
//    Since gcd(3, 32) = 1, the access pattern cycles through all banks
//    without collision.
// =====================================================================
__global__ void kernel_stride3(float* out) {
  __shared__ float smem[128];

  int tid = threadIdx.x;

  smem[tid * 3] = (float)tid;
  __syncthreads();

  int idx = tid * 3;  // stride = 3 words → conflict-free
  float v = smem[idx];

  out[tid] = v;
}

// =====================================================================
// 4. Irregular access: conflict-free random permutation
//    Access index = (tid * 13) % 32
//    Since 13 and 32 are coprime, this permutation maps threads to
//    all 32 banks exactly once → no conflicts.
// =====================================================================
__global__ void kernel_irregular_perm(float* out) {
  __shared__ float smem[32];

  int tid = threadIdx.x;

  if (tid < 32) smem[tid] = (float)tid;
  __syncthreads();

  int idx = (tid * 13) % 32;  // pseudo-random but permutation
  float v = smem[idx];

  out[tid] = v;
}

// 5. Irregular access with a specific mapping:
//    Mapping pattern (thread -> word index):
//      2  -> 3
//      3,4,6,7,9 -> 5   (all broadcast from word 5 in bank 5)
//      5  -> 4
//      10 -> 9
//      13 -> 14
//      14 -> 13
//      26 -> 29
//      29 -> 26
//    All other threads: tid -> tid
//
//    Threads 3,4,6,7,9 reading the same word 5 form a broadcast in bank 5;
//    the rest either access unique words or form pairwise permutations,
//    so overall this pattern is conflict-free.
__global__ void kernel_irregular_same_word_in_bank5(float* out) {
  __shared__ float smem[32];

  int tid = threadIdx.x;

  if (tid < 32) smem[tid] = (float)tid;
  __syncthreads();

  int idx = tid;  // default: thread x -> word x

  if (tid == 2)
    idx = 3;
  else if (tid == 3)
    idx = 5;
  else if (tid == 4)
    idx = 5;
  else if (tid == 5)
    idx = 4;
  else if (tid == 6)
    idx = 5;
  else if (tid == 7)
    idx = 5;
  else if (tid == 9)
    idx = 5;
  else if (tid == 10)
    idx = 9;
  else if (tid == 13)
    idx = 14;
  else if (tid == 14)
    idx = 13;
  else if (tid == 26)
    idx = 29;
  else if (tid == 29)
    idx = 26;

  float v = smem[idx];
  out[tid] = v;
}

// 6. Partial broadcast pattern:
//    Mapping pattern (thread -> word index):
//      0–6   -> 12
//      7     -> 20
//      8–14  -> 12
//      15    -> 20
//      16    -> 12
//      17–22 -> 20
//      23    -> 12
//      24–25 -> 20
//      26    -> 12
//      27–28 -> 20
//      29    -> 12
//      30–31 -> 20
//
//    This creates two broadcast groups: one on word 12 (bank 12),
//    and one on word 20 (bank 20). Each bank is only accessed at a
//    single address, so all accesses are served as broadcasts and
//    remain conflict-free.
__global__ void kernel_broadcast(float* out) {
  __shared__ float smem[32];

  int tid = threadIdx.x;

  // Initialize the two words that will be broadcast
  if (tid == 0) {
    smem[12] = 12.0f;
    smem[20] = 20.0f;
  }
  __syncthreads();

  int idx;
  if (tid <= 6)
    idx = 12;
  else if (tid == 7)
    idx = 20;
  else if (tid <= 14)
    idx = 12;  // 8–14
  else if (tid == 15)
    idx = 20;
  else if (tid == 16)
    idx = 12;
  else if (tid <= 22)
    idx = 20;  // 17–22
  else if (tid == 23)
    idx = 12;
  else if (tid <= 25)
    idx = 20;  // 24–25
  else if (tid == 26)
    idx = 12;
  else if (tid <= 28)
    idx = 20;  // 27–28
  else if (tid == 29)
    idx = 12;
  else
    idx = 20;  // 30–31

  float v = smem[idx];
  out[tid] = v;
}

// =====================================================================
int main() {
  const int WARPSIZE = 32;
  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, WARPSIZE * 6 * sizeof(float)));

  dim3 grid(1);
  dim3 block(WARPSIZE);

  kernel_stride1<<<grid, block>>>(d_out + WARPSIZE * 0);
  kernel_stride2<<<grid, block>>>(d_out + WARPSIZE * 1);
  kernel_stride3<<<grid, block>>>(d_out + WARPSIZE * 2);
  kernel_irregular_perm<<<grid, block>>>(d_out + WARPSIZE * 3);
  kernel_irregular_same_word_in_bank5<<<grid, block>>>(d_out + WARPSIZE * 4);
  kernel_broadcast<<<grid, block>>>(d_out + WARPSIZE * 5);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
