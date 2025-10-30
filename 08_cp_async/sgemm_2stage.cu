#include <cuda_runtime.h>
#include <torch/extension.h>

template <int BLOCK_SIZE = 32>
__global__ void sgemm_shmem_naive_kernel(float* __restrict__ c_matrix,
                                         float const* __restrict__ a_matrix,
                                         float const* __restrict__ b_matrix,
                                         int const M, int const N,
                                         int const K) {
  a_matrix += BLOCK_SIZE * blockIdx.y * K;
  b_matrix += BLOCK_SIZE * blockIdx.x;
  c_matrix += BLOCK_SIZE * blockIdx.y * N + BLOCK_SIZE * blockIdx.x;

  __shared__ float smem_block_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float smem_block_b[BLOCK_SIZE][BLOCK_SIZE];

  float acc{0.0f};
#pragma unroll
  for (int k = 0; k < K; k += BLOCK_SIZE) {
    smem_block_a[threadIdx.y][threadIdx.x] =
        a_matrix[threadIdx.y * K + threadIdx.x];
    smem_block_b[threadIdx.y][threadIdx.x] =
        b_matrix[threadIdx.y * N + threadIdx.x];
    __syncthreads();

    a_matrix += BLOCK_SIZE;
    b_matrix += BLOCK_SIZE * N;

    float partial_acc{0.0f};
#pragma unroll
    for (int k_in_block = 0; k_in_block < BLOCK_SIZE; k_in_block++) {
      partial_acc += smem_block_a[threadIdx.y][k_in_block] *
                     smem_block_b[k_in_block][threadIdx.x];
    }
    acc += partial_acc;
    __syncthreads();
  }

  c_matrix[threadIdx.y * N + threadIdx.x] = acc;
}

// simt mma, instruction shape: 1x1x1
// only tested with BLOCK MNK = (128, 128, 8), WARP MNK = (32, 64, 8)
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K, int SMEM_PADDING_A = 0>
__global__ void sgemm_warptiling_kernel(float* __restrict__ c_matrix,
                                        float const* __restrict__ a_matrix,
                                        float const* __restrict__ b_matrix,
                                        int const M, int const N, int const K) {
  static_assert(BLOCK_K == WARP_K);
  extern __shared__ float smem[];

  float* smem_block_a = smem;
  float* smem_block_b = smem + (BLOCK_M + SMEM_PADDING_A) * BLOCK_K;

  a_matrix += BLOCK_M * blockIdx.y * K;
  b_matrix += BLOCK_N * blockIdx.x;

  constexpr int warp_size = 32;
  constexpr int num_threads =
      (BLOCK_M / WARP_M) * (BLOCK_N / WARP_N) * warp_size;

  int const warp_id = threadIdx.x / warp_size;
  int const lane_id = threadIdx.x % warp_size;
  constexpr int num_warp_m = BLOCK_M / WARP_M;
  constexpr int num_warp_n = BLOCK_N / WARP_N;

  // define warp-level variables
  constexpr int warp_num_threads_m = (WARP_M > WARP_N) ? 8 : 4;
  constexpr int warp_num_threads_n = warp_size / warp_num_threads_m;
  constexpr int thread_tile_m = WARP_M / warp_num_threads_m;
  constexpr int thread_tile_n = WARP_N / warp_num_threads_n;
  int const warp_num_threads_m_idx = lane_id / warp_num_threads_n;
  int const warp_num_threads_n_idx = lane_id % warp_num_threads_n;
  static_assert(!(WARP_M % warp_num_threads_m) &&
                !(WARP_N % warp_num_threads_n));

  float frag_a[thread_tile_m] = {0.f};
  float frag_b[thread_tile_n] = {0.f};
  float accum[thread_tile_m][thread_tile_n] = {0.f};

  // main loop
  int gemm_k_iterations = K / BLOCK_K;
  for (; gemm_k_iterations > 0; gemm_k_iterations--) {
    {
      // load a from global memory to shared memory
      constexpr int load_a_iterations = BLOCK_M * BLOCK_K / num_threads;
      int col_idx = threadIdx.x % BLOCK_K;
      int row_idx = threadIdx.x / BLOCK_K;
      constexpr int row_stride = num_threads / BLOCK_K;
#pragma unroll
      for (int i = 0; i < load_a_iterations; i++) {
        smem_block_a[col_idx * (BLOCK_M + SMEM_PADDING_A) + row_idx] =
            a_matrix[row_idx * K + col_idx];
        row_idx += row_stride;
      }
    }
    {
      // load b from global memory to shared memory
      constexpr int load_b_iterations = BLOCK_K * BLOCK_N / num_threads;
      int col_idx = threadIdx.x % BLOCK_N;
      int row_idx = threadIdx.x / BLOCK_N;
      constexpr int row_stride = num_threads / BLOCK_N;
#pragma unroll
      for (int i = 0; i < load_b_iterations; i++) {
        smem_block_b[row_idx * BLOCK_N + col_idx] =
            b_matrix[row_idx * N + col_idx];
        row_idx += row_stride;
      }
    }
    __syncthreads();

    // advance matrix a and b
    a_matrix += BLOCK_K;
    b_matrix += BLOCK_K * N;

    // warp-level matrix multiply-accumulate
    static_assert(BLOCK_K / WARP_K == 1);
    constexpr int warp_gemm_iterations = WARP_K;
    float* smem_warp_a = smem_block_a + warp_id % num_warp_m * WARP_M;
    float* smem_warp_b = smem_block_b + warp_id / num_warp_m * WARP_N;

#pragma unroll
    for (int warp_k = 0; warp_k < warp_gemm_iterations; warp_k++) {
      // load a and b fragments from shared memory
#pragma unroll
      for (int tm = 0; tm < thread_tile_m; tm++) {
        frag_a[tm] = smem_warp_a[warp_num_threads_m_idx * thread_tile_m + tm];
      }
#pragma unroll
      for (int tn = 0; tn < thread_tile_n; tn++) {
        frag_b[tn] = smem_warp_b[warp_num_threads_n_idx * thread_tile_n + tn];
      }

      // compute mma for each thread tiles
#pragma unroll
      for (int tm = 0; tm < thread_tile_m; tm++) {
#pragma unroll
        for (int tn = 0; tn < thread_tile_n; tn++) {
          accum[tm][tn] += frag_a[tm] * frag_b[tn];
        }
      }

      // advance shared memory pointer
      smem_warp_a += (BLOCK_M + SMEM_PADDING_A);
      smem_warp_b += BLOCK_N;
    }
    __syncthreads();
  }

  // store accumulators to global memory
  c_matrix += (BLOCK_M * blockIdx.y + (warp_id % num_warp_m) * WARP_M) * N +
              (BLOCK_N * blockIdx.x + (warp_id / num_warp_m) * WARP_N);
  for (int tm = 0; tm < thread_tile_m; tm++) {
    for (int tn = 0; tn < thread_tile_n; tn++) {
      c_matrix[(warp_num_threads_m_idx * thread_tile_m + tm) * N +
               (warp_num_threads_n_idx * thread_tile_n + tn)] = accum[tm][tn];
    }
  }
}

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

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K, int SMEM_PADDING_A = 0>
__global__ void sgemm_warptiling_2stage_kernel(
    float* __restrict__ c_matrix, float const* __restrict__ a_matrix,
    float const* __restrict__ b_matrix, int const M, int const N, int const K) {
  static_assert(BLOCK_K == WARP_K);
  extern __shared__ float smem[];

  float* smem0 = smem;
  float* smem1 =
      smem + (BLOCK_M + SMEM_PADDING_A) * BLOCK_K + BLOCK_N * BLOCK_K;

  a_matrix += BLOCK_M * blockIdx.y * K;
  b_matrix += BLOCK_N * blockIdx.x;

  constexpr int warp_size = 32;
  constexpr int num_threads =
      (BLOCK_M / WARP_M) * (BLOCK_N / WARP_N) * warp_size;

  int const warp_id = threadIdx.x / warp_size;
  int const lane_id = threadIdx.x % warp_size;
  constexpr int num_warp_m = BLOCK_M / WARP_M;
  constexpr int num_warp_n = BLOCK_N / WARP_N;

  // define warp-level variables
  constexpr int warp_num_threads_m = (WARP_M > WARP_N) ? 8 : 4;
  constexpr int warp_num_threads_n = warp_size / warp_num_threads_m;
  constexpr int thread_tile_m = WARP_M / warp_num_threads_m;
  constexpr int thread_tile_n = WARP_N / warp_num_threads_n;
  int const warp_num_threads_m_idx = lane_id / warp_num_threads_n;
  int const warp_num_threads_n_idx = lane_id % warp_num_threads_n;
  static_assert(!(WARP_M % warp_num_threads_m) &&
                !(WARP_N % warp_num_threads_n));

  float frag_a[thread_tile_m] = {0.f};
  float frag_b[thread_tile_n] = {0.f};
  float accum[thread_tile_m][thread_tile_n] = {0.f};

  // pre-load first block to shared memory
  {
    float* smem_block_a = smem0;
    constexpr int load_a_iterations = BLOCK_M * BLOCK_K / num_threads;
    int col_idx = threadIdx.x % BLOCK_K;
    int row_idx = threadIdx.x / BLOCK_K;
    constexpr int row_stride = num_threads / BLOCK_K;
#pragma unroll
    for (int i = 0; i < load_a_iterations; i++) {
      cp_async<4>(&smem_block_a[col_idx * (BLOCK_M + SMEM_PADDING_A) + row_idx],
                  &a_matrix[row_idx * K + col_idx]);
      row_idx += row_stride;
    }
  }
  {
    float* smem_block_b = smem0 + (BLOCK_M + SMEM_PADDING_A) * BLOCK_K;
    constexpr int load_b_iterations = BLOCK_K * BLOCK_N / num_threads;
    int col_idx = threadIdx.x % BLOCK_N;
    int row_idx = threadIdx.x / BLOCK_N;
    constexpr int row_stride = num_threads / BLOCK_N;
#pragma unroll
    for (int i = 0; i < load_b_iterations; i++) {
      cp_async<4>(&smem_block_b[row_idx * BLOCK_N + col_idx],
                  &b_matrix[row_idx * N + col_idx]);
      row_idx += row_stride;
    }
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  // main loop
  int gemm_k_iterations = K / BLOCK_K - 1;
  int stage = 0;
  for (; gemm_k_iterations > 0; gemm_k_iterations--) {
    // preload next block asynchronously
    {
      // advance matrix a and b
      a_matrix += BLOCK_K;
      b_matrix += BLOCK_K * N;
      float* next_smem = (stage == 0) ? smem1 : smem0;
      {
        float* next_smem_block_a = next_smem;
        constexpr int load_a_iterations = BLOCK_M * BLOCK_K / num_threads;
        int col_idx = threadIdx.x % BLOCK_K;
        int row_idx = threadIdx.x / BLOCK_K;
        constexpr int row_stride = num_threads / BLOCK_K;
#pragma unroll
        for (int i = 0; i < load_a_iterations; i++) {
          cp_async<4>(&next_smem_block_a[col_idx * (BLOCK_M + SMEM_PADDING_A) +
                                         row_idx],
                      &a_matrix[row_idx * K + col_idx]);
          row_idx += row_stride;
        }
      }
      {
        float* next_smem_block_b =
            next_smem + (BLOCK_M + SMEM_PADDING_A) * BLOCK_K;
        constexpr int load_b_iterations = BLOCK_K * BLOCK_N / num_threads;
        int col_idx = threadIdx.x % BLOCK_N;
        int row_idx = threadIdx.x / BLOCK_N;
        constexpr int row_stride = num_threads / BLOCK_N;
#pragma unroll
        for (int i = 0; i < load_b_iterations; i++) {
          cp_async<4>(&next_smem_block_b[row_idx * BLOCK_N + col_idx],
                      &b_matrix[row_idx * N + col_idx]);
          row_idx += row_stride;
        }
      }
    }
    cp_async_commit();

    // select current shared memory buffer
    float* smem_block_a = (stage == 0) ? smem0 : smem1;
    float* smem_block_b = smem_block_a + (BLOCK_M + SMEM_PADDING_A) * BLOCK_K;

    // warp-level matrix multiply-accumulate
    static_assert(BLOCK_K / WARP_K == 1);
    constexpr int warp_gemm_iterations = WARP_K;
    float* smem_warp_a = smem_block_a + warp_id % num_warp_m * WARP_M;
    float* smem_warp_b = smem_block_b + warp_id / num_warp_m * WARP_N;

#pragma unroll
    for (int warp_k = 0; warp_k < warp_gemm_iterations; warp_k++) {
      // load a and b fragments from shared memory
#pragma unroll
      for (int tm = 0; tm < thread_tile_m; tm++) {
        frag_a[tm] = smem_warp_a[warp_num_threads_m_idx * thread_tile_m + tm];
      }
#pragma unroll
      for (int tn = 0; tn < thread_tile_n; tn++) {
        frag_b[tn] = smem_warp_b[warp_num_threads_n_idx * thread_tile_n + tn];
      }

      // compute mma for each thread tiles
#pragma unroll
      for (int tm = 0; tm < thread_tile_m; tm++) {
#pragma unroll
        for (int tn = 0; tn < thread_tile_n; tn++) {
          accum[tm][tn] += frag_a[tm] * frag_b[tn];
        }
      }

      // advance shared memory pointer
      smem_warp_a += (BLOCK_M + SMEM_PADDING_A);
      smem_warp_b += BLOCK_N;
    }

    // move to next stage
    stage ^= 1;

    // wait until next block is fully loaded
    cp_async_wait<0>();
    __syncthreads();
  }

  // compute the last stage
  {
    float* smem_block_a = (stage == 0) ? smem0 : smem1;
    float* smem_block_b = smem_block_a + (BLOCK_M + SMEM_PADDING_A) * BLOCK_K;

    // warp-level matrix multiply-accumulate
    static_assert(BLOCK_K / WARP_K == 1);
    constexpr int warp_gemm_iterations = WARP_K;
    float* smem_warp_a = smem_block_a + warp_id % num_warp_m * WARP_M;
    float* smem_warp_b = smem_block_b + warp_id / num_warp_m * WARP_N;

#pragma unroll
    for (int warp_k = 0; warp_k < warp_gemm_iterations; warp_k++) {
      // load a and b fragments from shared memory
#pragma unroll
      for (int tm = 0; tm < thread_tile_m; tm++) {
        frag_a[tm] = smem_warp_a[warp_num_threads_m_idx * thread_tile_m + tm];
      }
#pragma unroll
      for (int tn = 0; tn < thread_tile_n; tn++) {
        frag_b[tn] = smem_warp_b[warp_num_threads_n_idx * thread_tile_n + tn];
      }

      // compute mma for each thread tiles
#pragma unroll
      for (int tm = 0; tm < thread_tile_m; tm++) {
#pragma unroll
        for (int tn = 0; tn < thread_tile_n; tn++) {
          accum[tm][tn] += frag_a[tm] * frag_b[tn];
        }
      }

      // advance shared memory pointer
      smem_warp_a += (BLOCK_M + SMEM_PADDING_A);
      smem_warp_b += BLOCK_N;
    }
  }

  // store accumulators to global memory
  c_matrix += (BLOCK_M * blockIdx.y + (warp_id % num_warp_m) * WARP_M) * N +
              (BLOCK_N * blockIdx.x + (warp_id / num_warp_m) * WARP_N);
  for (int tm = 0; tm < thread_tile_m; tm++) {
    for (int tn = 0; tn < thread_tile_n; tn++) {
      c_matrix[(warp_num_threads_m_idx * thread_tile_m + tm) * N +
               (warp_num_threads_n_idx * thread_tile_n + tn)] = accum[tm][tn];
    }
  }
}

template <int VER>
torch::Tensor launch_sgemm(torch::Tensor const input, torch::Tensor const other,
                           torch::Tensor const output) {
  int const M = input.size(0);
  int const K = other.size(0);
  int const N = other.size(1);

  if constexpr (VER == 0) {
    constexpr int BLOCK_SIZE = 32;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    sgemm_shmem_naive_kernel<BLOCK_SIZE>
        <<<grid, block>>>(output.data_ptr<float>(), input.data_ptr<float>(),
                          other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 1) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 8;
    constexpr int WARP_M = 32;
    constexpr int WARP_N = 64;
    constexpr int WARP_K = 8;

    dim3 block((BLOCK_M / WARP_M) * (BLOCK_N / WARP_N) * 32);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    size_t shared_mem_size =
        (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(float);

    sgemm_warptiling_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K>
        <<<grid, block, shared_mem_size>>>(output.data_ptr<float>(),
                                           input.data_ptr<float>(),
                                           other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 2) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 8;
    constexpr int WARP_M = 32;
    constexpr int WARP_N = 64;
    constexpr int WARP_K = 8;
    constexpr int SMEM_PADDING_A = 4;

    dim3 block((BLOCK_M / WARP_M) * (BLOCK_N / WARP_N) * 32);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    size_t shared_mem_size =
        ((BLOCK_M + SMEM_PADDING_A) * BLOCK_K + BLOCK_K * BLOCK_N) *
        sizeof(float);

    sgemm_warptiling_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K,
                            SMEM_PADDING_A><<<grid, block, shared_mem_size>>>(
        output.data_ptr<float>(), input.data_ptr<float>(),
        other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 3) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 8;
    constexpr int WARP_M = 32;
    constexpr int WARP_N = 64;
    constexpr int WARP_K = 8;
    constexpr int SMEM_PADDING_A = 4;

    dim3 block((BLOCK_M / WARP_M) * (BLOCK_N / WARP_N) * 32);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    size_t shared_mem_size =
        2 * ((BLOCK_M + SMEM_PADDING_A) * BLOCK_K + BLOCK_K * BLOCK_N) *
        sizeof(float);

    sgemm_warptiling_2stage_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N,
                                   WARP_K, SMEM_PADDING_A>
        <<<grid, block, shared_mem_size>>>(output.data_ptr<float>(),
                                           input.data_ptr<float>(),
                                           other.data_ptr<float>(), M, N, K);
  } else {
    TORCH_CHECK(false, "Unsupported version");
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgemm_shmem", &launch_sgemm<0>, py::arg("input"), py::arg("other"),
        py::arg("out"));
  m.def("sgemm_warptiling_128x128x8_32x64x8", &launch_sgemm<1>,
        py::arg("input"), py::arg("other"), py::arg("out"));
  m.def("sgemm_warptiling_128x128x8_32x64x8_padding", &launch_sgemm<2>,
        py::arg("input"), py::arg("other"), py::arg("out"));
  m.def("sgemm_warptiling_128x128x8_32x64x8_padding_2stage", &launch_sgemm<3>,
        py::arg("input"), py::arg("other"), py::arg("out"));
}