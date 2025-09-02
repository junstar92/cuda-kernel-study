#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void sgemm_naive_kernel(float *__restrict__ c_matrix,
                                   float const *__restrict__ a_matrix,
                                   float const *__restrict__ b_matrix,
                                   int const M, int const N, int const K) {
  int const m = blockDim.y * blockIdx.y + threadIdx.y;
  int const n = blockDim.x * blockIdx.x + threadIdx.x;

  if (m < M && n < N) {
    float acc{0.0f};
    for (int k = 0; k < K; k++) {
      acc += a_matrix[m * K + k] * b_matrix[k * N + n];
    }
    c_matrix[m * N + n] = acc;
  }
}

template <int BLOCK_SIZE = 32>
__global__ void sgemm_shmem_naive_kernel(float *__restrict__ c_matrix,
                                         float const *__restrict__ a_matrix,
                                         float const *__restrict__ b_matrix,
                                         int const M, int const N,
                                         int const K) {
  a_matrix += BLOCK_SIZE * blockIdx.y * K;
  b_matrix += BLOCK_SIZE * blockIdx.x;
  c_matrix += BLOCK_SIZE * blockIdx.y * N + BLOCK_SIZE * blockIdx.x;

  __shared__ float a_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float b_block[BLOCK_SIZE][BLOCK_SIZE];

  float acc{0.0f};
#pragma unroll
  for (int k = 0; k < K; k += BLOCK_SIZE) {
    a_block[threadIdx.y][threadIdx.x] = a_matrix[threadIdx.y * K + threadIdx.x];
    b_block[threadIdx.y][threadIdx.x] = b_matrix[threadIdx.y * N + threadIdx.x];
    __syncthreads();

    a_matrix += BLOCK_SIZE;
    b_matrix += BLOCK_SIZE * N;

    float partial_acc{0.0f};
#pragma unroll
    for (int k_in_block = 0; k_in_block < BLOCK_SIZE; k_in_block++) {
      partial_acc +=
          a_block[threadIdx.y][k_in_block] * b_block[k_in_block][threadIdx.x];
    }
    acc += partial_acc;
    __syncthreads();
  }

  c_matrix[threadIdx.y * N + threadIdx.x] = acc;
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M>
__global__ void sgemm_1d_blocktile_kernel(float *__restrict__ c_matrix,
                                          float const *__restrict__ a_matrix,
                                          float const *__restrict__ b_matrix,
                                          int const M, int const N,
                                          int const K) {
  extern __shared__ float smem[];
  float *a_block = smem;
  float *b_block = smem + BLOCK_M * BLOCK_K;

  a_matrix += BLOCK_M * blockIdx.y * K;
  b_matrix += BLOCK_N * blockIdx.x;
  c_matrix += BLOCK_M * blockIdx.y * N + BLOCK_N * blockIdx.x;

  int const thread_row = threadIdx.x / BLOCK_N;
  int const thread_col = threadIdx.x % BLOCK_N;

  int constexpr num_threads = (BLOCK_M * BLOCK_N) / THREAD_M;
  int const a_load_idx = threadIdx.x;
  int const b_load_idx = threadIdx.x;

  float acc[THREAD_M] = {0.0f};
  for (int k = 0; k < K; k += BLOCK_K) {
    int constexpr a_elements = BLOCK_M * BLOCK_K;
    int constexpr loads_per_thread_a = a_elements / num_threads;
#pragma unroll
    for (int i = 0; i < loads_per_thread_a; i++) {
      int load_idx = a_load_idx + i * num_threads;
      int a_row = load_idx / BLOCK_K;
      int a_col = load_idx % BLOCK_K;

      a_block[a_row * BLOCK_K + a_col] = a_matrix[a_row * K + a_col];
    }

    int constexpr b_elements = BLOCK_K * BLOCK_N;
    int constexpr loads_per_thread_b = b_elements / num_threads;
#pragma unroll
    for (int i = 0; i < loads_per_thread_b; i++) {
      int load_idx = b_load_idx + i * num_threads;
      int b_row = load_idx / BLOCK_N;
      int b_col = load_idx % BLOCK_N;

      b_block[b_row * BLOCK_N + b_col] = b_matrix[b_row * N + b_col];
    }
    __syncthreads();

    a_matrix += BLOCK_K;
    b_matrix += BLOCK_K * N;

#pragma unroll
    for (int inner_k = 0; inner_k < BLOCK_K; inner_k++) {
      float b_elem = b_block[inner_k * BLOCK_N + thread_col];
#pragma unroll
      for (int thread_m = 0; thread_m < THREAD_M; thread_m++) {
        acc[thread_m] +=
            a_block[(thread_row * THREAD_M + thread_m) * BLOCK_K + inner_k] *
            b_elem;
      }
    }
    __syncthreads();
  }

  for (int thread_m = 0; thread_m < THREAD_M; thread_m++) {
    c_matrix[(thread_row * THREAD_M + thread_m) * N + thread_col] =
        acc[thread_m];
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void sgemm_2d_blocktile_kernel(float *__restrict__ c_matrix,
                                          float const *__restrict__ a_matrix,
                                          float const *__restrict__ b_matrix,
                                          int const M, int const N,
                                          int const K) {
  extern __shared__ float smem[];
  float *a_block = smem;
  float *b_block = smem + BLOCK_M * BLOCK_K;

  a_matrix += BLOCK_M * blockIdx.y * K;
  b_matrix += BLOCK_N * blockIdx.x;
  c_matrix += BLOCK_M * blockIdx.y * N + BLOCK_N * blockIdx.x;

  int constexpr num_threads = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
  int const thread_row = threadIdx.x / (BLOCK_N / THREAD_N);
  int const thread_col = threadIdx.x % (BLOCK_N / THREAD_N);

  float acc[THREAD_M][THREAD_N] = {0.0f};
  float thread_m[THREAD_M] = {0.0f};
  float thread_n[THREAD_N] = {0.0f};
  for (int k = 0; k < K; k += BLOCK_K) {
    int constexpr a_elements = BLOCK_M * BLOCK_K;
    int constexpr loads_per_thread_a = a_elements / num_threads;
#pragma unroll
    for (int i = 0; i < loads_per_thread_a; i++) {
      int load_idx = threadIdx.x + i * num_threads;
      int a_row = load_idx / BLOCK_K;
      int a_col = load_idx % BLOCK_K;

      a_block[a_row * BLOCK_K + a_col] = a_matrix[a_row * K + a_col];
    }

    int constexpr b_elements = BLOCK_K * BLOCK_N;
    int constexpr loads_per_thread_b = b_elements / num_threads;
#pragma unroll
    for (int i = 0; i < loads_per_thread_b; i++) {
      int load_idx = threadIdx.x + i * num_threads;
      int b_row = load_idx / BLOCK_N;
      int b_col = load_idx % BLOCK_N;

      b_block[b_row * BLOCK_N + b_col] = b_matrix[b_row * N + b_col];
    }
    __syncthreads();

    a_matrix += BLOCK_K;
    b_matrix += BLOCK_K * N;

#pragma unroll
    for (int inner_k = 0; inner_k < BLOCK_K; inner_k++) {
#pragma unroll
      for (int tm = 0; tm < THREAD_M; tm++) {
        thread_m[tm] =
            a_block[(thread_row * THREAD_M + tm) * BLOCK_K + inner_k];
      }
#pragma unroll
      for (int tn = 0; tn < THREAD_N; tn++) {
        thread_n[tn] = b_block[inner_k * BLOCK_N + thread_col * THREAD_N + tn];
      }

#pragma unroll
      for (int tm = 0; tm < THREAD_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < THREAD_N; tn++) {
          acc[tm][tn] += thread_m[tm] * thread_n[tn];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int tm = 0; tm < THREAD_M; tm++) {
#pragma unroll
    for (int tn = 0; tn < THREAD_N; tn++) {
      c_matrix[(thread_row * THREAD_M + tm) * N + thread_col * THREAD_N + tn] =
          acc[tm][tn];
    }
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void sgemm_vec_2d_blocktile_kernel(
    float *__restrict__ c_matrix, float const *__restrict__ a_matrix,
    float const *__restrict__ b_matrix, int const M, int const N, int const K) {
  extern __shared__ float smem[];
  float *a_block = smem;
  float *b_block = smem + BLOCK_M * BLOCK_K;

  a_matrix += BLOCK_M * blockIdx.y * K;
  b_matrix += BLOCK_N * blockIdx.x;
  c_matrix += BLOCK_M * blockIdx.y * N + BLOCK_N * blockIdx.x;

  int constexpr num_threads = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
  int const thread_row = threadIdx.x / (BLOCK_N / THREAD_N);
  int const thread_col = threadIdx.x % (BLOCK_N / THREAD_N);

  float acc[THREAD_M][THREAD_N] = {0.0f};
  float thread_m[THREAD_M] = {0.0f};
  float thread_n[THREAD_N] = {0.0f};
  for (int k = 0; k < K; k += BLOCK_K) {
    int constexpr a_elements = BLOCK_M * BLOCK_K;
    int constexpr vectorized_loads_a =
        a_elements / 4; // loads 16 bytes per thread
    int constexpr loads_per_thread_a = vectorized_loads_a / num_threads;
#pragma unroll
    for (int i = 0; i < loads_per_thread_a; i++) {
      int load_idx = threadIdx.x + i * num_threads;
      int linear_idx = load_idx * 4;
      int a_row = linear_idx / BLOCK_K;
      int a_col = linear_idx % BLOCK_K;

      float4 tmp =
          reinterpret_cast<float4 const &>(a_matrix[a_row * K + a_col]);
      // transposed storage for better access pattern
      a_block[(a_col + 0) * BLOCK_M + a_row] = tmp.x;
      a_block[(a_col + 1) * BLOCK_M + a_row] = tmp.y;
      a_block[(a_col + 2) * BLOCK_M + a_row] = tmp.z;
      a_block[(a_col + 3) * BLOCK_M + a_row] = tmp.w;
    }

    int constexpr b_elements = BLOCK_K * BLOCK_N;
    int constexpr vectorized_loads_b = b_elements / 4;
    int constexpr loads_per_thread_b = vectorized_loads_b / num_threads;
#pragma unroll
    for (int i = 0; i < loads_per_thread_b; i++) {
      int load_idx = threadIdx.x + i * num_threads;
      int linear_idx = load_idx * 4;
      int b_row = linear_idx / BLOCK_N;
      int b_col = linear_idx % BLOCK_N;

      reinterpret_cast<float4 &>(b_block[b_row * BLOCK_N + b_col]) =
          reinterpret_cast<float4 const &>(b_matrix[b_row * N + b_col]);
    }
    __syncthreads();

    a_matrix += BLOCK_K;
    b_matrix += BLOCK_K * N;

#pragma unroll
    for (int inner_k = 0; inner_k < BLOCK_K; inner_k++) {
#pragma unroll
      for (int tm = 0; tm < THREAD_M; tm++) {
        thread_m[tm] = a_block[inner_k * BLOCK_M + thread_row * THREAD_M + tm];
      }
#pragma unroll
      for (int tn = 0; tn < THREAD_N; tn++) {
        thread_n[tn] = b_block[inner_k * BLOCK_N + thread_col * THREAD_N + tn];
      }

#pragma unroll
      for (int tm = 0; tm < THREAD_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < THREAD_N; tn++) {
          acc[tm][tn] += thread_m[tm] * thread_n[tn];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int tm = 0; tm < THREAD_M; tm++) {
#pragma unroll
    for (int tn = 0; tn < THREAD_N; tn += 4) {
      reinterpret_cast<float4 &>(c_matrix[(thread_row * THREAD_M + tm) * N +
                                          thread_col * THREAD_N + tn]) =
          reinterpret_cast<float4 const &>(acc[tm][tn]);
    }
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
          int SKEW_A, int SKEW_B>
__global__ void sgemm_vec_2d_blocktile_pad_kernel(
    float *__restrict__ c_matrix, float const *__restrict__ a_matrix,
    float const *__restrict__ b_matrix, int const M, int const N, int const K) {
  extern __shared__ float smem[];

  // Use per-THREAD_N-group skew for B so adjacent threads do not stride by
  // exactly THREAD_N words (which causes bank conflicts). We add SKEW_B pad
  // elements after every THREAD_N columns.
  constexpr int LDA = BLOCK_M + SKEW_A;
  constexpr int GROUPS_N = BLOCK_N / THREAD_N;
  constexpr int LDB = BLOCK_N + SKEW_B * GROUPS_N;

  float *a_block = smem;
  float *b_block = a_block + LDA * BLOCK_K;

  a_matrix += BLOCK_M * blockIdx.y * K;
  b_matrix += BLOCK_N * blockIdx.x;
  c_matrix += BLOCK_M * blockIdx.y * N + BLOCK_N * blockIdx.x;

  int constexpr num_threads = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
  int const thread_row = threadIdx.x / (BLOCK_N / THREAD_N);
  int const thread_col = threadIdx.x % (BLOCK_N / THREAD_N);

  float acc[THREAD_M][THREAD_N] = {0.0f};
  float thread_m[THREAD_M] = {0.0f};
  float thread_n[THREAD_N] = {0.0f};
  for (int k = 0; k < K; k += BLOCK_K) {
    int constexpr a_elements = BLOCK_M * BLOCK_K;
    int constexpr vectorized_loads_a =
        a_elements / 4; // loads 16 bytes per thread
    int constexpr loads_per_thread_a = vectorized_loads_a / num_threads;
#pragma unroll
    for (int i = 0; i < loads_per_thread_a; i++) {
      int load_idx = threadIdx.x + i * num_threads;
      int linear_idx = load_idx * 4;
      int a_row = linear_idx / BLOCK_K;
      int a_col = linear_idx % BLOCK_K;

      float4 tmp =
          reinterpret_cast<float4 const &>(a_matrix[a_row * K + a_col]);
      // transposed storage for better access pattern
      a_block[(a_col + 0) * LDA + a_row] = tmp.x;
      a_block[(a_col + 1) * LDA + a_row] = tmp.y;
      a_block[(a_col + 2) * LDA + a_row] = tmp.z;
      a_block[(a_col + 3) * LDA + a_row] = tmp.w;
    }

    int constexpr b_elements = BLOCK_K * BLOCK_N;
    int constexpr vectorized_loads_b = b_elements / 4;
    int constexpr loads_per_thread_b = vectorized_loads_b / num_threads;
#pragma unroll
    for (int i = 0; i < loads_per_thread_b; i++) {
      int load_idx = threadIdx.x + i * num_threads;
      int linear_idx = load_idx * 4;
      int b_row = linear_idx / BLOCK_N;
      int b_col = linear_idx % BLOCK_N;

      // Store B into shared with per-group padding to avoid bank conflicts
      float4 tmp =
          reinterpret_cast<float4 const &>(b_matrix[b_row * N + b_col]);
      // Map column index with padding: after every THREAD_N columns, add SKEW_B
      int g0 = (b_col + 0) / THREAD_N; int i0 = (b_col + 0) % THREAD_N;
      int g1 = (b_col + 1) / THREAD_N; int i1 = (b_col + 1) % THREAD_N;
      int g2 = (b_col + 2) / THREAD_N; int i2 = (b_col + 2) % THREAD_N;
      int g3 = (b_col + 3) / THREAD_N; int i3 = (b_col + 3) % THREAD_N;
      b_block[b_row * LDB + g0 * (THREAD_N + SKEW_B) + i0] = tmp.x;
      b_block[b_row * LDB + g1 * (THREAD_N + SKEW_B) + i1] = tmp.y;
      b_block[b_row * LDB + g2 * (THREAD_N + SKEW_B) + i2] = tmp.z;
      b_block[b_row * LDB + g3 * (THREAD_N + SKEW_B) + i3] = tmp.w;
    }
    __syncthreads();

    a_matrix += BLOCK_K;
    b_matrix += BLOCK_K * N;

#pragma unroll
    for (int inner_k = 0; inner_k < BLOCK_K; inner_k++) {
#pragma unroll
      for (int tm = 0; tm < THREAD_M; tm++) {
        thread_m[tm] = a_block[inner_k * LDA + thread_row * THREAD_M + tm];
      }
#pragma unroll
      for (int tn = 0; tn < THREAD_N; tn++) {
        int col_base = thread_col * (THREAD_N + SKEW_B);
        thread_n[tn] = b_block[inner_k * LDB + col_base + tn];
      }

#pragma unroll
      for (int tm = 0; tm < THREAD_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < THREAD_N; tn++) {
          acc[tm][tn] += thread_m[tm] * thread_n[tn];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int tm = 0; tm < THREAD_M; tm++) {
#pragma unroll
    for (int tn = 0; tn < THREAD_N; tn += 4) {
      reinterpret_cast<float4 &>(c_matrix[(thread_row * THREAD_M + tm) * N +
                                          thread_col * THREAD_N + tn]) =
          reinterpret_cast<float4 const &>(acc[tm][tn]);
    }
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
          int WARP_M, int WARP_N>
__global__ void sgemm_warptiling_kernel(float *__restrict__ c_matrix,
                                        float const *__restrict__ a_matrix,
                                        float const *__restrict__ b_matrix,
                                        int const M, int const N, int const K) {
  extern __shared__ float smem[];

  constexpr int LDA = BLOCK_M;
  constexpr int LDB = BLOCK_N;
  constexpr int WARP_SIZE = 32;

  float *a_block = smem;
  float *b_block = smem + LDA * BLOCK_K;

  a_matrix += BLOCK_M * blockIdx.y * K;
  b_matrix += BLOCK_N * blockIdx.x;
  c_matrix += BLOCK_M * blockIdx.y * N + BLOCK_N * blockIdx.x;

  int constexpr num_threads = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
  // warp-level coordinates
  int const warp_id = threadIdx.x / WARP_SIZE;
  int const lane_id = threadIdx.x % WARP_SIZE;
  int const warps_per_block = num_threads / WARP_SIZE;
  int const warp_row = warp_id / (BLOCK_N / WARP_N);
  int const warp_col = warp_id % (BLOCK_N / WARP_N);

  // thread-level coordinates
  int const thread_row_in_warp = lane_id / (WARP_N / THREAD_N);
  int const thread_col_in_warp = lane_id % (WARP_N / THREAD_N);
  int const thread_row = warp_row * (WARP_M / THREAD_M) + thread_row_in_warp;
  int const thread_col = warp_col * (WARP_N / THREAD_N) + thread_col_in_warp;

  float acc[THREAD_M][THREAD_N] = {0.0f};
  float thread_m[THREAD_M] = {0.0f};
  float thread_n[THREAD_N] = {0.0f};
  for (int k = 0; k < K; k += BLOCK_K) {
    int constexpr a_elements = BLOCK_M * BLOCK_K;
    int constexpr vectorized_loads_a = a_elements / 4;
    int constexpr loads_per_thread_a = vectorized_loads_a / num_threads;
#pragma unroll
    for (int i = 0; i < loads_per_thread_a; i++) {
      int load_idx = threadIdx.x + i * num_threads;
      int linear_idx = load_idx * 4;
      int a_row = linear_idx / BLOCK_K;
      int a_col = linear_idx % BLOCK_K;

      float4 tmp =
          reinterpret_cast<float4 const &>(a_matrix[a_row * K + a_col]);
      a_block[(a_col + 0) * LDA + a_row] = tmp.x;
      a_block[(a_col + 1) * LDA + a_row] = tmp.y;
      a_block[(a_col + 2) * LDA + a_row] = tmp.z;
      a_block[(a_col + 3) * LDA + a_row] = tmp.w;
    }

    int constexpr b_elements = BLOCK_K * BLOCK_N;
    int constexpr vectorized_loads_b = b_elements / 4;
    int constexpr loads_per_thread_b = vectorized_loads_b / num_threads;
#pragma unroll
    for (int i = 0; i < loads_per_thread_b; i++) {
      int load_idx = threadIdx.x + i * num_threads;
      int linear_idx = load_idx * 4;
      int b_row = linear_idx / BLOCK_N;
      int b_col = linear_idx % BLOCK_N;

      reinterpret_cast<float4 &>(b_block[b_row * LDB + b_col]) =
          reinterpret_cast<float4 const &>(b_matrix[b_row * N + b_col]);
    }
    __syncthreads();

    a_matrix += BLOCK_K;
    b_matrix += BLOCK_K * N;

#pragma unroll
    for (int inner_k = 0; inner_k < BLOCK_K; inner_k++) {
#pragma unroll
      for (int tm = 0; tm < THREAD_M; tm++) {
        thread_m[tm] = a_block[inner_k * LDA + thread_row * THREAD_M + tm];
      }
#pragma unroll
      for (int tn = 0; tn < THREAD_N; tn++) {
        thread_n[tn] = b_block[inner_k * LDB + thread_col * THREAD_N + tn];
      }

#pragma unroll
      for (int tm = 0; tm < THREAD_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < THREAD_N; tn++) {
          acc[tm][tn] += thread_m[tm] * thread_n[tn];
        }
      }
    }
    __syncthreads();
  }

  for (int tm = 0; tm < THREAD_M; tm++) {
    for (int tn = 0; tn < THREAD_N; tn++) {
      c_matrix[(thread_row * THREAD_M + tm) * N + thread_col * THREAD_N + tn] =
          acc[tm][tn];
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
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    sgemm_naive_kernel<<<grid, block>>>(output.data_ptr<float>(),
                                        input.data_ptr<float>(),
                                        other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 10) {
    constexpr int BLOCK_SIZE = 32;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    sgemm_shmem_naive_kernel<BLOCK_SIZE>
        <<<grid, block>>>(output.data_ptr<float>(), input.data_ptr<float>(),
                          other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 20) {
    constexpr int BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 8, THREAD_M = 8;

    size_t shmem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(float);
    dim3 block(BLOCK_N * BLOCK_M / THREAD_M);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    sgemm_1d_blocktile_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M>
        <<<grid, block, shmem_size>>>(output.data_ptr<float>(),
                                      input.data_ptr<float>(),
                                      other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 21) {
    constexpr int BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 16, THREAD_M = 8;

    size_t shmem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(float);
    dim3 block(BLOCK_N * BLOCK_M / THREAD_M);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    sgemm_1d_blocktile_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M>
        <<<grid, block, shmem_size>>>(output.data_ptr<float>(),
                                      input.data_ptr<float>(),
                                      other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 30) {
    constexpr int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 8, THREAD_M = 8,
                  THREAD_N = 8;

    size_t shmem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(float);
    dim3 block((BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N));
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    sgemm_2d_blocktile_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N>
        <<<grid, block, shmem_size>>>(output.data_ptr<float>(),
                                      input.data_ptr<float>(),
                                      other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 40) {
    constexpr int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 8, THREAD_M = 8,
                  THREAD_N = 8;

    size_t shmem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(float);
    dim3 block((BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N));
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    sgemm_vec_2d_blocktile_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N>
        <<<grid, block, shmem_size>>>(output.data_ptr<float>(),
                                      input.data_ptr<float>(),
                                      other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 50) {
    constexpr int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 8, THREAD_M = 8,
                  THREAD_N = 8, SKEW_A = 1, SKEW_B = 1; // per-group pad = 1

    size_t shmem_size = (BLOCK_M + SKEW_A +
                         BLOCK_N + SKEW_B * (BLOCK_N / THREAD_N)) *
                        BLOCK_K * sizeof(float);
    dim3 block((BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N));
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    sgemm_vec_2d_blocktile_pad_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M,
                                      THREAD_N, SKEW_A, SKEW_B>
        <<<grid, block, shmem_size>>>(output.data_ptr<float>(),
                                      input.data_ptr<float>(),
                                      other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 51) {
    constexpr int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 16, THREAD_M = 8,
                  THREAD_N = 8, SKEW_A = 1, SKEW_B = 1; // per-group pad = 1

    size_t shmem_size = (BLOCK_M + SKEW_A +
                         BLOCK_N + SKEW_B * (BLOCK_N / THREAD_N)) *
                        BLOCK_K * sizeof(float);
    dim3 block((BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N));
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    sgemm_vec_2d_blocktile_pad_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M,
                                      THREAD_N, SKEW_A, SKEW_B>
        <<<grid, block, shmem_size>>>(output.data_ptr<float>(),
                                      input.data_ptr<float>(),
                                      other.data_ptr<float>(), M, N, K);
  } else if constexpr (VER == 60) {
    constexpr int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 16, THREAD_M = 8,
                  THREAD_N = 8, WARP_M = 64, WARP_N = 32;

    size_t shmem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(float);
    dim3 block((BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N));
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    sgemm_warptiling_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N,
                            WARP_M, WARP_N>
        <<<grid, block, shmem_size>>>(output.data_ptr<float>(),
                                      input.data_ptr<float>(),
                                      other.data_ptr<float>(), M, N, K);
  } else {
    TORCH_CHECK(false, "Unsupported version");
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgemm_naive", launch_sgemm<0>, py::arg("input"), py::arg("other"),
        py::arg("out"));
  m.def("sgemm_shmem", launch_sgemm<10>, py::arg("input"), py::arg("other"),
        py::arg("out"));
  m.def("sgemm_64x64x8_1d", launch_sgemm<20>, py::arg("input"),
        py::arg("other"), py::arg("out"));
  m.def("sgemm_64x64x16_1d", launch_sgemm<21>, py::arg("input"),
        py::arg("other"), py::arg("out"));
  m.def("sgemm_128x128x8_8x8x1", launch_sgemm<30>, py::arg("input"),
        py::arg("other"), py::arg("out"));
  m.def("sgemm_vec_128x128x8_8x8x1", launch_sgemm<40>, py::arg("input"),
        py::arg("other"), py::arg("out"));
  m.def("sgemm_vec_128x128x8_8x8x1_pad", launch_sgemm<50>, py::arg("input"),
        py::arg("other"), py::arg("out"));
  m.def("sgemm_vec_128x128x16_8x8x1_pad", launch_sgemm<51>, py::arg("input"),
        py::arg("other"), py::arg("out"));
  m.def("sgemm_warptiling_128x128x16_64x32x1_8x8x1", launch_sgemm<60>,
        py::arg("input"), py::arg("other"), py::arg("out"));
}
