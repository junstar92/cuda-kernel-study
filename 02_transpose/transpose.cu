#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// upper-bound - all memory accesses are coalesced
template <typename T>
__global__ void copy_kernel(T* out, T const* in, size_t const M,
                            size_t const N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t m = idx / M;
  size_t n = idx % N;

  if (m < M && n < N) {
    out[m * N + n] = in[m * N + n];
  }
}

template <typename T, size_t BLOCK_SIZE = 256>
void copy(T* out, T const* in, size_t const M, size_t const N) {
  size_t grid = (M * N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  copy_kernel<<<grid, BLOCK_SIZE>>>(out, in, M, N);
}

// row-read column-write
template <typename T>
__global__ void transpose_row_kernel(T* out, T const* in, size_t const M,
                                     size_t const N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t m = idx / M;
  size_t n = idx % N;

  if (m < M && n < N) {
    out[n * M + m] = in[m * N + n];
  }
}

template <typename T, size_t BLOCK_SIZE = 256>
void transpose_row(T* out, T const* in, size_t const M, size_t const N) {
  size_t grid = (M * N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  transpose_row_kernel<<<grid, BLOCK_SIZE>>>(out, in, M, N);
}

// column-read row-write
template <typename T>
__global__ void transpose_col_kernel(T* out, T const* in, size_t const M,
                                     size_t const N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t m = idx % M;
  size_t n = idx / N;

  if (m < M && n < N) {
    out[n * M + m] = in[m * N + n];
  }
}

template <typename T, size_t BLOCK_SIZE = 256>
void transpose_col(T* out, T const* in, size_t const M, size_t const N) {
  size_t grid = (M * N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  transpose_col_kernel<<<grid, BLOCK_SIZE>>>(out, in, M, N);
}

// column-read row-write unrolling 4
template <typename T>
__global__ void transpose_col_unroll_4_kernel(T* out, T const* in, size_t M,
                                              size_t N) {
  constexpr size_t unroll_factor = 4;
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < M * N / unroll_factor) {
    size_t idx = tid * unroll_factor;

#pragma unroll unroll_factor
    for (size_t i = 0; i < unroll_factor; i++) {
      size_t m = (idx + i) % M;
      size_t n = (idx + i) / N;

      out[n * M + m] = in[m * N + n];
    }

    int remainder = (M * N) % unroll_factor;
    if (tid == M * N / unroll_factor - 1 && remainder) {
      while (remainder) {
        idx = M * N - remainder--;
        size_t m = idx % M;
        size_t n = idx / N;

        out[n * M + m] = in[m * N + n];
      }
    }
  }
}

template <typename T, size_t BLOCK_SIZE = 256>
void transpose_col_unroll_4(T* out, T const* in, size_t const M,
                            size_t const N) {
  size_t grid = ((M * N) / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;

  transpose_col_unroll_4_kernel<<<grid, BLOCK_SIZE>>>(out, in, M, N);
}

// column-read row-vectorized-write unrolling n
template <typename T>
__global__ void transpose_col_unroll_n_kernel(T* out, T const* in, size_t M,
                                              size_t N) {
  constexpr size_t unroll_factor = 128 / sizeof(T) / 8;
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < M * N / unroll_factor) {
    size_t idx = tid * unroll_factor;

    T tmp[unroll_factor];
#pragma unroll unroll_factor
    for (size_t i = 0; i < unroll_factor; i++) {
      size_t m = (idx + i) % M;
      size_t n = (idx + i) / N;

      tmp[i] = in[m * N + n];
    }
    reinterpret_cast<float4&>(out[idx]) = *reinterpret_cast<float4*>(tmp);

    int remainder = (M * N) % unroll_factor;
    if (tid == M * N / unroll_factor - 1 && remainder) {
      while (remainder) {
        idx = M * N - remainder--;
        size_t m = idx % M;
        size_t n = idx / N;

        out[n * M + m] = in[m * N + n];
      }
    }
  }
}

template <typename T, size_t BLOCK_SIZE = 256>
void transpose_col_unroll_n(T* out, T const* in, size_t const M,
                            size_t const N) {
  constexpr size_t unroll_factor = 128 / sizeof(T) / 8;
  size_t grid = ((M * N) / unroll_factor + BLOCK_SIZE - 1) / BLOCK_SIZE;

  transpose_col_unroll_n_kernel<<<grid, BLOCK_SIZE>>>(out, in, M, N);
}

// row-read row-write with shared memory
template <typename T, size_t TILE_M = 16, size_t TILE_N = 16, size_t PAD_N = 0>
__global__ void transpose_shm_kernel(T* out, T const* in, size_t const M,
                                     size_t const N) {
  __shared__ T smem[(TILE_N + PAD_N) * TILE_M];

  size_t in_m = blockDim.y * blockIdx.y + threadIdx.y;
  size_t in_n = blockDim.x * blockIdx.x + threadIdx.x;

  size_t block_idx = blockDim.x * threadIdx.y + threadIdx.x;
  size_t block_row = block_idx / blockDim.y;
  size_t block_col = block_idx % blockDim.y;
  size_t out_m = blockDim.y * blockIdx.y + block_col;
  size_t out_n = blockDim.x * blockIdx.x + block_row;

  smem[threadIdx.y * TILE_N + threadIdx.x] =
      (in_m < M && in_n < N) ? in[in_m * N + in_n] : T{0.f};
  __syncthreads();

  if (out_m < M && out_n < N) {
    out[out_n * M + out_m] = smem[block_col * TILE_N + block_row];
  }
}

template <typename T, size_t TILE_M = 16, size_t TILE_N = 16, size_t PAD_M = 0>
__global__ void transpose_shm_v1_kernel(T* out, T const* in, size_t const M,
                                        size_t const N) {
  __shared__ T smem[TILE_N * (TILE_M + PAD_M)];

  size_t in_m = blockDim.y * blockIdx.y + threadIdx.y;
  size_t in_n = blockDim.x * blockIdx.x + threadIdx.x;

  size_t block_idx = blockDim.x * threadIdx.y + threadIdx.x;
  size_t block_row = block_idx / blockDim.y;
  size_t block_col = block_idx % blockDim.y;
  size_t out_m = blockDim.y * blockIdx.y + block_col;
  size_t out_n = blockDim.x * blockIdx.x + block_row;

  smem[threadIdx.x * TILE_M + threadIdx.y] =
      (in_m < M && in_n < N) ? in[in_m * N + in_n] : T{0.f};
  __syncthreads();

  if (out_m < M && out_n < N) {
    out[out_n * M + out_m] = smem[block_row * TILE_M + block_col];
  }
}

// row-read row-write with shared memory + unrolling 2
template <typename T, size_t TILE_M = 16, size_t TILE_N = 16, size_t PAD_N = 0>
__global__ void transpose_shm_unroll_2_kernel(T* out, T const* in,
                                              size_t const M, size_t const N) {
  __shared__ T smem[TILE_N * 2 * TILE_M];

  size_t in_m = blockDim.y * blockIdx.y + threadIdx.y;
  size_t in_n = 2 * blockDim.x * blockIdx.x + threadIdx.x;

  size_t block_idx = blockDim.x * threadIdx.y + threadIdx.x;
  size_t block_row = block_idx / blockDim.y;
  size_t block_col = block_idx % blockDim.y;
  size_t out_m = blockDim.y * blockIdx.y + block_col;
  size_t out_n = 2 * blockDim.x * blockIdx.x + block_row;

  smem[threadIdx.y * TILE_N * 2 + threadIdx.x] =
      (in_m < M && in_n < N) ? in[in_m * N + in_n] : T{0.f};
  smem[threadIdx.y * TILE_N * 2 + threadIdx.x + TILE_N] =
      (in_m < M && (in_n + TILE_N) < N) ? in[in_m * N + in_n + TILE_N] : T{0.f};
  __syncthreads();

  if (out_m < M && out_n < N) {
    out[out_n * M + out_m] = smem[block_col * 2 * TILE_N + block_row];
    if (out_n + TILE_N < N) {
      out[(out_n + blockDim.x) * M + out_m] =
          smem[block_col * 2 * TILE_N + block_row + TILE_N];
    }
  }

  if ((N % 2) &&
      ((N + 2 * blockDim.x - 1) / (2 * blockDim.x)) == (gridDim.x + 1) &&
      (blockIdx.x == gridDim.x - 1) && threadIdx.x == 0) {
    size_t m = blockDim.y * blockIdx.y + threadIdx.y;
    size_t n = 2 * blockDim.x * gridDim.x;
    out[n * M + m] = in[m * N + n];
  }
}

template <typename T>
void transpose_shm(T* out, T const* in, size_t const M, size_t const N,
                   int ver = 0) {
  if (ver == 0) {
    constexpr size_t TILE_M = 16, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};
    transpose_shm_kernel<T, TILE_M, TILE_N><<<grid, block>>>(out, in, M, N);
  } else if (ver == 1) {
    constexpr size_t TILE_M = 16, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};
    transpose_shm_v1_kernel<T, TILE_M, TILE_N><<<grid, block>>>(out, in, M, N);
  } else if (ver == 2) {
    constexpr size_t TILE_M = 32, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};
    transpose_shm_v1_kernel<T, TILE_M, TILE_N><<<grid, block>>>(out, in, M, N);
  } else if (ver == 3) {
    // ver 0 + padding 2
    constexpr size_t TILE_M = 32, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};
    transpose_shm_kernel<T, TILE_M, TILE_N, 2><<<grid, block>>>(out, in, M, N);
  } else if (ver == 4) {
    // ver 1 + padding 2
    constexpr size_t TILE_M = 32, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};
    transpose_shm_v1_kernel<T, TILE_M, TILE_N, 2>
        <<<grid, block>>>(out, in, M, N);
  } else if (ver == 5) {
    // unroll 2
    constexpr size_t TILE_M = 32, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N / 2 + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    transpose_shm_unroll_2_kernel<T, TILE_M, TILE_N>
        <<<grid, block>>>(out, in, M, N);
  }
}

// functions for launching kernels
void launch_copy(torch::Tensor& dst, torch::Tensor const& src) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  switch (dst.dtype().toScalarType()) {
    case at::ScalarType::Float:
      copy(dst.data_ptr<float>(), src.data_ptr<float>(), M, N);
      break;

    case at::ScalarType::Double:
      copy(dst.data_ptr<double>(), src.data_ptr<double>(), M, N);
      break;

    case at::ScalarType::Half:
      copy(reinterpret_cast<half*>(dst.data_ptr<at::Half>()),
           reinterpret_cast<half const*>(src.data_ptr<at::Half>()), M, N);
      break;

    case at::ScalarType::BFloat16:
      copy(reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
           reinterpret_cast<__nv_bfloat16 const*>(src.data_ptr<at::BFloat16>()),
           M, N);
      break;

    case at::ScalarType::Int:
      copy(dst.data_ptr<int>(), src.data_ptr<int>(), M, N);
      break;

    case at::ScalarType::Long:
      copy(dst.data_ptr<long>(), src.data_ptr<long>(), M, N);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

void launch_transpose_row(torch::Tensor& dst, torch::Tensor const& src) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  switch (dst.dtype().toScalarType()) {
    case at::ScalarType::Float:
      transpose_row(dst.data_ptr<float>(), src.data_ptr<float>(), M, N);
      break;

    case at::ScalarType::Double:
      transpose_row(dst.data_ptr<double>(), src.data_ptr<double>(), M, N);
      break;

    case at::ScalarType::Half:
      transpose_row(reinterpret_cast<half*>(dst.data_ptr<at::Half>()),
                    reinterpret_cast<half const*>(src.data_ptr<at::Half>()), M,
                    N);
      break;

    case at::ScalarType::BFloat16:
      transpose_row(
          reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16 const*>(src.data_ptr<at::BFloat16>()),
          M, N);
      break;

    case at::ScalarType::Int:
      transpose_row(dst.data_ptr<int>(), src.data_ptr<int>(), M, N);
      break;

    case at::ScalarType::Long:
      transpose_row(dst.data_ptr<long>(), src.data_ptr<long>(), M, N);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

void launch_transpose_col(torch::Tensor& dst, torch::Tensor const& src) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  switch (dst.dtype().toScalarType()) {
    case at::ScalarType::Float:
      transpose_col(dst.data_ptr<float>(), src.data_ptr<float>(), M, N);
      break;

    case at::ScalarType::Double:
      transpose_col(dst.data_ptr<double>(), src.data_ptr<double>(), M, N);
      break;

    case at::ScalarType::Half:
      transpose_col(reinterpret_cast<half*>(dst.data_ptr<at::Half>()),
                    reinterpret_cast<half const*>(src.data_ptr<at::Half>()), M,
                    N);
      break;

    case at::ScalarType::BFloat16:
      transpose_col(
          reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16 const*>(src.data_ptr<at::BFloat16>()),
          M, N);
      break;

    case at::ScalarType::Int:
      transpose_col(dst.data_ptr<int>(), src.data_ptr<int>(), M, N);
      break;

    case at::ScalarType::Long:
      transpose_col(dst.data_ptr<long>(), src.data_ptr<long>(), M, N);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

void launch_transpose_col_unroll_4(torch::Tensor& dst,
                                   torch::Tensor const& src) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  switch (dst.dtype().toScalarType()) {
    case at::ScalarType::Float:
      transpose_col_unroll_4(dst.data_ptr<float>(), src.data_ptr<float>(), M,
                             N);
      break;

    case at::ScalarType::Double:
      transpose_col_unroll_4(dst.data_ptr<double>(), src.data_ptr<double>(), M,
                             N);
      break;

    case at::ScalarType::Half:
      transpose_col_unroll_4(
          reinterpret_cast<half*>(dst.data_ptr<at::Half>()),
          reinterpret_cast<half const*>(src.data_ptr<at::Half>()), M, N);
      break;

    case at::ScalarType::BFloat16:
      transpose_col_unroll_4(
          reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16 const*>(src.data_ptr<at::BFloat16>()),
          M, N);
      break;

    case at::ScalarType::Int:
      transpose_col_unroll_4(dst.data_ptr<int>(), src.data_ptr<int>(), M, N);
      break;

    case at::ScalarType::Long:
      transpose_col_unroll_4(dst.data_ptr<long>(), src.data_ptr<long>(), M, N);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

void launch_transpose_col_unroll_n(torch::Tensor& dst,
                                   torch::Tensor const& src) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  switch (dst.dtype().toScalarType()) {
    case at::ScalarType::Float:
      transpose_col_unroll_n(dst.data_ptr<float>(), src.data_ptr<float>(), M,
                             N);
      break;

    case at::ScalarType::Double:
      transpose_col_unroll_n(dst.data_ptr<double>(), src.data_ptr<double>(), M,
                             N);
      break;

    case at::ScalarType::Half:
      transpose_col_unroll_n(
          reinterpret_cast<half*>(dst.data_ptr<at::Half>()),
          reinterpret_cast<half const*>(src.data_ptr<at::Half>()), M, N);
      break;

    case at::ScalarType::BFloat16:
      transpose_col_unroll_n(
          reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16 const*>(src.data_ptr<at::BFloat16>()),
          M, N);
      break;

    case at::ScalarType::Int:
      transpose_col_unroll_n(dst.data_ptr<int>(), src.data_ptr<int>(), M, N);
      break;

    case at::ScalarType::Long:
      transpose_col_unroll_n(dst.data_ptr<long>(), src.data_ptr<long>(), M, N);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

void launch_transpose_shm(torch::Tensor& dst, torch::Tensor const& src,
                          int ver = 0) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  switch (dst.dtype().toScalarType()) {
    case at::ScalarType::Float:
      transpose_shm(dst.data_ptr<float>(), src.data_ptr<float>(), M, N, ver);
      break;

    case at::ScalarType::Double:
      transpose_shm(dst.data_ptr<double>(), src.data_ptr<double>(), M, N, ver);
      break;

    case at::ScalarType::Half:
      transpose_shm(reinterpret_cast<half*>(dst.data_ptr<at::Half>()),
                    reinterpret_cast<half const*>(src.data_ptr<at::Half>()), M,
                    N, ver);
      break;

    case at::ScalarType::BFloat16:
      transpose_shm(
          reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16 const*>(src.data_ptr<at::BFloat16>()),
          M, N, ver);
      break;

    case at::ScalarType::Int:
      transpose_shm(dst.data_ptr<int>(), src.data_ptr<int>(), M, N, ver);
      break;

    case at::ScalarType::Long:
      transpose_shm(dst.data_ptr<long>(), src.data_ptr<long>(), M, N, ver);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy", &launch_copy, "Copy kernel for CUDA tensors");
  m.def("transpose_row", &launch_transpose_row,
        "Transpose kernel for CUDA tensors");
  m.def("transpose_col", &launch_transpose_col,
        "Transpose kernel for CUDA tensors");
  m.def("transpose_col_unroll_4", &launch_transpose_col_unroll_4,
        "Transpose kernel for CUDA tensors");
  m.def("transpose_col_unroll_n", &launch_transpose_col_unroll_n,
        "Transpose kernel for CUDA tensors");
  m.def("transpose_shm", &launch_transpose_shm,
        "Transpose kernel for CUDA tensors");
}