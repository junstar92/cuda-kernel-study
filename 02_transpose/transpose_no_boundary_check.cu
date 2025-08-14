#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// The performance upper bound: all memory accesses are perfectly coalesced
template <typename T>
__global__ void copy_kernel(T* out, T const* in, size_t const M,
                            size_t const N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t m = idx / N;
  size_t n = idx % N;

  out[m * N + n] = in[m * N + n];
}

template <typename T, size_t BLOCK_SIZE = 256>
void copy(T* out, T const* in, size_t const M, size_t const N) {
  size_t grid = (M * N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  copy_kernel<<<grid, BLOCK_SIZE>>>(out, in, M, N);
}

// Reads are coalesced (row-wise), but writes are scattered across columns
template <typename T>
__global__ void transpose_row_kernel(T* out, T const* in, size_t const M,
                                     size_t const N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t m = idx / N;
  size_t n = idx % N;

  out[n * M + m] = in[m * N + n];
}

template <typename T, size_t BLOCK_SIZE = 256>
void transpose_row(T* out, T const* in, size_t const M, size_t const N) {
  size_t grid = (M * N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  transpose_row_kernel<<<grid, BLOCK_SIZE>>>(out, in, M, N);
}

// Reads are scattered across a column, but writes are coalesced (row-wise in
// output)
template <typename T>
__global__ void transpose_col_kernel(T* out, T const* in, size_t const M,
                                     size_t const N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t m = idx % M;
  size_t n = idx / M;

  out[n * M + m] = in[m * N + n];
}

template <typename T, size_t BLOCK_SIZE = 256>
void transpose_col(T* out, T const* in, size_t const M, size_t const N) {
  size_t grid = (M * N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  transpose_col_kernel<<<grid, BLOCK_SIZE>>>(out, in, M, N);
}

// Each thread processes 4 elements; loop is unrolled for ILP
template <typename T>
__global__ void transpose_col_unroll_4_kernel(T* out, T const* in, size_t M,
                                              size_t N) {
  constexpr size_t unroll_factor = 4;
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  size_t idx = tid * unroll_factor;

#pragma unroll
  for (size_t i = 0; i < unroll_factor; i++) {
    size_t m = (idx + i) % M;
    size_t n = (idx + i) / M;

    out[n * M + m] = in[m * N + n];
  }
}

template <typename T, size_t BLOCK_SIZE = 256>
void transpose_col_unroll_4(T* out, T const* in, size_t const M,
                            size_t const N) {
  size_t grid = ((M * N) / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;

  transpose_col_unroll_4_kernel<<<grid, BLOCK_SIZE>>>(out, in, M, N);
}

// Vectorized store (float4) to reduce global memory transactions
template <typename T>
__global__ void transpose_col_unroll_n_kernel(T* out, T const* in, size_t M,
                                              size_t N) {
  constexpr size_t unroll_factor = 128 / sizeof(T) / 8;
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  size_t idx = tid * unroll_factor;

  T tmp[unroll_factor];
#pragma unroll
  for (size_t i = 0; i < unroll_factor; i++) {
    size_t m = (idx + i) % M;
    size_t n = (idx + i) / M;

    tmp[i] = in[m * N + n];  // scattered global loads
  }
  reinterpret_cast<float4&>(out[idx]) =
      *reinterpret_cast<float4*>(tmp);  // vectorized store
}

template <typename T, size_t BLOCK_SIZE = 256>
void transpose_col_unroll_n(T* out, T const* in, size_t const M,
                            size_t const N) {
  constexpr size_t unroll_factor = 128 / sizeof(T) / 8;
  size_t grid = ((M * N) / unroll_factor + BLOCK_SIZE - 1) / BLOCK_SIZE;

  transpose_col_unroll_n_kernel<<<grid, BLOCK_SIZE>>>(out, in, M, N);
}

// Version 0: Shared-memory tiling (potential bank conflicts on transpose)
template <typename T, size_t TILE_M = 16, size_t TILE_N = 16, size_t PAD_N = 0>
__global__ void transpose_shm_v0_kernel(T* out, T const* in, size_t const M,
                                        size_t const N) {
  __shared__ T smem[(TILE_N + PAD_N) * TILE_M];

  size_t in_m = blockDim.y * blockIdx.y + threadIdx.y;
  size_t in_n = blockDim.x * blockIdx.x + threadIdx.x;
  // Compute output indices corresponding to transposed block
  size_t block_idx = blockDim.x * threadIdx.y + threadIdx.x;
  size_t block_row = block_idx / blockDim.y;
  size_t block_col = block_idx % blockDim.y;
  size_t out_m = blockDim.y * blockIdx.y + block_col;
  size_t out_n = blockDim.x * blockIdx.x + block_row;
  // Coalesced load from global to shared (row-major in shared)
  smem[threadIdx.y * TILE_N + threadIdx.x] = in[in_m * N + in_n];
  __syncthreads();
  // Each thread now reads a column from shared and write to global (transposed)
  out[out_n * M + out_m] = smem[block_col * TILE_N + block_row];
}

// Version 1: Shared-memory tiling (reindexed to reduce bank conflicts)
template <typename T, size_t TILE_M = 16, size_t TILE_N = 16, size_t PAD_M = 0>
__global__ void transpose_shm_v1_kernel(T* out, T const* in, size_t const M,
                                        size_t const N) {
  __shared__ T smem[TILE_N * (TILE_M + PAD_M)];

  size_t in_m = blockDim.y * blockIdx.y + threadIdx.y;
  size_t in_n = blockDim.x * blockIdx.x + threadIdx.x;
  // Transposed indexing for shared memory
  size_t block_idx = blockDim.x * threadIdx.y + threadIdx.x;
  size_t block_row = block_idx / blockDim.y;
  size_t block_col = block_idx % blockDim.y;
  size_t out_m = blockDim.y * blockIdx.y + block_col;
  size_t out_n = blockDim.x * blockIdx.x + block_row;
  // Colesced load from global, but store into shared in column-major order
  smem[threadIdx.x * TILE_M + threadIdx.y] = in[in_m * N + in_n];
  __syncthreads();
  // Coalesced write from shared (read row-major from shared)
  out[out_n * M + out_m] = smem[block_row * TILE_M + block_col];
}

// Tiled transpose (32x16 tile) + block-level unrolling (process 2 tiles per
// block)
template <typename T, size_t TILE_M = 32, size_t TILE_N = 16>
__global__ void transpose_shm_v1_unroll_2_kernel(T* out, T const* in,
                                                 size_t const M,
                                                 size_t const N) {
  __shared__ T smem[TILE_N * 2 * TILE_M];

  size_t in_m = blockDim.y * blockIdx.y + threadIdx.y;
  size_t in_n = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  // Calculate indices for two tiles
  size_t block_idx = blockDim.x * threadIdx.y + threadIdx.x;
  size_t block_row = block_idx / blockDim.y;
  size_t block_col = block_idx % blockDim.y;
  size_t out_m = blockDim.y * blockIdx.y + block_col;
  size_t out_n = 2 * blockDim.x * blockIdx.x + block_row;
  // Load two tiles into shared memory (coalesced global loads)
  smem[threadIdx.y * TILE_N * 2 + threadIdx.x] = in[in_m * N + in_n];
  smem[threadIdx.y * TILE_N * 2 + threadIdx.x + TILE_N] =
      in[in_m * N + in_n + TILE_N];
  __syncthreads();
  // Store transposed tiles from shared to global (coalesced global stores)
  out[out_n * M + out_m] = smem[block_col * 2 * TILE_N + block_row];
  out[(out_n + blockDim.x) * M + out_m] =
      smem[block_col * 2 * TILE_N + block_row + TILE_N];
}

template <typename T>
void transpose_shm(T* out, T const* in, size_t const M, size_t const N,
                   int ver = 0) {
  if (ver == 0) {
    constexpr size_t TILE_M = 16, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};
    transpose_shm_v0_kernel<T, TILE_M, TILE_N><<<grid, block>>>(out, in, M, N);
  } else if (ver == 1) {
    constexpr size_t TILE_M = 16, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};
    transpose_shm_v1_kernel<T, TILE_M, TILE_N><<<grid, block>>>(out, in, M, N);
  } else if (ver == 2) {
    // v0 + (32, 16) tile
    constexpr size_t TILE_M = 32, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};
    transpose_shm_v0_kernel<T, TILE_M, TILE_N><<<grid, block>>>(out, in, M, N);
  } else if (ver == 3) {
    // v1 + (32, 16) tile
    constexpr size_t TILE_M = 32, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};
    transpose_shm_v1_kernel<T, TILE_M, TILE_N><<<grid, block>>>(out, in, M, N);
  } else if (ver == 4) {
    // v1 + (32, 16) tile + padding 1
    constexpr size_t TILE_M = 32, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};
    transpose_shm_v1_kernel<T, TILE_M, TILE_N, 1>
        <<<grid, block>>>(out, in, M, N);
  } else if (ver == 5) {
    // v1 + (32, 16) tile + unroll 2
    constexpr size_t TILE_M = 32, TILE_N = 16;
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N / 2 + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    transpose_shm_v1_unroll_2_kernel<T, TILE_M, TILE_N>
        <<<grid, block>>>(out, in, M, N);
  }
}

// functions for launching kernels
void launch_copy(torch::Tensor& dst, torch::Tensor const& src) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  auto type = dst.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, type, "copy", [&]() {
        copy(dst.data_ptr<scalar_t>(), src.data_ptr<scalar_t>(), M, N);
      });
}

void launch_transpose_row(torch::Tensor& dst, torch::Tensor const& src) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  auto type = dst.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "transpose_row", [&]() {
                               transpose_row(dst.data_ptr<scalar_t>(),
                                             src.data_ptr<scalar_t>(), M, N);
                             });
}

void launch_transpose_col(torch::Tensor& dst, torch::Tensor const& src) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  auto type = dst.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "transpose_col", [&]() {
                               transpose_col(dst.data_ptr<scalar_t>(),
                                             src.data_ptr<scalar_t>(), M, N);
                             });
}

void launch_transpose_col_unroll_4(torch::Tensor& dst,
                                   torch::Tensor const& src) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  auto type = dst.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "transpose_col_unroll_4", [&]() {
                               transpose_col_unroll_4(dst.data_ptr<scalar_t>(),
                                                      src.data_ptr<scalar_t>(),
                                                      M, N);
                             });
}

void launch_transpose_col_unroll_n(torch::Tensor& dst,
                                   torch::Tensor const& src) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  auto type = dst.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "transpose_col_unroll_n", [&]() {
                               transpose_col_unroll_n(dst.data_ptr<scalar_t>(),
                                                      src.data_ptr<scalar_t>(),
                                                      M, N);
                             });
}

void launch_transpose_shm(torch::Tensor& dst, torch::Tensor const& src,
                          int ver = 0) {
  size_t const M = src.size(0);
  size_t const N = src.size(1);

  auto type = dst.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "transpose_shm", [&]() {
                               transpose_shm(dst.data_ptr<scalar_t>(),
                                             src.data_ptr<scalar_t>(), M, N,
                                             ver);
                             });
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