#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr size_t warp_size = 32;

template <typename T>
__global__ void shuffle_kernel(T* vars, int src_lane, size_t n,
                               int width = warp_size) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    int lane_id = threadIdx.x & 0x1f;
    T var;
    if (lane_id == src_lane) var = vars[idx];
    // Why doesn't it make an error when only some threads in a warp take part ?
    // Does it result in just an undefined error ?
    var = __shfl_sync(0xffffffff, var, src_lane, width);

    vars[idx] = var;
  }
}

template <>
__global__ void shuffle_kernel(c10::Half* vars, int src_lane, size_t n,
                               int width) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    int lane_id = threadIdx.x & 0x1f;
    __half var;
    if (lane_id == src_lane) var = __half{vars[idx]};
    var = __shfl_sync(0xffffffff, var, src_lane, width);

    vars[idx] = var;
  }
}

void shuffle_sync(torch::Tensor& tensor, int src_lane, int width = warp_size) {
  size_t constexpr block = 256;
  size_t const n = tensor.numel();
  size_t const grid = (n + block - 1) / block;

  auto type = tensor.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "shuffle_kernel", [&]() {
                               shuffle_kernel<<<grid, block>>>(
                                   tensor.data_ptr<scalar_t>(), src_lane, n,
                                   width);
                             });

  TORCH_CHECK(cudaGetLastError() == cudaError_t::cudaSuccess);
}

template <typename T>
__global__ void shuffle_up_sync_kernel(T* vars, int delta, size_t n,
                                       int width = warp_size) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    T var = vars[idx];
    var = __shfl_up_sync(0xffffffff, var, delta, width);

    vars[idx] = var;
  }
}

template <>
__global__ void shuffle_up_sync_kernel(c10::Half* vars, int delta, size_t n,
                                       int width) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    __half var = __half{vars[idx]};
    var = __shfl_up_sync(0xffffffff, var, delta, width);

    vars[idx] = var;
  }
}

void shuffle_up_sync(torch::Tensor& tensor, int delta, int width = warp_size) {
  size_t constexpr block = 256;
  size_t const n = tensor.numel();
  size_t const grid = (n + block - 1) / block;

  auto type = tensor.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "shuffle_up_sync_kernel", [&]() {
                               shuffle_up_sync_kernel<<<grid, block>>>(
                                   tensor.data_ptr<scalar_t>(), delta, n,
                                   width);
                             });

  TORCH_CHECK(cudaGetLastError() == cudaError_t::cudaSuccess);
}

template <typename T>
__global__ void shuffle_down_sync_kernel(T* vars, int delta, size_t n,
                                         int width = warp_size) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    T var = vars[idx];
    var = __shfl_down_sync(0xffffffff, var, delta, width);

    vars[idx] = var;
  }
}

template <>
__global__ void shuffle_down_sync_kernel(c10::Half* vars, int delta, size_t n,
                                         int width) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    __half var = __half{vars[idx]};
    var = __shfl_down_sync(0xffffffff, var, delta, width);

    vars[idx] = var;
  }
}

void shuffle_down_sync(torch::Tensor& tensor, int delta,
                       int width = warp_size) {
  size_t constexpr block = 256;
  size_t const n = tensor.numel();
  size_t const grid = (n + block - 1) / block;

  auto type = tensor.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "shuffle_down_sync_kernel", [&]() {
                               shuffle_down_sync_kernel<<<grid, block>>>(
                                   tensor.data_ptr<scalar_t>(), delta, n,
                                   width);
                             });

  TORCH_CHECK(cudaGetLastError() == cudaError_t::cudaSuccess);
}

template <typename T>
__global__ void shuffle_xor_sync_kernel(T* vars, int lane_mask, size_t n,
                                        int width = warp_size) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    T var = vars[idx];
    var = __shfl_xor_sync(0xffffffff, var, lane_mask, width);

    vars[idx] = var;
  }
}

template <>
__global__ void shuffle_xor_sync_kernel(c10::Half* vars, int lane_mask,
                                        size_t n, int width) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    __half var = __half{vars[idx]};
    var = __shfl_xor_sync(0xffffffff, var, lane_mask, width);

    vars[idx] = var;
  }
}

void shuffle_xor_sync(torch::Tensor& tensor, int lane_mask,
                      int width = warp_size) {
  size_t constexpr block = 256;
  size_t const n = tensor.numel();
  size_t const grid = (n + block - 1) / block;

  auto type = tensor.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "shuffle_xor_sync_kernel", [&]() {
                               shuffle_xor_sync_kernel<<<grid, block>>>(
                                   tensor.data_ptr<scalar_t>(), lane_mask, n,
                                   width);
                             });

  TORCH_CHECK(cudaGetLastError() == cudaError_t::cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shuffle_sync", shuffle_sync, py::arg("tensor"), py::arg("src_lane"),
        py::arg("width") = warp_size);
  m.def("shuffle_up_sync", shuffle_up_sync, py::arg("tensor"), py::arg("delta"),
        py::arg("width") = warp_size);
  m.def("shuffle_down_sync", shuffle_down_sync, py::arg("tensor"),
        py::arg("delta"), py::arg("width") = warp_size);
  m.def("shuffle_xor_sync", shuffle_xor_sync, py::arg("tensor"),
        py::arg("lane_mask"), py::arg("width") = warp_size);
}