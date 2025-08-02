#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__global__ void load_without_ldg_kernel(T* output, T const* input, size_t n) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    size_t non_coalesced_idx = (idx * 2) % n;
    output[idx] = input[non_coalesced_idx];
  }
}

template <typename T>
__global__ void load_with_ldg_kernel(T* output, T const* __restrict__ input,
                                     size_t const n) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    size_t non_coalesced_idx = (idx * 2) % n;
    output[idx] = __ldg(&input[non_coalesced_idx]);
  }
}

void load_without_ldg(torch::Tensor& output, torch::Tensor const& input) {
  size_t constexpr block = 256;
  size_t const n = input.numel();
  size_t const grid = (n + block - 1) / block;

  auto type = input.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "load_without_ldg_kernel", [&]() {
                               load_without_ldg_kernel<<<grid, block>>>(
                                   output.data_ptr<scalar_t>(),
                                   input.data_ptr<scalar_t>(), n);
                             });
}

void load_with_ldg(torch::Tensor& output, torch::Tensor const& input) {
  size_t constexpr block = 256;
  size_t const n = input.numel();
  size_t const grid = (n + block - 1) / block;

  auto type = input.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "load_with_ldg_kernel", [&]() {
                               load_with_ldg_kernel<<<grid, block>>>(
                                   output.data_ptr<scalar_t>(),
                                   input.data_ptr<scalar_t>(), n);
                             });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_without_ldg", load_without_ldg);
  m.def("load_with_ldg", load_with_ldg);
}