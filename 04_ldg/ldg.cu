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

  switch (input.dtype().toScalarType()) {
    case at::ScalarType::Float:
      load_without_ldg_kernel<<<grid, block>>>(output.data_ptr<float>(),
                                               input.data_ptr<float>(), n);
      break;

    case at::ScalarType::Double:
      load_without_ldg_kernel<<<grid, block>>>(output.data_ptr<double>(),
                                               input.data_ptr<double>(), n);
      break;

    case at::ScalarType::Half:
      load_without_ldg_kernel<<<grid, block>>>(
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          reinterpret_cast<half*>(input.data_ptr<at::Half>()), n);
      break;

    case at::ScalarType::BFloat16:
      load_without_ldg_kernel<<<grid, block>>>(
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>()), n);
      break;

    case at::ScalarType::Int:
      load_without_ldg_kernel<<<grid, block>>>(output.data_ptr<int>(),
                                               input.data_ptr<int>(), n);
      break;

    case at::ScalarType::Long:
      load_without_ldg_kernel<<<grid, block>>>(output.data_ptr<long>(),
                                               input.data_ptr<long>(), n);
      break;
  }

  TORCH_CHECK(cudaGetLastError() == cudaError_t::cudaSuccess);
}

void load_with_ldg(torch::Tensor& output, torch::Tensor const& input) {
  size_t constexpr block = 256;
  size_t const n = input.numel();
  size_t const grid = (n + block - 1) / block;

  switch (input.dtype().toScalarType()) {
    case at::ScalarType::Float:
      load_with_ldg_kernel<<<grid, block>>>(output.data_ptr<float>(),
                                            input.data_ptr<float>(), n);
      break;

    case at::ScalarType::Double:
      load_with_ldg_kernel<<<grid, block>>>(output.data_ptr<double>(),
                                            input.data_ptr<double>(), n);
      break;

    case at::ScalarType::Half:
      load_with_ldg_kernel<<<grid, block>>>(
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          reinterpret_cast<half*>(input.data_ptr<at::Half>()), n);
      break;

    case at::ScalarType::BFloat16:
      load_with_ldg_kernel<<<grid, block>>>(
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>()), n);
      break;

    case at::ScalarType::Int:
      load_with_ldg_kernel<<<grid, block>>>(output.data_ptr<int>(),
                                            input.data_ptr<int>(), n);
      break;

    case at::ScalarType::Long:
      load_with_ldg_kernel<<<grid, block>>>(output.data_ptr<long>(),
                                            input.data_ptr<long>(), n);
      break;
  }

  TORCH_CHECK(cudaGetLastError() == cudaError_t::cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_without_ldg", load_without_ldg);
  m.def("load_with_ldg", load_with_ldg);
}