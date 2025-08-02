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

template <typename T>
__global__ void copy_kernel(T const* in, T* out, size_t n) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    out[idx] = in[idx];
  }
}

template <typename T, size_t BLOCK_SIZE = 256>
void copy(T const* in, T* out, size_t n) {
  size_t grids = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  copy_kernel<<<grids, BLOCK_SIZE>>>(in, out, n);
}

template <typename T, size_t NUM_ELEMS_IN_VECTOR = 128 / sizeof(T) / 8>
__global__ void vectorized_copy_kernel(T const* in, T* out, size_t n) {
  size_t idx = NUM_ELEMS_IN_VECTOR * (blockDim.x * blockIdx.x + threadIdx.x);

  reinterpret_cast<float4&>(out[idx]) =
      reinterpret_cast<float4 const&>(in[idx]);

  int remainder = n % NUM_ELEMS_IN_VECTOR;
  if (remainder && idx == n - remainder - NUM_ELEMS_IN_VECTOR) {
    while (remainder) {
      idx = n - remainder--;
      out[idx] = in[idx];
    }
  }
}

template <typename T, size_t BLOCK_SIZE = 256,
          size_t NUM_ELEMS_IN_VECTOR = 128 / sizeof(T) / 8>
void vectorized_copy(T const* in, T* out, size_t n) {
  size_t grids = (n / NUM_ELEMS_IN_VECTOR + BLOCK_SIZE - 1) / BLOCK_SIZE;

  vectorized_copy_kernel<<<grids, BLOCK_SIZE>>>(in, out, n);
}

void launch_copy(torch::Tensor const& in, torch::Tensor& out) {
  size_t n = in.numel();

  auto type = in.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, type, "copy",
      [&]() { copy(in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n); });
}

void launch_vectorized_copy(torch::Tensor const& in, torch::Tensor& out) {
  size_t n = in.numel();

  auto type = in.scalar_type();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                             type, "vectorized_copy", [&]() {
                               vectorized_copy(in.data_ptr<scalar_t>(),
                                               out.data_ptr<scalar_t>(), n);
                             });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy", &launch_copy, "Copy kernel for CUDA tensors");
  m.def("vectorized_copy", &launch_vectorized_copy,
        "Vectorized copy kernel for CUDA tensors");
}