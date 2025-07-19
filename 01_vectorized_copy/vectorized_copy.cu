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

  switch (in.dtype().toScalarType()) {
    case at::ScalarType::Float:
      copy(in.data_ptr<float>(), out.data_ptr<float>(), n);
      break;

    case at::ScalarType::Double:
      copy(in.data_ptr<double>(), out.data_ptr<double>(), n);
      break;

    case at::ScalarType::Half:
      copy(reinterpret_cast<half const*>(in.data_ptr<at::Half>()),
           reinterpret_cast<half*>(out.data_ptr<at::Half>()), n);
      break;

    case at::ScalarType::BFloat16:
      copy(reinterpret_cast<__nv_bfloat16 const*>(in.data_ptr<at::BFloat16>()),
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), n);
      break;

    case at::ScalarType::Int:
      copy(in.data_ptr<int>(), out.data_ptr<int>(), n);
      break;

    case at::ScalarType::Long:
      copy(in.data_ptr<long>(), out.data_ptr<long>(), n);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

void launch_vectorized_copy(torch::Tensor const& in, torch::Tensor& out) {
  size_t n = in.numel();

  switch (in.dtype().toScalarType()) {
    case at::ScalarType::Float:
      vectorized_copy(in.data_ptr<float>(), out.data_ptr<float>(), n);
      break;

    case at::ScalarType::Double:
      vectorized_copy(in.data_ptr<double>(), out.data_ptr<double>(), n);
      break;

    case at::ScalarType::Half:
      vectorized_copy(reinterpret_cast<half const*>(in.data_ptr<at::Half>()),
                      reinterpret_cast<half*>(out.data_ptr<at::Half>()), n);
      break;

    case at::ScalarType::BFloat16:
      vectorized_copy(
          reinterpret_cast<__nv_bfloat16 const*>(in.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), n);
      break;

    case at::ScalarType::Int:
      vectorized_copy(in.data_ptr<int>(), out.data_ptr<int>(), n);
      break;

    case at::ScalarType::Long:
      vectorized_copy(in.data_ptr<long>(), out.data_ptr<long>(), n);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy", &launch_copy, "Copy kernel for CUDA tensors");
  m.def("vectorized_copy", &launch_vectorized_copy,
        "Vectorized copy kernel for CUDA tensors");
}