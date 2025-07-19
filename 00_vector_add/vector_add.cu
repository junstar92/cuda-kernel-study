#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector_types.h>

template <typename T>
__global__ void add_kernel(T const *a, T const *b, T *out, size_t n) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

template <typename T>
__global__ void add_element2_kernel(T const *a, T const *b, T *out, size_t n) {
  size_t idx = 2 * blockDim.x * blockIdx.x + threadIdx.x * 2;

  if (idx < n) {
    out[idx] = a[idx] + b[idx];
    out[idx + 1] = a[idx + 1] + b[idx + 1];
  }

  if (n % 2 == 1 && idx == n - 3) {
    out[n - 1] = a[n - 1] + b[n - 1];
  }
}

// Note: __fadd2 intrinsics require compute capability >= 10.0
// template <>
// __global__ void add_element2_kernel(float const *a, float const *b, float
// *out,
//                            size_t n) {
//   int size_t = blockDim.x * blockIdx.x + threadIdx.x * 2;
//   reinterpret_cast<float2 &>(out[idx]) =
//       __fadd2_rn(reinterpret_cast<float2 const &>(a[idx]),
//                  reinterpret_cast<float2 const &>(b[idx]));

//   if (idx == n - 2 && n % 2 == 1) {
//     out[n - 1] = a[n - 1] + b[n - 1];
//   }
// }

template <>
__global__ void add_element2_kernel(half const *a, half const *b, half *out,
                                    size_t n) {
  size_t idx = 2 * blockDim.x * blockIdx.x + threadIdx.x * 2;

  if (idx < n) {
    reinterpret_cast<half2 &>(out[idx]) =
        reinterpret_cast<half2 const &>(a[idx]) +
        reinterpret_cast<half2 const &>(b[idx]);
  }

  if (n % 2 == 1 && idx == n - 3) {
    out[n - 1] = a[n - 1] + b[n - 1];
  }
}

template <>
__global__ void add_element2_kernel(__nv_bfloat162 const *a,
                                    __nv_bfloat162 const *b,
                                    __nv_bfloat162 *out, size_t n) {
  size_t idx = 2 * blockDim.x * blockIdx.x + threadIdx.x * 2;

  if (idx < n) {
    reinterpret_cast<__nv_bfloat162 &>(out[idx]) =
        reinterpret_cast<__nv_bfloat162 const &>(a[idx]) +
        reinterpret_cast<__nv_bfloat162 const &>(b[idx]);
  }

  if (n % 2 == 1 && idx == n - 3) {
    out[n - 1] = a[n - 1] + b[n - 1];
  }
}

template <typename T>
__global__ void add_element2_interleaved_kernel(T const *a, T const *b, T *out,
                                                size_t n) {
  size_t idx = 2 * blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    out[idx] = a[idx] + b[idx];
    out[blockDim.x + idx] = a[blockDim.x + idx] + b[blockDim.x + idx];
  }

  if (n % 2 == 1 && idx == gridDim.x * blockDim.x - 1) {
    out[n - 1] = a[n - 1] + b[n - 1];
  }
}

template <typename T, size_t BLOCK_SIZE = 256>
void add(T const *a, T const *b, T *out, size_t n) {
  size_t grids = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaProfilerStart();
  add_kernel<<<grids, BLOCK_SIZE>>>(a, b, out, n);
  cudaProfilerStop();
}

template <typename T, size_t BLOCK_SIZE = 256>
void add_element2(T const *a, T const *b, T *out, size_t n) {
  size_t grids = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaProfilerStart();
  add_element2_kernel<<<grids, BLOCK_SIZE>>>(a, b, out, n);
  cudaProfilerStop();
}

template <typename T, size_t BLOCK_SIZE = 256>
void add_element2_interleaved(T const *a, T const *b, T *out, size_t n) {
  size_t grids = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaProfilerStart();
  add_element2_interleaved_kernel<<<grids, BLOCK_SIZE>>>(a, b, out, n);
  cudaProfilerStop();
}

void launch_add(const torch::Tensor &a, const torch::Tensor &b,
                torch::Tensor &out) {
  // Get the size of the tensors
  size_t n = a.numel();

  // Launch the add function
  switch (a.dtype().toScalarType()) {
    case at::ScalarType::Float:
      add(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
      break;

    case at::ScalarType::Double:
      add(a.data_ptr<double>(), b.data_ptr<double>(), out.data_ptr<double>(),
          n);
      break;

    case at::ScalarType::Half:
      add((half const *)(a.data_ptr<at::Half>()),
          (half const *)(b.data_ptr<at::Half>()),
          (half *)(out.data_ptr<at::Half>()), n);
      break;

    case at::ScalarType::BFloat16:
      add((__nv_bfloat16 const *)(a.data_ptr<at::BFloat16>()),
          (__nv_bfloat16 const *)(b.data_ptr<at::BFloat16>()),
          (__nv_bfloat16 *)(out.data_ptr<at::BFloat16>()), n);
      break;

    case at::ScalarType::Int:
      add(a.data_ptr<int>(), b.data_ptr<int>(), out.data_ptr<int>(), n);
      break;

    case at::ScalarType::Long:
      add(a.data_ptr<long>(), b.data_ptr<long>(), out.data_ptr<long>(), n);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

void launch_add_element2(const torch::Tensor &a, const torch::Tensor &b,
                         torch::Tensor &out) {
  // Get the size of the tensors
  size_t n = a.numel();

  // Launch the add function
  switch (a.dtype().toScalarType()) {
    case at::ScalarType::Float:
      add_element2(a.data_ptr<float>(), b.data_ptr<float>(),
                   out.data_ptr<float>(), n);
      break;

    case at::ScalarType::Double:
      add_element2(a.data_ptr<double>(), b.data_ptr<double>(),
                   out.data_ptr<double>(), n);
      break;

    case at::ScalarType::Half:
      add_element2((half const *)(a.data_ptr<at::Half>()),
                   (half const *)(b.data_ptr<at::Half>()),
                   (half *)(out.data_ptr<at::Half>()), n);
      break;

    case at::ScalarType::BFloat16:
      add_element2((__nv_bfloat16 const *)(a.data_ptr<at::BFloat16>()),
                   (__nv_bfloat16 const *)(b.data_ptr<at::BFloat16>()),
                   (__nv_bfloat16 *)(out.data_ptr<at::BFloat16>()), n);
      break;

    case at::ScalarType::Int:
      add_element2(a.data_ptr<int>(), b.data_ptr<int>(), out.data_ptr<int>(),
                   n);
      break;

    case at::ScalarType::Long:
      add_element2(a.data_ptr<long>(), b.data_ptr<long>(), out.data_ptr<long>(),
                   n);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

void launch_add_element2_interleaved(const torch::Tensor &a,
                                     const torch::Tensor &b,
                                     torch::Tensor &out) {
  // Get the size of the tensors
  size_t n = a.numel();

  // Launch the add function
  switch (a.dtype().toScalarType()) {
    case at::ScalarType::Float:
      add_element2_interleaved(a.data_ptr<float>(), b.data_ptr<float>(),
                               out.data_ptr<float>(), n);
      break;

    case at::ScalarType::Double:
      add_element2_interleaved(a.data_ptr<double>(), b.data_ptr<double>(),
                               out.data_ptr<double>(), n);
      break;

    case at::ScalarType::Half:
      add_element2_interleaved((half const *)(a.data_ptr<at::Half>()),
                               (half const *)(b.data_ptr<at::Half>()),
                               (half *)(out.data_ptr<at::Half>()), n);
      break;

    case at::ScalarType::BFloat16:
      add_element2_interleaved(
          (__nv_bfloat16 const *)(a.data_ptr<at::BFloat16>()),
          (__nv_bfloat16 const *)(b.data_ptr<at::BFloat16>()),
          (__nv_bfloat16 *)(out.data_ptr<at::BFloat16>()), n);
      break;

    case at::ScalarType::Int:
      add_element2_interleaved(a.data_ptr<int>(), b.data_ptr<int>(),
                               out.data_ptr<int>(), n);
      break;

    case at::ScalarType::Long:
      add_element2_interleaved(a.data_ptr<long>(), b.data_ptr<long>(),
                               out.data_ptr<long>(), n);
      break;

    default:
      TORCH_CHECK(false, "Unsupported tensor dtype");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &launch_add, "Add kernel (1 element) for CUDA tensors");
  m.def("add_element2", &launch_add_element2,
        "Add kernel (2 elements) for CUDA tensors");
  m.def("add_element2_interleaved", &launch_add_element2_interleaved,
        "Add kernel (2 unrolling) for CUDA tensors");
}