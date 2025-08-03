#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cub/block/block_reduce.cuh>

namespace cg = cooperative_groups;

// This kernel uses atomic operations to accumulate the individual elements
// in a single global memory.
template <typename T>
__global__ void reduction_atomic_kernel(T* out, T const* in, size_t const n) {
  size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    atomicAdd(out, in[idx]);
  }
}

// Each block can accumulate partial results in isolated shared memory.
// The final result is then accumulated into the global memory.
// This approach reduces the number of atomic operations, which can be a
// bottleneck.
template <typename T>
__global__ void reduction_atomic_shm_kernel(T* out, T const* in,
                                            size_t const n) {
  __shared__ T partial_sum;
  size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (threadIdx.x == 0) {
    partial_sum = T{0};
  }
  __syncthreads();

  if (idx < n) {
    atomicAdd(&partial_sum, in[idx]);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(out, partial_sum);
  }
}

// This kernel uses shared memory to accumulate results within each block.
// The final result is then accumulated into the global memory.
template <typename T>
__global__ void reduce_shm_kernel(T* out, T const* in, size_t const n) {
  extern __shared__ char smem[];
  T* partial_sum = reinterpret_cast<T*>(smem);
  size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;

  partial_sum[threadIdx.x] = (idx < n) ? in[idx] : T{0};

  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    __syncthreads();
    if (threadIdx.x < i) {
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
    }
  }

  if (threadIdx.x == 0) {
    atomicAdd(out, partial_sum[0]);
  }
}

// This kernel uses warp-level primitives to reduce the number of threads
// in last 5 iterations of the reduction.
template <typename T>
__global__ void reduce_shuffle(T* out, T const* in, size_t const n) {
  extern __shared__ char smem[];
  T* partial_sum = reinterpret_cast<T*>(smem);
  size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;

  partial_sum[threadIdx.x] = (idx < n) ? in[idx] : T{0};

  // this reduction in shared memory stops at 32 partial results
  for (int i = blockDim.x / 2; i > 16; i /= 2) {
    __syncthreads();
    if (threadIdx.x < i) {
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
    }
  }

  // now we have 32 partial results, we can use shuffle to reduce them with
  // warp-level primitives
  T final_sum = partial_sum[threadIdx.x];
  if (threadIdx.x < 32) {
    // This replace the last 5 iterations of the `reduce_shm_kernel` kernel.
    // In each shuffle, at least half of the threads only participate in the
    // reduction, so the number of threads is halved each time.
    final_sum += __shfl_sync(0xffffffff, final_sum, threadIdx.x + 16);
    final_sum += __shfl_sync(0xffffffff, final_sum, threadIdx.x + 8);
    final_sum += __shfl_sync(0xffffffff, final_sum, threadIdx.x + 4);
    final_sum += __shfl_sync(0xffffffff, final_sum, threadIdx.x + 2);
    final_sum += __shfl_sync(0xffffffff, final_sum, 1);
  }

  if (threadIdx.x == 0) {
    atomicAdd(out, final_sum);
  }
}

// This kernel uses unrolling to optimize the last 6 iterations of the
// reduction in shared memory. It is similar to `reduce_shuffle`, but it
// unrolls the last 6 iterations instead of using warp-level primitives.
template <typename T>
__global__ void reduce_unroll(T* out, T const* in, size_t const n) {
  extern __shared__ char smem[];
  T* partial_sum = reinterpret_cast<T*>(smem);
  size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;

  partial_sum[threadIdx.x] = (idx < n) ? in[idx] : T{0};

  // this reduction in shared memory stops at 64 partial results
  for (int i = blockDim.x / 2; i > 32; i /= 2) {
    __syncthreads();
    if (threadIdx.x < i) {
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
    }
  }

  // unrolling the last 6 iterations of the `reduce_shm_kernel` kernel
  if (threadIdx.x < 32) {
    // volatile qualifier is used to prevent the compiler from optimizing
    // the memory access, which is necessary for correctness in this case
    volatile T* vmem = partial_sum;
    vmem[threadIdx.x] += vmem[threadIdx.x + 32];
    vmem[threadIdx.x] += vmem[threadIdx.x + 16];
    vmem[threadIdx.x] += vmem[threadIdx.x + 8];
    vmem[threadIdx.x] += vmem[threadIdx.x + 4];
    vmem[threadIdx.x] += vmem[threadIdx.x + 2];
    vmem[threadIdx.x] += vmem[threadIdx.x + 1];
  }

  if (threadIdx.x == 0) {
    atomicAdd(out, partial_sum[0]);
  }
}

// This kernel uses cooperative groups to reduce the number of threads
// in the last 6 iterations of the reduction.
template <typename T>
__global__ void reduce_with_cg(T* out, T const* in, size_t const n) {
  extern __shared__ char smem[];
  T* partial_sum = reinterpret_cast<T*>(smem);
  size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;

  partial_sum[threadIdx.x] = (idx < n) ? in[idx] : T{0};

  for (int i = blockDim.x / 2; i > 32; i /= 2) {
    __syncthreads();
    if (threadIdx.x < i) {
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
    }
  }

  // the last 64 values can be handled with cooperative groups
  auto threadblock = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(threadblock);
  if (warp.meta_group_rank() == 0) {
    int warp_lane = warp.thread_rank();
    T sum = partial_sum[warp_lane] + partial_sum[warp_lane + 32];
    sum = cg::reduce(warp, sum, cg::plus<T>());
    if (warp_lane == 0) {
      atomicAdd(out, sum);
    }
  }
}

template <typename T>
__global__ void reduce_with_cg_unroll_2(T* out, T const* in, size_t const n) {
  extern __shared__ char smem[];
  T* partial_sum = reinterpret_cast<T*>(smem);
  size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;

  partial_sum[threadIdx.x] = (idx < n / 2) ? in[idx] : T{0};
  partial_sum[threadIdx.x] += (idx + n / 2 < n) ? in[idx + n / 2] : T{0};

  for (int i = blockDim.x / 2; i > 32; i /= 2) {
    __syncthreads();
    if (threadIdx.x < i) {
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
    }
  }

  // the last 64 values can be handled with cooperative groups
  auto threadblock = cg::this_thread_block();
  if (n % 2 == 1 &&
      threadblock.group_index().x == cg::this_grid().dim_blocks().x - 1 &&
      threadIdx.x == 0) {
    // If the input size is odd, we need to handle the last element separately
    partial_sum[threadIdx.x] += in[n - 1];
  }
  auto warp = cg::tiled_partition<32>(threadblock);
  if (warp.meta_group_rank() == 0) {
    int warp_lane = warp.thread_rank();
    T sum = partial_sum[warp_lane] + partial_sum[warp_lane + 32];
    sum = cg::reduce(warp, sum, cg::plus<T>());
    if (warp_lane == 0) {
      atomicAdd(out, sum);
    }
  }
}

template <typename T, size_t BLOCK_SIZE>
__global__ void reduction_cub_kernel(T* out, T const* in, size_t const n) {
  using BlockReduce = cub::BlockReduce<T, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;
  T sum = 0;
  if (idx < n) {
    sum += in[idx];
  }
  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    atomicAdd(out, sum);
  }
}

template <size_t VER>
void launch_reduce(torch::Tensor& out, torch::Tensor const& in) {
  // TORCH_CHECK(out.scalar_type() == at::ScalarType::Int &&
  //                 in.scalar_type() == at::ScalarType::Int,
  //             "Both input and output tensors must be of type int.");
  size_t const n = in.numel();
  size_t block_size = 256;
  size_t grid_size = (n + block_size - 1) / block_size;

  auto type = in.scalar_type();

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Int, type, "reduction_kernel", [&]() {
        if constexpr (VER == 0) {
          reduction_atomic_kernel<<<grid_size, block_size>>>(
              out.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), n);
        } else if constexpr (VER == 1) {
          reduction_atomic_shm_kernel<<<grid_size, block_size>>>(
              out.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), n);
        } else if constexpr (VER == 2) {
          reduce_shm_kernel<<<grid_size, block_size,
                              block_size * sizeof(scalar_t)>>>(
              out.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), n);
        } else if constexpr (VER == 3) {
          reduce_shuffle<<<grid_size, block_size,
                           block_size * sizeof(scalar_t)>>>(
              out.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), n);
        } else if constexpr (VER == 4) {
          reduce_unroll<<<grid_size, block_size,
                          block_size * sizeof(scalar_t)>>>(
              out.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), n);
        } else if constexpr (VER == 5) {
          reduce_with_cg<<<grid_size, block_size,
                           block_size * sizeof(scalar_t)>>>(
              out.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), n);
        } else if constexpr (VER == 6) {
          grid_size = (n / 2 + block_size - 1) / block_size;
          reduce_with_cg_unroll_2<<<grid_size, block_size,
                                    block_size * sizeof(scalar_t)>>>(
              out.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), n);
        } else if constexpr (VER == 7) {
          reduction_cub_kernel<scalar_t, 256>
              <<<grid_size, block_size, block_size * sizeof(scalar_t)>>>(
                  out.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), n);
        } else {
          // nothing to do, VER is not supported
        }
      });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduction_atomic", &launch_reduce<0>);
  m.def("reduction_atomic_shm", &launch_reduce<1>);
  m.def("reduction_shm", &launch_reduce<2>);
  m.def("reduction_shuffle", &launch_reduce<3>);
  m.def("reduction_unroll", &launch_reduce<4>);
  m.def("reduction_cg", &launch_reduce<5>);
  m.def("reduction_cg_unroll_2", &launch_reduce<6>);
  m.def("reduction_cub", &launch_reduce<7>);
}