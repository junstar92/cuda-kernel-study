# CUDA Kernel Study: Reduction

Parallel reduction (summing an array on the GPU) is deceptively tricky: naive kernels suffer from atomic **contention** and **serialization**. On an NVIDIA A6000, we start with a naive `atomicAdd` approach and progressively apply shared-memory, warp-level, and library-based optimizations. We benchmark both float32 and int32 and explain why their performance trends diverge. All techniques follow NVIDIA’s official guidance.

## Naive AtomicAdd 

A simple reduction lets every thread do `atomicAdd(out, in[idx])`. It’s correct but forces all updates through one location, creating extreme contention. Global atomics are hundreds of cycles; with many threads, throughput collapses—especially for float32, where the hardware can’t freely reorder/aggregate updates.

```c++
// This kernel uses atomicAdd operations to accumulate each input element into a
// single global output variable.
template <typename T>
__global__ void reduction_atomic_kernel(T* out, T const* in, size_t const n) {
  size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    atomicAdd(out, in[idx]);
  }
}
```

## Block-Wise Accumulation with Shared Memory (Atomic per Block)

Reduce global contention by first accumulating per block in shared memory, then doing one global `atomicAdd` per block. This turns `N` global atomics into roughly `B` (blocks), a huge win—though threads still contend on the shared accumulator.

```c++
// Each block accumulates a partial sum in its shared memory, then the block’s
// result is atomically added to the global output. By reducing global atomic
// operations (one per block instead of one per element), this approach
// alleviates the atomic contention bottleneck.
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
```

## In-Block Parallel Reduction (Shared Memory with Manual Summation)

Eliminate per-thread atomics entirely: load values to shared memory and do a **tree reduction** within the block. Only thread 0 performs a single global `atomicAdd`.

```c++
// This kernel performs a block-wide reduction using shared memory. Each block’s
// threads iteratively halve the active range and sum values, and the final
// block result is atomically added to the global output.
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
```

## Warp Shuffle Optimization for the Last Steps

Once only 32 values remain, finish inside the warp with register shuffles (`__shfl_sync`) - no shared memory or `__syncthreads()` needed.

```c++
// This kernel uses warp-level shuffle operations to complete the final steps of
// the reduction once 32 partial values remain. Using __shfl_sync intrinsics
// replaces the last 5 iterations of the shared-memory reduction loop, halving
// the number of active threads at each step without additional __syncthreads.
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
```

It shaves off the final syncs and shared-memory ops; modest but consistent gains.

## Unrolling the Final Reduction Loop (Manual Warp Reduction)

Before shuffles were available, people manually unrolled the last warp using `volatile` shared-memory pointers. It achieves the same sync-free tail reduction.

```c++
// This kernel manually unrolls the final 6 iterations of the reduction loop in
// shared memory (reducing 64 values down to 1). It is similar in intent to
// `reduce_shuffle`, but instead of using warp intrinsics, it explicitly
// performs the last 6 add operations in code (using volatile memory to ensure
// correctness).
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
```

It is comparable to the shuffle version; tiny wins/losses depend on compiler/arch.

## Using Cooperative Groups for Warp Reduction

CUDA's Cooperative Groups (introduced in CUDA 9) provides concise, correct warp collectives. Partition the block into warps and call `cg::reduce`.

```c++
// This kernel uses cooperative groups to perform the final reduction at warp
// level when 64 partial values remain. It partitions the block into warps and
// uses `cg::reduce` on a 32-thread warp (combining values from two warps),
// eliminating the need for manual unrolling or explicit shuffle intrinsics in
// the last 6 steps.
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
```

It is as fast as low-level shuffles/unrolling, with cleaner code.

## Unrolling Input Loading (Processing 2 Elements per Threads)

Give each thread more work: load and sum two elements before the block reduction. This improves memory utilization and reduces per-thread overhead. It is consistently the fastest in our tests for large `N`.

```c++
// This kernel is a variant of `reduce_with_cg` that unrolls the input by a
// factor of 2: each thread sums two elements from the input. By processing two
// elements per thread, it halves the number of threads (and iterations) needed
// before the cooperative groups reduction handles the final 64 values similarly
// to `reduce_with_cg`.
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
```

## Using CUB's Optimized BlockReduce

Finally, we implemented a reduction using the CUB library which provides highly optimized primitives for CUDA. CUB offers `BlockREduce` templates that handle the intra-block reduction efficiently (with warp-level optimizations, etc), and we simply use `BlockReduce<T, BLOCK_SIZE>::Sum()` to get each block's sum, then have thread 0 do an `atomicAdd` to the global output. The rest of the kernel is similar to our manual shared-memory reduction.

```c++
// This kernel uses the CUB library’s BlockReduce to sum values within each
// block. CUB’s optimized reduction routine computes the block’s partial sum
// (stored in shared memory), and the thread 0 then atomically adds that partial
// sum to the global output.
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
```

NVIDIA's documentation and blogs often suggest using CUB or similar libraries for reductions and scans, because they are *tuned for each architecture* and can save you the effort of writing and tuning your own code. CUB will, for example, choose the best way to reduce (using shuffle on newer GPUs or other tricks) and manage edge cases.

## Benchmark Results

We benchmarked all the above kernel versions on an NVIDIA A6000 for various array size `N`. We measured the kernel execution time (in milliseconds) for summing an array of length `N`. For reference, we also include the time taken by PyTorch's built-in `.sum()` (which we denote as "Torch"), which is highly optimized. We did tests for both **float32** and **int32** data types to highlight the differences in behavior.

### Float32 Reduction Performance

| N | Torch | Atomic (global) | Atomic (shared) | Shared | Shuffle | Unroll | CG | CG (unroll 2) | CUB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65,536 | 0.016384 | 0.118784 | 0.015360 | 0.005120 | 0.005120 | 0.005120 | 0.005120 | 0.005120 | 0.005120 |
| 131,072 | 0.009216 | 0.222208 | 0.023552 | 0.006144 | 0.005120 | 0.005120 | 0.005120 | 0.006112 | 0.005120 |
| 262,144 | 0.010240 | 0.441344 | 0.035840 | 0.007168 | 0.006144 | 0.006144 | 0.006144 | 0.006144 | 0.006144 |
| 524,288 | 0.011264 | 0.878592 | 0.059392 | 0.010368 | 0.008192 | 0.008192 | 0.008192 | 0.008192 | 0.008192 |
| 1,048,576 | 0.015360 | 1.752064 | 0.106496 | 0.016384 | 0.012288 | 0.011264 | 0.012288 | 0.011264 | 0.012208 |
| 2,097,152 | 0.022528 | 3.500032 | 0.200704 | 0.029696 | 0.021504 | 0.020480 | 0.020480 | 0.018432 | 0.019456 |
| 4,194,304 | 0.034912 | 6.994944 | 0.387072 | 0.054272 | 0.035840 | 0.034320 | 0.034816 | 0.030720 | 0.032768 |
| 8,388,608 | 0.057344 | 13.985200 | 0.760704 | 0.103424 | 0.066560 | 0.062464 | 0.062464 | 0.054240 | 0.060416 |
| 16,777,216 | 0.102400 | 27.966463 | 1.506304 | 0.202624 | 0.126976 | 0.119808 | 0.117760 | 0.098304 | 0.114688 |
| 33,554,432 | 0.207872 | 55.927807 | 3.020992 | 0.399360 | 0.248832 | 0.233472 | 0.229376 | 0.187392 | 0.228352 |

Looking at the float32 results:

- **Global atomic** scales poorly (55.9 ms @ 33M), >200x slower than best.
- **Block atomic** helps, but in-block float atomics still serialize (3.02 ms @ 33M).
- **Shared-memory tree** reduction removes in-block atomics (0.399 ms @ 33M).
- **Warp-scope tail** (shuffle/unroll/CG) gives final ~1.5-2x boost (down to ~0.23-0.25 ms).
- **Unrolling inputs (x2)** is best overall (0.187 ms @ 33M), slightly beating Torch and CUB in our setup.

For float32, **PyTorch's built-in reduction** (likely using its own optimized kernel, possibly leveraging something like CUB or a tree reduction) clocks in at 0.2079 ms for 33M. Interestingly, our best custom kernel (0.187 ms) slightly outperforms PyTorch's in this case. This could be due to PyTorch carrying some extra overhead or not using the exact method we did - or simply measurement variance. Regardless, all the optimized approaches (shuffle, unroll, CG, CUB) are within the same order of magnitude and far beyond the naive methods.

### Int32 Reduction Performance

| N  | Torch | Atomic (global) | Atomic (shared) | Shared | Shuffle | Unroll | CG | CG (unroll 2) | CUB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65,536 | 0.018432 | 0.005120 | 0.004096 | 0.005120 | 0.004128 | 0.004224 | 0.004096 | 0.005120 | 0.004096 |
| 131,072 | 0.013312 | 0.007168 | 0.005120 | 0.006144 | 0.005120 | 0.005120 | 0.005120 | 0.005120 | 0.005120 |
| 262,144 | 0.014336 | 0.009216 | 0.006144 | 0.008064 | 0.006144 | 0.006144 | 0.006144 | 0.006144 | 0.006144 |
| 524,288 | 0.020480 | 0.013312 | 0.007168 | 0.010240 | 0.008192 | 0.008192 | 0.008192 | 0.008192 | 0.008192 |
| 1,048,576 | 0.039936 | 0.023552 | 0.011264 | 0.017408 | 0.012288 | 0.012288 | 0.011264 | 0.011264 | 0.011264 |
| 2,097,152 | 0.072704 | 0.043008 | 0.018432 | 0.029696 | 0.021504 | 0.020480 | 0.020480 | 0.018432 | 0.018592 |
| 4,194,304 | 0.130048 | 0.081312 | 0.031744 | 0.054272 | 0.036864 | 0.034816 | 0.033792 | 0.030720 | 0.031696 |
| 8,388,608 | 0.244736 | 0.157696 | 0.053392 | 0.104448 | 0.067584 | 0.063488 | 0.060272 | 0.053248 | 0.054272 |
| 16,777,216 | 0.473088 | 0.311296 | 0.098304 | 0.203776 | 0.130048 | 0.120832 | 0.112640 | 0.098304 | 0.099328 |
| 33,554,432 | 0.946176 | 0.617472 | 0.187392 | 0.401408 | 0.253952 | 0.237568 | 0.217088 | 0.187392 | 0.188416 |

The int32 results reveal some interesting differences compared to float32:

- Even **naive global atomic** is fairly strong (~0.62 ms @ 33M), much better than float32.
- **Block atomic** already reaches top-tier (~0.187 ms @ 33M).
- Tree + warp optimizations converge in the same ~0.18-0.25 ms band.
- Torch's int path is slower in out tests (likely different accumulation strategy).

In summary, for int32 data, even a naive atomic approach leverages GPU hardware capabilities to achieve high throughput. The benefit of elaborate optimizations is less pronounced, though they do still yield the best results. For float32, each layer of optimization was necessary to climb out of the pit of atomic contention and reach peak performance.

### Why Float32 and Int32 Behave So Differently

**Associativity matters**: Integer addition is associative; floating-point is not. Hardware can safely aggregate many **int** atomics (e.g., warp-level coalescing or L2 update combining) without changing results, but doing so for **float** would after rounding/order and violate atomic semantics.

**Hardware paths differ**: For int, modern GPUs effectively turn many contended aotmics into fewer updates; for float, updates are applied individually. In shared memory, float atomics may be implemented via CAS loops on some architectures, amplifying contention cost.

For float32, avoid atomics - use block reductions + warp tiles. For int32, atomics are already efficient; classic reduction still help but are less critical.

## Conclusion

We started with a naive reduction kernel and improved it step by step using shared memory, warp-level operations, and library routines. The final outcomes demonstrate how critical it is to minimize atomic operations and exploit the GPU’s parallel capabilities:

- Using shared memory to accumulate results per block dramatically reduces contention and is the first big leap in performance.
- Within each block, doing parallel reductions (avoiding atomic operations inside the block) yields another huge speedup, especially for float data.
- Warp-level optimizations (like shuffle or manual unrolling) shave off the last bits of overhead, bringing the implementation to near-optimal performance.
- Advanced features like cooperative groups can make the code cleaner without sacrificing speed, and processing multiple elements per thread (unrolling) taps into additional performance by better using memory bandwidth.
- The CUB library provides these optimizations, and its performance is on par with the best custom code, which is a testament to writing high-quality, portable code using established libraries.

Ultimately, the reduction problem exemplifies many key concepts of CUDA optimization: memory hierarchy (global vs shared vs registers), synchronization costs, warp granularity, and the importance of algorithmic associativity. By progressively applying these optimizations, we turned an algorithm that took tens of milliseconds into one that completes in a few tenths of a millisecond – over two orders of magnitude faster – all while doing the same amount of arithmetic. This kind of performance difference can be critical in large-scale GPU applications.

# References

- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler)
- stackoverflow: [Shared Memory's atomicAdd with int and float have different SASS](https://stackoverflow.com/questions/70912964/shared-memorys-atomicadd-with-int-and-float-have-different-sass)
- [Shared Memory Atomic througput on RTX 4090](https://girl.surgery/shmem_atomic)
- [NVIDIA/cccl](https://github.com/NVIDIA/cccl/tree/main)