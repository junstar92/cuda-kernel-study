# CUDA Kernel Study: Vectorized Copy

In the world of GPU computing, performance is often limited not by computational power, but by memory bandwidth - the speed at which data can be moved to and from the GPU's memory. For many tasks, especially those with simple arithmetic like copying data, the primary bottleneck is memory access. Therefore, optimizing how a CUDA kernel reads and writes data is crucial for achieving maximum performance.

This study explores one of the most effective techniques for maximizing memory bandwidth: **vectorized memory access**. We will analyze this by comparing two CUDA kernels that perform a simple memory copy. The first is a standard, element-by-element copy, and the second is an optimized version that uses vector data types to move large chunks of data at once. By examining this simple copy operation, we can isolate the impact of memory access patterns on performance.

## Baseline Kernel: A Standard Coalesced Copy

First, let's look at a standard implementation of a copy kernel. This approach assigns one thread to copy one element of data.

```c++
template <typename T>
__global__ void copy_kernel(T const* in, T* out, size_t n) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    out[idx] = in[idx];
  }
}
```

This kernel is a solid starting point. The line `size_t idx = blockDim.x * blockIdx.x + threadIdx.x;` is the standard way to give each thread a unique index to work on. This indexing scheme ensures that threads within the same execution unit (a "warp" of 32 threads) access consecutive memory locations. This pattern is known as **memory coalescing**.

Memory coalescing is a critical hardware feature where the GPU bundles the memory requests from all 32 threads in a warp into a single, large transaction. This is the most efficient way to access global memory. Therefore, this baseline kernel is already well-optimized from a coalescing perspective. Any further improvements must come from a different technique.

## Optimized Kernel: Introducing Vectorized Memory Access

To improve upon the baseline, we can modify the kernel to handle more data per thread. This is achieved by using vector data types, such as `float4`, which represents 4 floating-point numbers packed together.

```c++
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
```

The key to this kernel is the line:
```c++
reinterpret_cast<float4&>(out[idx]) = reinterpret_cast<float4 const&>(in[idx]);
```

Here's what's happening:

1. **Wider Data Access**: Instead of reading and writing a single `float` (4 bytes), each thread now reads and writes a `float4` (16 bytes). This means each thread does 4 times the work of the original kernel.

2. **Compiler Instruction**: The `reinterpret_cast` tells the compiler that it's safe to use a special, wide memory instruction. Instead of generating a standard 32-bit load instruction, it generates a 128-bit vectorized load instruction (e.g., `LDG.E.128`). This single instruction moves 16 bytes of data, whereas the original kernel would have needed 4 separate instructions to move the same amount. In practice, profiling showed that the number of global memory load/store requests decreased from 2,052 to 267.

3. **Efficiency Gains**: Using fewer, wider instructions is more efficient. It reduces the total number of instructions the GPU has to execute and allows the memory system to operate closer to its peak bandwidth.

4. **Alignment**: This technique requires that the memory addresses be properly aligned. For a `float4` (16 bytes), the address must be a multiple of 16. Fortunately, memory alloced with `cudaMalloc` is guaranteed to have sufficient alignment (at least 256 bytes), and the strided index calculation in the kernel maintains this alignment.

The final part of the kernel is a simple loop to handle any leftover lements if the total array size isn't perfectly divisible by the vector size (4, in this case).

## Performance Analysis

The provided performance data compares the execution time of the stardard `copy_kernel` and the `vectorized_copy_kernel` in NVIDIA A6000 GPU (`fp16`), along with optimized libraries like PyTorch and Triton.

```
Copy Performance (ms):
               N    PyTorch     Triton  COPY (CUDA)  Vectorized COPY (CUDA)
0   4.096000e+03   0.004096   0.004096     0.004096                0.004096
1   8.192000e+03   0.004096   0.004096     0.004096                0.004096
2   1.638400e+04   0.004096   0.004096     0.004096                0.004096
3   3.276800e+04   0.004096   0.004096     0.004096                0.004096
4   6.553600e+04   0.004096   0.004096     0.004096                0.004096
5   1.310720e+05   0.005120   0.005120     0.005120                0.005120
6   2.621440e+05   0.005120   0.005120     0.006144                0.005120
7   5.242880e+05   0.007168   0.007168     0.007168                0.007168
8   1.048576e+06   0.010240   0.010176     0.010240                0.009216
9   2.097152e+06   0.016384   0.016384     0.016384                0.015360
10  4.194304e+06   0.028672   0.027648     0.028672                0.027648
11  8.388608e+06   0.052224   0.051200     0.052224                0.051200
12  1.677722e+07   0.099328   0.098304     0.100352                0.097280
13  3.355443e+07   0.191488   0.191408     0.196608                0.190464
14  6.710886e+07   0.377856   0.376832     0.389120                0.375808
15  1.342177e+08   0.750592   0.750592     0.775168                0.748544
16  2.684355e+08   1.497088   1.492064     1.547264                1.491104
17  5.368709e+08   2.987008   2.982912     3.092480                2.979840
18  1.073742e+09   5.964800   5.958656     6.179840                5.955584
19  2.147484e+09  11.955200  11.918336    12.367872               11.900928
```

- For small data size, the performance is similar across all methods. This is because the fixed overhead of launching a CUDA kernel dominates the total time.
- As the data size (`N`) grows, a clear trend emerges: the `vectorized_copy_kernel` is consistently faster than the standard `copy_kernel`.
- At the largest scale (over 2 billion elements), the vectorized version is about **3.9% faster** than the standard version. While this may seem small, in high-performance computing, such gains are significant, especially for a fundamental operation like memory copy.
- Notably, the performance of the hand-written `vectorized_copy_kernel` is competitive with, and sometimes even slightly better than, the highly optimized PyTorch and Triton frameworks. This demonstrates that vectorization is a key technique used in professional-grade libraries.

## Conclusion

This study highlights that while memory coalescing is a fundamental requirement for good performance in CUDA, it is not the final step in optimization. For memory-bound operations, **vectorized memory access** provides an additional layer of performance by increasing the efficiency of the GPU's memory subsystem.

By using vector types like `float4`, we instruct the compiler to use fewer, wider memory instructions. This reduces instruction overhead and allows for higher data throughput, resulting in faster kernel execution. As shown by the benchmarks, this simple change leads to measurable performance improvements that are on par with industry-standard deep learning frameworks. For any developer looking to maximize memory bandwidth in CUDA, vectorization is an essential tool.