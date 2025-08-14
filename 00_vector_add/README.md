# CUDA Kernel Study: Vector Addition

This document provides a basic introduction to CUDA programming using a simple vector addition kernel. We will cover this basic programming model, and then explore optimization technique like increasing workload per thread and using vectorized memory access.

## The CUDA Programming Model: A Brief Overview

CUDA uses a model where the **host** (the CPU) controls the overall application and lanches computations on the **device** (the GPU). These computations are defined in special functions called **kernels**.

- **Kernel** (`__global__`): A function, marked with `__global__`, that runs on the GPU. It is executed by a large number of threads in parallel.
- **Threads**: The basic unit of execution. Each thread runs the same kernel code but has a unique ID, allowing it to work on different parts of the data.
- **Blocks**: Threads are grouped into **blocks**. Threads within the same block can cooperate efficiently.
- **Grid**: Blocks are organized into a **grid**. The grid represents the entire set of threads launched for a single kernel execution.

When launching a kernel, we define the dimensions of the grid and blocks, which determines the total number of threads.

## A Basic CUDA Kernel: Vector Addition

The most straightforward way to write a parallel vector addition is to assign one thread to compute the sum of one element from each input vector.

### The Kernel Code

Here is a basic kernel that adds two vectors, `a` and `b`, and stores the result in `out`.
```c++
template <typename T>
__global__ void add_kernel(T const *a, T const *b, T *out, size_t n) {
  // Calculate the unique global index for this thread
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  // Check boundary to prevent writing out of bounds
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}
```

### How It Works

1. `__global__ void add_kernel(...)`: Defines `add_kernel` as a function that will be executed on the GPU and can be called from the CPU.
2. `size_t idx = ...`: This is the most critical line for a simple 1D kernel.
    - `threadIdx.x`: The index of the current thread within its block (e.g., 0 to 255).
    - `blockIdx.x`: The index of the current block within the grid.
    - `blockDim.x`: The number of threads in each block.
    - This formula gives each thread a unique global index (`idx`) across the entire grid, so it know which element of the vectors `a`, `b`, and `out` to handle.
3. `if (idx < n)`: This boundary check is essential. We often launch a grid of threads that is slightly larger than the data size (for convenience, as the data size `n` might not be a perfect multiple of the block size). This `if` statement ensures that threads with an `idx` greater than or equal to `n` do nothing, preventing memory corruption.

## Optimizing the Kernel: Vectorized Access

A common optimizing strategy is to make each thread to more work. This can reduce overhead. An even better optimization for memory-intensive tasks is to use **vectorized data types** to perform memory operations more efficiently.

### Coalesced Memory Access

The most significant factor for performance in kernels like this is memory bandwidth. GPUs achieve maximum bandwidth when threads in a **warp** (a group of 32 threads) access consecutive memory locations simultaneously. This is called **coalesced memory access**. A single, wide memory transaction can service all 32 threads at once. If threads access scattered memory locations, the GPU may need to perform multiple, separate memory transactions, which is much slower.

The `add_kernel` already uses a coalesced access pattern because adjacent threads (e.g., thread 0, 1, 2, ...) access adjacent memory locations (`a[0]`, `a[1]`, `a[2]`, ...).

### Using Vectorized Types for Better Coalescing

We can further improve memory efficiency by explicitly loading multiple data elements at once. For data types like `half` (16-bit float), we can use `half2`, which is a struct containing two `half` values. The GPU can load a `half2` vector (32 bits) in a single transaction.

```c++
// Specialization for 'half' data type using the 'half2' vector type
template <>
__global__ void add_element2_kernel(half const *a, half const *b, half *out,
                                    size_t n) {
  // Each thread now processes two elements, so we stride by 2
  size_t idx = 2 * blockDim.x * blockIdx.x + threadIdx.x * 2;

  if (idx < n) {
    // Perform a single vectorized load, add, and store for 2 elements
    reinterpret_cast<half2 &>(out[idx]) =
        reinterpret_cast<half2 const &>(a[idx]) +
        reinterpret_cast<half2 const &>(b[idx]);
  }

  // ... (boundary handling for odd n)
}
```

Here, `reinterpret_cast<half2&>` tells the compiler to treat two consecutive `half` elements as a single `half2` unit. This allows the hardware to issue a single, wider memory instruction to load/store both elements, ensuring optimal use of memory bandwidth.

### Performance Analysis: Why Didn't the Optimization Work?

You provided performance results from an `NVIDIA A6000 GPU`. This data reveals an important lesson in optimization: the theoretically "better" kernel is not always faster in practice.

```
Add Performance (ms):
               N    PyTorch     Triton        add  add 2 elems  add 2 elems (interleaved)
0   4.096000e+03   0.004096   0.004096   0.004096     0.004096                   0.004096
1   8.192000e+03   0.004096   0.004096   0.004096     0.004096                   0.004096
2   1.638400e+04   0.004096   0.004096   0.004096     0.004096                   0.004096
3   3.276800e+04   0.004096   0.004096   0.004096     0.004096                   0.004096
4   6.553600e+04   0.005120   0.005120   0.005120     0.005120                   0.005120
5   1.310720e+05   0.005120   0.005120   0.005120     0.005216                   0.005216
6   2.621440e+05   0.006144   0.006144   0.007168     0.007168                   0.007168
7   5.242880e+05   0.009216   0.009216   0.009216     0.009216                   0.009216
8   1.048576e+06   0.013312   0.014176   0.013312     0.014336                   0.013312
9   2.097152e+06   0.022528   0.023552   0.022528     0.023552                   0.022528
10  4.194304e+06   0.040960   0.041984   0.039936     0.041984                   0.041984
11  8.388608e+06   0.077824   0.078848   0.075776     0.078848                   0.077824
12  1.677722e+07   0.151552   0.152528   0.148480     0.153600                   0.150528
13  3.355443e+07   0.297984   0.299008   0.292864     0.304128                   0.294912
14  6.710886e+07   0.590976   0.590848   0.580608     0.604160                   0.583680
15  1.342177e+08   1.178624   1.177536   1.157120     1.209344                   1.161216
16  2.684355e+08   2.351104   2.349056   2.308096     2.415616                   2.316288
17  5.368709e+08   4.694016   4.692992   4.614144     4.836352                   4.625408
18  1.073742e+09   9.392640   9.372672   9.223680     9.649152                   9.244672
19  2.147484e+09  18.752512  18.737152  18.447359    19.334145                  18.477057
```

For large inputs, basic `add` kernel is the **fastest** of the custom kernels, performing on par with or even slightly better then PyTorch's default. Surprisingly, the `add 2 elems` kernel, which uses vectorized access, is consistently the **slowest**.

This result might seem counter-intuitive, but it highlights critical concepts in GPU performance.

1. **The Operation is Memory-Bound**: This remains the most important factor. Vector addition is fundamentally limited by the speed at which the GPU can fetch data from memory (memory bandwidth), not by how fast it can perform addition. The A6000 has extremely high memory bandwidth, and the simple `add_kernel` already uses a perfectly **coalesced** access pattern that takes full advantage of it. 

2. **Compiler & Hardware are Highly Optimized**: The CUDA compiler (NVCC) and modern hardware like the A6000 are exceptionally good at optimizing simple, clean code. The compiler can analyze the basic `add_kernel` and generate machine code that is nearly perfect in terms of memory access, likely using wide memory operations automatically. Your manual `half2` vectorization is essentially telling the compiler to do something it was probably already doing.

3. **Boundary Checks and Logical Complexity**: This is the likely culprit for the slowest in `add 2 elems`. Your optimized kernel introduces more complex logic.
    - **Simple Check** (`add_kernel`): The `if (idx < n)` check is simple and creates minimal **thread divergence**. In a warp of 32 threads, only the very last warp that processes the end of the array will have some threads active and some inactive.
    - **Complex Check** (`add_element2_kernel`): This kernel has a more complex boundary condition to handle arrays with an odd number of elements (`if (n % 2 == 1 && ...)`). This check creates overhead. Even though it's only true for a single thread in the entire grid, the condition must be evaluated by many more threads, potentially causing divergence and hurting performance. The slight overhead from this more complex logic outweighs the benefit of explicit vectorization.

4. **Framework Overhead**: The measured times include not just the kernel execution but also the overhead from the PyTorch C++ extension API for launching the kernel. For very fast kernels like vector addition, this framework overhead can be a significant portion of the total measured time, masking any small performance differences between the kernels themselves.

In summary, for a memory-bound kernel on a powerful GPU, the simplest code often wins. The `add_kernel` is the fastest because it has the most straightforward, perfectly coalesced memory access pattern and the simplest possible logic, which allows the advanced compiler and hardware to generate the most efficient execution plan without being hindered by complex manual optimizations.