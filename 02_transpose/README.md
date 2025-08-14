# CUDA Kernel Study: Matrix Transpose

The matrix transpose is a fundamental operation in high-performance computing (HPC), data science, and AI. It appears everywhere—from linear-algebra solvers to the attention blocks in large language models. On the surface, it looks trivial: for an input matrix $A$, compute its transpose $A^T$ such that $A^T[j][i] = A[i][j]$. Implementing this efficiently on a massively parallel architecture like a GPU is a classic, instructive challenge.

Transpose performance is limited not by arithmetic but by **data movement**. It is a **memory-bound** problem, making it an ideal case study for the GPU memory hierarchy. Achieving high performance requires understanding access patterns, on-chip memory, and the hardware’s execution model.

This walks through a systematic optimization of a CUDA transpose kernel. We first establish a practical performance ceiling and diagnose bottlenecks in naïve implementations. We then apply optimizations—loop unrolling, vectorized I/O, and finally the cornerstone: **shared-memory tiling with bank-conflict avoidance**. Each step is backed by benchmarks and hardware counters, demonstrating not just *what* works but *why*.

> **Note**: For brevity, example kernels omit boundary checks and edge handling.

# Problem Definition and Measurement

## Objective and Primary Metric

We aim to maximize the performance of an out-of-place transpose for an $M \times N$ tensor of type $T$ (here, for `float16`). Performance is reported as **Effective Memory Bandwidth** (GB/s):

$$ \text{Effective BW (GB/s)} = \frac{\text{(bytes read + bytes written)}}{\text{elapsed time seconds} \times 10^9} $$

This metric reflects how efficiently the kernel utilizes the available memory bandwidth of the GPU. A full transpose reads and writes the entire matrix once, so total traffic is $2 \times M \times N \times \text{sizeof(T)} $.

## Diagnostic Metrics via NVIDIA Nsight Compute

While effective bandwidth tells us the final performance, understanding the *why* behind performance changes requires peering into the GPU's hardware behavior. Using the NVIDIA Nsight Compute profiler, the following diagnostic metrics will be used to correlate code changes with their hardware-level impact:

- **DRAM Throughput (GB/s)**: This measures the actual data transfer rate to and from the GPU's off-chip DRAM. It will be compared against the device's theoretical peak bandwidth to gauge efficiency.
- **L2 Hit Rate (%)**: The L2 cache is the last level of on-chip cache before accessing slower DRAM. A high L2 hit rate indicates that the cache is effectively servicing memory requests, which is critical for performance.
- **Global Memory Transactions**: This is the total number of requests issued to the memory subsystem. A primary goal of optimization is to minimize this number. Metrics such as `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` provide a precise count of the sectors transferred for global load operations.
- **Achieved Occupancy (%)**: Occupancy is the ratio of active warps per Streaming Multiprocessor (SM) to the maximum number of warps the SM can support. High occupancy is essential for hiding the high latency of global memory accesses, as it allows the SM's warp scheduler to switch to other ready warps while one is stalled waiting for data.
- **Shared Memory Bank Conflict**: When using shared memory, multiple therads in a warp accessing different addresses in the same memory bank will cause those accesses to be serialized. Profiler counters, such as `smsp__warp_serialize.sum`, directly measure the number of cycles warps are stalled due to these conflicts.

## Experimental Setup

The key specifications of the test GPU and the list of kernel variants under study are summarized below:

- **GPU**: NVIDIA RTX A6000 (48 GB GDDR6, 768 GB/s peak memory bandwidth)
- **Kernels Under Study**:
    - `copy_kernel`: Upper-bound benchmark with ideal memory access
    - `transpose_row_kernel`: Naive - Coalesced loads, scattered stores
    - `transpose_col_kernel`: Naive - Scattered loads, coalesced stores
    - `transpose_col_unroll_4_kernel`: Scalar loop unrolling to reduce instruction overhead
    - `transpose_col_unroll_n_kernel`: Vectorized I/O(`float4`) to reduce transactions
    - `transpose_shm_v0_kernel`: Tiled approach using shared memory
    - `transpose_shm_v1_kernel`: Alternative shared memory indexing for tiling
    - `transpose_shm_v1_unroll_2_kernel`: Tiled approach with unrolling to increase work/thread

# The Upper Bound: `copy_kernel`

To optimize effectively, we first need a target. For a memory-bound operation, the practical limit is the GPU’s sustained DRAM bandwidth. A simple, perfectly coalesced copy sets this upper bound:

```c++
// The performance upper bound: all memory accesses are perfectly coalesced
template <typename T>
__global__ void copy_kernel(T* out, T const* in, size_t const M,
                            size_t const N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t m = idx / N;
  size_t n = idx % N;

  out[m * N + n] = in[m * N + n];
}
```

## Analysis of Memory Access

The key to this kernel's performance lies in its memory access pattern. It uses a 1D grid of threads where each thread computes a unique linear index, `idx` (`m` and `n`). The core operation, `out[m * N + n] = in[m * N + n]`, is perfectly linear and sequential.

In the CUDA execution model, threads are grouped into warps of 32. When a warp executes the copy instruction, thread `t` within the warp accesses `in[base_idx + t]` and writes to `out[base_idx + t]`. Because these 32 threads access 32 contiguous memory locations, the GPU's memory subsystem can **coalesce** these individual requests into a minimal number of large memory transactions. On modern architecture, these 32 requests for 4-byte floats (128 bytes total) can often be serviced by a single 128-byte cache line fetch from L2, representing the most efficient possible use of the memory bus.

## Role as an Upper Bound

Because `copy_kernel` minimizes both instruction overhead and memory transaction overhead, its measured effective bandwidth serves as a practical upper bound for any kernel whose primary job is moving data. The theoretical peak memory bandwidth of a high-end GPU like the NVIDIA H100 is over 3000 GB/s, and a well-written copy can approach a significant fraction of this peak in practice.

We will measure the performance of all subsequent transpose kernels against this gold standard. The gap between a transpose kernel’s performance and the copy kernel’s performance essentially quantifies the overhead introduced by the extra work (index calculations, uncoalesced accesses, synchronization, etc.) required for transposition.

**Measured Bandwidth (GB/s)** - *Copy kernel as baseline*
| Matrix Size (N = M) | `copy_kernel` (GB/s) |
| -- | -- |
| 512 | 186.712 |
| 1024 | 409.600 |
| 2048 | 586.780 |
| 4096 | 668.735 |
| 8192 | 689.853 |
| 16384 | 695.364 |
| 32768 | 696.381 |

*These measurement (on RTX 6000) show the copy kernel approaching ~90% of the hardware's 768 GB/s bandwidth at large sizes, setting an upper bound for a fully optimized transpose.*

# Naive Transpose

The fundamental challenge in optimizing matrix transpose is the conflict between the operation’s data access pattern and the row-major memory layout used in C/C++ and CUDA. Efficient memory access on a GPU hinges on coalescing, where threads within a warp access contiguous memory locations. A naive transpose implementation inevitably breaks this pattern for either its read or its write operations.

## `transpose_row_kernel`: Coalesced Reads, Scattered Writes

The first naive approach maps threads to the input matrix in row-major order (each warp reads a contiguous row from the input):

```c++
// Reads are coalesced (row-wise), but writes are scattered across columns
template <typename T>
__global__ void transpose_row_kernel(T* out, T const* in, size_t const M,
                                     size_t const N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t m = idx / N;
  size_t n = idx % N;

  out[n * M + m] = in[m * N + n];
}
```

- **Loads**: Contiguous across a warp (coalesced).
- **Stores**: Stride by $M$ (scattered, poor coalescing).

## `transpose_col_kernel`: Scattered Reads, Coalesced Writes

The second approach maps threads to the output matrix in row-major order (each warp writes a contiguous row in the output):

```c++
// Reads are scattered across a column, but writes are coalesced (row-wise in output)
template <typename T>
__global__ void transpose_col_kernel(T* out, T const* in, size_t const M,
                                     size_t const N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t m = idx % M;
  size_t n = idx / M;

  out[n * M + m] = in[m * N + n];
}
```

- **Loads**: Stride by $N$ (scattered).
- **Stores**: Contiguous across a row (coalesced).

## Benchmark and Analysis

**Measured Bandwidth (GB/s)** - *Naive transpose variants vs. copy*
| Matrix Size (N = M) | `copy_kernel` | `transpose_row_kernel` | `transpose_col_kernel` |
| -- | -- | -- | -- |
| 512 | 186.712 | 85.333 | 113.778 |
| 1024 | 409.600 | 120.471 | 227.556 |
| 2048 | 586.780 | 137.681 | 282.483 |
| 4096 | 668.735 | 140.635 | 276.523 |
| 8192 | 689.853 | 137.392 | 226.768 |
| 16384 | 695.364 | 132.480 | 199.423 |
| 32768 | 696.381 | 74.739 | 165.776 |

Across all sizes, the `transpose_col_kernel` (scattered read / coalesced write) consistently **outperformed** the `transpose_row_kernel` (coalesced read / scattered write). The performance gap can be explained by how warps access memory and how the GPU's memory hierarchy handles these accesses:

- When a warp accesses **consecutive addresses in memory**, the memory controller can merge the requests into a single transaction. This is **coalesced access**, which maximizes memory throughput.
- In contrast, if threads in a warp access **strided or scattered addresses**, the controller must issue multiple smaller transactions. This dramatically reduces effective bandwidth because: (1) each transaction has a fixed overhead (so more transactions means more overhead), and (2) bandwidth utilization per transaction drops when many fetched bytes are not used by the warp.

In our case, one of the two operations (read or write) is always strided. Modern GPUs have sizeable L1 and L2 caches, which can **partially hide the cost of scattered reads** if there is spatial locality - e.g., even though each thread loads from a different row, nearby rows might reside in the same cache lines. **Scatter writes**, however, are particularly troublesome: global memory stores are generally **write-through to L2** (and eventually DRAM), and they cannot be coalesced as easily. Each thread writing to a different 4-byte location separated by 4KB means essentially 32 separate transactions for a warp. The GPU can buffer some of these writes, but merging them is limited by the stride pattern.

This explains the benchmark trends observed above. For small to medium matrix sizes, `transpose_col_kernel` clearly outperformed `transpose_row_kernel` because coalesced writes are much more efficient than scattered writes (even though its reads are scattered). As `N` grew larger, both patterns suffered from reduced cache effectiveness (once the working set exceeded the cache capacity), but the column-major kernel maintained its lead. At very large sizes (e.g., 32768), the performance of both kernels dropped sharply, indicating that caching could no longer hide the penalty of non-coalesced accesses - but the version with coalesced writes still retained a significant advantage.

*In summary, a naive transpose is __heavily limited by memory access patterns__. To make further gains, we need to tackle the strided access head-on*. [NVIDIA's own example](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc) shows a naive transpose achieving only a fraction of copy bandwidth due to the large stride in either the read or write access. We next explore optimizations to mitigate these issues.

# Unrolling and Vectorization

With the primary bottleneck identified as non-coalesced memory access, our optimization journey continues with two foundational techniques: **loop unrolling** (to enhance instruction-level parallelism) and **vectorized I/O** (to perform wider memory transactions). These optimizations aim to reduce overhead and make better use of the GPU’s wide memory bus. However, as we’ll see, they are not cure-alls – especially when the underlying access pattern remains unfriendly to the memory system.

## Loop Unrolling for Instruction-Level Parallelism (ILP)

Loop unrolling is a compiler optimization that reduces the overhead of loop control logic by replicating the loop body multiple times. This eliminates many branch instructions and loop index updates, creating a longer straight-line sequence of instructions. The GPU’s warp scheduler can then find more independent instructions to issue in parallel, helping hide the latency of arithmetic and memory operations. Our `transpose_col_unroll_4_kernel` demonstrates this principle by having each thread transpose four elements instead of one:

```c++
// Each thread processes 4 elements; loop is unrolled for ILP
template <typename T>
__global__ void transpose_col_unroll_4_kernel(T* out, T const* in, size_t M,
                                              size_t N) {
  constexpr size_t unroll_factor = 4;
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  size_t idx = tid * unroll_factor;

#pragma unroll
  for (size_t i = 0; i < unroll_factor; i++) {
    size_t m = (idx + i) % M;
    size_t n = (idx + i) / M;

    out[n * M + m] = in[m * N + n];
  }
}
```

The `#pragma unroll` directive tells the compiler to fully unroll the loop, effectively inlining four copies of the loop body. This amortizes the cost of calculating indices over four elements and removes the per-iteration branch overhead. We end up with more instructions that the hardware can overlap, improving ILP. In theory, this should hide memory latency better and slightly reduce the instruction count per element.

**Measured Bandwidth (GB/s)**: *Naive vs. Unrolled*
| Matrix Size (N = M) | `transpose_col_kernel` | `transpose_col_unroll_4` |
| -- | -- | -- |
| 512 | 113.778 | 146.286 |
| 1024 | 227.556 | 227.556 |
| 2048 | 282.483 | 221.405 |
| 4096 | 276.523 | 227.556 |
| 8192 | 226.768 | 210.135 |
| 16384 | 199.423 | 143.690 |
| 32768 | 165.776 | 88.892 |

**Expected Benefit**: By unrolling, we reduce the number of instructions and give the GPU more work to do in parallel within each thread. This primarily targets compute and instruction overhead – which is useful if those were limiting factors.

**Reality Check**: In our matrix transpose (a memory-bound problem), simple unrolling yielded only modest gains, and in some cases even hurt performance. For smaller matrices we did see a bump (e.g. for a 1024×1024 matrix, the unrolled kernel reached ~227 GB/s, eliminating the gap that existed with the naive version). But as we scaled up, the benefit evaporated and even reversed. By 2048×2048, `transpose_col_unroll_4` achieved ~221 GB/s versus ~282 GB/s for the naive column kernel – **slower than the baseline**. What happened is that once global memory access dominated the runtime, shaving off a few arithmetic instructions or loop overhead didn’t help much. The kernel was still performing the same number of non-coalesced global loads, so memory bandwidth remained the bottleneck. In fact, the extra index arithmetic introduced by unrolling added its own overhead. In short, unrolling improved ILP but did not address the fundamental issue of memory access pattern, so its impact on this memory-bound kernel was limited.

## Vectorized I/O for Memory-Level Parallelism

Modern GPUs can load and store 32, 64, or 128 bytes per memory transaction. **Vectorized memory access** leverages this by using wider types (e.g., loading a `float4` instead of four `float`s) to move multiple values in one operation. The goal is to perform one 128-bit memory operation (`LDG.E.128` or `STG.E.128` in SASS) to move four 32-bit values, instead of four separate 32-bit instructions. This reduces the total number of memory transactions and can increase effective bandwidth.

Our `transpose_col_unroll_n_kernel` applies this strategy to the **store** operations:

```c++
// Vectorized store (float4) to reduce global memory transactions
template <typename T>
__global__ void transpose_col_unroll_n_kernel(T* out, T const* in, size_t M,
                                              size_t N) {
  constexpr size_t unroll_factor = 128 / sizeof(T) / 8;
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  size_t idx = tid * unroll_factor;

  T tmp[unroll_factor];
#pragma unroll
  for (size_t i = 0; i < unroll_factor; i++) {
    size_t m = (idx + i) % M;
    size_t n = (idx + i) / M;
    tmp[i] = in[m * N + n]; // scattered global loads
  }
  reinterpret_cast<float4&>(out[idx]) = *reinterpret_cast<float4*>(tmp); // vectorized store
}
```

Each thread still loads 4 elements from the input (one at a time, scattered across 4 rows of that column), but then writes them out with **one coalesced 128-bit store**. A warp of 32 threads executing this will write `32 × 16 bytes = 512 bytes` in one go. If those addresses are contiguous (and they are, since each thread writes a block of 4 elements right next to its neighbors), the memory system can service the whole warp’s stores with just 4 transactions of 128 bytes each, instead of 16 transactions of 32 bytes for a scalar approach. This 4× reduction in transaction count directly improves memory bandwidth utilization and cuts down the executed store instructions (one store instruction per 4 values, instead of four). On paper, this is a **big win** for a memory-bound kernel.

However, vectorization comes with some strict prerequisites and trade-offs:

- **Alignment**: The memory address for a vector load/store must be aligned to the vector’s size. For a `float4` (16 bytes), this means the starting address must be a multiple of 16. In our code, we ensure alignment by using `idx = tid * 4` for floats, so each thread’s starting index is a multiple of 4. (Memory allocated by `cudaMalloc` is typically 256-byte aligned, so the base pointer is aligned. The main concern is the index offset.)
- **Boundary Condition**: The total number of elements must be a multiple of the vector width. For simplicity, we assumed $M \times N$ is divisible by 4 (and chose dimensions accordingly). In a general implementation, any leftover elements (if the total size isn’t a multiple of 4) would need a fallback handling to avoid out-of-bound vector accesses.

**Measured Bandwidth (GB/s)**: *Naive vs. Unrolled vs. Vectorized*
| Matrix Size (N = M) | `transpose_col_kernel` | `transpose_col_unroll_4` | `transpose_col_unroll_n_kernel` |
| -- | -- | -- | -- |
| 512 | 113.778 | 146.286 | 146.286 |
| 1024 | 227.556 | 227.556 | 310.966 |
| 2048 | 282.483 | 221.405 | 282.483 |
| 4096 | 276.523 | 227.556 | 273.067 |
| 8192 | 226.768 | 210.135 | 154.384 |
| 16384 | 199.423 | 143.690 | 93.099 |
| 32768 | 165.776 | 88.892 | 59.751 |

**Expected Benefit**: By bundling four 32-bit stores into one 128-bit store, we drastically cut down the number of memory transactions on the write path. If the global stores were the limiting factor, we would expect nearly a 4x improvement. Even though the reads are still strided, halving the total number of global memory operations (4 loads + 1 store per thread, instead of 4 loads + 4 stores) should improve throughput. Indeed, in theory a memory-bound transpose should benefit more from reducing memory transactions than from reducing ALU instructions. We hoped that vectorized I/O would significantly outperform the simpler unroll-4 approach.

**Reality Check**: Initially, results looked promising for the vectorized kernel. For smaller matrices, it **did** deliver higher bandwidth – for example, at 1024×1024, the vectorized variant reached ~311 GB/s, versus ~227 GB/s for both the naive and unroll-4 versions. This confirmed that when the problem size was moderate (and perhaps fits in cache), reducing the number of memory operations had a positive effect on performance. However, as we scaled to larger sizes, the performance of the vectorized kernel not only plateaued but eventually **fell below** the naive implementation. By 8192×8192, `transpose_col_unroll_n` managed only ~154 GB/s, whereas the naive kernel was ~226 GB/s (unroll-4 was ~210 GB/s). At 16384×16384, the gap widened further: the vectorized kernel hit ~93 GB/s, barely half of the ~199 GB/s achieved by the naive version. This result was perplexing at first – how could an “optimized” approach perform worse than no optimization?

**Diagnosing the Downfall**: The disappointing outcome highlights that optimizing one aspect of memory access isn’t enough if another part remains a bottleneck. In our case, vectorizing the stores didn’t fix the fundamental issue of **non-coalesced loads** from global memory. Every thread was still reading 4 elements from a column of the input matrix, meaning warps were reading 32 scattered addresses that spanned multiple 128-byte lines. Those scattered loads dominated the runtime for large matrices (especially once the working set exceeded cache and every access went to DRAM). The combined effect was that the input side of the kernel continued to saturate the memory system with inefficient access patterns. The coalesced wide stores helped on the output side, but the overall throughput could not exceed what those strided reads allowed.

Moreover, the vectorized approach introduced some new overhead of its own:

- **Increased Register Pressure**: Each thread now had to hold 4 loaded values (`tmp[0..3]`) in registers before writing them out. This inflated the register usage per thread and can reduce the total number of warps that the GPU can keep in flight (since registers are a limited per-SM resource). For a memory-bound kernel, having fewer active warps is harmful – we want *more* warps to hide latency. NVIDIA’s guidance notes that while vectorized memory operations usually improve throughput, they *increase register pressure and reduce overall parallelism* ([link](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/#:~:text=In%20almost%20all%20cases%2C%20vectorized,you%20cannot%20use%20vectorized%20loads)). In our case, the `float4` store likely pushed the register usage high enough to lower the occupancy. At small matrix sizes this wasn’t apparent, but at large sizes it meant not enough warps were available to cover the long memory latencies, which led to lower overall throughput.


- **Instruction pipeline and scheduling**: In the unroll-4 kernel, each iteration’s load and store happen in sequence (and the compiler might even intermix instructions from different loop iterations to overlap latencies). In the vectorized kernel, we perform all the loads first, then execute one wide store. This alters the instruction mix and timing. It’s possible that the memory pipeline can’t overlap operations as effectively. For instance, a warp has to issue four load instructions back-to-back (for each thread’s loop iterations) and then one store. The benefit is fewer store instructions, but the cost is that those loads still serialize per thread and the final store is deferred. If the compiler or hardware cannot hide the latency of those loads as well as in the unrolled-scalar case, the advantage of fewer store instructions might be negated.

**Takeaway**: Loop unrolling and vectorized I/O were out first remedies to improve the transpose, and they did reduce per-thread instruction overhead and the number of memory transactions *on paper*. But the measured results underscore that these optimization alone **couldn't overcome a fundamentally poor access pattern**. Once the matrix size grows large enough that global memory accesses dominate runtime, the only way to speed things up is to make those accesses themselves more efficient (i.e., **more coalesced and cache-friendly**). Unrolling and vectorization gave us a small boost initially, but ultimately left the main bottleneck (strided global memory loads) intact - even introducing side effects like higher register usage that hurt performance at scale.

In our case, reducing the number of memory instructions helped the *output* side but the *input* side was still incurring 4x more memory transactions than an ideal coalesced approach would. The result was that `transpose_col_unroll_n` didn't significantly improve overall throughput, and its additional overheads actually made it slower for very large matrices. This is a powerful reminder that optimizing at the instruction level only goes so far if you don't fix the data access pattern.

# Tiling with Shared Memory for Coalesced Access

The most effective way to tackle the strided access bottleneck is **tiling with shared memory**. By using fast on-chip shared memory as a manual scratchpad, we can reduce memory accesses to be fully coalesced. The tiled transpose strategy works as follows:

1. **Coalesced Load into Shared Memory**: Each thread block loads a small 2D **tile** (sub-matrix) from the input matrix into a shared memory array. Threads read along rows of the matrix so that contiguous threads access contiguous global memory locations, achieving coalesced reads. For example, if we use a 32x16 tile, a warp of 32 threads will read 32 consecutive elements from a row at once - an optimal coalesced transaction.

2. **Synchronization**: All threads in the block synchroize (`__syncthreads()`) after the tile is loaded. This barrier ensures every element of the tile is in shared memory before any thread proceeds to the next step. It prevents race conditions where some threads might start using the tile data while others are still writing it.

3. **Coalesced Store from Shared Memory**: The threads then write the tile to the output matrix, but with transposed indices. In shared memory, the tile is effectively transposed relative to the global layout. Each thread now reads *down the column of the shared tile* and writes the value to the output. By carefully assigning threads to output coordinates, we ensure these writes are contiguous in the output matrix. Thus, the global stores become perfectly coalesced as well.

Tow CUDA kernels implementing the tiled transpose are given below. They differ in how the shared memory is indexed (which affects bank conflicts, discussed shortly):

```c++
// Version 0: Shared-memory tiling (potential bank conflicts on transpose)
template <typename T, size_t TILE_M = 16, size_t TILE_N = 16, size_t PAD_N = 0>
__global__ void transpose_shm_v0_kernel(T* out, T const* in, size_t const M,
                                        size_t const N) {
  __shared__ T smem[(TILE_N + PAD_N) * TILE_M];

  size_t in_m = blockDim.y * blockIdx.y + threadIdx.y;
  size_t in_n = blockDim.x * blockIdx.x + threadIdx.x;
  // Compute output indices corresponding to transposed block
  size_t block_idx = blockDim.x * threadIdx.y + threadIdx.x;
  size_t block_row = block_idx / blockDim.y;
  size_t block_col = block_idx % blockDim.y;
  size_t out_m = blockDim.y * blockIdx.y + block_col;
  size_t out_n = blockDim.x * blockIdx.x + block_row;
  // Coalesced load from global to shared (row-major in shared)
  smem[threadIdx.y * TILE_N + threadIdx.x] = in[in_m * N + in_n];
  __syncthreads();
  // Each thread now reads a column from shared and write to global (transposed)
  out[out_n * M + out_m] = smem[block_col * TILE_N + block_row];
}

// Version 1: Shared-memory tiling (reindexed to reduce bank conflicts)
template <typename T, size_t TILE_M = 16, size_t TILE_N = 16, size_t PAD_M = 0>
__global__ void transpose_shm_v1_kernel(T* out, T const* in, size_t const M,
                                        size_t const N) {
  __shared__ T smem[TILE_N * (TILE_M + PAD_M)];

  size_t in_m = blockDim.y * blockIdx.y + threadIdx.y;
  size_t in_n = blockDim.x * blockIdx.x + threadIdx.x;
  // Transposed indexing for shared memory
  size_t block_idx = blockDim.x * threadIdx.y + threadIdx.x;
  size_t block_row = block_idx / blockDim.y;
  size_t block_col = block_idx % blockDim.y;
  size_t out_m = blockDim.y * blockIdx.y + block_col;
  size_t out_n = blockDim.x * blockIdx.x + block_row;
  // Colesced load from global, but store into shared in column-major order
  smem[threadIdx.x * TILE_M + threadIdx.y] = in[in_m * N + in_n];
  __syncthreads();
  // Coalesced write from shared (read row-major from shared)
  out[out_n * M + out_m] = smem[block_row * TILE_M + block_col];
}
```

The above code uses a tile size of 16x16 by default for illustration. In both versions, each 16x16 tile is handled by a block of 256 threads. We will later adjust the tile size for better performance

**Measured Bandwidth (GB/s)**: *Tiled traspose vs. naive*
| Matrix Size (N = M) | `transpose_col_kernel` | `transpose_shm_v0_kernel` (16x16 tile) | `transpose_shm_v1_kernel` (16x16 tile) |
| -- | -- | -- | -- |
| 512 | 113.778 | 170.667 | 170.667 |
| 1024 | 227.556 | 372.364 | 372.364 |
| 2048 | 282.483 | 528.516 | 528.516 |
| 4096 | 276.523 | 528.516 | 537.180 |
| 8192 | 226.768 | 526.394 | 530.656 |
| 16384 | 199.423 | 487.256 | 490.218 |
| 32768 | 165.776 | 298.527 | 298.921 |

Even the initial tiled kernel (`transpose_shm_v0`) massively outperforms the naive version, especially at moderate sizes - achieving over **500 GB/s** for matrix sizes up to 8K, which is close to 75% of the device's peak. This is a huge gain compared to the ~200-280 GB/s of the best naive version. The improvement comes from the fact that both the loads and stores are now coalesced, meaning we're making optimal use of each memory transaction. The GPU's memory controllers see large, contiguous 128-byte requests instead of many small scattered requests.

However, careful readers will note the drop-off at N = 32768 (where performance falls to ~299 GB/s). This hints that something is still suboptimal at very large sizes. To investigate, we need to discuss an important detail: **shared memory bank conflicts**.

## The Challenge of Shared Memory Bank Conflicts

Tiling solves the global memory access problem, but it introduces a new subtle bottleneck: **shared memory bank conflict**. Shared memory is divided into 32 banks (on NVIDIA GPUs, for 32-bit words), and consecutive 32-bit words map to consecutive banks. If threads in the **same warp** access different addresses that reside in the **same bank**, these accesses will serialize (the bank services one thread at a time). This serialization negates the speed of parallel shared memory access.

Unfortunately, a native transepose tile is a textbook case for bank conflicts. Consider a 32x32 shared memory tile stored in row-major order. When threads try to read the tile *column-wise* (which is what happens when transposing), thread 0 accesses `tile[0][col]`, thread 1 accesses `tile[1][col]`, and so on up to thread 31 accessing `tile[31][col]`. These addresses in memory are separated by 32 elements - and since there are 32 banks, each address falls into the **same bank** (bank index = `col` for all of them). In effect, reading a column of 32 elements results in a 32-way bank conflict - [the worst possible scenario](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/#:~:text=For%20a%20shared%20memory%20tile,elements%20wide%20rather%20than%2032). This means that a warp's shared memory reads (or writes) of that column are completely serialized, hurting performance.

In our initial tiled kernel (`transpose_shm_v0`), we indeed encountered this issue. Threads wrote to shared memory in row-major order (which was fine for the coalesced global load), but when they later read out the columns of the shared tile to write to global memory, those shared memory accesses suffered heavy bank conflicts. The warp's reads from shared memory were effectively serialized, and as a result the performance gain, while significant, was not as high as it could be. This was especially evident at very large matrix sizes (like 32768), where the benefit of tiling plateaued or dropped - indicating the shared memory itself had become a bottleneck.

### Padding: The Classic Solution

The standard remedy for shared memory bank conflicts is **padding** the shared memory array. By adding an extra dummy column (or row) to the shared memory tile, we change the alignment of data in memory so that threads in a warp access different banks. In [NVIDIA's example](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/#:~:text=For%20a%20shared%20memory%20tile,elements%20wide%20rather%20than%2032), changing a 32x32 tile to 33 columns eliminates the 32-way conflict. The code simply declares: `__shared__ float tile[TILE_DIM][TILE_DIM + 1];`. This one-column pad breaks the alignment that caused all threads to hit the same bank. Now, when thread `t` reads `tile[t][col]`, the address is `t * (TILE_DIM + 1) + col`, and the bank index becomes `(t * (TILE_DIM + 1) + col) mod 32`. Because `(TILE_DIM + 1)` is not a multiple of 32, each thread's access falls into a different bank. In practice, this removed the shared memory conflicts and brought transpose performance up to ~95% of the device's copy bandwidth in NVIDIA's example.

In our implementation, we applied this padding idea. The `transpose_shm_v1_kernel` uses a slightly different indexing scheme and includes padding to achieve conflict-free accesses. Specifically, for a 32×16 tile, we added a padding of 1 in the first dimension (making the shared memory array effectively 33×16). This changes the memory stride between what would have been “columns” in shared memory. As a result, when threads read the tile in the transposed order, the addresses they access are skewed across the 32 banks instead of colliding.

The difference between our v0 and v1 kernels in how they index the shared memory:

- In v0, we stored the tile in straightforward row-major order in shared memory (`smem[row * TILE_N + col]`). The conflict occurs when reading `smem[col * TILE_N + row]` (column-wise).
- In v1, we stored the tile in a transposed layout within shared memory (`smem[col * TILE_M + row]`). Essentially, each thread writes to shared memory in column-major order (which would ordinarily cause conflicts on the write, but the padding prevents it), and then after the barrier, threads read from shared memory in row-major order (which is conflict-free). Both global reads and global writes remain coalesced, and now the shared memory accesses are also conflict-free.

However, in our benchmarks the padded version (`transpose_shm_v1_kernel`) showed only a *marginal* improvement over the unpadded version.

## Choosing an Optimal Tile Size (16x16 vs 32x16)

Tiling introduces two tunable parameters: the tile width (number of threads per row) and tile height (number of threads per column) in a block. A `32×32` tile might seem natural (cover a full warp per row and per column), but using 1024 threads per block can reduce occupancy and flexibility. We found that a `32×16` tile hits a sweet spot for this problem. Here’s why:

- **Warp-Aligned Width**: A width of 32 threads means each row load or store is handled by exactly one warp. This guarantees that memory accesses along a row are perfectly coalesced - each warp accesses 32 contiguous values (128 bytes for 32 floats), which the hardware can service in as few transactions as possible. In constrast, a 16-thread row (as in a 16x16 tile) means a warp covers two rows of 16. While global memory coalescing can still work with half-warps, using the full warp width for each memory transaction tends to maximize utilization. When we increased our block size from 16x16 to 32x16, we observed a modest performance gain, attributable to better memory coalescing and alignment of accesses with the warp size. Essentially, the 32x16 tile allowed each warp to read/write contiguous 32-element segments, making more efficient use of the memory bus.

- **Moderate Tile Height for Occupancy**: We chose 16 for the tile's height to keep the shared memory usage and per-block thread count moderate. A 32x16 block has 512 threads, which is half the threads of a 32x32 block. This means two 32x16 blocks can potentially reside concurrently on one streaming multiprocessor (SM), instead of one 32x32 block. Having multiple active blocks per SM often helps **increase occupancy**, allowing the GPU to better hide memory latency by swapping warps from different blocks. The 16-row tile also uses half the shared memory of a 32-row tile, which can further allow more blocks to be active if shared memory is a limiting factor. In summary, the 32x16 tile strikes a balance: it's wide enough for full coalescing, but not so tall that it starves the SM of resources. This change from 16x16 to 32x16 in our kernel yielded a slight performance improvement, thanks to these memory and occupancy benefits.

**Measured Bandwidth (GB/s)**: *Impact of tile size (with padding)*
| Matrix Size (N = M) | `transpose_shm_v1_kernel` (16x16 tile) | `transpose_shm_v1_kernel` (32x16 tile) | `transpose_shm_v1_kernel` (32x16 tile + Padding 1) | 
| -- | -- | -- | -- |
| 512 | 170.667 | 170.667 | 170.667 |
| 1024 | 372.364 | 372.364 | 372.364 |
| 2048 | 528.516 | 496.485 | 496.485 |
| 4096 | 537.180 | 550.723 | 550.723 |
| 8192 | 530.656 | 519.145 | 522.199 |
| 16384 | 490.218 | 494.378 | 494.611 |
| 32768 | 298.921 | 434.508 | 433.565 |

## Increasing Throughput with Block-Level Unrolling

Finally, we also experimented with increasing the **work per thread block** to further boost throughput. The idea was to have each block transpose more than one tile, reusing the same threads to do extra work before finishing. Our `transpose_shm_v1_unroll_2_kernel` is an example: each thread block processes **two adjacent tiles** (each of size 32×16) in one go. This is accomplished by extending the indices so that after handling the first tile, the same threads immediately load a second tile from a neighboring region of the matrix and transpose it as well. We allocate a larger shared memory array to hold 2×(32×16) elements (with padding as needed). By unrolling the block’s work in this manner, we **amortize overheads** – fewer kernel launches and loop iterations are needed per matrix, and index calculations or synchronizations are reduced per element processed. In our results, this block-level unrolling gave another small bump in throughput. Essentially, it improves the ratio of useful memory operations to overhead (arithmetic and synchronization), pushing us closer to the hardware’s peak bandwidth.

```c++
// Tiled transpose (32x16 tile) + block-level unrolling (process 2 tiles per block)
template <typename T, size_t TILE_M = 32, size_t TILE_N = 16>
__global__ void transpose_shm_v1_unroll_2_kernel(T* out, T const* in,
                                                 size_t const M,
                                                 size_t const N) {
  __shared__ T smem[TILE_N * 2 * TILE_M];

  size_t in_m = blockDim.y * blockIdx.y + threadIdx.y;
  size_t in_n = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  // Calculate indices for two tiles
  size_t block_idx = blockDim.x * threadIdx.y + threadIdx.x;
  size_t block_row = block_idx / blockDim.y;
  size_t block_col = block_idx % blockDim.y;
  size_t out_m = blockDim.y * blockIdx.y + block_col;
  size_t out_n = 2 * blockDim.x * blockIdx.x + block_row;
  // Load two tiles into shared memory (coalesced global loads)
  smem[threadIdx.y * TILE_N * 2 + threadIdx.x] = in[in_m * N + in_n];
  smem[threadIdx.y * TILE_N * 2 + threadIdx.x + TILE_N] =
      in[in_m * N + in_n + TILE_N];
  __syncthreads();
  // Store transposed tiles from shared to global (coalesced global stores)
  out[out_n * M + out_m] = smem[block_col * 2 * TILE_N + block_row];
  out[(out_n + blockDim.x) * M + out_m] =
      smem[block_col * 2 * TILE_N + block_row + TILE_N];
}
```

**Measured Bandwidth (GB/s)**: *Impact of block-level unrolling*
| Matrix Size (N = M) | `transpose_shm_v1_kernel` (16x16 tile) | `transpose_shm_v1_kernel` (32x16 tile) | `transpose_shm_v1_unroll_2_kernel` (32x16 tile + block-level unrolling) | 
| -- | -- | -- | -- |
| 512 | 170.667 | 170.667 | 170.667 |
| 1024 | 372.364 | 372.364 | 372.364 |
| 2048 | 528.516 | 496.485 | 512.000 |
| 4096 | 537.180 | 550.723 | 537.180 |
| 8192 | 530.656 | 519.145 | 560.137 |
| 16384 | 490.218 | 494.378 | 523.813 |
| 32768 | 298.921 | 434.508 | 476.301 |

# Conclusion and Future Work

Through this step-by-step optimization, we transformed a naive GPU transpose that achieved barely 20~40% of peak bandwidth into highly optimized version reaching about 70% of peak memory bandwidth on the RTX A6000. By diagnosing the bottlenecks at each stage, we learned the following lessons:

- **Memory access patterns dominates** in memory-bound kernels. Coalescing global accesses (both loads and stores) is absolutely critical for performance, far outweighing minor tweaks to arithmetic or loop overhead.
- **Instruction-level tweaks** (unrolling, etc.) provided limited benefit when the memory pattern was poor. They became effective only after we had mostly fixed the memory access pattern.
- **Vectorized memory operations** can reduce the number of transactions and instructions, but one must watch out for side-effects like increased register usage which can reduce parallelism. Balance is key.
- **Tiling and use of shared memory** allowed us to reorganize accesses and recover coalescing, at the cost of added synchronization. The payoff was huge, but we had to manage **shared memory bank conflicts** through padding (a classic trick).
- **Tile size and shape matter**: Using a tile width equal to a warp (32) ensured maximal coalescing, while keeping the tile height moderate preserved occupancy and flexibility. More threads per block isn't always better if it prevents full SM utilization.
- **Amortizing overhead** by doing more work per thread/block (unrolling at the block level) can give incremental gains once the main bottlenecks are resolved.

In our final implementation, for large matrices, we acheived around ~560 GB/s effective bandwidth, which is about 80% of the copy kernel's throughput on this GPU. This is a significant improvement over the naive transpose's ~200 GB/s range, and it illustrates the power of optimizing memory access patterns.

**Further optimizations** could push performance even closer to the hardware limits. One avenue is to leverage **asynchronous copy** (available in CUDA 11+ on newer architectures like Ampere) to overlap the data transfer to shared memory with computation. The new `cp.async` and `commit_group` instructions allow a warp to load a tile into shared memory without stalling, and to synchronize more efficiently, which could reduce the overhead of the `__syncthreads()` barrier. Additionally, for very large matrices, techniques to improve cache utilizaion (such as splitting the matrix into sub-tiles to increase L2 locality) or using **double buffering** in shared memory (overlapping the processing of one tile with the loading of the next) might yield benefits.

Lastly, it's important to handle edges cases (matrices that aren't multiples of the tile size) is a robust implementation. In practice, one might launch an extra set of "edge" thread blocks or add conditional checks for bounds inside the kernel. These add a bit of overhead, but the techniques discussed remain the same.

By understanding and addressing each bottleneck in turn - from global memory coalescing to shared memory banking - we arrived at a solution that is not only fast, but also illustrates core principles of GPU performance engineering. This exercise in optimizing a transpose kernel serves as a microcosm for many GPU optimization problems: **know your hardware, mind your memory accesses, and profile each change** to guide the next improvement.
