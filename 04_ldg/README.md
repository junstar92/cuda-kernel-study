# CUDA Kernel Study: `__ldg` for Read-Only Data Caching

```c++
__device__ T __ldg(const T* address);
```

The `__ldg()` intrinsic is a CUDA device function that performs a **read-only data cache load** from global memory. In simpler terms, it gives a hint to the compiler/hardware that the value being loaded from the given address will **not be modified** during the kernel's execution, allowing the load to be serviced through a special cache (the texture or read-only cache) rather than the normal path. This can reduce memory access latency for data that is accessed repeatedly by threads.

## What is `__ldg` and Why Use It?

On NVIDIA GPUs of compute capability 3.5 and above, global memory accesses can be cached not only in the L2 cache (which all devices have) but also in a fast **read-only data cache** (part of the L1/texture cache) if the data is known to be read-only. The `__ldg()` intrinsic was introduced with the Kepler architecture (SM 3.5) to explicitly take advantage of this read-only cache path. When a load is done via `__ldg`, the hardware knows to fetch it through the texture pipeline (read-only cache) instead of the regular L1 path. This read-only cache has **lower latency** for cached data compared to normal global memory access, which can significantly speed up memory-bound kernels where values are reused many times. For example, in a matrix multiplication or other algorithms where many threads read the same input values repeatedly, caching those values in the read-only L1 can yield a huge performance boost.

By default, **global memory loads are always cached in L2**. On older GPUs (compute capability 3.0), they were *not* cached in L1 at all. The read-only cache was introduced in Kepler to allow a subset of data (the read-only data) to also be cached-on-chip (similar to how texture fetches worked). On later architectures (Maxwell and beyond), the L1 cache and texture cache were **unified into one unit**. Even so, marking data as read-only (via `__ldg` or pointer qualifiers) still provides a hint to use that cache move effectively for non-coherent (read-only) accesses. In other words, the concept of a "read-only cache" lives on as a *cache policy hint*, even if the physical caches are unified.

## How to Use `__ldg` in Your Code

Using `__ldg` is straightforward. You call it with the address of the value you want to load. For example:

```c++
// Explicitly using __ldg to laod through read-only cache:
float val = __ldg(&input[idx]);
output[idx] = val * 2.0f;
```

In the above, `input` is a pointer to global memory (e.g, a `float*` passed to the kernel). The `__ldg(&input[idx])` will fetch `input[idx]` via the read-only cache path, assuming the hardware supports it. This is equivalent to the PTX instruction `LDG.E` (load global, cached in the texture/non-coherent cache) under the hood.

**Alternative approach**: You may not need to call `__ldg` explicitly if the compiler can tell a load is from read-only data. The CUDA compiler will *automatically* use a read-only cache load if it can prove that a global memory pointer is only read and never written to. Marking a pointer as both `const` and `__restrict__` helps the compiler make this determination. For example, if you declare your kernel as `myKernel(const float* __restrict__ data, ...)` and only read from `data` inside, the compiler is likely to issue `LDG` loads for `data[i]` accesses. In practice, there are two ways to ensure read-only caching:

- **A. Use the `__ldg` intrinsic explicitly**: e.g., `output[i] = __ldg(&input[j]);` - this forces the use of the read-only cache for that load
- **B. Qualify pointers as read-only**: e.g., declare `const float* __restrict__ input` and then just do `output[i] = input[j];`. The compiler may recognize that `input` is read-only and use the `LDG` instruction automatically.

Both approaches achieve the same end result (loading via the texture/read-only cache). Approach B relies on compiler analysis, which *might not cache every case*. Approach A (explicit `__ldg`) **bypasses compiler analysis and guarantees** the use of the read-only cache for that load. In fact, experienced developers often favor `__ldg` intrinsics in performance-critical code because it provides a clear, explicit intent to the compiler and hardware.

> **Note**: Marking your pointer with `const __restrict__` is still highly recommended (even if you use `__ldg` explicitly) because it tells the compiler that the data is not modified and not aliased. This can enable other optimizations and increases the likelihood the compiler will automatically use the read-only cache on its own.

## Best Practices and Considerations

**1. Only use __ldg for truly read-only data**: It's crucial that the memory you load with `__ldg` is not written to by any thread for the duration of the kernel. The hardware does not maintain coherence for the texture/read-only cache (it assumes the data doesn't change). If you violate this (e.g., one thread modifies data while another uses `__ldg` on it), you could get stale reads or undefined behavior. In general, use `__ldg` for *inputs* that remains constant during the kernel and for data that is read frequently.

**2. Expect benefits in memory-bound scenarios**: The advantage of `__ldg` comes from caching and memory locality. If your kernel is bottlenecked by global memory bandwidth and reuses certain values across threads or iterations, `__ldg` can help by serving those repeated reads from faster cache memory. As NVIDIA's documentation notes, the read-only cache loads have much lower latency than normal global loads. This was demonstrated in cases like matrix multiplication, where using the read-only cache led to significant speedups. Another case is when multiple threads (especially within a warp) read the same address - using the texture cache can efficiently service all those threads with one cache line fetch.

**3. Don't expect miracles if access patterns have low reuse**: Like any cache, the read-only cache is effective only if data access exhibits **locality** (temporal or spatial). If each thread is reading unique data that is never reused or has poor locality, then `__ldg` may not provide a noticeable benefit - at worst, it will behave similarly to a normal global load (a cache miss just falls back to L2/DRAM). In fact, empirical tuning has shown that in some cases with very low reuse or highly random access patterns, forcing use of the read-only cache can slightly **decrease** L2 cache hit rates (because you might be diverting some accesses through texture cache unnecessarily). The takeaway is to use `__ldg` when you have reason to believe caching will help. If unsure, you can A/B test performance with and without `__ldg`. As one expert notes, as long as your data has some spatial/temporal locality, caches (including the read-only cache) will usually help.

**4. Modern GPUs and __ldg**: With newer GPU architectures, NVIDIA has expanded and unified caches, and sometimes they automatically cache more operations. Even so, the `__ldg` intrinsic and the use of `const __restrict__` remain relevant for indicating non-modified data. On architectures like Volta, Turing, Ampere, all global loads go through a unified L1/texture cache, but the compiler's notion of a "cache hint" still matters. Using `__ldg` can be seen as setting a **cache policy hint** to treat that load as non-coherent (read-only). This can effect how the memory system prioritizes or places that data. In summary, continue to use `__ldg` for read-only data on modern GPUs - it won't harm, and it may still help optimize caching behavior (especially in cases of divergent access patterns, since the texture path has slightly different capabilities).

## Conclusion

The `__ldg()` intrinsic is a handy tool in CUDA for optimizing memory reads that are read-only. By leveraging the GPU’s read-only data cache, it can significantly lower memory access latencies for data that is reused, leading to better performance in memory-bound kernels. You can either rely on the compiler (using `const __restrict__` qualifiers) or explicitly call `__ldg` to fetch through this cache. Remember to use it only for truly read-only data and compile for the appropriate architecture. When used in the right scenarios, `__ldg` is an easy way to squeeze more performance out of your GPU kernels by making smarter use of the memory hierarchy.

# References

- [NVIDIA CUDA Programming Guide - Global Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-5-x)
- [NVIDIA CUDA Programming Guide - Read-Only Data Cache Load Function](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ldg-function)
- [stackoverflow: What is the difference bitween `__ldg()` intrinsic and a normal execution](https://stackoverflow.com/questions/26603188/what-is-the-difference-between-ldg-intrinsic-and-a-normal-execution#:~:text=From%20the%20CUDA%20C%20Programming,Guide)
- [CUDA Optimization with NVIDIA Tools](https://calcul.math.cnrs.fr/attachments/spip/IMG/pdf/CUDA-Optimization-Julien-Demouth.pdf)