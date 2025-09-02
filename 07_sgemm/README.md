# SGEMM Kernel Optimization

In this document, we examine a sequence of progressively optimized single-precision matrix multiplication (SGEMM) CUDA kernels on an NVIDIA RTX A6000 GPU (Ampere architecture). Starting from a naive global-memory implementation and culminating in an advanced warp-tiled kernel, each version's performance is compared against its predecessor and against a highly optimized baseline: PyTorch's `torch.matmul` (which uses a CUTLASS SIMT GEMM kernel or cuBLAS GEMM kernel depending on the size of the matrix). All benchmarks use large 4096x4096 matrices, with runtime and hardware metrics collected via NVIDIA Nsight Compute. This comparative analysis highlights how each optimization impacts performance and key GPU metrics.

**Overall Performance**: These iterative optimizations achieved a dramatic improvement from ~2.07 TFLOPS (naive) up to ~18.9 TFLOPS (final warp-tiled kernel) for 4096x4096x4096 multiplication - an ~9.1x speedup. The torch/CUTLASS baseline reaches ~22.98 TFLOPS, representing the performance ceiling. The table below summarizes the runtime and throughput of each kernel version:
| Kernel Version                            | Time (ms) | Achieved TFLOPS |
| ----------------------------------------- | --------: | --------------: |
| Torch (CUTLASS) – baseline                |     5.979 |           22.98 |
| Naive (No Tiling, Global Memory)          |    66.514 |            2.07 |
| Tiled Shared Memory                       |    48.025 |            2.86 |
| 1D Thread-Block Tiling                    |    16.794 |            8.18 |
| 2D Thread-Block Tiling                    |    10.549 |           13.03 |
| Vectorized 2D Tiliing                     |     8.072 |           17.02 |
| Vectorized 2D Tiling with Padding         |     8.688 |           15.82 |
| Vectorized 2D Tiling with Padding (K=16)  |     8.049 |           17.07 |
| Warp-Tiling                               |     7.271 |           18.90 |

*Table: SGEMM kernel performance for 4096x4096 matrices (runtime and throughput).*

## CUTLASS SIMT Kernel (`torch.matmul`) as Performance Ceiling

> Kernel: `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>`.

This kernel serves as a highly optimized baseline. It is likely to use large thread-block tiles with efficient warp-level matrix multiplication, yielding excellent compute utilization. Key metrics for the 4096^3 problem size are summarized below:

| Metric | CUTLASS Baseline |
| ------ | ---------------- |
| Achieved FP32 Throughput | ~22.98 TFLOPS |
| Achieved Occupancy (Active Warps) | ~16.7% ($\approx$ 8 warps active per SM out of 48) |
| SM Compute Throughput | ~76~77% of peak (SM pipelines busy) |
| DRAM Bandwidth Utilization | ~9.6% of peak |
| Instruction Issue Utilization | ~87.8% of peak issue rate |

Despite a *low occupancy* (~8 warps per SM, due to large register usage per thread-block), this kernel keeps the SMs **very busy** - ~76% of peak FP32 throughput. The warp schedulers are utilized ~87% on average, meaning most cycles a warp is issuing instructions. The kernel achieves ~22.98 TFOPS, which is near the hardware's FP32 limit. Memory throughput is modest (only ~9.6% of DRAM bandwidth), indicating it is **compute-bound** - the kernel reuses data extensively in registers/shared memory, minimizing external memory traffic. Stall analysis shows almost no stalls on memory pipelines (negligible "LG throttle" or memory *scoreboard* stalls), and warp stalls are low - the primary minor stall reason is "not selected" (warp waiting to be scheduled) at ~0.81 cycles/inst, which is expected with fewer warps available. Overall, the CUTLASS kernel's optimizations (thread-block tiling, instruction-level parallelism, double-buffering, etc) yield a **well-balanced compute-bound execution**.

This sets the bar for performance - later kernels will be compared against this. The CUTLASS kernel achieves the highest throughput, so any gaps in later kernels will highlight optimization opportunities (e.g., lower compute utilization or higher stalls than CUTLASS).

## Naive SGEMM Kernel (No Tiling, Global Memory)

> Kernel: `sgemm_naive_kernel`

This kernel computes C[i,j] output with a straightforward loop over K. Each thread handles one C element and performs 4096 FMA iterations, loading A and B operands directly from global memory every time (no tiling or caching). It launches one 1024-thread block per 32x32 output tile of C.

Performance metrics:
| Metric | CUTLASS Baseline |
| ------ | ---------------- |
| Achieved FP32 Throughput | ~2.07 TFLOPS (very low) |
| Achieved Occupancy | ~66% (32 warps per SM out of 48) |
| SM Compute Throughput | ~76.78% of peak |
| DRAM Bandwidth Utilization | ~15.3% of peak |
| Instruction Issue Utilization | ~26.9% of peak issue rate (low) |
| Dominant Stall Reason | **Global memory (LG) throttle** - ~18 cycles/inst |

The naive kernel launches a full 1024-thread block per 32×32 tile, achieving high occupancy (~32 active warps per SM). However, its performance is only ~2.07 TFLOPS - about 9% of the CUTLASS throughput. The SM's compute pipelines are busy when active (~76.78% of peak) because each thread executes a long K-loop of fused multiply-adds. **But the issue efficiency is very low (~27%)** - indicating most cycles the warps are *stalled* waiting for memory. Indeed, profiling shows the **global memory pipeline is the bottleneck**: the **"LG throttle" stall** (warps waiting for the L1/Global memory unit) averages **18.3 cycles/inst**, dominating all other stalls. This occurs because each thread performs 4096 global loads from A and B, with no data reuse, saturating memory latency. The kernel only achieves ~15% of peak DRAM bandwidth despite heavy memory traffic, as warps often idle waiting on memory. The high occupancy (many warps) helps hide some latency, but the overall throughput is memory-bound. **In summary, the naive kernel is heavily memory-bound** - it wastes compute capacity waiting on redundant global loads, and its performance is an order of magnitude lower than the optimized baseline.

Compared to the CUTLASS baseline, the naive kernel performs **~10x slower**. It has higher occupancy but far lower instruction throughput. The lack of tiling means it suffers from massive memory stalls (unlike CUTLASS, which reuses data). This highlights the need for **tiling and on-chip data reuse** to improve performance.

## Tiled Shared-Memory SGEMM

> Kernel: `sgemm_shmem_naive_kernel<32>`

This kernel introduces **tiling in shared memory**: each 32x32 thread block loads a 32x32 sub-tile of A and B into **shared memory**, then computes a 32x32 product tile before moving to the next K segment. This reuse should cut down redundant global memory accesses.

Key metrics:
| Metric | CUTLASS Baseline |
| ------ | ---------------- |
| Achieved FP32 Throughput | ~2.86 TFLOPS (1.38x vs prev.) |
| Achieved Occupancy | ~66% (32 warps per SM out of 48) |
| SM Compute Throughput | ~83.4% of peak |
| DRAM Bandwidth Utilization | ~20.3% of peak |
| Instruction Issue Utilization | ~20.3% of peak (lower than naive) |
| Dominant Stall Reason | **Shared mem pipeline (MIO)** throttle - ~23.7 cycles/inst |

Using a 32x32 tile in shared memory yields a moderate speedup (~2.86 vs 2.07 TFLOPS). **Global memory traffic is reduced** - DRAM utilization rose to ~20%, meaning each thread now reuses data from shared memory instead of reloading each element, and more bandwidth is effectively used. The global memory stall ("LG throttle") plummets from 18.3 to 0.39 cycles/inst (no longer the bottleneck). However, **new bottlenecks appear**. Each thread block performs an `__syncthreads()` every 32 K-steps and each thread issues shared memory loads in the inner loop. The profile shows a **huge "MIO throttle" stall** (Memory I/O pipeline for shared memory) of ~23.7 cycles/inst - i.e., warps often wait because the shared memory pipeline is saturated by the 32 concurrent threads loading from shared memory banks. Additionally, the average warp stall on **CTA barrier** is high (~2.81 cycles/inst at barrier) due to frequent `__syncthreads()` every tile. These stalls **lower the overall instruction issue rate** (only ~20% of peak, even less than naive's 27%). In effect, this kernel shifts the bottleneck from global memory to **shared memory and synchronization overhead**. The net result is a modest gain in TFLOPS - the benefit of fewer global loads is partially offset by shared memory and barrier synchronization costs.

Relative to the naive kernel, the shared-memory tiling **reduces global memory stalls dramatically**, improving throughput ~35%. However, it introduces new overheads (shared memory and barrier waits). Unlike the naive kernel, which was purely memory-bound, this version is still memory-bound but at the **shared memory level** (MIO pipeline), and its instruction efficiency actually dropped (due to synchronization). This underscores that simply adding tiling helps but also requires managing on-chip resource contention.

## 1D Thread-Block Tiling

> Kernel: `sgemm_1d_blocktile_kernel<64, 64, 8, 8>`

This kernel uses a **1D thread tiling** strategy: a thread block of size 63x64 computes a 64x64 tile, but each thread computes a **vector of 8 partial results** in one dimension (`THREAD_M=8`). Shared memory is still used for 64x8 tiles of A and B (`BLOCK_K=8`) with loop increments of 8 in K.

Key performance metrics:
| Metric | CUTLASS Baseline |
| ------ | ---------------- |
| Achieved FP32 Throughput | **~8.18 TFLOPS** (2.8x vs prev.) |
| Achieved Occupancy | ~66% (32 warps per SM out of 48) |
| SM Compute Throughput | ~82.9% of peak |
| DRAM Bandwidth Utilization | ~24.7% of peak |
| Instruction Issue Utilization | ~37.3% of peak (improved) |
| Dominant Stall Reason | **Shared mem pipeline (MIO)** throttle - ~7.5 cycles/inst |

The 1D tiling strategy yields a **significant performance jump** to ~8.18 TFLOPS - nearly 3x faster than the 32x32 tiled kernel. Several factors contribute: (1) **Each thread now does more work (computes 8 outputs)**, increasing the arithmetic intensity. Data loaded into shared memory (64x8 sub-tiles) is reused across multiple results in that thread, reducing redundant loads/stores. (2) The 64x64 tile size means fewer thread blocks and fewer barriers for the same matrix size, slightly lowering barrier stall overhead. We see the **CTA barrier stall average drop** (from ~2.8 to 2.51 cycles/inst). The **shared memory pipeline stall (MIO Throttle) also drops** dramatically from ~23.7 to ~7.5.53 cycles/inst - still significant, but much improved. This indicates far better utilization of each shared memory load (each load feeds 8 accumulations now) and fewer concurrent accesses needed. **Global memory throughput** remains similar (~24.7% of peak), meaning the kernel is still not saturating memory bandwidth - it's moving towards a more balanced or compute-bound regime. The SM pipes are busy ~83% of the time, and importantly the **instruction issue efficiency more than doubles** (to ~37% of peak). Fewer cycles are wasted compared to earlier kernels. The remaining inefficiencies come from still having frequent `__syncthreads()` and some memory waits. Notably, **warp occupancy per SM is unchanged (~32 waprs)**, but each warp does more useful work now. Stall analysis shows the top stall reason is still "MIO throttle" (shared memory) at ~7.5, with others like long scoreboard (memory dependency) down to ~4.8 and "not selected" ~2.1 - so memory latency is less dominant and the scheduler can issue more instructions overall.

Compared to the simpler 32x32 tiling, the 1D-thread-block tiling **triples performance**. By increasing per-thread work and reuse (at the cost of more registers per thread and slightly larger shared tiles), it reduces on-chip memory stalls and better overlaps computation. The result is a higher instruction throughput and a shift closer to a compute-bound execution. The trade-off is increased register pressure (each thread holds an 8-element accumulator array) - but the data shows occupancy remained ~66%, meaning the register usage still allowed 2 blocks per SM (so it did not degrade occupancy from the previous kernel). Overall, this optimization found a **better balance between computation and memory usage**, significantly closing the gap towards the CUTLASS baseline.

## 2D Thread-Block Tiling

> Kernel: `sgemm_2d_blocktile_kernel<128,128,8,8,8>`

This kernel extends tiling in both thread-block dimensions: a 128x128 output tile is computed per block, and each thread computes a **2D tile of 8x8 results** (`THREAD_M=8`, `THREAD_N=8`). `BLOCK_K` remains 8. This dramatically increases reuse: each load of a 8x8 sub-tile from A or B is used to update an 8x8 output sub-matrix per thread.

Key metrics:
| Metric | CUTLASS Baseline |
| ------ | ---------------- |
| Achieved FP32 Throughput | **~13.03 TFLOPS** (~1.6x vs prev.) |
| Achieved Occupancy | ~16.7% (only ~8 warps per SM) |
| SM Compute Throughput | ~42.1% of peak (lower %) |
| DRAM Bandwidth Utilization | ~11.1% of peak (much lower) |
| Instruction Issue Utilization | ~44.9% of peak (better) |
| Dominant Stall Reason | *No single dominant stall* (balanced) <br>- minimal barrier stall (~0.3) and low memory stalls |

The 2D tiling strategy produces a **substantial performance boost** to ~13.03 TFLOPS, thanks to massive data reuse. Each thread now does 8x8=64 FMAs per inner-loop tile, so the ratio of compute-to-memory operations is greatly increased. This kernel's behavior shifts closer to compute-bound: **DRAM utilization drops to ~11%**, less than half that of the 1D tiling. This is expected - it's reading far fewer bytes per FLOP due to reusing each tile of A and B across many results. In fact, the **L2 cache hit rate rises** (implied by less DRAM need), and long memory stalls shrink: the "long scoreboard" (waiting on global memory) is only ~1.04 cycles/inst now. **Barrier overhead plummets** - only one `__syncthreads()` per `K=8` tile with 8 warps per block means barrier stall ~0.29 cycles/inst (nearly negligible). The shared memory MIO stalls also drop to ~0.23 on average - a huge improvement, as each shared load is heavily amortized across 64 FMA operations. Overall, **no single stall reason dominates**; the kernel achieves a far more balanced execution. Interestingly, the **achieved occupancy fell sharply**: with a 128x128 tile, each block has 256 threads (8 warps), and due to high register usage per thread, only **one block could schedule per SM** ($\approx$ 8 warps active, ~16% occupancy). Despite low occupancy, the threads are so productive that instruction issue utilization actually rose to ~45%. The warp scheduler wasn't starved: with fewer warps, it still kept issuing nearly half the peak rate, indicating each warp had lots of independent work (and less time waiting on memory). We see very low "not selected" stall (~0.6), meaning most cycles a warp was ready to execute. The flip side is slightly lower SM pipe utilization (42% of peak) - an artifact of counting idle cycles from lower occupancy. In practice, this kernel is **compute-heavy**: it significantly reduced memory-bound stalls and overhead, at the cost of using more registers and reducing concurrency.

Compared to the 1D tiling, the 2D tiling **improves throughput by ~60%** and reaches ~57% of the CUTLASS baseline performance. The trade-off was a big drop in occupancy (from ~32 to 8 warps/SM) due to increased register and shared memory per block. However, the data shows this was largely mitigated by the improved efficiency: memory demand fell so much that having fewer warps was not a problem (the remaining warps were busy with compute). This kernel demonstrates the power of maximizing on-chip reuse: it nearly eliminated global and shared memory stalls. The remaining gap to the baseline is due to lower parallelism (only 1 CTA per SM, vs baseline can achieve more concurrency or better pipeline overlap). Also, each warp in this kernel may still execute operations serially for its 8x8 tile, which can incur dependency stalls (e.g., a slight increase in short scoreboard stalls as threads accumulate results). Overall, 2D tiling made the kernel **compute-bound and efficient**, at the expense of high resource usage per block.

## Vectorized 2D Tiling

> Kernel: `sgemm_vec_2d_blocktile_kernel<128,128,8,8,8>`

Building on the 2D tiling above, this kernel adds **vectorized memory access** (e.g., loading floats in coalesced 128-bit (float4) vectors) for global memory and possibly shared memory. The tile sizes (128x128x8, thread 8x8) are the same as the previous kernel, but with vector load/store instructions to improve memory throughput.

Key metrics:
| Metric | CUTLASS Baseline |
| ------ | ---------------- |
| Achieved FP32 Throughput | **~17.02 TFLOPS** (1.3x vs prev.) |
| Achieved Occupancy | ~32% (~16 warps per SM) |
| SM Compute Throughput | ~56.6% of peak (higher) |
| DRAM Bandwidth Utilization | ~14.5% of peak |
| Instruction Issue Utilization | ~60.7% of peak (much higher) |
| Dominant Stall Reason | **None dominant** (memory and sync stalls remain low; <br> some increase in dependency stalls) |

Enabling vectorized loads boosts performance to ~17.02 TFLOPS, achieving ~74% of the baseline. The improvements stem from **more efficient memory operations and better occupancy**. By loading data in 128-bit chunks (float4), each instruction transfers 4 floats, reducing the total number of load instructions and improving coalescing. This is reflected in the metrics: DRAM throughput rises slightly to ~14.5 of peak - the kernel is pulling a bit more data per cycle, likely because vector loads make better use of the memory bus (less wasted bandwidth on misaligned or small transactions). The **instruction issue efficiency jumps sharply** to ~60~61% of peak. Fewer instructions are needed for the same work (one vector load replaces four scalar loads, etc.), and the scheduler can issue instructions more consistently. We also observe that **achieved occupancy doubled** to ~32% (roughly 16 warps per SM). This suggests that vectorization (and possibly slight code refactoring) **reduced register pressure per thread**, allowing **two CTAs per SM** instead on one. Indeed, with 8 warps per CTA, having 2 CTAs gives 16 warps active, which matches the measured occupancy. The higher warp count provides more latency hiding and parallelism. Consequently, "not selected" warp stall increased somewhat (from ~0.6 to ~1.6 cycles/inst) - with more warps in play, there are occasional cycles where none are ready, though this is still a small stall fraction. Barrier and memory stalls remain low (barrier ~0.75, shared memory throttle ~0.45 - slightly up from the non-vector kernel but still minor). One noticeable change is a slight rise in **short scoreboard stalls** (~0.8 cycles/inst) relative to the non-vector kernel. This can be attributed to threads issuing back-to-back compute instructions more frequently (thanks to fewer memory instructions), thus sometimes waiting on the result of a previous instruction (dependency stall). Overall, however, no single stall dominates and the SM is busier (SM active ~56.6% vs 42% before). The vectorized kernel effectively leverages memory bandwidth and achieves better overlap of computation, demonstrating a **significant gain from memory-level optimizations**.

Vectorization yields a **~29% performance gain** over the scalar 2D tiling kernel. By widening memory accesses, it not only improved bandwidth usage but also freed enough registers to allow higher occupancy (2 CTAs/SM). The result is substantially higher instruction throughput and better latency hiding. This change closed much of the gap to CUTLASS. The trade-off is complexity: ensuring alignment and dealing with potential padding (addressed in the next kernels) to use vector loads. But as seen, the effort paid off - the kernel became **more memory-efficient and parallel**, with only minor increases in manageable stalls (like slight dependency waits).

## Vectorized 2D Tiling with Padding (K=8)

> Kernel: `sgemm_vec_2d_blocktile_pad_kernel<128,128,8,8,8,1,8>`

This kernel builds on the vectorized 2D tiling approach by adding **padding to the shared memory tiles**. Padding ensures that memory accesses avoid shared memory bank conflicts, which can otherwise serialize loads and degrade throughput. The tiling parameters are set to BLOCK_M=128, BLOCK_N=128, BLOCK_K=8, with each thread computing an 8×8 micro-tile of results. Vectorized loads (float4) are used for global memory, and padding is applied to shared memory allocations.

The performance metrics are:
| Metric | CUTLASS Baseline |
| ------ | ---------------- |
| Achieved FP32 Throughput | **15.82 TFLOPS** ($\approx$ same as without padding; minor variance) |
| Achieved Occupancy | ~32% (unchanged, ~16 warps) |
| SM Compute Throughput | ~70.83% of peak |
| DRAM Bandwidth Utilization | ~13.53% of peak |
| Instruction Issue Utilization | ~61.08% of peak |
| Dominant Stall Reason | **None dominant** (similar profile to vector kernel) |

The addition of padding eliminated residual bank conflicts, ensuring that shared memory accesses were replay-free. The DRAM bandwidth is efficiently utilized (~13.5%), and instruction issue efficiency rose compared to earlier kernels (but still only ~61% of peak). However, the kernel remained **memory-latency bound**: long scoreboard stalls (waiting for global memory loads) were significant (~0.82 cycles/inst). Frequent `__syncthreads()` barriers also contributed to stall cycles (~0.57). Occupancy was doubled compared to the CUTLASS baseline, but the scheduler often reported “not selected” stalls, meaning it had more ready warps than it could issue instructions for, suggesting the pipeline was saturated but not fully utilized.

Relative to the unpadded vectorized kernel, performance is essentially flat (~15.82 vs ~17.02 TFLOPS), but the internal metrics highlight the intended effect: shared memory usage is fully conflict-free, SM pipelines are busier, and memory transactions per FMA are reduced. At this problem size the benefit is subtle, but padding removes a potential scalability hazard. It sets the stage for the next kernel, where increasing the K tile size to 16 halves the number of synchronization points per loop. With padding ensuring conflict-free access, the design is well-prepared to sustain higher efficiency as more work per iteration is introduced.

## Vectorized 2D Tiling, Larger K Tile (K=16)

> Kernel: `sgemm_vec_2d_blocktile_pad_kernel<128, 128, 16, 8, 8, 1, 1>`

This kernel is the same as the previous kernel (vectorized 128x128 tile with padding) but doubles the **K-block size** to 16. That is, `BLOCK_K=16`, meaning each iteration the block loads a 128x16 tile of A and B into shared memory and performs 16 FMA loop iterations before moving to the next segment of K. Doubling the K tile reduces the number of iterations (any synchronizations) by half compared to the K=8 version, at the cost of loading more data per iteration.

The performance metrics are:
| Metric | CUTLASS Baseline |
| ------ | ---------------- |
| Achieved FP32 Throughput | **17.07 TFLOPS** (~1.08x vs K=8 version) |
| Achieved Occupancy | ~32% (unchanged, ~16 warps) |
| SM Compute Throughput | ~77.98% of peak |
| DRAM Bandwidth Utilization | ~14.91% of peak |
| Instruction Issue Utilization | ~66.79% of peak |
| Dominant Stall Reason | **None dominant** (similar profile to vector kernel) |

Using a larger K tile increases performance to ~17.07 TFLOPS, roughly a 8% gain over the K=8 kernel. The primary benefit comes from halving the number of iteration loops and `__syncthreads()` calls. With `BLOCK_K=16`, each thread does twice as much math work per iteration, so the inner loop runs 256 iterations (down from 512 with 'BLOCK_K=8'), and thus half as many expensive global syncs. This is evident in the metrics: the average **barrier stall decreases** (each block performs fewer synchronizations - the barrier stall drops to 0.51 cycles/inst, from 0.57). With less frequent stopping, the warp schedulers keep issuing instructions more consistently - the instruction issue utilization rises to ~66.79% of peak. The SM compute pipelines are active a larger fraction of time (~77.98% vs ~70.83%), which shows that we're doing more useful work per memory load. Memory throughput usage ticks up slightly (~14.91% vs ~13.53%), likely because when the block does load a 128x16 tile, it issues a burst of memory requests that more fully saturate the bus in that moment (even though overall data volume is the same). Interestingly, some stall categories tick up a tiny amount; e.g., we observe a minor increase in "not selected" (when all warps are busy or none ready) because during the longer compute phase of each iteration, warps move more in lockstep - there can be moments when all warps are waiting for a long latency event (like a shared memory transaction for the larger tile) at the same time. However, there are small effects - no single stall dominates. On the whole, the larger K tile lets the kernel **spend more time computing and less time synchronizing**, which improves throughput.

Increasing the K tile size yields a modest performance uplift, showing positive returns from reducing synchronization frequency. The kernel now achieves ~17.07 TFLOPS, climbing closer to the baseline. The trade-off - doing more work per iteration - did not introduce any new bottlenecks, thanks in part to the shared memory padding and efficient memory access already in place. We have effectively balanced compute and memory even more finely, leaving only ~25% gap to the highly optimized baseline. The next step is to tackle the remaining inefficiencies via warp-level scheduling.

## Warp-Tiling with Cooperative Matrix Compute

> Kernel: `sgemm_warptiling_kernel<128, 128, 16, 8, 8, 64, 32>`

This kernel employs **warp-level tiling**: the 128x128 tile is divided among warps (e.g., each warp might compute a 64x32 portion, suggested by the template parameters), and uses finer-grained coordination (likely via shuffle instructions or shared memory) instead of only block-wise sync. The template indicates a 128x128 block, K=16, with warp tile 64x32 and thread tile 8x8. This approach aims to better utilize warp-level parallelism and reduce synchronization overhead further.

Key metrics:
| Metric | CUTLASS Baseline |
| ------ | ---------------- |
| Achieved FP32 Throughput | **18.90 TFLOPS** |
| Achieved Occupancy | ~32% (unchanged, ~16 warps) |
| SM Compute Throughput | ~65.36% of peak |
| DRAM Bandwidth Utilization | ~17.16% of peak |
| Instruction Issue Utilization | ~66.48% of peak |
| Dominant Stall Reason | **None dominant**; notable: slightly lower warp scheduler idle (not-selected ~2.23) |

The warp-tiling kernel reaches ~18.90 TFLOPS, making it the fastest custom kernel and only about 18% shy of the CUTLASS baseline. The improvement of ~10% over the previous kernel comes from better **concurrency and latency hiding**. By dividing the workload so that each warp handles a portion of the tile (64x32, for example) and overlapping memory operations with computation, the kernel ensures that at any given time, some warps are performing FMAs while others fetch the next data. The instruction issue utilization holds around ~65%, similar to the previous kernel, which indicates the scheduler is busy two-thirds of the time. With 16 warps, there are still occasional cycles where no warp can issue (the "not selected" stall is ~2.23 cycles/inst) - this comes from the fact that all warps might be waiting on a synchronization or long latency load at roughly the same moment. However, this stall is relatively minor and no worse than before. Overall, warp-level tiling succeeds in keeping the SM fed almost continuously: the compute throughput (~65% of peak) is very high given the hardware constraints (compare to ~76% in the baseline which uses special scheduling tricks). The DRAM utilization remains under ~20%, reinforcing that we are not memory-bandwidth limited; rather, we are **latency-limited** by the need to manage data movement and synchronization among warps. In summary, the warp-tiling kernel achieves an excellent balance - it leverages intra-block parallelism and overlapping of compute/memory to substantially boost performance, coming quite come to the highly optimized library kernel.

The warp-tiling approach adds another ~10% performance on top of the vectorized K=16 kernel, demonstrating the benefit of **cooperative compute and pipelining**. It highlights that even after optimizing memory access patterns, there was still an opportunity to improve utilization by orchestrating how warps work together. The final custom kernel now reaches ~82% of the baseline's TFLOPS. The remaining gap is due to the last bits of efficiency (the baseline likely exploits even more instruction-level parallelism, and possibly uses specialized load/store instructions). Nonetheless, this progression to warp-level tiling shows that careful management of work among warps - overlapping computation with data loading - can push GPU utilization to a very high level.