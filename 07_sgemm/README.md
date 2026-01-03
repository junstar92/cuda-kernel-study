# CUDA Kernel Study: SGEMM Optimization

This README summarizes how each SGEMM kernel evolves along the optimization path described in [`how_to_optimize_sgemm.md`](./how_to_optimize_sgemm.md), based on Nsight Compute profiling and benchmark measurements. The main goal is not to focus on absolute per-kernel numbers in isolation, but to track which structural changes produced the largest performance jumps and how the bottleneck moved from one stage to the next.

The current best result is `20.99 TFLOPS` from `sgemm_warptiling_v4<128, 128, 16, 32, 64, 16>`. `v5` later experimented with a CUTLASS-style epilogue, but although it improved the store path, it did not surpass that peak because the shared-memory residency cost was too high. What follows is a step-by-step account of the path from the naive kernel through shared tiling, register tiling, 2D outer product, vec4 ingest, warptiling refinements, and finally the epilogue experiments.

The analyzed kernels follow this sequence.

1. `sgemm_naive_row`
2. `sgemm_smem_tiling`
3. `sgemm_blocktiling_1d<64, 64, 8, 8, 4>`
4. `sgemm_blocktiling_1d<64, 64, 8, 16, 4>`
5. `sgemm_blocktiling_1d<128, 128, 8, 64, 4>`
6. `sgemm_blocktiling_1d<128, 128, 16, 64, 4>`
7. `sgemm_blocktiling_2d<128, 128, 8, 8, 8, 4>`
8. `sgemm_blocktiling_2d<128, 128, 16, 8, 8, 4>`
9. `sgemm_blocktiling_2d_vec4<128, 128, 8, 8, 8, 4>`
10. `sgemm_blocktiling_2d_vec4<128, 128, 16, 8, 8, 4>`
11. `sgemm_warptiling_v0<128, 128, 8, 32, 64, 8>`
12. `sgemm_warptiling_v0<128, 128, 8, 64, 32, 8>`
13. `sgemm_warptiling_v1<128, 128, 8, 32, 64, 8>`
14. `sgemm_warptiling_v1<128, 128, 8, 64, 32, 8>`
15. `sgemm_warptiling_v2<128, 128, 8, 32, 64, 8>`
16. `sgemm_warptiling_v2<128, 128, 8, 64, 32, 8>`
17. `sgemm_warptiling_v2<128, 128, 16, 32, 64, 16>`
18. `sgemm_warptiling_v3<128, 128, 8, 32, 64, 8>`
19. `sgemm_warptiling_v3<128, 128, 8, 64, 32, 8>`
20. `sgemm_warptiling_v3<128, 128, 16, 32, 64, 16>`
21. `sgemm_warptiling_v4<128, 128, 8, 32, 64, 8>`
22. `sgemm_warptiling_v4<128, 128, 8, 64, 32, 8>`
23. `sgemm_warptiling_v4<128, 128, 16, 32, 64, 16>`
24. `sgemm_warptiling_v5<128, 128, 8, 32, 64, 8>`
25. `sgemm_warptiling_v5<128, 128, 8, 64, 32, 8>`
26. `sgemm_warptiling_v5<128, 128, 16, 32, 64, 16>`

The `v5` section also includes an additional experiment that changes the epilogue address-generation pattern relative to the default implementation.

## Analysis Setup

- GPU: NVIDIA RTX A6000, SM86
- Problem size: `M=N=K=4096`
- Profiling: Nsight Compute with `ncu --set full`
- Kernel implementation files
  - [`01_naive.cuh`](./01_naive.cuh)
  - [`02_smem_tiling.cuh`](./02_smem_tiling.cuh)
  - [`03_blocktiling_1d.cuh`](./03_blocktiling_1d.cuh)
  - [`04_blocktiling_2d.cuh`](./04_blocktiling_2d.cuh)
  - [`05_blocktiling_2d_vec4.cuh`](./05_blocktiling_2d_vec4.cuh)
  - [`06_warptiling.cuh`](./06_warptiling.cuh)
  - helper load/store utilities: [`utils.cuh`](./utils.cuh)
- Reference optimization flow: [`how_to_optimize_sgemm.md`](./how_to_optimize_sgemm.md)

The core flow from [`how_to_optimize_sgemm.md`](./how_to_optimize_sgemm.md) can be summarized as follows.

1. Start with a naive kernel that uses only global memory.
2. Reduce global reloads with shared-memory tiling.
3. Increase per-thread output reuse with register tiling.
4. Turn 1D accumulation into a 2D outer-product structure to increase ILP.
5. Increase `BLOCK_K` to reduce barrier and loop overhead.
6. Clean up the memory path with padding and vectorized load/store.
7. Make the `threadblock -> warp -> thread` hierarchy explicit with warp tiling.
8. Separate global-to-shared staging into thread-private register fragments so load/store scheduling becomes more explicit.

One important detail is that the `blocktiling_*` kernels analyzed here already use `4` as the last template argument, which means the `A` shared-memory layout already includes padding. In other words, the speedups from `blocktiling_1d` and `blocktiling_2d` are not purely from register/block tiling alone. They also include the effect of a cleaner shared-memory layout.

## Performance At a Glance

| Kernel | Core idea | Elapsed (ms) | TFLOPS | Dominant bottleneck |
| --- | --- | ---: | ---: | --- |
| `sgemm_naive_row` | global-only baseline | 67.32 | 2.04 | `LG Throttle`, low issue rate |
| `sgemm_smem_tiling` | shared-memory tiling | 46.86 | 2.93 | `MIO Throttle`, barrier, shared-store conflict |
| `blocktiling_1d<64,64,8,8,4>` | start of 1D register tiling | 22.04 | 6.23 | long scoreboard, MIO, barrier |
| `blocktiling_1d<64,64,8,16,4>` | larger thread tile | 19.36 | 7.10 | scoreboard dependency |
| `blocktiling_1d<128,128,8,64,4>` | larger CTA tile + larger register tile | 11.84 | 11.61 | short scoreboard, MIO |
| `blocktiling_1d<128,128,16,64,4>` | `BLOCK_K=16` | 11.46 | 11.99 | short scoreboard, residual shared overhead |
| `blocktiling_2d<128,128,8,8,8,4>` | 2D outer product | 8.56 | 16.05 | shared wavefront excess |
| `blocktiling_2d<128,128,16,8,8,4>` | 2D + `BLOCK_K=16` | 7.90 | 17.39 | shared wavefront excess, not selected |
| `blocktiling_2d_vec4<128,128,8,8,8,4>` | vec4 global path | 8.15 | 16.86 | shared wavefront excess |
| `blocktiling_2d_vec4<128,128,16,8,8,4>` | vec4 + `BLOCK_K=16` | 7.55 | 18.20 | shared wavefront excess |
| `warptiling_v0<128,128,8,32,64,8>` | first warp tiling, wide-N warp tile | 8.50 | 16.16 | shared bank conflict, uncoalesced store |
| `warptiling_v0<128,128,8,64,32,8>` | first warp tiling, wide-M warp tile | 8.15 | 16.86 | uncoalesced store, register-limited occupancy |
| `warptiling_v1<128,128,8,32,64,8>` | refined lane mapping + `4x4` island MMA/store | 7.95 | 17.28 | register-limited occupancy, scheduler/not selected |
| `warptiling_v1<128,128,8,64,32,8>` | refined lane mapping + `4x4` island MMA/store | 7.91 | 17.37 | register-limited occupancy, scheduler/not selected |
| `warptiling_v2<128,128,8,32,64,8>` | explicit register-staged global->shared | 7.53 | 18.25 | register-limited occupancy, scheduler/not selected |
| `warptiling_v2<128,128,8,64,32,8>` | explicit register-staged global->shared | 7.69 | 17.86 | register-limited occupancy, scheduler/not selected |
| `warptiling_v2<128,128,16,32,64,16>` | register-staged global->shared + `BLOCK_K=16` | 7.18 | 19.13 | scheduler/not selected, uncoalesced global store |
| `warptiling_v3<128,128,8,32,64,8>` | 1-stage register prefetch | 7.34 | 18.72 | scheduler/not selected, residual scoreboard |
| `warptiling_v3<128,128,8,64,32,8>` | 1-stage register prefetch | 7.29 | 18.84 | scheduler/not selected, residual scoreboard |
| `warptiling_v3<128,128,16,32,64,16>` | prefetch + `BLOCK_K=16` | 8.01 | 17.15 | register cliff, scheduler starvation |
| `warptiling_v4<128,128,8,32,64,8>` | vec4 global ingest + retained prefetch skeleton | 6.79 | 20.22 | scheduler/not selected, residual scalar epilogue |
| `warptiling_v4<128,128,8,64,32,8>` | vec4 global ingest + retained prefetch skeleton | 6.86 | 20.04 | scheduler/not selected, residual scalar epilogue |
| `warptiling_v4<128,128,16,32,64,16>` | vec4 global ingest + `BLOCK_K=16` | 6.55 | 20.99 | scheduler/not selected, residual scalar epilogue |
| `warptiling_v5<128,128,8,32,64,8>` | CUTLASS-style epilogue shared staging + cooperative vec4 store | 7.50 | 18.33 | shared-memory cliff, reduced issue rate |
| `warptiling_v5<128,128,8,64,32,8>` | CUTLASS-style epilogue shared staging + cooperative vec4 store | 7.51 | 18.29 | shared-memory cliff, reduced issue rate |
| `warptiling_v5<128,128,16,32,64,16>` | CUTLASS-style epilogue shared staging + `BLOCK_K=16` | 6.91 | 19.88 | shared-memory cliff, reduced issue rate |

The overall arc can be summarized in one line.

- Bottleneck movement: `LG Throttle -> MIO Throttle + barrier -> scoreboard -> shared wavefront excess -> scheduler/not selected + residual scoreboard -> BK16 register cliff(v3) -> BK16 recovered by vec4 ingest + residual scalar epilogue -> store path cleaned up but blocked by a shared-memory cliff in the CUTLASS-style epilogue(v5)`
- Final performance: `2.04 -> 20.99 TFLOPS`, about `10.3x`
- Current best point: `warptiling_v4<128x128x16, 32x64, BK=16>` at `20.99 TFLOPS`, `6.55 ms`

> The largest performance jumps did not come from shared tiling alone. They came when shared reuse was extended into register reuse, and when 1D accumulation was restructured into a 2D outer product.
>
> `v5` confirms that the scalar epilogue really was inefficient, but full CTA epilogue staging costs too much residency in the current `128x128` family.

## 1. `sgemm_naive_row`

### Structure

`sgemm_naive_row` in [`01_naive.cuh`](./01_naive.cuh) is the most straightforward possible form: one thread computes one `C[row, col]` value from start to finish.

- Every loop iteration loads `A[row, k]` and `B[k, col]` directly from global memory.
- Each thread owns exactly one output accumulator.
- There is no reuse along the `K` dimension, so the same `A` rows and `B` columns are repeatedly fetched through DRAM, L2, and L1 by many different threads.

### Key NCU Metrics

- Duration: `78.30 ms`
- Throughput: `2.04 TFLOPS`
- Achieved occupancy: `66.66%`
- Eligible warps per scheduler: `1.16`
- Issued warps per scheduler: `0.27`
- No eligible: `73.08%`
- Dominant stalls: `Stall LG Throttle 18.32`, `Long Scoreboard 3.02`

### Interpretation

This kernel is not slow because occupancy is low. Occupancy is actually around 66%. The real problem is that the scheduler frequently cannot find a warp it can issue.

- `Issued Warp 0.27` means instruction issue is extremely sparse.
- `No Eligible 73.08%` means most warps are waiting on memory or cannot issue their next instruction in time.
- The fact that `LG Throttle` dominates shows that the global-memory and L1TEX load path is being heavily over-pressured.

So the starting bottleneck is not "too little arithmetic." It is the access structure itself: `A` and `B` are reloaded from global memory far too often. That matches the starting point in [`how_to_optimize_sgemm.md`](./how_to_optimize_sgemm.md) exactly.

## 2. `sgemm_smem_tiling`

### Structure

[`02_smem_tiling.cuh`](./02_smem_tiling.cuh) loads `BM x BK` and `BK x BN` tiles into shared memory and lets the threads in a block reuse them.

- Once a tile of `A` or `B` is loaded from global memory, many threads within the CTA share it.
- The kernel walks along `K` in `BK`-sized chunks and repeats `load -> sync -> compute -> sync`.
- Global reloads drop, but shared-memory load/store and `__syncthreads()` now become the hot path.

### Key NCU Metrics

- Duration: `60.20 ms`
- Throughput: `2.93 TFLOPS`
- Achieved occupancy: `66.66%`
- Eligible warps per scheduler: `0.89`
- Issued warps per scheduler: `0.20`
- No eligible: `79.79%`
- Dominant stalls: `Stall MIO Throttle 23.77`, `Long Scoreboard 5.97`, `Barrier 2.76`
- Nsight warning: shared-store average `1.2-way` bank conflict, bank conflicts `29,216,048`

### Interpretation

Shared-memory tiling is clearly the right direction, but this implementation still pays a large synchronization cost with a relatively heavy block shape.

- Performance improves from `2.04 -> 2.93 TFLOPS`, but not by much.
- The `LG Throttle` bottleneck recedes, but `MIO Throttle` becomes prominent.
- That means the shared-memory instruction path has now become the main bottleneck.
- `Barrier 2.76` is direct evidence that the per-tile sync cost is real.
- The shared-store bank conflict warning shows that memory reuse was gained, but the shared-memory path is still not clean.

This stage mainly moves the bottleneck from global memory to shared memory. The next step is to reuse each shared-memory tile more aggressively by increasing the number of outputs computed per thread.

## 3. `sgemm_blocktiling_1d<64, 64, 8, 8, 4>`

### Structure

The 1D block-tiling kernel in [`03_blocktiling_1d.cuh`](./03_blocktiling_1d.cuh) gives each thread not one output but `THREAD_M` outputs in registers.

- The CTA computes a `64 x 64` tile of `C`.
- It stages data in shared memory with `BLOCK_K=8`.
- Each thread computes a strip of `THREAD_M=8` output rows.
- `A` is consumed from a transposed shared-memory layout, with padding `4`.

### Key NCU Metrics

- Duration: `22.47 ms`
- Throughput: `6.23 TFLOPS`
- Registers per thread: `41`
- Threads per block: `512`
- Achieved occupancy: `66.31%`
- Issued warps per scheduler: `0.32`
- No eligible: `68.21%`
- Dominant stalls: `Long Scoreboard 7.49`, `MIO Throttle 7.30`, `Barrier 6.34`
- `L1 Wavefronts Shared Excessive = 0`

### Interpretation

This is the first major jump: `2.93 -> 6.23 TFLOPS`.

Two things matter most here.

1. A `B[k, col]` value loaded from shared memory is reused across multiple outputs within the same thread.
2. With multiple accumulators in registers, each shared load feeds many more FMAs.

In other words, if `sgemm_smem_tiling` moved data into shared memory but still had limited thread-level reuse, this stage pulls that reuse into registers.

The padding also matters.

- The `pad4` version analyzed here has `L1 Wavefronts Shared Excessive = 0`.
- A no-pad variant shows a large amount of excessive wavefront traffic.

So the speedup here is not only from register tiling. It also includes the benefit of a cleaner shared-memory layout.

The remaining bottlenecks are still `Long Scoreboard`, `MIO Throttle`, and `Barrier`. The kernel still has a long load-to-use dependency chain because it reads from shared memory and immediately performs scalar accumulation.

## 4. `sgemm_blocktiling_1d<64, 64, 8, 16, 4>`

### Structure

This version keeps the CTA tile and `BLOCK_K` fixed, and increases only `THREAD_M` from `8 -> 16`.

- The number of accumulators per thread increases.
- Fewer threads are needed to cover the same output tile.
- Each thread consumes more of the operands it reads from shared memory.

### Key NCU Metrics

- Duration: `19.35 ms`
- Throughput: `7.10 TFLOPS`
- Registers per thread: `55`
- Threads per block: `256`
- Achieved occupancy: `65.68%`
- Issued warps per scheduler: `0.35`
- No eligible: `65.20%`
- Dominant stalls: `Long Scoreboard 8.62`, `Barrier 6.86`, `Short Scoreboard 2.47`, `MIO Throttle 2.38`
- `L1 Wavefronts Shared Excessive = 0`

### Interpretation

This is a small but useful experiment about whether a larger thread tile is always better.

- Performance improves from `6.23 -> 7.10 TFLOPS`.
- `MIO Throttle` drops sharply from `7.30 -> 2.38`, so the shared-memory instruction path gets lighter.
- But registers rise from `41 -> 55`, and `Long Scoreboard` actually gets worse at `8.62`.

So per-thread reuse improved, but the dependency chain also became longer. The pattern is clear.

- Smaller tile: more instructions, less reuse.
- Larger tile: better reuse, but longer register dependencies.

This kernel is still a net win. But the next improvement should not be "keep growing `THREAD_M` forever." It should be to increase block-level reuse by enlarging the CTA tile itself.

## 5. `sgemm_blocktiling_1d<128, 128, 8, 64, 4>`

### Structure

Here the CTA tile grows from `64x64` to `128x128`, and each thread holds `64` accumulators.

- One CTA computes a larger tile of `C`.
- Each shared-memory tile contributes to more outputs.
- Per-thread work increases, so occupancy drops, but ILP and reuse increase.

### Key NCU Metrics

- Duration: `13.39 ms`
- Throughput: `11.61 TFLOPS`
- Registers per thread: `116`
- Threads per block: `256`
- Achieved occupancy: `32.32%`
- Eligible warps per scheduler: `0.83`
- Issued warps per scheduler: `0.49`
- No eligible: `51.04%`
- Dominant stalls: `MIO Throttle 2.28`, `Short Scoreboard 2.30`, `Long Scoreboard 0.71`, `Barrier 0.34`

### Interpretation

At first glance, the large occupancy drop from `65% -> 32%` looks bad. But actual performance rises from `7.10 -> 11.61 TFLOPS`.

This is one of the clearest messages in [`how_to_optimize_sgemm.md`](./how_to_optimize_sgemm.md).

- Higher occupancy is not automatically better.
- If reuse and ILP improve enough, fewer resident warps can still lead to better scheduler efficiency.

This kernel is much healthier than the `64x64` family in several ways.

- `Issued Warp 0.49` is a large improvement.
- `Warp Cycles Per Issued Instruction` drops significantly.
- `Long Scoreboard 0.71` is far lower than in the previous kernels.
- `Barrier 0.34` shows much lower synchronization overhead.

So the larger CTA tile increases shared-tile reuse enough to more than offset the loss in occupancy.

## 6. `sgemm_blocktiling_1d<128, 128, 16, 64, 4>`

### Structure

This version keeps the `128x128` 1D structure and increases `BLOCK_K` from `8 -> 16`.

- The loop runs fewer iterations.
- `__syncthreads()` is called fewer times.
- More compute happens after each tile load.

### Key NCU Metrics

- Duration: `13.04 ms`
- Throughput: `11.99 TFLOPS`
- Registers per thread: `116`
- Achieved occupancy: `32.36%`
- Issued warps per scheduler: `0.50`
- No eligible: `50.18%`
- Dominant stalls: `Short Scoreboard 2.20`, `MIO Throttle 2.54`, `Long Scoreboard 0.46`, `Barrier 0.23`

### Interpretation

The gain is modest, `11.61 -> 11.99 TFLOPS`, but the direction is clear.

- `Barrier` falls from `0.34 -> 0.23`.
- `Long Scoreboard` drops from `0.71 -> 0.46`.
- `MIO Throttle` is also somewhat reduced.

So increasing `BLOCK_K` works here as a cleanup pass that reduces loop and barrier overhead after the larger reuse structure is already in place. Because the shared-memory footprint also grows, this would not have been the right first optimization earlier in the pipeline.

This stage is closer to fine-tuning than to a major structural change.

## 7. `sgemm_blocktiling_2d<128, 128, 8, 8, 8, 4>`

### Structure

[`04_blocktiling_2d.cuh`](./04_blocktiling_2d.cuh) moves from a 1D accumulation strip to a 2D outer-product structure.

- Each thread owns an `8 x 8` output tile in registers.
- The inner loop loads `A` and `B` fragments as row and column fragments and forms an outer product.
- The total number of accumulators is still `64`, the same as 1D `THREAD_M=64`, but the shape of immediately issueable FMA groups changes.

### Key NCU Metrics

- Duration: `8.98 ms`
- Throughput: `16.05 TFLOPS`
- Registers per thread: `104`
- Achieved occupancy: `32.34%`
- Eligible warps per scheduler: `1.60`
- Issued warps per scheduler: `0.62`
- No eligible: `38.16%`
- Dominant stalls: `Not Selected 1.59`, `Short Scoreboard 0.87`, `Long Scoreboard 0.84`, `Barrier 0.62`
- `L1 Wavefronts Shared: 704,643,072`
- Ideal shared wavefronts: `436,207,616`
- Shared excessive wavefronts: `268,435,456`

### Interpretation

This is the second major turning point: `11.99 -> 16.05 TFLOPS`.

1D register tiling increased reuse, but accumulation was still biased toward a single strip direction. Once the kernel switches to a 2D outer product, the same 64 accumulators can be updated in a form that is much more scheduler-friendly.

#### Why is 2D faster even with the same 64 accumulators?

The clearest comparison is `1d<128,128,16,64,4>` versus `2d<128,128,16,8,8,4>`.

- Both use a `128 x 128` CTA tile.
- Both use `BLOCK_K=16`.
- Both use `64` accumulators per thread.
- Both use `256` threads per block.

So the key difference is not "more arithmetic." It is how much operand state each thread must prepare in order to update the same 64 outputs.

The inner loop of the 1D kernel is conceptually like this.

```cpp
b = B[k, col]
for m in 0..63:
  a = A[k, row + m]
  acc[m] += a * b
```

This has the advantage that one `b` value is reused 64 times. But to produce the same 64 accumulator updates, the thread needs one `B` element and 64 `A` elements in sequence. So `B` reuse is excellent, but `A` is barely reused. The computation is shaped like a long 1D dot product.

The 2D kernel looks more like this.

```cpp
for m in 0..7:
  a = A[k, row + m]
  for n in 0..7:
    b = B[k, col + n]
    acc[m][n] += a * b
```

Now the same 64 accumulator updates require only 8 `A` elements and 8 `B` elements. In other words, `8 + 8 = 16` operand fragments produce `8 x 8 = 64` FMAs. The shape is no longer a long strip. It is a small multiplication table.

That same point can be summarized like this.

| Form | Outputs | Required A fragments | Required B fragments | Total |
| --- | ---: | ---: | ---: | ---: |
| `64 x 1` 1D | `64` | `64` | `1` | `65` |
| `8 x 8` 2D | `64` | `8` | `8` | `16` |

The key is that 2D reuses both `A` and `B`. 1D is biased toward reuse on one side. With 2D, once the operands are prepared, many FMAs can be issued in succession more easily.

That is why the benefit of 2D is better understood not as "more accumulators," but as "the same 64 results produced from a much smaller operand set."

The NCU numbers support exactly that interpretation. Moving from `1d<128,128,16,64,4>` to `2d<128,128,16,8,8,4>` gives:

- Throughput: `11.99 -> 17.39 TFLOPS`
- Eligible warps per scheduler: `0.90 -> 1.90`
- Issued warps per scheduler: `0.50 -> 0.66`
- No eligible: `50.18% -> 34.02%`
- Short scoreboard: `2.20 -> 0.63`

The drop in `Short Scoreboard` is especially important. It does not mean 2D magically perfects memory access. It means the immediate load-to-use dependency chain has been reduced substantially.

That improvement shows up directly in scheduler metrics.

- `Eligible Warp 1.60` is much higher than in 1D.
- `Issued Warp 0.62` also rises significantly.
- `No Eligible 38.16%` drops noticeably.

So the scheduler spends much less time with nothing ready to issue.

There is still a new bottleneck, though.

- Shared wavefront traffic is far above ideal.
- Excessive wavefronts reach `268,435,456`.

So the compute structure improves dramatically, but shared-memory load access is still not fully cleaned up. At this point, the main bottleneck is shifting away from pure compute and toward inefficiencies in the shared-memory path.

## 8. `sgemm_blocktiling_2d<128, 128, 16, 8, 8, 4>`

### Structure

This is the 2D outer-product kernel with `BLOCK_K=16`.

- It keeps the advantages of 2D register blocking.
- It reduces loop and barrier count.
- Each shared tile is used for a longer compute phase.

### Key NCU Metrics

- Duration: `8.25 ms`
- Throughput: `17.39 TFLOPS`
- Registers per thread: `108`
- Achieved occupancy: `32.35%`
- Eligible warps per scheduler: `1.90`
- Issued warps per scheduler: `0.66`
- No eligible: `34.02%`
- Dominant stalls: `Not Selected 1.89`, `MIO Throttle 0.67`, `Long Scoreboard 0.41`, `Short Scoreboard 0.63`
- Shared excessive wavefronts: `285,212,672`

### Interpretation

This is the fastest scalar kernel in the requested family.

- `16.05 -> 17.39 TFLOPS`
- `Eligible Warp 1.60 -> 1.90`
- `Issued Warp 0.62 -> 0.66`

So the effect of increasing `BLOCK_K` becomes more pronounced once it is combined with the 2D structure. Each shared tile now supports a longer compute phase, which makes the reduction in loop overhead more valuable.

One notable point is that `Not Selected` becomes the representative stall.

- That does not mean "there are no warps available."
- It means "enough warps are available that the scheduler spends time choosing among them."

At this stage the kernel is already structurally quite clean, and the remaining headroom is more about memory-path detail and deeper pipeline optimization.

## 9. `sgemm_blocktiling_2d_vec4<128, 128, 8, 8, 8, 4>`

### Structure

[`05_blocktiling_2d_vec4.cuh`](./05_blocktiling_2d_vec4.cuh) keeps the 2D compute structure but moves the global load/store path to `float4`.

- `A` and `B` tile preload use vectorized transactions.
- `C` store also uses a vectorized path.
- This stage improves data movement efficiency more than the compute structure itself.

### Key NCU Metrics

- Duration: `8.41 ms`
- Throughput: `16.86 TFLOPS`
- Registers per thread: `111`
- Achieved occupancy: `32.37%`
- Eligible warps per scheduler: `1.54`
- Issued warps per scheduler: `0.64`
- No eligible: `36.05%`
- Dominant stalls: `Not Selected 1.42`, `Long Scoreboard 1.09`, `Short Scoreboard 0.96`, `MIO Throttle 0.39`
- Shared excessive wavefronts: `268,435,456`
- `L2 Theoretical Sectors Global Excessive`: `2.10 MB`

### Interpretation

This result has to be compared against the scalar 2D kernel with the same `BLOCK_K`.

- scalar `2d<..., BK=8>`: `8.56 ms`, `16.05 TFLOPS`
- vec4 `2d_vec4<..., BK=8>`: `8.15 ms`, `16.86 TFLOPS`

So vec4 is clearly beneficial. The apparent confusion comes from the fact that the immediately preceding kernel in the requested order is the scalar `BK=16` version, which is already stronger.

The real message is simple.

- Vec4 improves global-memory transaction efficiency.
- `L2 Theoretical Sectors Global Excessive` drops substantially.
- Shared excessive wavefronts remain almost unchanged.

So vec4 is not solving the shared-memory access-pattern problem. Its gain comes from cleaning up the global path, while the shared-memory side becomes the more clearly exposed remaining bottleneck.

## 10. `sgemm_blocktiling_2d_vec4<128, 128, 16, 8, 8, 4>`

### Structure

This is the end point of the block-tiling family.

- 2D outer product
- `BLOCK_K=16`
- padded shared-memory layout
- vec4 global load/store

This is the point where all of the major optimization axes developed so far are combined.

### Key NCU Metrics

- Duration: `7.82 ms`
- Throughput: `18.20 TFLOPS`
- Registers per thread: `103`
- Achieved occupancy: `32.35%`
- Eligible warps per scheduler: `1.92`
- Issued warps per scheduler: `0.68`
- No eligible: `32.16%`
- Dominant stalls: `Not Selected 1.83`, `Short Scoreboard 0.73`, `MIO Throttle 0.54`, `Long Scoreboard 0.56`
- Shared excessive wavefronts: `285,212,672`
- `L2 Theoretical Sectors Global Excessive`: `2.10 MB`

### Interpretation

The best point in the block-tiling family is already far ahead of the naive baseline.

- `2.04 -> 18.20 TFLOPS`
- elapsed time `67.32 -> 7.55 ms`
- about `9x` speedup

This kernel is fast because each optimization stage removed a different bottleneck in sequence.

- shared tiling: reduces global reloads
- register tiling: increases reuse of shared operands
- 2D outer product: increases compute density and ILP
- `BLOCK_K=16`: reduces barrier and loop overhead
- vec4: improves global transaction efficiency

But this is still not the end.

- Shared excessive wavefront traffic is still large.
- The remaining dominant stalls are no longer raw global-memory stalls. They have become a mix of shared-memory-path inefficiency and scheduler selection overhead.

So the headroom left at this point is no longer mainly about the global path. It is mostly about shared-memory access and deeper software pipelining.

For the warptiling family, the benchmark tables below come from independent reruns of each kernel, while the 4096 NCU tables later in each section come from representative Nsight Compute profiles. As a result, the benchmark Duration/TFLOPS and the NCU Duration/TFLOPS values within the same section do not always match exactly.

## 11. `sgemm_warptiling_v0<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v0<128, 128, 8, 64, 32, 8>`

### Structure

`v0` in [`06_warptiling.cuh`](./06_warptiling.cuh) is the first version in this flow that explicitly encodes the `threadblock tile -> warp tile -> thread tile` hierarchy in code.

- CTA tile: still `128x128x8`
- `8` warps per block
- each thread still owns an `8x8 = 64`-element accumulator tile
- the only difference between the two variants is whether each warp owns a `32x64` tile or a `64x32` tile

So the difference is not in accumulator count. It is in how the same `128x128` CTA tile is partitioned across warps.

This `v0` is intentionally simple.

- Global-to-shared preload still uses scalar helpers.
- `warp_mma()` does not yet use a CUTLASS-like `4x4 x 4` fragment structure. Each thread updates its own `8x8` micro-tile more directly.
- The epilogue also writes each thread's `8x8` tile directly to global memory with scalar stores.
- There is still no register prefetch, software pipeline, or vec4 path.

This stage is therefore a baseline for answering a specific question: what does introducing a warp hierarchy improve, and what does it leave unresolved?

### Benchmark Results

Elapsed time:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v0<128,128,8,32,64,8>` | 0.033987 | 0.053378 | 0.091939 | 0.204156 | 1.232713 | 8.503943 |
| `warptiling_v0<128,128,8,64,32,8>` | 0.039039 | 0.059792 | 0.095734 | 0.197789 | 1.179455 | 8.150416 |

TFLOPS:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v0<128,128,8,32,64,8>` | 0.122447 | 0.626165 | 2.914024 | 10.508573 | 13.929825 | 16.157845 |
| `warptiling_v0<128,128,8,64,32,8>` | 0.106599 | 0.558997 | 2.798484 | 10.846855 | 14.558822 | 16.858698 |

At 4096, the `64x32` warp tile is consistently faster and reaches `8.15 ms`, `16.86 TFLOPS`. That is clearly better than the non-vec4 `blocktiling_2d<128,128,8,8,8,4>` baseline (`8.56 ms`, `16.05 TFLOPS`). By contrast, the `32x64` warp tile stays at `8.50 ms`, `16.16 TFLOPS`, roughly `4.2%` lower than `64x32`.

So warp tiling itself is heading in the right direction, but the benefit almost disappears if the warp-tile orientation and lane mapping do not line up well.

### 4096 Key NCU Metrics

`warptiling_v0<128,128,8,32,64,8>`:

- Duration: `9.01 ms`
- Throughput: `15.93 TFLOPS`
- Registers per thread: `125`
- Achieved occupancy: `32.29%`
- Eligible warps per scheduler: `1.71`
- Issued warps per scheduler: `0.61`
- No eligible: `38.76%`
- Dominant stalls: `Not Selected 1.79`, `Long Scoreboard 0.79`, `Barrier 0.68`, `MIO Throttle 0.67`, `Short Scoreboard 0.44`
- Shared bank conflicts: `268,435,456`
- Shared excessive wavefronts: `268,435,456`
- `L2 Theoretical Sectors Global Excessive`: `14.68 MB`
- Global store average utilization: `4 byte / sector`

`warptiling_v0<128,128,8,64,32,8>`:

- Duration: `8.55 ms`
- Throughput: `16.65 TFLOPS`
- Registers per thread: `125`
- Achieved occupancy: `32.22%`
- Eligible warps per scheduler: `1.92`
- Issued warps per scheduler: `0.65`
- No eligible: `35.13%`
- Dominant stalls: `Not Selected 1.95`, `Long Scoreboard 0.79`, `Barrier 0.68`, `MIO Throttle 0.32`, `Short Scoreboard 0.23`
- Shared bank conflicts: `0`
- Shared excessive wavefronts: `0`
- `L2 Theoretical Sectors Global Excessive`: `14.68 MB`
- Global store average utilization: `4 byte / sector`

### Interpretation

This result shows what really matters in warptiling. Simply adding a warp-level hierarchy does not automatically make the kernel faster. The warp tile shape and the lane layout must align with the shared-memory access pattern.

The two kernels are almost identical in every obvious dimension.

- same CTA tile: `128x128x8`
- same block size: `256`
- same thread tile: `8x8`
- same accumulator count: `64`
- same register pressure: `125 regs/thread`

And yet the result is quite different.

- `32x64`: `16.16 TFLOPS`
- `64x32`: `16.86 TFLOPS`

So the difference is not "how many outputs are computed." It is how the same outputs are partitioned and consumed inside a warp.

The biggest difference appears immediately in the shared-memory path.

- `32x64`: shared bank conflicts `268,435,456`, excessive wavefronts `268,435,456`
- `64x32`: both are `0`

That strongly suggests that under the simple row-major lane mapping used in `v0`, the `32x64` warp tile mismatches the shared-memory consumption pattern, especially on the `B` side. In contrast, `64x32` narrows the `N` extent per warp, which makes shared loads much cleaner and reduces both `MIO Throttle` and `Short Scoreboard`.

- `MIO Throttle`: `0.67 -> 0.32`
- `Short Scoreboard`: `0.44 -> 0.23`
- `Eligible Warp`: `1.71 -> 1.92`
- `Issued Warp`: `0.61 -> 0.65`

So the gain from `64x32` comes from a cleaner shared-consumption path, not from more arithmetic.

The limitations of `v0` are equally clear.

First, register pressure is already high.

- both variants use `125 regs/thread`
- occupancy remains around `32%`

Second, the global-store path is still very rough.

- both variants show `L2 Theoretical Sectors Global Excessive = 14.68 MB`
- global-store utilization is only `4 byte / sector`

The current scalar epilogue writes each thread's `8x8` tile directly to global memory, so even if the shared path improves, the final store path is still far from the cleanliness of the later vec4 kernels.

Third, `v0` still does not use a CUTLASS-like lane-fragment layout.

Each thread still treats its `8x8` result tile as a dense local block. The fact that `32x64` breaks the shared path so badly is exactly what motivates the next step: a more refined lane mapping and `4x4` island organization similar in spirit to CUTLASS SIMT layouts.

So the role of `warptiling v0` is easy to summarize.

- Good news: the `64x32` variant is clearly faster than the `2D BK=8` baseline.
- Limitation: if the warp tile orientation is wrong, the shared path breaks immediately.
- Lesson: the real value of warptiling is not "adding a warp layer" by itself, but choosing a warp-tile shape and lane-fragment layout that match the shared/global path.

## 12. `sgemm_warptiling_v1<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v1<128, 128, 8, 64, 32, 8>`

### Structure

`v1` in [`06_warptiling.cuh`](./06_warptiling.cuh) keeps the same CTA, block, and thread scales as `v0`, but refines the lane mapping and the epilogue.

- CTA tile: still `128x128x8`
- number of warps: still `8`
- thread tile: still `8x8`
- accumulator count: still `64`
- register pressure: still `125 regs/thread`

So the point of `v1` is not to do more work. It is to execute the same work with a better warp-internal layout.

Three changes matter most.

1. The lane mapping moves away from a simple row-major layout toward something closer to `RowMajorInterleaved<2>`.
2. `warp_mma()` no longer treats the logical `8x8` tile as one dense block. It consumes it as four `4x4` islands.
3. The epilogue also stops using a simple row-major scalar store and instead writes data in `4x4` groups.

The goals are straightforward.

- Stabilize the shared-memory consumption path that was shape-sensitive in `v0`.
- Make the lane-fragment layout within each thread tile line up better with the shared-memory bank structure.
- Improve global-store sector utilization relative to the scalar epilogue in `v0`.

So `v1` is best understood as the first partial adoption of a CUTLASS-like SIMT lane-fragment organization, built on top of the warp hierarchy proven by `v0`.

### Benchmark Results

Elapsed time:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v1<128,128,8,32,64,8>` | 0.025396 | 0.043198 | 0.079007 | 0.178549 | 1.094341 | 7.951270 |
| `warptiling_v1<128,128,8,64,32,8>` | 0.033271 | 0.054355 | 0.087460 | 0.183464 | 1.113879 | 7.910420 |

TFLOPS:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v1<128,128,8,32,64,8>` | 0.163867 | 0.773729 | 3.390979 | 12.015687 | 15.691154 | 17.280938 |
| `warptiling_v1<128,128,8,64,32,8>` | 0.125080 | 0.614914 | 3.063248 | 11.693794 | 15.415925 | 17.370176 |

At 4096 the two kernels are now essentially in the same performance class.

- `32x64`: `7.95 ms`, `17.28 TFLOPS`
- `64x32`: `7.91 ms`, `17.37 TFLOPS`

That means the strong warp-shape sensitivity seen in `v0` is nearly gone. The `64x32` variant is still slightly ahead, but only marginally. Both variants now beat the best `warptiling v0` point and also edge past `blocktiling_2d_vec4<128,128,8,8,8,4>` at `16.86 TFLOPS`. They still do not reach the `18.20 TFLOPS` block-tiling peak at `blocktiling_2d_vec4<128,128,16,8,8,4>`.

### 4096 Key NCU Metrics

| Metric | `32x64` | `64x32` |
| --- | ---: | ---: |
| Duration (ms) | 8.25 | 8.32 |
| TFLOPS | 17.01 | 16.95 |
| Compute Throughput | 62.29% | 62.73% |
| Memory Throughput | 41.04% | 41.33% |
| DRAM Bandwidth | 130.77 GB/s | 127.99 GB/s |
| L2 Bandwidth | 520.54 GB/s | 515.80 GB/s |
| Registers / thread | 125 | 125 |
| Achieved Occupancy | 32.30% | 32.17% |
| Eligible Warps / Scheduler | 2.09 | 2.09 |
| Issued Warps / Scheduler | 0.67 | 0.67 |
| No Eligible | 33.41% | 33.33% |
| L2 Hit Rate | 78.40% | 79.12% |
| L1/TEX Hit Rate | 2.45% | 4.35% |
| Global Store Avg Bytes / Sector | 8 | 8 |
| L2 Theoretical Sectors Global Excessive | 6.29 MB | 6.29 MB |

The more important point is what the two kernels now share.

- shared bank conflict: both `0`
- shared excessive wavefronts: both `0`
- top stalls shift toward `Not Selected`, `Long Scoreboard`, and `Barrier`
- `MIO Throttle` is very low in both kernels, around `0.20`
- global-store average utilization rises to `8 byte / sector`

So by `v1`, the shared-memory path is no longer the main visible problem. The bottleneck has moved upward: high register pressure, lower occupancy, and scheduler behavior now matter more than basic shared-memory cleanliness.

### `v0 -> v1` Interpretation

The significance of `v1` is not just that it is slightly faster. It directly addresses the two weaknesses exposed by `v0`.

First, the pathological shape sensitivity of the `32x64` warp tile disappears.

- `TFLOPS`: `16.16 -> 17.28`
- `Duration`: `8.50 -> 7.95 ms`
- shared bank conflicts: `268,435,456 -> 0`
- shared excessive wavefronts: `268,435,456 -> 0`
- `MIO Throttle`: `0.67 -> 0.20`
- `Short Scoreboard`: `0.44 -> 0.09`

So the problem in `v0 32x64` was not the warp tile shape alone. It was the fact that the lane mapping and fragment/store layout did not support that shape well. `v1` fixes exactly that.

Second, even the already-cleaner `64x32` variant improves further in the global-store path.

- `TFLOPS`: `16.86 -> 17.37`
- `Duration`: `8.15 -> 7.91 ms`
- global-store average utilization: `4 -> 8 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: `14.68 -> 6.29 MB`

So `64x32` was already structurally sound in `v0`, but `v1` still gains more by refining the epilogue and lane-fragment layout.

Third, the remaining bottlenecks become clearer.

- register pressure is still `125 regs/thread`
- occupancy is still around `32%`
- the dominant stalls are now `Not Selected`, `Long Scoreboard`, and `Barrier`

That means the next headroom is no longer about removing shared bank conflicts. It is about increasing load-to-use distance and reducing scheduler idle time in a low-occupancy regime. That points directly toward register prefetch, software pipelining, and a better global/store path.

In short, `warptiling v1` does four things.

- It turns the promise of `v0` into a real performance improvement.
- It drastically reduces warp-shape sensitivity.
- It removes most of the shared-path bottleneck and shifts the problem toward scheduling and scoreboard behavior.
- It still leaves room for further pipeline and memory-path refinement before reaching the final vec4 + BK16 peak.

## 13. `sgemm_warptiling_v2<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v2<128, 128, 8, 64, 32, 8>` vs `sgemm_warptiling_v2<128, 128, 16, 32, 64, 16>`

### Structure

`v2` in [`06_warptiling.cuh`](./06_warptiling.cuh) keeps the warp MMA, lane mapping, and epilogue of `v1`, and changes only the CTA tile ingest path.

- `v1` uses helpers that load each thread's global element and place it directly into shared-memory layout.
- `v2` first loads into thread-private register arrays `tb_frag_a` and `tb_frag_b`.
- It then stores those fragments into shared memory with explicit helper functions.
- At the source level, the data path becomes `global -> register fragment -> shared -> warp_mma`.

The important point is that this is not `cp.async` and not a hardware direct global-to-shared path.

- The code still explicitly stages through thread-private fragments.
- NCU also reports `smsp__inst_executed_op_ldgsts.sum = 0`.

So the improvement here does not come from `LDGSTS` or `cp.async`. It comes from making the scheduling of regular global loads and shared stores more explicit.

The `BLOCK_K=16` variant matters as well.

- It keeps the same register-staged load path.
- It cuts the outer-loop trip count in half.
- That reduces both barrier frequency and loop overhead.

So the essence of `v2` is not another warp-layout change. It is explicit software staging for the global-to-shared ingest path, on top of the already-clean `v1` layout.

### Benchmark Results

Elapsed time:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v2<128,128,8,32,64,8>` | 0.025253 | 0.043205 | 0.078327 | 0.177671 | 1.060437 | 7.529288 |
| `warptiling_v2<128,128,8,64,32,8>` | 0.033650 | 0.055470 | 0.088835 | 0.183905 | 1.098456 | 7.694295 |
| `warptiling_v2<128,128,16,32,64,16>` | 0.028275 | 0.047618 | 0.081872 | 0.155086 | 1.032581 | 7.182624 |

TFLOPS:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v2<128,128,8,32,64,8>` | 0.164793 | 0.773597 | 3.420404 | 12.075049 | 16.192838 | 18.249455 |
| `warptiling_v2<128,128,8,64,32,8>` | 0.123673 | 0.602548 | 3.015833 | 11.665735 | 15.632374 | 17.858087 |
| `warptiling_v2<128,128,16,32,64,16>` | 0.147183 | 0.701912 | 3.272305 | 13.833521 | 16.629667 | 19.130252 |

At 4096, `v2` shows three things.

- `BK=8, 32x64`: `17.28 -> 18.25 TFLOPS`, `7.95 -> 7.53 ms`
- `BK=8, 64x32`: `17.37 -> 17.86 TFLOPS`, `7.91 -> 7.69 ms`
- `BK=16, 32x64`: reaches `19.13 TFLOPS`, `7.18 ms`, a new overall best at the time

So explicit register staging clearly helps both BK8 variants over `v1`, and when paired with `BLOCK_K=16`, it is the first point in the flow that breaks `19 TFLOPS`.

### 4096 Key NCU Metrics

Throughput / scheduler:

| Kernel | Duration (ms) | TFLOPS | Compute Throughput | Memory Throughput | Registers / thread | Achieved Occupancy | Eligible Warps / Scheduler | Issued Warps / Scheduler | No Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2<128,128,8,32,64,8>` | 7.83 | 17.76 | 65.37% | 43.59% | 125 | 32.31% | 2.08 | 0.70 | 30.34% |
| `v2<128,128,8,64,32,8>` | 8.02 | 17.39 | 63.80% | 42.54% | 119 | 32.30% | 2.06 | 0.68 | 32.13% |
| `v2<128,128,16,32,64,16>` | 7.60 | 18.55 | 68.28% | 52.36% | 127 | 32.23% | 2.39 | 0.73 | 27.26% |

Memory path:

| Kernel | DRAM BW | L2 BW | L1/TEX Hit Rate | L2 Hit Rate | Global Store Avg Bytes / Sector | Shared Bank Conflict | Shared Excessive Wavefront | L2 Theoretical Sectors Global Excessive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2<128,128,8,32,64,8>` | 129.34 GB/s | 548.29 GB/s | 2.40% | 79.57% | 8 | 0 | 0 | 6.29 MB |
| `v2<128,128,8,64,32,8>` | 124.39 GB/s | 535.26 GB/s | 4.43% | 79.78% | 8 | 0 | 0 | 6.29 MB |
| `v2<128,128,16,32,64,16>` | 125.60 GB/s | 565.09 GB/s | 2.62% | 80.23% | 8 | 0 | 0 | 6.29 MB |

The message here is clear.

- shared bank conflicts were already `0` in `v1`, and they remain `0` in `v2`
- shared excessive wavefronts were already `0` in `v1`, and they remain `0` in `v2`
- average global-store utilization is still `8 byte / sector`

So the `v2` speedup did not come from solving a new shared-layout problem. It came from making global-to-shared staging and inner-loop scheduling more explicit.

### `v1 -> v2` Interpretation

First, the gain in `v2` is better understood as "tighter control over load staging on top of an already-clean shared path" than as "further cleanup of the shared path itself."

In `v1`, bank conflicts and excessive wavefronts were already essentially resolved. Even so, both BK8 variants get faster again in `v2`.

- `32x64`: `17.28 -> 18.25 TFLOPS`
- `64x32`: `17.37 -> 17.86 TFLOPS`
- occupancy stays around `32%` for both
- average global-store utilization and excessive sectors also stay at `8 byte / sector` and `6.29 MB`

That means the improvement is not coming from the store path or from occupancy. The more plausible interpretation is that the source-level change makes register-fragment lifetimes more explicit and creates more scheduling freedom between global loads, shared stores, and the next FMA.

Second, `v2` does not reintroduce warp-shape sensitivity. It keeps `32x64` as the top variant.

In `v0`, `32x64` was heavily affected by shared-path issues. In `v1`, that sensitivity was mostly removed. `v2` preserves that state while keeping `32x64` at the top.

- `32x64 BK8`: `18.25 TFLOPS`
- `64x32 BK8`: `17.86 TFLOPS`

So explicit register staging does not recreate warp-shape instability. It adds another gain on top of the stabilized warp layout that `v1` established.

Third, the `BLOCK_K=16` variant is the core result of this generation.

- `19.13 TFLOPS`, `7.18 ms`
- Compute Throughput `68.28%`
- Issued Warps / Scheduler `0.73`
- Eligible Warps / Scheduler `2.39`
- `Stall Barrier 0.48`, `Stall Long Scoreboard 0.48`

Relative to `v2 BK=8 32x64`, the main changes are:

- Compute Throughput: `65.37 -> 68.28%`
- Issued Warp: `0.70 -> 0.73`
- `Stall Barrier`: `0.62 -> 0.48`
- `Stall Long Scoreboard`: `0.72 -> 0.48`

That is exactly what you would expect when `BLOCK_K=16` reduces outer-loop and barrier exposure on top of explicit register staging.

Fourth, the remaining bottlenecks also become clearer.

- all three variants stay near `33%` theoretical occupancy and around `32%` achieved occupancy
- the top stall is still `Not Selected`
- NCU still reports `8 byte / sector` global-store utilization and `6.29 MB` excessive sectors

So after `v2`, the remaining headroom is no longer about shared-bank conflicts. It is about deeper prefetch/pipeline behavior and a better epilogue store path.

Auxiliary experiment note: the table below comes from a separate rerun outside the current mainline benchmark path, so the absolute values matter less than the relative changes caused by helper rewrites.

### 3-Way Comparison of load/store fragment helpers

Even within the same `v2` kernel, the exact implementation of `load_*_thread_fragment()` and `store_*_thread_fragment_to_smem()` produces visible performance differences.

- Version 1: original `elem_idx -> row/col` style for both load and store
- Version 2: load changed to `offset += advance_offset`
- Version 3: starting from version 2, store changed the same way

The 4096 TFLOPS summary is:

| Kernel | Version 1 | Version 2 | Version 3 | v1 -> v2 | v2 -> v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2<128,128,8,32,64,8>` | 17.82 | 18.16 | 17.45 | `+1.96%` | `-3.94%` |
| `v2<128,128,8,64,32,8>` | 17.53 | 18.03 | 17.71 | `+2.81%` | `-1.74%` |
| `v2<128,128,16,32,64,16>` | 15.72 | 19.31 | 19.58 | `+22.84%` | `+1.40%` |

The key 4096 NCU metrics tell the same story.

| Kernel | Version | Duration | Compute | Memory | Reg/thread | Achieved Occupancy | No Eligible | DRAM BW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2<128,128,8,32,64,8>` | v1 | 8.44 ms | 65.25% | 40.43% | 125 | 33.01% | 29.20% | 69.05 GB/s |
| `v2<128,128,8,32,64,8>` | v2 | 7.83 ms | 65.33% | 43.56% | 125 | 32.29% | 30.28% | 126.00 GB/s |
| `v2<128,128,8,32,64,8>` | v3 | 8.08 ms | 63.19% | 41.92% | 127 | 32.29% | 32.56% | 121.42 GB/s |
| `v2<128,128,8,64,32,8>` | v1 | 8.63 ms | 63.98% | 39.58% | 127 | 33.04% | 30.55% | 67.53 GB/s |
| `v2<128,128,8,64,32,8>` | v2 | 8.08 ms | 63.80% | 42.54% | 119 | 32.20% | 32.48% | 121.34 GB/s |
| `v2<128,128,8,64,32,8>` | v3 | 8.23 ms | 63.02% | 41.80% | 127 | 32.29% | 32.98% | 128.40 GB/s |
| `v2<128,128,16,32,64,16>` | v1 | 10.13 ms | 55.29% | 39.62% | 141 | 16.66% | 41.04% | 98.45 GB/s |
| `v2<128,128,16,32,64,16>` | v2 | 7.60 ms | 68.29% | 52.37% | 127 | 32.23% | 27.24% | 125.76 GB/s |
| `v2<128,128,16,32,64,16>` | v3 | 7.42 ms | 69.86% | 53.59% | 125 | 32.18% | 25.56% | 128.46 GB/s |

All three profiles still show `L2 Theoretical Sectors Global Excessive = 6.29 MB`. So this helper experiment is not about fixing the epilogue store path. It is about changing address generation inside the main loop and thereby changing codegen and scheduling.

The important point is not whether `%` and `/` disappeared in the source. Under the current shape, divisors such as `BK=8/16` and `N=128` are compile-time powers of two, so the compiler can already strength-reduce many of those operations.

The real difference is the final address form used by the load and store helpers.

- The old load form keeps runtime `lda/ldb`-based address generation inside the loop.
- The old store form is already close to `base + const * iter`.

That is why the load rewrite behaves like a real optimization.

- `offset += advance_offset` reduces dynamic per-iteration address generation.
- For `load_a_thread_fragment()`, the old form effectively recalculates the address from scratch every iteration.
- The new form computes the first address and the increment once, then only advances the offset in the loop.

This is why:

- the two `BK=8` kernels improve modestly from v1 to v2
- the `BK=16` kernel jumps from `15.72 -> 19.31 TFLOPS`
- NCU also improves in registers, occupancy, `No Eligible`, and compute throughput

By contrast, the store-side recurrence transform is much less clearly beneficial in this shape.

- The old shared-store path is already very simple.
- The new form adds a loop-carried dependency through the running `offset`.
- That becomes a codegen and scheduling tradeoff, not a memory-access-pattern improvement.

That matches the observed data:

- `BK=8` sees small regressions from v2 to v3
- `BK=16` sees only a small additional gain

So the right interpretation is:

- the load recurrence transform is a real optimization
- the store recurrence transform is, in this shape, mostly a codegen/scheduling tradeoff
- that is why `v1 -> v2` matters a lot, especially for `BK=16`, while `v2 -> v3` is only a second-order effect

There is another important consequence: the epilogue store inefficiency is still one of the most explanatory residual inefficiencies after `v2`.

- `Global Store Avg Bytes / Sector = 8`
- `Maximum Bytes Per Sector For Global Stores = 32`
- `L2 Theoretical Sectors Global Excessive = 6.29 MB`

That is why the next big target after `v2` is not shared layout. It is pipeline depth and the epilogue/store path.

## 14. `sgemm_warptiling_v3<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v3<128, 128, 8, 64, 32, 8>` vs `sgemm_warptiling_v3<128, 128, 16, 32, 64, 16>`

### Structure

`v3` in [`06_warptiling.cuh`](./06_warptiling.cuh) adds a 1-stage register prefetch on top of the explicit register staging in `v2`.

- `tile 0` is loaded into `tb_frag_a` and `tb_frag_b` before the loop begins.
- Inside the loop, the current fragment is stored to shared memory, the current tile is computed, and then the next tile fragment is loaded into registers.
- From a steady-state point of view, the global load that lived at the beginning of the next iteration in `v2` is moved to the end of the current iteration in `v3`.
- But there is still only one shared-memory stage. The next tile is not written into a second shared buffer immediately. `register -> shared` still happens at the head of the next iteration.

So `v3` is neither `cp.async` nor CTA-level double buffering. It is best thought of as partial prefetch: ordinary global loads are issued earlier so that the next tile gets a longer load-to-use window.

What matters is not that current-tile FMA and next-tile global loads fully overlap. The next-tile load still happens after the current `warp_mma()` completes. The point is that it has been moved forward by roughly one iteration boundary, which gives the next tile more latency-hiding room across the following `__syncthreads()` and shared-store phase.

### Benchmark Results

Elapsed time:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v3<128,128,8,32,64,8>` | 0.023094 | 0.038546 | 0.069104 | 0.135797 | 1.047821 | 7.341302 |
| `warptiling_v3<128,128,8,64,32,8>` | 0.031305 | 0.047996 | 0.082558 | 0.143212 | 1.026991 | 7.292531 |
| `warptiling_v3<128,128,16,32,64,16>` | 0.027904 | 0.047996 | 0.077314 | 0.151259 | 1.165612 | 8.013312 |

TFLOPS:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v3<128,128,8,32,64,8>` | 0.180199 | 0.867098 | 3.876945 | 15.798469 | 16.387793 | 18.716760 |
| `warptiling_v3<128,128,8,64,32,8>` | 0.132934 | 0.696371 | 3.245111 | 14.980510 | 16.720191 | 18.841937 |
| `warptiling_v3<128,128,16,32,64,16>` | 0.149136 | 0.696385 | 3.465253 | 14.183541 | 14.731736 | 17.147142 |

At 4096, `v3` produces a very asymmetric result.

- `BK=8, 32x64`: `18.25 -> 18.72 TFLOPS`, `7.53 -> 7.34 ms`
- `BK=8, 64x32`: `17.86 -> 18.84 TFLOPS`, `7.69 -> 7.29 ms`
- `BK=16, 32x64`: `19.13 -> 17.15 TFLOPS`, `7.18 -> 8.01 ms`

So the same 1-stage prefetch is effective for BK8, but causes a major regression for BK16.

### 4096 Key NCU Metrics

Throughput / scheduler:

| Kernel | Duration (ms) | TFLOPS | Compute Throughput | Memory Throughput | Registers / thread | Achieved Occupancy | Eligible Warps / Scheduler | Issued Warps / Scheduler | No Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v3<128,128,8,32,64,8>` | 7.46 | 18.08 | 71.85% | 45.76% | 127 | 32.36% | 2.43 | 0.77 | 23.49% |
| `v3<128,128,8,64,32,8>` | 7.26 | 18.26 | 73.71% | 46.95% | 128 | 32.35% | 2.35 | 0.78 | 21.59% |
| `v3<128,128,16,32,64,16>` | 9.00 | 16.73 | 59.17% | 44.43% | 152 | 16.67% | 1.16 | 0.63 | 36.92% |

Memory path:

| Kernel | DRAM BW | L2 BW | Global Store Avg Bytes / Sector | Shared Bank Conflict | L2 Theoretical Sectors Global Excessive |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v3<128,128,8,32,64,8>` | 129.20 GB/s | 575.68 GB/s | 8 | 0 | 6.29 MB |
| `v3<128,128,8,64,32,8>` | 146.61 GB/s | 591.15 GB/s | 8 | 0 | 6.29 MB |
| `v3<128,128,16,32,64,16>` | 110.87 GB/s | 477.56 GB/s | 8 | 469,497 | 6.29 MB |

From a stall point of view, the contrast between BK8 and BK16 is equally sharp.

- `v3 BK8 32x64`: `Long Scoreboard 0.09`, `Barrier 0.67`, `Dispatch Stall 0.48`
- `v3 BK8 64x32`: `Long Scoreboard 0.03`, `Barrier 0.74`, `Short Scoreboard 0.29`
- `v3 BK16 32x64`: `Long Scoreboard 0.45`, `Dispatch Stall 0.26`, `Wait 0.19`

### `v2 -> v3` Interpretation

First, the gain in BK8 comes from main-loop scheduling, not from epilogue or shared-layout changes.

Going from `v2 -> v3`, the following remain essentially unchanged.

- global-store average utilization: still `8 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: still `6.29 MB`
- shared bank conflict: still `0`
- number of shared stages: still `1`

And yet both BK8 variants improve.

- `32x64`: `18.25 -> 18.72 TFLOPS`
- `64x32`: `17.86 -> 18.84 TFLOPS`

So the gain comes from issuing the next-tile global load earlier and increasing the load-to-use distance.

This is especially clean in `BK8 32x64`.

- Compute Throughput: `65.37 -> 71.85%`
- Eligible Warps / Scheduler: `2.08 -> 2.43`
- Issued Warps / Scheduler: `0.70 -> 0.76`
- No Eligible: `30.34 -> 23.49%`
- `Stall Long Scoreboard`: `0.72 -> 0.09`

Occupancy and store efficiency stay the same, but scoreboard and scheduler behavior improve. That makes the BK8 gain a very direct partial-prefetch effect.

Second, `v3` also changes the winning warp shape inside BK8.

- In `v2`, BK8 was led by `32x64`.
- In `v3`, `64x32` becomes the fastest BK8 variant at `18.84 TFLOPS`.

That difference is better read as a scheduler and memory-path fit under the same prefetch hoist, not as an epilogue difference.

Third, the BK16 regression should not be read as "prefetch is bad." It should be read as "the register cliff was crossed."

The `v2 BK16` to `v3 BK16` change is:

- TFLOPS: `19.13 -> 17.15`
- Duration: `7.18 -> 8.01 ms`
- Registers / thread: `127 -> 152`
- Achieved Occupancy: `32.23 -> 16.67%`
- Eligible Warps / Scheduler: `2.39 -> 1.16`
- Issued Warps / Scheduler: `0.73 -> 0.63`
- No Eligible: `27.26 -> 36.92%`

The key point is that with `256 threads/block`, around `128 regs/thread` is effectively the boundary for `2 blocks/SM`.

- `v2 BK16` at `127 regs/thread` just barely stays in the `2 blocks/SM` tier.
- `v3 BK16` at `152 regs/thread` falls to `1 block/SM`.

So in BK16, the larger fragment live range introduced by prefetch cost more than the extra load-to-use distance was worth.

The minor new shared signal in BK16, `0 -> 469,497` bank conflicts, is not large enough to explain the regression. The decisive change is the rise in register count and the resulting residency collapse.

The lesson of `v3` is therefore simple.

- The same prefetch idea is effective when the kernel stays in the same occupancy tier.
- The same idea fails when it pushes the kernel over the register cliff.
- The epilogue store inefficiency still remains unresolved.

Auxiliary experiment note: the table below comes from an additional rerun, separate from the current mainline benchmark path.

### If fragment loads are moved to just before `warp_mma()`

Inside the same `v3` idea, moving the next-tile fragment load from just after `warp_mma()` to just before it produces a mixed result. In current standalone measurements, the two BK8 variants are essentially unchanged or slightly slower, while only BK16 improves clearly. In other words, the practical beneficiary is the BK16 path that had fallen off the register cliff.

4096 summary:

| Kernel | Original `v3` | load moved | Change |
| --- | ---: | ---: | ---: |
| `BK8 32x64` | `18.72 TFLOPS` | `18.71 TFLOPS` | `-0.05%` |
| `BK8 64x32` | `18.84 TFLOPS` | `18.66 TFLOPS` | `-0.96%` |
| `BK16 32x64` | `17.15 TFLOPS` | `18.20 TFLOPS` | `+6.14%` |

4096 profile comparison:

| Kernel | Compute Throughput | Registers / thread | Achieved Occupancy | Eligible / Issued | No Eligible | DRAM BW | Notable point |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `BK8 32x64` | `71.85 -> 70.79%` | `127 -> 122` | `32.36 -> 32.45%` | `2.43 / 0.77 -> 2.36 / 0.75` | `23.49 -> 24.61%` | `129.20 -> 139.55 GB/s` | global store `8 byte/sector`, excessive `6.29 MB` unchanged |
| `BK8 64x32` | `73.71 -> 70.94%` | `128 -> 126` | `32.35 -> 32.52%` | `2.35 / 0.78 -> 2.33 / 0.75` | `21.59 -> 24.65%` | `146.61 -> 158.93 GB/s` | global store `8 byte/sector`, excessive `6.29 MB` unchanged |
| `BK16 32x64` | `59.17 -> 66.50%` | `152 -> 128` | `16.67 -> 32.37%` | `1.16 / 0.63 -> 2.35 / 0.71` | `36.92 -> 29.13%` | `110.87 -> 125.95 GB/s` | shared bank conflict `469,497 -> 0`, excessive `6.29 MB` unchanged |

The interpretation can be summarized in three lines.

- For BK8, the benchmark change is essentially flat or slightly negative. The profile also gets slightly worse in compute throughput, eligible/issued warps, and `No Eligible`, so there is no robust latency-hiding gain.
- For BK16, the reason for improvement is clear. Moving the load shortens fragment live range, registers drop from `152 -> 128`, occupancy recovers from `16.67 -> 32.37%`, and the register cliff is largely undone.
- All three profiles still show `8 byte / sector` and `L2 Theoretical Sectors Global Excessive = 6.29 MB`, so this is still a main-loop scheduling/register-allocation experiment, not an epilogue/store-path fix.

## 15. `sgemm_warptiling_v4<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v4<128, 128, 8, 64, 32, 8>` vs `sgemm_warptiling_v4<128, 128, 16, 32, 64, 16>`

### Structure

`v4` in [`06_warptiling.cuh`](./06_warptiling.cuh) keeps the overall loop skeleton of `v3`, but rewrites the CTA tile ingest helpers around `float4`.

- `load_a_thread_fragment_vec4()` and `load_b_thread_fragment_vec4()` read global `A/B` as `float4`.
- Shared stores are not symmetric. `A` must preserve a transposed shared layout, so each `float4` fragment is scattered as four scalar stores.
- `B` uses a row-major contiguous shared layout, so its `float4` shared store can remain vectorized.
- The epilogue still calls `v1::store_accum_to_gmem()`. A `store_accum_to_gmem_vec4()` helper exists, but the main benchmark path does not enable it.

So `v4` should not be read as "everything is vectorized all the way through the final `C` store." It is better read as "global ingest becomes vec4, and the shared path is vectorized only where the layout actually allows it."

### Benchmark Results

Elapsed time:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v4<128,128,8,32,64,8>` | 0.021948 | 0.035884 | 0.063512 | 0.143237 | 0.961416 | 6.793994 |
| `warptiling_v4<128,128,8,64,32,8>` | 0.030454 | 0.047177 | 0.082126 | 0.153376 | 0.973981 | 6.856581 |
| `warptiling_v4<128,128,16,32,64,16>` | 0.026093 | 0.041997 | 0.074253 | 0.125690 | 0.931647 | 6.547315 |

TFLOPS:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v4<128,128,8,32,64,8>` | 0.189610 | 0.931432 | 4.218276 | 14.977867 | 17.860620 | 20.224539 |
| `warptiling_v4<128,128,8,64,32,8>` | 0.136651 | 0.708473 | 3.262194 | 13.987729 | 17.630193 | 20.039929 |
| `warptiling_v4<128,128,16,32,64,16>` | 0.159488 | 0.795855 | 3.608095 | 17.068839 | 18.431322 | 20.986528 |

The picture is now very clear.

- `BK8 32x64`: `18.72 -> 20.22 TFLOPS`, `7.34 -> 6.79 ms`
- `BK8 64x32`: `18.84 -> 20.04 TFLOPS`, `7.29 -> 6.86 ms`
- `BK16 32x64`: `17.15 -> 20.99 TFLOPS`, `8.01 -> 6.55 ms`

In standalone benchmark terms, `BK16 32x64` improves by `+22.39%` over `v3`, and even relative to `v2 BK16` it rises from `19.13 -> 20.99 TFLOPS` (`+9.70%`). `v4` does more than recover the BK16 regression from `v3`. It creates the current best overall point.

### 4096 Key NCU Metrics

Throughput / scheduler:

| Kernel | Duration (ms) | TFLOPS | Compute Throughput | Memory Throughput | Registers / thread | Achieved Occupancy | Eligible Warps / Scheduler | Issued Warps / Scheduler | No Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v4<128,128,8,32,64,8>` | 6.77 | 19.63 | 74.48% | 48.88% | 128 | 32.16% | 2.40 | 0.79 | 21.32% |
| `v4<128,128,8,64,32,8>` | 6.85 | 19.39 | 73.50% | 48.49% | 128 | 32.19% | 2.38 | 0.78 | 22.14% |
| `v4<128,128,16,32,64,16>` | 6.53 | 20.34 | 77.70% | 51.22% | 128 | 32.27% | 2.70 | 0.83 | 17.24% |

Memory path:

| Kernel | DRAM BW | L2 BW | L1/TEX Hit Rate | Global Store Avg Bytes / Sector | Shared Bank Conflict | L2 Theoretical Sectors Global Excessive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v4<128,128,8,32,64,8>` | 141.74 GB/s | 633.80 GB/s | 2.38% | 8 | 0 | 6.29 MB |
| `v4<128,128,8,64,32,8>` | 139.73 GB/s | 626.78 GB/s | 4.51% | 8 | 0 | 6.29 MB |
| `v4<128,128,16,32,64,16>` | 147.43 GB/s | 657.49 GB/s | 2.50% | 8 | 0 | 6.29 MB |

From a stall perspective, `v4 BK16` is especially striking.

- `v4 BK8 32x64`: `Not Selected 2.05`, `Barrier 0.80`, `Dispatch Stall 0.44`, `Short Scoreboard 0.20`, `Long Scoreboard 0.12`
- `v4 BK8 64x32`: `Not Selected 2.05`, `Barrier 0.81`, `Dispatch Stall 0.44`, `Short Scoreboard 0.20`, `Long Scoreboard 0.12`
- `v4 BK16 32x64`: `Not Selected 2.27`, `Barrier 0.52`, `Dispatch Stall 0.43`, `Short Scoreboard 0.16`, `Long Scoreboard 0.01`

So BK16 is not just getting more bandwidth. The long-scoreboard problem and scheduler starvation that hurt `v3 BK16` have also been largely cleaned up.

### `v3 -> v4` Interpretation

First, `v4` overturns the simplistic reading that "`BK16 + prefetch` does not work."

The representative NCU comparison for BK16 is:

- TFLOPS: `16.73 -> 20.34`
- Duration: `9.00 -> 6.53 ms`
- Registers / thread: `152 -> 128`
- Achieved Occupancy: `16.67 -> 32.27%`
- Eligible Warps / Scheduler: `1.16 -> 2.70`
- Issued Warps / Scheduler: `0.63 -> 0.83`
- No Eligible: `36.92 -> 17.24%`
- DRAM BW: `110.87 -> 147.43 GB/s`
- L2 BW: `477.56 -> 657.49 GB/s`
- `Stall Long Scoreboard`: `0.45 -> 0.01`

The core of `v4` is that BK16 comes back down to around `128 regs/thread`, which restores the `2 blocks/SM` tier. That makes the `v3 BK16` failure look less like a fundamental limit of `BK16 + prefetch` and more like a specific codegen/register-allocation failure that crossed the register cliff.

Second, the BK8 gain should still be read primarily as a load-side ingest improvement, not an epilogue improvement.

Even after `v3 -> v4`, the following are still unchanged.

- global-store average utilization: still `8 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: still `6.29 MB`
- epilogue: still scalar
- shared bank conflict: still `0`

And yet both BK8 variants now exceed `20 TFLOPS`.

- `BK8 32x64`: `18.72 -> 20.22 TFLOPS`
- `BK8 64x32`: `18.84 -> 20.04 TFLOPS`

So the BK8 gain in `v4` is much more convincingly attributed to a more efficient global-load instruction/transaction path than to the final `C` store.

Third, the BK8 gain is better described as "lower ingest cost in the same occupancy tier" than as "every stall was improved."

For `BK8 32x64`:

- Compute Throughput: `71.85 -> 74.48%`
- DRAM BW: `129.20 -> 141.74 GB/s`
- L2 BW: `575.68 -> 633.80 GB/s`
- Issued Warps / Scheduler: `0.77 -> 0.79`
- No Eligible: `23.49 -> 21.32%`

But the stall mix is not a simple across-the-board cleanup. That is why the safest reading is that vec4 load helpers reduced load-side ingest cost within the same register/occupancy regime.

Fourth, from the shared-memory perspective, `v4` shows that "vectorize everything" is less important than "vectorize where the layout allows it."

- `A` still needs scalar scatter because of the transposed shared layout.
- `B` can keep vec4 stores because its shared layout is contiguous.
- Even with that asymmetry, shared bank conflicts remain `0`.

So the practical gain from vec4 comes first from global ingest and naturally contiguous paths, not from forcing vectorization across every store.

Auxiliary experiment note: the table below comes from a separate rerun, not the current mainline benchmark path.

### If `store_accum_to_gmem_vec4()` is enabled

If the final epilogue in `v4` is switched from `v1::store_accum_to_gmem()` to `store_accum_to_gmem_vec4()`, the result is: the store path gets cleaner, but the kernel as a whole gets slower.

4096 summary:

| Kernel | current `v4` scalar epilogue | vec4 epilogue | Change |
| --- | ---: | ---: | ---: |
| `BK8 32x64` | `20.22 TFLOPS` | `19.46 TFLOPS` | `-3.78%` |
| `BK8 64x32` | `20.04 TFLOPS` | `19.74 TFLOPS` | `-1.48%` |
| `BK16 32x64` | `20.99 TFLOPS` | `19.91 TFLOPS` | `-5.11%` |

4096 profile comparison:

| Kernel | Compute Throughput | Registers / thread | Achieved Occupancy | Eligible / Issued | DRAM BW | Store-path change |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `BK8 32x64` | `74.48 -> 72.79%` | `128 -> 121` | `32.16 -> 32.36%` | `2.40 / 0.79 -> 2.21 / 0.78` | `141.74 -> 139.82 GB/s` | global store `8 -> 32 byte/sector`, excessive `6.29 MB -> 0` |
| `BK8 64x32` | `73.50 -> 72.80%` | `128 -> 121` | `32.19 -> 32.39%` | `2.38 / 0.78 -> 2.22 / 0.78` | `139.73 -> 139.99 GB/s` | global store `8 -> 32 byte/sector`, excessive `6.29 MB -> 0` |
| `BK16 32x64` | `77.70 -> 70.36%` | `128 -> 147` | `32.27 -> 16.64%` | `2.70 / 0.83 -> 1.39 / 0.75` | `147.43 -> 138.15 GB/s` | global store `8 -> 32 byte/sector`, excessive `6.29 MB -> 0`, shared bank conflict `0 -> 864,274` |

So the vec4 epilogue itself does succeed at the narrow store-path goal. But because CUDA fully inlines the kernel and performs register allocation and instruction scheduling globally, the extra `float4` packing and address-generation temporaries perturb the overall codegen. BK8 loses a little while staying in the same occupancy tier. BK16 gets pushed back across the `128 regs/thread` boundary and pays a much larger price. That is why the best current `v4` point keeps vec4 global ingest while leaving the final `C` store scalar.

## 16. `sgemm_warptiling_v5<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v5<128, 128, 8, 64, 32, 8>` vs `sgemm_warptiling_v5<128, 128, 16, 32, 64, 16>`

### Structure

`v5` in [`06_warptiling.cuh`](./06_warptiling.cuh) keeps the `v4` main loop and vec4 ingest path, and replaces the epilogue with a CUTLASS-style shared-memory staging structure.

- The main loop still uses `v4::load_*_thread_fragment_vec4()` and `v4::store_*_thread_fragment_to_smem_vec4()`.
- The difference is entirely in the writeback. Instead of storing each thread's accumulators directly to global memory with scalar stores, the kernel first rearranges them into a CTA-wide epilogue shared-memory tile with `store_accum_to_epilogue_smem()`.
- After `__syncthreads()`, `store_epilogue_smem_to_gmem()` writes the whole CTA tile out with cooperative `float4` global stores.
- `KernelTraits` adds `kEpilogueSmemStride = BLOCK_N + 4`, `kEpilogueSmemNumElems = BLOCK_M * kEpilogueSmemStride`, and `kV5SmemBytes = max(kTotalSmemElems, kEpilogueSmemNumElems)`.

So the intent of `v5` is very clear: fix the poor store-sector utilization of the residual scalar epilogue in `v4` by using shared-memory reordering.

> `v5` is the first mainline attempt that directly targets the residual scalar epilogue in `v4`, but it also introduces a new cost: CTA-wide epilogue scratch space.

The default `v5` uses the first implementation of `store_epilogue_smem_to_gmem()`, which recomputes `vec_idx`, `row`, and `vec_col` inside the loop. The version that switches to `smem_offset/gmem_offset += advance` is treated separately as an additional experiment.

### Benchmark Results

Elapsed time:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v5<128,128,8,32,64,8>` | 0.022032 | 0.034767 | 0.064054 | 0.128296 | 1.063786 | 7.496521 |
| `warptiling_v5<128,128,8,64,32,8>` | 0.023729 | 0.041022 | 0.074049 | 0.132619 | 1.064461 | 7.514165 |
| `warptiling_v5<128,128,16,32,64,16>` | 0.022692 | 0.038729 | 0.065036 | 0.125585 | 0.983340 | 6.913477 |

TFLOPS:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v5<128,128,8,32,64,8>` | 0.188883 | 0.961343 | 4.182578 | 16.722162 | 16.141862 | 18.329222 |
| `warptiling_v5<128,128,8,64,32,8>` | 0.175376 | 0.814759 | 3.618043 | 16.177043 | 16.131618 | 18.286185 |
| `warptiling_v5<128,128,16,32,64,16>` | 0.183393 | 0.863007 | 4.119419 | 17.083148 | 17.462396 | 19.875007 |

At 4096, all three `v5` variants fail to beat `v4`.

- `BK8 32x64`: `20.22 -> 18.33 TFLOPS`
- `BK8 64x32`: `20.04 -> 18.29 TFLOPS`
- `BK16 32x64`: `20.99 -> 19.88 TFLOPS`

So the CUTLASS-style epilogue has a clear purpose, but in mainline results the residency loss is larger than the store-path gain.

### 4096 Key NCU Metrics

Throughput / residency:

| Kernel | Duration (ms) | Compute Throughput | Memory Throughput | Registers / thread | Dynamic Smem / block | Theoretical / Achieved Occupancy | Eligible / Issued | No Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v5<128,128,8,32,64,8>` | 7.90 | 63.94% | 41.80% | 126 | 67.58 KB | `16.67% / 16.66%` | `1.19 / 0.68` | 31.81% |
| `v5<128,128,8,64,32,8>` | 7.89 | 64.01% | 41.85% | 126 | 67.58 KB | `16.67% / 16.64%` | `1.20 / 0.68` | 31.62% |
| `v5<128,128,16,32,64,16>` | 7.28 | 69.84% | 45.94% | 139 | 67.58 KB | `16.67% / 16.66%` | `1.35 / 0.74` | 25.58% |

Store path / residency:

| Kernel | Global Store Avg Bytes / Sector | L2 Theoretical Sectors Global Excessive | L1 Wavefronts Shared Excessive | Block Limit Shared Mem | Block Limit Registers |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v5<128,128,8,32,64,8>` | 32 | 0 | 0 | 1 | 2 |
| `v5<128,128,8,64,32,8>` | 32 | 0 | 0 | 1 | 2 |
| `v5<128,128,16,32,64,16>` | 32 | 0 | 0 | 1 | 1 |

The first thing to read from this table is that the store path and residency move to opposite extremes at the same time.

- In `v4`, global-store average utilization is `8 byte / sector` and `L2 Theoretical Sectors Global Excessive = 6.29 MB`.
- In `v5`, those become `32 byte / sector` and `0`.
- But dynamic shared memory jumps from `8.32 KB` in `v4 BK8` and `16.51 KB` in `v4 BK16` to `67.58 KB` in all three `v5` variants.
- As a result, `Block Limit Shared Mem = 1`, theoretical occupancy falls to `16.67%`, and eligible/issued warp counts drop sharply relative to `v4`.

### `v4 -> v5` Interpretation

First, `v5` confirms that scalar epilogue inefficiency was a real problem.

- global-store average utilization: `8 -> 32 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: `6.29 MB -> 0`

So the direction of the CUTLASS-style epilogue itself is correct. `v5` makes the residual store-path inefficiency in `v4` much easier to see.

Second, the cost is simply too high. This time the kernel does not hit a register cliff. It hits a shared-memory cliff.

- `BK8 32x64`: `Eligible Warp 2.40 -> 1.19`, `Issued Warp 0.79 -> 0.68`, `No Eligible 21.32 -> 31.81%`
- `BK8 64x32`: `Eligible Warp 2.38 -> 1.20`, `Issued Warp 0.78 -> 0.68`, `No Eligible 22.14 -> 31.62%`
- `BK16 32x64`: `Eligible Warp 2.70 -> 1.35`, `Issued Warp 0.83 -> 0.74`, `No Eligible 17.24 -> 25.58%`

`v5` does not really damage the shared path itself. `L1 Wavefronts Shared Excessive = 0`, and the shared-conflict signal is too small to be the main story. But once the epilogue scratch reaches `67.58 KB` per CTA, residency is effectively cut in half and the scheduler simply has a much smaller warp pool to choose from. The latency-hiding loss is bigger than the gain from the cleaner store path.

Third, the fact that `Stall Not Selected` drops relative to `v4` should not be read as a clear improvement. In an almost `1 block/SM` environment, the scheduler has fewer warps to choose from in the first place. In this stage, occupancy, eligible warps, issued warps, and `No Eligible` are much more explanatory metrics.

> `8 byte/sector -> 32 byte/sector` is a real improvement, but a `67.58 KB` CTA epilogue scratch buffer is too expensive for the current `128x128` family.

So `v5` reproduces, in a cleaner form, the earlier observation that fixing the store path really does improve sector efficiency. But this time performance is blocked not by a register cliff, but by a shared-memory cliff.

### Additional experiment: changing the epilogue address-generation pattern

The default `v5` computes `vec_idx / kEpilogueVectorsPerRow` and `vec_idx % kEpilogueVectorsPerRow` inside the loop to determine the row and column position. The additional experiment precomputes `smem_offset`, `gmem_offset`, `smem_advance`, and `gmem_advance` outside the loop, and then only performs `offset += advance` inside the loop.

Elapsed time:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v5<128,128,8,32,64,8>` | 0.021357 | 0.032987 | 0.060800 | 0.124343 | 1.042751 | 10.081835 |
| `warptiling_v5<128,128,8,64,32,8>` | 0.020717 | 0.035596 | 0.062220 | 0.125772 | 1.043407 | 10.437112 |
| `warptiling_v5<128,128,16,32,64,16>` | 0.020347 | 0.033262 | 0.059686 | 0.120263 | 0.995882 | 6.839624 |

TFLOPS:

| Kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v5<128,128,8,32,64,8>` | 0.194858 | 1.013242 | 4.406457 | 17.253772 | 16.467474 | 13.629008 |
| `warptiling_v5<128,128,8,64,32,8>` | 0.200871 | 0.938968 | 4.305858 | 17.057765 | 16.457121 | 13.165079 |
| `warptiling_v5<128,128,16,32,64,16>` | 0.204526 | 1.004858 | 4.488695 | 17.839117 | 17.242492 | 20.089612 |

4096 profile comparison:

| Kernel | Duration (ms) | Compute Throughput | Registers / thread | Eligible / Issued | No Eligible | Dominant stall change |
| --- | --- | --- | --- | --- | --- | --- |
| `BK8 32x64` | `7.90 -> 10.55` | `63.94 -> 49.36%` | `126 -> 121` | `1.19 / 0.68 -> 0.87 / 0.52` | `31.81 -> 47.78%` | `Barrier 0.38 -> 1.23`, `Long Scoreboard 0.07 -> 0.56` |
| `BK8 64x32` | `7.89 -> 9.97` | `64.01 -> 52.96%` | `126 -> 121` | `1.20 / 0.68 -> 0.92 / 0.55` | `31.62 -> 44.95%` | `Barrier 0.37 -> 1.24`, `Long Scoreboard 0.07 -> 0.49` |
| `BK16 32x64` | `7.28 -> 7.27` | `69.84 -> 69.46%` | `139 -> 147` | `1.35 / 0.74 -> 1.39 / 0.75` | `25.58 -> 25.01%` | `Barrier 0.18 -> 0.19`, `Long Scoreboard 0.07 -> 0.10` |

The crucial point is that the larger store-path and residency picture does not change at all.

- global-store average utilization: still `32 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: still `0`
- dynamic shared memory: still `67.58 KB`
- `Block Limit Shared Mem`: still `1`

So what changes here is not the memory access pattern itself. It is the codegen and scheduling behavior induced by a different form of address generation in the epilogue loop.

The two BK8 variants collapse badly.

- benchmark: `18.33 -> 13.63 TFLOPS`, `18.29 -> 13.17 TFLOPS`
- profile: `Eligible Warp`, `Issued Warp`, and Compute Throughput all drop
- `Barrier` and `Long Scoreboard` rise sharply

BK16 improves slightly.

- benchmark: `19.88 -> 20.09 TFLOPS`
- profile: `Eligible Warp 1.35 -> 1.39`, `Issued Warp 0.74 -> 0.75`
- but it still does not beat `v4 BK16` at `20.99 TFLOPS`

So the right interpretation is not "`/` and `%` disappeared, therefore it should be faster." It is that expressing the same address sequence as a running offset introduced a dependency structure or codegen pattern that hurt BK8 much more than it helped. Even though registers drop from `126 -> 121` for BK8, shared memory already forces the kernel into `1 block/SM`, so that small register reduction cannot restore residency.

> It mattered less that `%` and `/` disappeared inside the loop than how the same address sequence was expressed as a dependency chain.

## Key Takeaways

### 1. There are two major structural performance jumps

- `sgemm_smem_tiling -> blocktiling_1d<64,64,8,8,4>`
  - `2.93 -> 6.23 TFLOPS`
  - the effect of extending shared reuse into register reuse
- `blocktiling_1d<128,128,16,64,4> -> blocktiling_2d<128,128,8,8,8,4>`
  - `11.99 -> 16.05 TFLOPS`
  - the effect of changing 1D accumulation into a 2D outer product

In this dataset, changes to register/blocking structure matter much more than shared tiling alone.

### 2. Occupancy is an intermediate metric, not the goal

Faster kernels do not necessarily have higher occupancy.

- naive/smem stages: around `66%`
- fast `128x128` family: around `32%`

And yet the latter is much faster. That shows that better reuse and ILP can more than offset lower occupancy.

More importantly, the low occupancy in the current warptiling family is largely an unavoidable tradeoff.

- `v2` variants live in the `119~127 regs/thread` range
- the `8x8` output tile alone already needs `64` float accumulators per thread
- operand fragments, address generation, and staging temporaries naturally push register pressure higher

So the current `~32%` occupancy is not primarily an implementation mistake. It is the cost of buying reuse and ILP. The current best kernel, `warptiling_v4<128,128,16,32,64,16>`, also runs at `128 regs/thread` and `32.27%` occupancy.

`v3` adds one important qualification. Low occupancy by itself is not the problem. The real danger is losing an entire occupancy tier when registers increase.

- `v2 BK16`: `127 regs/thread`, `32.23%` occupancy, `19.13 TFLOPS`
- `v3 BK8`: `127~128 regs/thread`, `~32%` occupancy, `18.72~18.84 TFLOPS`
- `v3 BK16`: `152 regs/thread`, `16.67%` occupancy, `17.15 TFLOPS`

The same principle appears again in `v5`, but through shared memory instead of registers.

- `v4 BK8`: `8.32 KB` dynamic shared memory
- `v4 BK16`: `16.51 KB`
- `v5`: `67.58 KB` for all three variants
- result: `Block Limit Shared Mem = 1`, theoretical occupancy `16.67%`

So the key is not low occupancy itself. The key is dropping to the next lower residency tier, whether because of registers or shared memory.

### 3. The bottleneck moves stage by stage

The bottleneck moves through the flow as follows.

1. naive: `LG Throttle`
2. smem: `MIO Throttle + Barrier + shared-store conflict`
3. 1D block tiling: `scoreboard dependency`
4. after 2D block tiling: `shared wavefront excess`
5. refined warptiling (`v1`): `Not Selected + residual scoreboard + barrier`
6. explicit register staging (`v2`): `Not Selected + residual long scoreboard + epilogue store inefficiency`
7. 1-stage register prefetch (`v3`): less scoreboard in BK8, but a BK16 register cliff and a surge in `No Eligible`
8. vec4 global ingest (`v4`): BK16 recovers, but the remaining bottlenecks are `Not Selected` and residual scalar epilogue inefficiency
9. CUTLASS-style epilogue staging (`v5`): store path is cleaned up, but eligible warp count and issue rate fall again because of a shared-memory cliff

Optimization is therefore best understood as repeatedly removing one bottleneck and exposing the next.

### 4. Vec4 should be used on load-side ingest before trying to vectorize everything

`v4` makes the effective scope of vec4 more concrete.

- global `A/B` loads become vec4
- transposed shared stores for `A` stay scalar scatter
- only the contiguous shared stores for `B` remain vec4
- the final `C` epilogue is still scalar

And even so, `v4 BK16` reaches `20.99 TFLOPS`. Meanwhile the profile still shows `8 byte / sector` and `6.29 MB` excessive sectors for global stores.

The lesson is simple: the biggest gain from vec4 comes first from the global-ingest side. You do not need to force every final store or transposed scatter into vector form all at once to get a large speedup.

### 5. The first success or failure of warptiling is determined jointly by warp-tile shape and lane mapping

The two `warptiling_v0` variants show this directly.

- both use an `8x8` output tile per thread
- both use `64` accumulators
- both use `125 regs/thread`

And yet `32x64` leaves large shared excessive wavefronts and bank conflicts, while `64x32` removes them in the same structure. So the first make-or-break condition in warptiling is not "did we add warps," but "did we choose a warp tile shape and lane mapping that fit the shared/global path?"

### 6. Refined warptiling reduces shape sensitivity and pushes the bottleneck upward into scheduler/scoreboard behavior

`warptiling_v1` translates that lesson into a real fix.

- `32x64`: `16.16 -> 17.28 TFLOPS`
- `64x32`: `16.86 -> 17.37 TFLOPS`
- both variants: shared bank conflict `0`, excessive wavefront `0`
- global-store average utilization: `4 -> 8 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: `14.68 -> 6.29 MB`

So the effect of `v1` is not just that it gets a bit faster. It cleans up shared-memory consumption and the epilogue store together, almost eliminates warp-shape sensitivity, and moves the bottleneck to higher-level scheduling and pipeline behavior.

### 7. Explicit register-staged global->shared is about load scheduling and loop overhead, not about fixing a shared-memory bottleneck

`warptiling_v2` keeps the lane mapping and warp MMA from `v1`, and decomposes the CTA tile ingest path into source-level register staging.

- `32x64 BK8`: `17.28 -> 18.25 TFLOPS`
- `64x32 BK8`: `17.37 -> 17.86 TFLOPS`
- `32x64 BK16`: `19.13 TFLOPS`, the new best point at the time

What matters is what did not change.

- shared bank conflict: still `0`
- shared excessive wavefront: still `0`
- global-store average utilization: still `8 byte / sector`
- `smsp__inst_executed_op_ldgsts.sum`: still `0`

So the gain in `v2` is not from a new hardware path such as async copy. It comes from making global-load and shared-store scheduling more explicit and then reducing barrier/loop overhead with `BLOCK_K=16`.

### 8. 1-stage register prefetch only works strongly while the kernel stays in the same occupancy tier

`warptiling_v3` moves the next-tile global load one iteration earlier on top of the `v2` register staging.

- `BK8 32x64`: `18.25 -> 18.72 TFLOPS`
- `BK8 64x32`: `17.86 -> 18.84 TFLOPS`
- `BK16 32x64`: `19.13 -> 17.15 TFLOPS`

The difference is not the prefetch idea itself. It is whether the larger live range introduced by that prefetch still fits inside the current kernel shape.

- BK8 stays at `127~128 regs/thread` and remains in the `2 blocks/SM` tier.
- BK16 rises to `152 regs/thread` and falls to `1 block/SM`.

So register prefetch really can reduce scoreboard pressure, but if it destroys the occupancy tier, the gain disappears immediately.

`v4` adds one important correction to that lesson. The more precise statement is not "`BK16 + prefetch` does not work." It is "`prefetch that pushes BK16 beyond the `128 regs/thread` boundary does not work."

### 9. `v4` restores BK16 from the register cliff and creates the new overall best point

`warptiling_v4` does not discard the partial-prefetch skeleton from `v3`. Instead, it keeps that skeleton and rewrites the global-ingest helpers around vec4, which improves both codegen and register pressure.

- `v4 BK8 32x64`: `20.22 TFLOPS`
- `v4 BK8 64x32`: `20.04 TFLOPS`
- `v4 BK16 32x64`: `20.99 TFLOPS`, the current best

The decisive BK16 change is:

- Registers / thread: `152 -> 128`
- Achieved Occupancy: `16.67 -> 32.27%`
- Eligible Warps / Scheduler: `1.16 -> 2.70`
- `Stall Long Scoreboard`: `0.45 -> 0.01`

So `v4` confirms that the deeper limit was the register cliff, not prefetch in the abstract. The current best combination is still closest to "refined warptiling + explicit register staging/prefetch + vec4 global ingest + BK16."

### 10. A CUTLASS-style epilogue fixes the store path, but CTA-wide staging is not free

`v5` is the first mainline version that directly targets the residual scalar epilogue inefficiency left in `v4`.

- global-store average utilization: `8 -> 32 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: `6.29 MB -> 0`

So the store-path goal is genuinely achieved. But the same design also pushes CTA-wide epilogue scratch up to `67.58 KB`, making `Block Limit Shared Mem = 1` and theoretical occupancy `16.67%`.

- `BK8 32x64`: `20.22 -> 18.33 TFLOPS`
- `BK8 64x32`: `20.04 -> 18.29 TFLOPS`
- `BK16 32x64`: `20.99 -> 19.88 TFLOPS`

This time the limit is not a register cliff. It is a shared-memory cliff. The additional address-generation experiment also shows that under the same store path and the same shared-memory footprint, BK8 can still collapse into the `13 TFLOPS` range if the dependency form and codegen of the epilogue loop are unfavorable.

So the direction of the CUTLASS-style epilogue is correct, but in the current `128x128` family it is not enough to transplant one helper. Epilogue design, residency, and pipeline structure all have to stay balanced together.

## Comparison with `torch.matmul` and CUTLASS

The current best local point is `warptiling_v4<128,128,16,32,64,16>` at `20.99 TFLOPS`. In a standalone 4096 benchmark, the relationship to library kernels looks like this.

| Kernel | 4096 TFLOPS | Relative to `v4 BK16` |
| --- | ---: | ---: |
| `warptiling_v4<128,128,16,32,64,16>` | 20.99 | baseline |
| `cutlass_simt_128x128x8_32x64x8_2stage` | 21.02 | `+0.15%` |
| `cutlass_simt_128x128x8_64x32x8_2stage` | 21.05 | `+0.29%` |
| `cutlass_universal_simt_128x256_8x4` | 21.64 | `+3.10%` |
| `torch.matmul` | 22.72 | `+8.27%` |

The most important point is what kind of kernels these are.

- `torch.matmul` calls `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` in representative Nsight Compute profiling. So this fp32 baseline is still a library-grade SIMT kernel, not a tensor-core kernel.
- The closest like-for-like comparison is the pair of `cutlass_simt_128x128x8_*_2stage` kernels. They are only slightly ahead of our `v4 BK16` within the same `128x128x8` family.
- `cutlass_universal_simt_128x256_8x4` and `torch.matmul` use larger CTA tiles such as `128x256` and `256x128`, which gives them a higher ceiling on larger sizes like `2048` and `4096`.

Representative Nsight Compute numbers look like this.

| Kernel | Compute Throughput | Registers / thread | Achieved Occupancy | Eligible / Issued | DRAM BW | Global Store Avg Bytes / Sector | L2 Excessive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `torch.matmul` | 76.76% | 202 | 16.67% | `1.59 / 0.88` | 69.74 GB/s | 32 | 0 |
| `cutlass_simt_128x128x8_32x64x8_2stage` | 82.52% | 122 | 32.41% | `2.47 / 0.88` | 155.01 GB/s | 32 | 0 |
| `cutlass_simt_128x128x8_64x32x8_2stage` | 82.46% | 122 | 32.40% | `2.47 / 0.88` | 141.13 GB/s | 32 | 0 |
| `cutlass_universal_simt_128x256_8x4` | 76.20% | 254 | 16.67% | `1.59 / 0.87` | 67.06 GB/s | 32 | 0 |
| `warptiling_v4<128,128,16,32,64,16>` | 77.70% | 128 | 32.27% | `2.70 / 0.83` | 147.43 GB/s | 8 | 6.29 MB |

There are six important takeaways from this comparison.

- First, the closest immediate target is CUTLASS SIMT 2-stage. The benchmark gap within the same `128x128x8` family is only `0.15~0.29%`, yet CUTLASS already reaches `Issued Warp / Scheduler = 0.88`, `Global Store Avg Bytes / Sector = 32`, and `L2 Excessive = 0`. The remaining gap is therefore less about a new grand principle and more about true 2-stage overlap and a better epilogue/store path.
- Second, the epilogue/store path is now the clearest first-order gap. The reference kernels all achieve `32 byte/sector` and `L2 Theoretical Sectors Global Excessive = 0`, while `v4 BK16` still sits at `8 byte/sector` and `6.29 MB`.
- Third, occupancy is still a constraint, not the goal. `torch.matmul` and `cutlass_universal` run at `16.67%` occupancy with `202/254 regs/thread` and still benchmark higher. That means a larger CTA tile and better schedule/pipeline may matter more than simply increasing occupancy.
- Fourth, that does not mean the next step is to jump blindly to `128x256` or `256x128`. `cutlass_universal` is much slower than `v4 BK16` at `1024` and only shows its ceiling at `2048/4096`. So the natural sequence is to first match CUTLASS SIMT 2-stage within the current `128x128` family, then explore larger CTA tiles.
- Fifth, shared bank conflict is no longer the first priority. Some reference kernels are still fast even when shared bank-conflict counters are not zero. Shared layout is now more of a diagnostic knob than the first item in the queue.
- Sixth, `v5` shows why copying a CUTLASS-style epilogue alone is not enough. It does achieve `32 byte/sector` and `L2 Excessive = 0`, but the `67.58 KB` epilogue scratch forces `1 block/SM` and prevents it from beating `v4`. The real difference from CUTLASS is not whether a particular helper exists, but how the whole epilogue, pipeline, and residency balance fit together.

## Suggested Next Steps

Relative to that comparison, the overall direction is correct, but the priorities are better arranged like this: first close the small gap to CUTLASS SIMT 2-stage, then move into the larger CTA-tile territory represented by `torch.matmul` and `cutlass_universal`.

1. 2-stage software pipeline / double buffering
2. Additional cleanup of global store / epilogue
3. CTA tile shape / swizzle exploration
4. Register / shared-memory budget control
5. Deeper shared-memory layout refinement

### 1. 2-stage software pipeline / double buffering

The most direct reference is `cutlass_simt_128x128x8_*_2stage`. It is already above `21 TFLOPS` in the same `128x128x8` family, with `Issued Warp / Scheduler` around `0.88`. So the first structural gap to close is a true 2-stage overlap.

- keep two shared-memory stages and ping-pong between them
- avoid holding the next tile in registers longer than needed
- overlap `global -> shared(next) -> compute(curr)` inside the steady-state loop

The real target here is not an abstract "deeper prefetch." It is to raise issue rate and latency hiding the way CUTLASS 2-stage does.

### 2. Further cleanup of the global store / epilogue

The reference kernels all already achieve `Global Store Avg Bytes / Sector = 32` and `L2 Excessive = 0`. By contrast, `v4 BK16` still has `8 byte/sector` and `6.29 MB` excessive sectors. So the epilogue is no longer a nice-to-have. It is a direct lever for closing the remaining gap to CUTLASS SIMT.

`v5` shows both the opportunity and the constraint.

- `32 byte/sector` and `L2 Excessive = 0` are achievable
- but full CTA epilogue staging costs `67.58 KB` and falls to `1 block/SM`

The next epilogue iteration should therefore:

- make output lane mapping more store-friendly
- introduce vectorized output stores only in ways that do not perturb the whole kernel the way `store_accum_to_gmem_vec4()` or `v5` did
- consider rearranging the accumulator layout itself if that helps the epilogue
- prioritize warp-local staging, smaller epilogue tiles, or cooperative strip-mined stores that do not explode shared-memory footprint

The core constraint is simple: the store path should move toward CUTLASS, but without crossing `129+ regs/thread` again in BK16 or creating another `67 KB` shared-memory cliff like `v5`.

### 3. CTA tile shape / swizzle exploration

`torch.matmul` uses `256x128_8x4`, and `cutlass_universal` uses `128x256_8x4`. Both show a higher ceiling than the current `v4` at `2048/4096`. So the next higher-level step is to explore larger CTA shapes, not only to keep optimizing within `128x128`.

But this is correctly third in priority.

- `cutlass_universal` is much slower than `v4 BK16` at `1024`
- larger CTA tiles are not "always better"
- they mainly raise the ceiling at larger problem sizes

So the natural order is: first clean up 2-stage overlap and the epilogue within the `128x128` family, then compare `256x128`, `128x256`, and swizzled mappings.

### 4. Register / shared-memory budget control

The failure of `v3 BK16` and the recovery in `v4 BK16` show that register budget is still a crucial guardrail. `v5` adds shared-memory budget to that same class of guardrail. But compared with the reference kernels, the real goal is not minimizing register pressure in isolation. It is introducing new structure without breaking residency and issue rate.

- In the `128x128 BK16` family, staying near `128 regs/thread` is a good practical guardrail.
- In the same family, once dynamic shared memory grows into the `67 KB` range, `1 block/SM` becomes a real risk, so epilogue/pipeline scratch must be tracked as well.
- For larger CTA tiles, `16.67%` occupancy can still produce better benchmark results, so occupancy alone should not be treated as failure.
- Instead, look at `Issued Warp / Scheduler`, `No Eligible`, and epilogue store efficiency together.

### 5. Deeper shared-memory layout refinement

In the current data, the shared path has already dropped out of the first-order bottleneck position. Some reference kernels remain faster even with nonzero shared bank-conflict counters. So deeper shared-layout work is not the automatic next step. It should be revisited only if a new pipeline or new CTA shape creates a fresh shared-memory bottleneck.

- respond when new designs bring back bank conflicts or wavefront excess
- revisit `A/B` layout, `4x4` island consumption order, or swizzle only then
- for now, pipeline, epilogue, and CTA shape are higher priorities

## Conclusion

The profiling results summarized here show that the staged approach in [`how_to_optimize_sgemm.md`](./how_to_optimize_sgemm.md) holds up well in real Nsight Compute numbers.

- starting from the naive kernel, the global-load path is the first bottleneck
- shared tiling alone is not enough, and register blocking is what produces the first major jump
- the true second turning point is the 2D outer product
- after that, `BLOCK_K` and vec4 are mainly refinements on top of an already improved structure
- `warptiling v0` shows the potential of a warp hierarchy
- `warptiling v1` proves that lane mapping and `4x4` island store refinement have real impact
- `warptiling v2` shows that explicit register-staged global->shared gives additional gains even at BK8, and creates a new best point when combined with `BLOCK_K=16`
- `warptiling v3` shows that 1-stage register prefetch works well in BK8 but can backfire immediately in BK16 through register pressure
- `warptiling v4` shows that under the same prefetch skeleton, vec4 global ingest and better codegen can pull BK16 back from the register cliff
- `warptiling v5` shows that CUTLASS-style epilogue staging can genuinely clean up the store path, but full CTA epilogue scratch creates a shared-memory cliff in the current `128x128` family and still cannot beat `v4`
- the additional `v5` address-generation experiment shows that the dependency form and codegen/scheduling of the epilogue loop matter more than simply reducing the visible arithmetic in the source

The current best point remains `sgemm_warptiling_v4<128, 128, 16, 32, 64, 16>` at `20.99 TFLOPS`. So the most coherent current mainline combination is still "refined warptiling + explicit register staging/prefetch + vec4 global ingest + BK16."

BK16 is no longer a weak region after `v4`.

- `v2 BK16`: `19.13 TFLOPS`
- `v3 BK16`: `17.15 TFLOPS`
- `v4 BK16`: `20.99 TFLOPS`

So the next jump is not mainly about adding deeper prefetch by itself. It is about keeping the register budget recovered by `v4`, while combining a real 2-stage pipeline with a better epilogue store path. As `v5` shows, the direction of fixing the store path is correct, but the implementation cannot destroy shared-memory residency. Occupancy is still not the goal in itself. It is the result of a tradeoff. The current data shows that the key boundary in that tradeoff is less "whether prefetch exists" and more "whether the kernel crosses the `128 regs/thread` boundary" and whether the CTA shared-memory footprint drops the kernel into the next lower residency tier.
