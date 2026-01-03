# A Practical Guide to Optimizing SGEMM

This document describes a practical order for applying CUDA FP32 SIMT SGEMM optimizations. The goal is not to reproduce a single implementation, but to describe a general optimization path that improves performance step by step by identifying the bottleneck at each stage.

The core principles are simple.

- Good thread mapping alone is not enough.
- The memory path, register reuse, warp hierarchy, pipeline, and epilogue all need to work together.
- Optimization is not about adding complexity. It is about removing the current bottleneck one step at a time.
- An optimization is not automatically beneficial just because you apply it early.

One hardware-oriented mental model makes this order easier to understand. On NVIDIA GPUs, SGEMM performance mostly comes from keeping warp schedulers busy, using the fixed on-chip register/shared-memory budgets efficiently, and minimizing the number of memory transactions each warp generates. In that sense, ILP is not best explained as "every instruction has its own private pipeline." The more useful model is that the scheduler benefits when a warp has multiple independent instructions ready while earlier results are still pending, and this becomes more important once SGEMM lowers occupancy by consuming many registers for accumulators.

The hardware explanations below are aligned with NVIDIA's CUDA C++ Programming Guide, CUDA C++ Best Practices Guide, architecture tuning guides, and CUTLASS GEMM documentation.

## Recommended Order at a Glance

| Step | Optimization Focus | Expected Benefit | What to Check First |
| --- | --- | --- | --- |
| 0 | Fix the measurement setup | Establish a comparable baseline | Correctness, TFLOPS, stall reasons |
| 1 | Shared memory tiling | Greatly reduce global reloads | Is DRAM the main bottleneck? |
| 2 | Register tiling | Increase reuse per thread | Register count and occupancy |
| 3 | 2D outer-product structure | Increase FMA density | Compute work per shared load |
| 4 | Tune `BLOCK_K` | Reduce loop and barrier overhead | Shared footprint, occupancy |
| 5 | Shared layout and padding | Reduce bank conflicts | Access pattern, wavefront excess |
| 6 | Vectorized memory path | Reduce memory instruction count | Alignment, contiguous packets |
| 7 | Warp tiling | Build a scheduler-friendly compute structure | Warp-level work decomposition inside the CTA |
| 8 | Warp microkernel scheduling | Improve operand reuse and dependency structure | Issue efficiency, scoreboard stalls |
| 9 | Register lookahead / prefetch | Hide load latency | Register cliff |
| 10 | Epilogue design | Optimize the output path | Store coalescing, extra barriers |
| 11 | 2-stage software pipeline | Strengthen overlap | Stability of the memory path and warp structure |
| 12 | `cp.async`, multistage, Tensor Core | Next-level optimization | Architecture, complexity, target performance |

## 0. Fix the Measurement Setup First

In SGEMM optimization, measurement comes before implementation. If the evaluation setup is not stable, you cannot tell whether a change is actually an improvement.

- What to do first: choose a fixed set of problem sizes and keep using the same set throughout your measurements. Sizes such as `1024`, `2048`, and `4096` are good tile-friendly defaults, but it is also reasonable to include real workload sizes.
- Metrics to watch together: TFLOPS, `ptxas` register count, occupancy, stall reasons such as `long scoreboard`, shared-memory bank conflicts, and global/shared load-store patterns.
- Practical rule: change one optimization at a time, and always verify both performance and correctness.
- Extra comparison worth making: test both tile-aligned cases and boundary cases. A kernel that looks good on full tiles may expose a completely different bottleneck once edge handling comes into play.

A common mistake is to move on by looking only at TFLOPS. Changes such as preload, padding, or epilogue restructuring often show their downsides first in register pressure or stall reasons before the top-line throughput number makes it obvious.

## 1. Introduce Shared Memory Tiling

The first optimization to apply is shared memory tiling.

- What changes: the CTA cooperatively loads `A` and `B` tiles from global memory into shared memory, and the `K` dimension is processed in `BLOCK_K` chunks using a `load -> sync -> compute` structure.
- Hardware reason: global memory is served in aligned transactions, and scattered warp addresses create extra transactions and wasted bandwidth. Shared memory turns those repeated off-chip fetches into one cooperative fetch plus many on-chip reuses.
- Expected benefit: it greatly reduces repeated DRAM reads of the same `A` rows and `B` columns by different threads.
- When to introduce it: when the baseline is clearly bound by global memory loads, with symptoms such as high `LG Throttle` or low issue rate.
- What to check: global traffic should go down, but shared load/store cost and barrier overhead may show up as the new bottleneck.
- Common failure mode: shared memory is introduced, but each thread still computes too little output. In that case, the bottleneck simply moves from DRAM to the shared-memory path. This stage is the foundation, not the final structure.

> Reference example: [`02_smem_tiling.cuh`](./02_smem_tiling.cuh)

## 2. Introduce Register Tiling

Even with shared-memory tiles, the arithmetic intensity is still too low if each thread computes only one output element. The next step is therefore to keep multiple accumulators per thread in registers.

- What changes: each thread computes a small strip or tile of `C` instead of a single element.
- Hardware reason: registers are the closest on-chip storage available to a thread, so keeping more accumulators in registers raises useful math per load. The tradeoff is that those accumulators consume the finite per-SM register file and can reduce the number of resident warps.
- Expected benefit: values loaded from shared memory can be reused more times, increasing the amount of useful work done per instruction.
- When to introduce it: after shared-memory tiling, if shared loads are still heavy and thread-level reuse is still weak.
- What to check: register count, occupancy, and the increase in FMA work per shared load.
- Common failure mode: making the thread tile too large and driving register pressure up too aggressively. Since later stages will also need room for warp-level hierarchy, it is usually better not to oversize the thread tile early.

> Reference example: [`03_blocktiling_1d.cuh`](./03_blocktiling_1d.cuh)

## 3. Move to a 2D Outer-Product Structure

After 1D register tiling, the next good step is to reorganize the computation into a 2D outer-product structure.

- What changes: treat each thread as owning a `TM x TN` sub-tile of `C`, and update multiple accumulators after loading one `A` operand and one `B` operand inside the inner loop.
- Hardware reason: one loaded `A` value and one loaded `B` value now feed several independent FFMA updates. On recent NVIDIA architectures, arithmetic dependency latency is only a few cycles, so multiple independent accumulator updates give the scheduler more ready work while previous FFMA results are still pending.
- Expected benefit: the same loads feed more FMAs, and register reuse becomes much clearer and more effective.
- When to introduce it: when a 1D strip structure stops improving how much compute you get from each shared-memory load.
- What to check: how many accumulators are updated per shared operand, and whether the load-use pattern becomes more regular.
- Common failure mode: the compute structure improves, but the memory path is still scalar, so the overall performance gain is limited. This stage organizes the compute structure; it does not complete the memory-side optimization.

> Reference example: [`04_blocktiling_2d.cuh`](./04_blocktiling_2d.cuh)

## 4. Tune `BLOCK_K`

Once the reuse structure is in place, tuning `BLOCK_K` becomes worthwhile.

- What changes: keep the CTA tile the same, but try values such as `8` and `16` for `BLOCK_K` to reduce the number of `K`-loop iterations and barriers.
- Hardware reason: a larger `BLOCK_K` amortizes loop-control instructions, iterator updates, and CTA-wide barriers over more math. The cost is a larger shared-memory stage and often more live register state, both of which reduce residency if pushed too far.
- Expected benefit: lower fixed overhead from loop control, iterator updates, and synchronization.
- When to introduce it: after shared-memory tiling and register tiling have already stabilized to some extent.
- What to check: shared-memory footprint, register usage, occupancy, and how the number of mainloop iterations changes.
- Common failure mode: treating larger `BLOCK_K` as automatically better. If reuse is still weak, the benefit will be small while shared-memory and register usage increase enough to hurt occupancy.

`BLOCK_K` is not a magic knob. It tends to pay off only after the earlier structure is already in good shape.

## 5. Tune Shared Memory Layout and Padding

In SGEMM, shared memory is not just a cache. It is also a data-reordering buffer. Layout design becomes especially important once `A` is staged with a transpose.

- What changes: design the shared-memory layouts of `A` and `B` to match the compute access pattern, and add padding to the stage stride when needed.
- Hardware reason: shared memory is banked, so if multiple addresses in a warp land on the same bank, hardware serializes the request into multiple conflict-free parts. Padding works because it changes the stride modulo the bank mapping.
- Expected benefit: even with the same number of shared-memory loads and stores, reducing bank conflicts can significantly improve actual throughput.
- When to introduce it: when shared wavefront excess, bank conflicts, or `MIO Throttle` become prominent.
- What to check: which operand is being transpose-staged, what stride pattern the warp uses when reading and writing shared memory, and whether the padding actually reduces conflicts.
- Common failure mode: adding `+1` or `+4` mechanically without looking at the access pattern. Padding is not a small trick. It changes the iterator math and the layout as a whole.

CUTLASS uses the same idea in its SIMT path by adding skew or padding to the stage layout to reduce bank concentration. The important part is not memorizing a specific number, but designing the stride so that it matches the actual access pattern and avoids bank collisions.

> Reference example: [`03_blocktiling_1d.cuh`](./03_blocktiling_1d.cuh), [`04_blocktiling_2d.cuh`](./04_blocktiling_2d.cuh)

## 6. Introduce a Vectorized Memory Path

Rebuilding the memory path around packetized operations such as `float4` is often one of the most effective steps.

- What changes: convert the parts of the global load/store path, shared load/store path, and epilogue store path that can form contiguous `4-float` packets into vectorized operations.
- Hardware reason: if a warp can cover contiguous aligned packets, the hardware can satisfy the access with fewer memory instructions and fewer transaction fragments. This is why vectorization helps most when the thread map already produces contiguous addresses.
- Expected benefit: the same amount of data moves in fewer memory instructions, and the load/store path becomes cleaner.
- When to introduce it: when reuse and layout are already in decent shape, but the memory instruction count is still too high and contiguous packets can be formed.
- What to check: alignment, thread mapping, edge predication, and whether vectorization is truly feasible in transpose-scatter regions.
- Common failure mode: forcing vectorization into every part of the kernel. In regions where destinations are scattered, a scalar path may still be simpler and faster.

The important point is not that a fast kernel must use per-thread `float4` loads. The real question is whether the overall memory path can create contiguous packets, and whether that benefit outweighs the added register pressure and predication cost.

> Reference example: [`05_blocktiling_2d_vec4.cuh`](./05_blocktiling_2d_vec4.cuh), `sgemm_warptiling_v4` in [`06_warptiling.cuh`](./06_warptiling.cuh)

## 7. Add Warp Tiling and Warp-Aware Decomposition

Once the CTA-level structure is reasonably clean, it is a good idea to reorganize the computation around warps inside the CTA.

- What changes: split the CTA tile into multiple warp tiles, and define explicitly which `A/B` fragments and `C` sub-tiles each warp owns.
- Hardware reason: scheduling and issue happen at warp granularity, not CTA granularity. A decomposition with clear warp ownership gives the schedulers regular ready work and makes shared -> register -> FMA movement easier to organize.
- Expected benefit: because the scheduler issues work at warp granularity, making the compute structure warp-aware creates a more regular shared -> register -> FMA flow.
- When to introduce it: when block tiling is already in place and you want a more structured decomposition of work inside the CTA.
- What to check: whether the role of each warp is clear, and whether the chosen warp tile shape works for both shared-memory access and output stores.
- Common failure mode: assuming warp tiling alone will solve everything. If the memory path or epilogue is still weak, warp decomposition on its own will still hit a ceiling.

> Reference example: `sgemm_warptiling_v0` and `sgemm_warptiling_v1` in [`06_warptiling.cuh`](./06_warptiling.cuh)

## 8. Refine Warp Microkernel Scheduling

After warp tiling, the next step is to organize the FMA order inside each warp.

- What changes: schedule the microkernel so that the `A/B` operands read by each lane are reused in small `2x2` or `4x4` islands.
- Hardware reason: the scheduler does not need vague "register locality"; it needs independent ready instructions. Grouping updates into small outer-product islands shortens dependency chains and lets one loaded operand pay for several nearby FFMA instructions before it is discarded.
- Expected benefit: operand reuse improves, and the dependency structure becomes more regular and easier for the scheduler to handle.
- When to introduce it: when warp tiling is already present but scoreboard stalls or issue efficiency still look poor.
- What to check: how quickly a loaded operand gets reused across multiple accumulator updates, and whether those accumulator updates are grouped into small regular blocks.
- Common failure mode: describing this in vague terms such as "register physical locality." The more useful concepts here are operand reuse, use distance, and dependency structure.

A good microkernel uses an operand immediately across multiple FMAs and extracts as much value as possible from it before moving on to the next operand.

## 9. Register Lookahead Still Matters in a Single-Stage Kernel

This step is often misunderstood. A `single-stage` kernel means there is one shared-memory stage. It does not mean register lookahead is impossible.

- What changes: while computing the current tile, preload part of the next tile into thread-private registers to increase load-use distance.
- Hardware reason: off-chip memory latency is far longer than arithmetic latency. Lookahead helps because it moves the load earlier in time, but the extra fragments also lengthen live ranges and increase pressure on the same register budget that controls occupancy.
- Expected benefit: even if the total traffic stays the same, global loads are pulled earlier off the direct dependency path, making latency easier to hide.
- When to introduce it: when warp tiling and the memory path are already reasonably organized, but you do not yet want to increase the number of shared-memory stages.
- What to check: preload fragment size, register count, occupancy, and changes in stall reasons.
- Common failure mode: adding lookahead when the fragment is already large, so all it does is extend live ranges. In kernels with scalar-heavy fragments or already large accumulator state, it can easily become a net loss.

In practice, lookahead tends to work well when fragments are small and the memory path is already vectorized. If the kernel already carries many accumulators and temporaries, it is much easier to fall off a register cliff instead.

> Reference example: `sgemm_warptiling_v3` and `sgemm_warptiling_v4` in [`06_warptiling.cuh`](./06_warptiling.cuh)

## 10. Design the Epilogue Separately from the Main Compute Structure

How `C` is written out matters just as much as the mainloop. It helps to separate `compute ownership` from `final output ownership`.

- What changes: decide whether to write accumulators directly to global memory or first reorder them through a shared-memory scratch space before storing them.
- Hardware reason: the thread map that is best for FFMA often is not the thread map that gives coalesced global stores. An epilogue scratch buffer is a remapping mechanism, and it only wins if the better store pattern outweighs the extra shared-memory footprint, barriers, and reorder instructions.
- Expected benefit: the output path can be optimized independently of the compute path.
- When to introduce it: when the mainloop is already fast, but the store path, edge predication, or output mapping is becoming a bottleneck.
- What to check: store coalescing, extra shared-memory footprint, extra barriers, and the cost of the reorder pass.
- Common failure mode: copying the epilogue structure of a more general framework as-is. A shared-memory scratch epilogue is useful when you need generality and separate mappings, but it can be slower for a specialized kernel.

One of the main reasons CUTLASS uses a scratch epilogue is that the best thread map for compute is not always the best thread map for final stores. But if each lane already knows its final `(row, col)` coordinates and can write a contiguous segment directly, direct store is often better.

> Reference example: a direct vector store path can be seen in [`05_blocktiling_2d_vec4.cuh`](./05_blocktiling_2d_vec4.cuh)

## 11. Move to a 2-Stage Software Pipeline

Once the memory path and warp hierarchy are reasonably organized, the next major step is a 2-stage software pipeline.

- What changes: keep two shared-memory stages, using one for the current compute work and the other to prepare the next tile. Separate the prologue, steady state, and epilogue explicitly.
- Hardware reason: SGEMM mainloops usually run with limited occupancy because accumulators and fragments consume a large fraction of each thread's registers. Double buffering compensates by overlapping memory movement and math inside the same threadblock instead of depending only on extra resident warps.
- Expected benefit: stronger overlap between global -> shared movement, shared -> register movement, and FFMA execution.
- When to introduce it: when a single-stage kernel with lookahead is already working reasonably well, but memory latency is still a major bottleneck.
- What to check: shared-memory budget, additional register usage, and whether the pipeline actually reaches a stable steady state.
- Common failure mode: trying to add pipelining before the memory path and warp decomposition are clean. In that case, complexity goes up while performance barely moves.

CUTLASS also leans heavily on this kind of pipelined structure in its SIMT mainloop. The key point is not to imitate a specific framework, but to confirm that your kernel is structurally ready to benefit from overlap.

## 12. Next Steps: `cp.async`, Multistage, and Tensor Core

From here on, you are moving beyond the core SIMT SGEMM path. If the earlier stages are not already in good shape, jumping here too early usually increases implementation complexity more than performance.

### `cp.async`

- Key idea: replace part of the software-managed global -> register -> shared path with asynchronous copies to strengthen global -> shared latency hiding.
- Hardware reason: on Ampere-class GPUs, async copy is hardware-accelerated, avoids the intermediate register normally used by a copy, and can optionally bypass L1.
- When to consider it: when the 2-stage software pipeline already works well and the target includes SM80 or newer hardware.

### Multistage

- Key idea: use a deeper pipeline than simple double buffering to hide memory latency more aggressively.
- When to consider it: when `BLOCK_K` is still small, memory latency is still dominant, and the shared-memory and register budgets allow it.

### Tensor Core

- Key idea: instead of continuing to fine-tune a SIMT FFMA kernel, switch the compute path itself to MMA-based execution.
- Why this is a separate step: tile shapes, fragment layouts, alignment requirements, and datatype choices all change. This is not so much the final stage of SIMT tuning as a different class of kernel.

## Summary of the Recommended Order

Do not try to apply every optimization at once. A safer order is:

1. Fix the measurement setup.
2. Reduce global reloads with shared memory tiling.
3. Increase per-thread reuse with register tiling.
4. Raise compute density with a 2D outer-product structure.
5. Tune `BLOCK_K` to reduce loop and barrier overhead.
6. Clean up the shared layout and padding to reduce bank conflicts.
7. Reduce load/store instruction count with a vectorized memory path.
8. Reorganize CTA work around warp-level roles.
9. Refine the dependency structure with a warp microkernel.
10. Hide more load latency with register lookahead.
11. Optimize the epilogue separately.
12. Strengthen overlap with a 2-stage software pipeline.
13. Move on to `cp.async`, multistage, or Tensor Core only if needed.

This order works because each step prepares the ground for the next one. If you increase register tiling before shared-memory tiling is in place, the DRAM bottleneck is still too large. If the memory path is still weak, making the epilogue or pipeline more sophisticated will not improve the overall kernel very much.

## Final Checklist

Every time you change a kernel, it is worth checking the following:

- Is the result still correct?
- Did TFLOPS actually improve?
- How much did the register count change?
- Did occupancy drop too far?
- How did stall reasons such as `long scoreboard`, `MIO Throttle`, or `not selected` change?
- Did shared-memory bank conflicts and wavefront excess go down?
- Does the global/shared load-store path actually match the intended width and alignment?
- Did prefetching make live ranges too long?
- Did the epilogue become more expensive than the mainloop?
- Are both aligned and boundary cases still correct and performant?

The essence of SGEMM optimization is not to write the perfect kernel in one shot. It is to keep checking whether the current bottleneck is global memory, shared memory, register pressure, scheduler behavior, or the epilogue, and then remove those bottlenecks one at a time.
