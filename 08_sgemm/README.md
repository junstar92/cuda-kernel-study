# CUDA Kernel Study: SGEMM Optimization

This document summarizes the SGEMM optimization path in this repository. It focuses on three questions:

1. What changed in code from the previous kernel?
2. Which metrics moved with that change?
3. Why did it get faster, and what is still missing?

Unless noted otherwise, the comparisons below use the current `M=N=K=4096` benchmark and profiling snapshot.

## Current Snapshot

Listed in implementation order, not performance order. The `Key metrics` column uses `Issued / No Eligible / L2 Hit`.

| # | Kernel | 4096 TFLOPS | Key metrics | Key note |
| ---: | --- | ---: | --- | --- |
| 1 | `torch_matmul` | 23.87 | Issued 0.88 / No Eligible 12.26% / L2 91.25% | Best overall reference; `LDGSTS` + `256x128x8`. |
| 2 | `cutlass_simt_128x128x8_32x64x8_2stage` | 22.64 | Issued 0.88 / No Eligible 12.02% / L2 93.11% | Best scalar reference; clean 2-stage. |
| 3 | `cutlass_universal_simt_256x128_8x4` | 22.28 | Issued 0.85 / No Eligible 14.56% / L2 92.92% | Larger-tile reference with more generic plumbing. |
| 4 | `naive_row` | 1.98 | Issued 0.27 / No Eligible 73.08% / L2 49.61% | Row-wise global-reload baseline. |
| 5 | `naive_col` | 0.28 | Issued 0.03 / No Eligible 96.52% / L2 59.04% | Worst access order; strided baseline. |
| 6 | `smem_tiling` | 2.64 | Issued 0.20 / No Eligible 79.72% / L2 49.14% | First CTA shared-memory staging step. |
| 7 | `blocktiling_1d_v0_64x64x8_8` | 8.14 | Issued 0.38 / No Eligible 62.39% / L2 60.65% | 1D register tiling baseline. |
| 8 | `blocktiling_1d_v1_64x64x8_8` | 8.13 | Issued 0.38 / No Eligible 62.06% / L2 61.43% | 1D tiling cleanup variant. |
| 9 | `blocktiling_1d_v2_64x64x8_8` | 8.96 | Issued 0.43 / No Eligible 56.90% / L2 61.05% | Improved `64x64` 1D variant. |
| 10 | `blocktiling_1d_v3_64x64x8_8` | 8.89 | Issued 0.43 / No Eligible 56.89% / L2 61.13% | Best `64x64` 1D variant. |
| 11 | `blocktiling_1d_v2_64x64x8_16` | 10.66 | Issued 0.47 / No Eligible 53.28% / L2 61.29% | Larger per-thread tile. |
| 12 | `blocktiling_1d_v2_128x128x8_64` | 11.91 | Issued 0.49 / No Eligible 51.17% / L2 79.64% | Larger CTA tile. |
| 13 | `blocktiling_1d_v2_128x128x16_64` | 12.16 | Issued 0.50 / No Eligible 49.76% / L2 79.55% | Larger CTA + `K` tile. |
| 14 | `blocktiling_2d_128x128x8_8x8` | 17.48 | Issued 0.62 / No Eligible 38.20% / L2 81.74% | First 2D outer-product kernel. |
| 15 | `blocktiling_2d_128x128x16_8x8` | 19.10 | Issued 0.67 / No Eligible 32.75% / L2 81.53% | Stronger 2D outer-product variant. |
| 16 | `blocktiling_2d_vec4_v0_128x128x8_8x8` | 17.93 | Issued 0.61 / No Eligible 38.65% / L2 80.15% | 2D + vec4 global path. |
| 17 | `blocktiling_2d_vec4_v0_128x128x16_8x8` | 19.96 | Issued 0.69 / No Eligible 31.27% / L2 80.07% | 2D + vec4 with `BK16`. |
| 18 | `blocktiling_2d_vec4_v1_128x128x8_8x8` | 19.05 | Issued 0.65 / No Eligible 35.29% / L2 80.12% | Cleaner vec4 thread map. |
| 19 | `blocktiling_2d_vec4_v1_128x128x16_8x8` | 20.13 | Issued 0.68 / No Eligible 31.79% / L2 80.01% | Best blocktiling-era kernel. |
| 20 | `warptiling_v0_128x128x8_32x64x8` | 17.67 | Issued 0.62 / No Eligible 37.74% / L2 81.59% | First warptiling attempt, `BK8`. |
| 21 | `warptiling_v0_128x128x16_32x64x16` | 19.43 | Issued 0.69 / No Eligible 31.34% / L2 81.50% | First strong warptiling baseline. |
| 22 | `warptiling_v1_128x128x8_32x64x8` | 19.44 | Issued 0.69 / No Eligible 31.35% / L2 80.44% | Lane-remap cleanup. |
| 23 | `warptiling_v2_128x128x8_32x64x8` | 20.70 | Issued 0.74 / No Eligible 26.05% / L2 80.35% | Clean scalar 2-stage overlap. |
| 24 | `warptiling_v2_vec4_128x128x8_32x64x8` | 23.01 | Issued 0.81 / No Eligible 19.40% / L2 79.82% | Clean 2-stage + vec4 ingress. |
| 25 | `warptiling_v3_128x128x8_32x64x8` | 20.96 | Issued 0.75 / No Eligible 24.64% / L2 82.03% | Helper-load cache hint. |
| 26 | `warptiling_v3_vec4_128x128x8_32x64x8` | 23.75 | Issued 0.84 / No Eligible 15.52% / L2 81.77% | Helper-load + vec4 sweet spot. |
| 27 | `warptiling_v4_128x128x8_32x64x8` | 21.66 | Issued 0.78 / No Eligible 21.52% / L2 81.78% | Predicated scalar steady-state. |
| 28 | `warptiling_v4_vec4_128x128x8_32x64x8` | 23.70 | Issued 0.84 / No Eligible 15.53% / L2 81.77% | Predicated vec4 plateau. |
| 29 | `warptiling_v5_128x128x8_32x64x8` | 21.79 | Issued 0.80 / No Eligible 19.69% / L2 81.67% | Deeper scalar pipeline. |
| 30 | `warptiling_v5_vec4_128x128x8_32x64x8` | 23.47 | Issued 0.84 / No Eligible 16.28% / L2 81.82% | Vec4 with deeper pipeline. |
| 31 | `warptiling_v6_128x128x8_32x64x8` | 21.85 | Issued 0.81 / No Eligible 18.50% / L2 73.93% | Deepest scalar overlap. |
| 32 | `warptiling_v6_vec4_128x128x8_32x64x8` | 22.83 | Issued 0.79 / No Eligible 20.59% / L2 80.67% | Too-deep vec4 pipeline. |
| 33 | `warptiling_v7_128x128x8_32x64x8` | 22.01 | Issued 0.82 / No Eligible 17.83% / L2 77.62% | CUTLASS-style epilogue reorder. |
| 34 | `warptiling_v7_vec4_128x128x8_32x64x8` | 23.76 | Issued 0.85 / No Eligible 14.79% / L2 81.35% | Current best study kernel. |
| 35 | `warptiling_v8_128x128x8_32x64x8` | 22.21 | Issued 0.83 / No Eligible 17.24% / L2 81.44% | Explicit `prologue -> steady -> drain`. |
| 36 | `warptiling_v8_vec4_128x128x8_32x64x8` | 23.70 | Issued 0.85 / No Eligible 14.87% / L2 81.40% | Same plateau, slightly behind `v7_vec4`. |
| 37 | `warptiling_v9_128x128x8_32x64x8` | 22.27 | Issued 0.82 / No Eligible 17.70% / L2 81.37% | Retimed next-tile load. |
| 38 | `warptiling_v10_128x128x8_32x64x8` | 22.74 | Issued 0.84 / No Eligible 15.93% / L2 81.38% | Same logic, faster with `cutlass` symbol. |
| 39 | `warptiling_v11_128x128x8_32x64x8` | 22.94 | Issued 0.84 / No Eligible 15.94% / L2 91.07% | `v9` + grouped threadblock swizzle; plain symbol wins. |

The big picture is already clear:

- The vec4 study path has plateaued around `23.7-23.8 TFLOPS`; `v7_vec4` now leads, but the family story is unchanged.
- The scalar path now peaks at `v11`, which beats both CUTLASS references and improves for a different reason than `v10`.
- The remaining gap is mostly about ingress quality, stage-boundary compression, inter-CTA cache residency, and arithmetic density.

## How To Read The Deltas

The most useful metrics in this study are:

- `TFLOPS`: the final result
- `Issued Warp Per Scheduler`: how often the scheduler actually issues work
- `No Eligible`: how often the scheduler cannot find a ready warp
- `L2 Hit` / `L2 Excessive`: whether the memory path is cache-friendly and waste-free
- `Global Store Avg Bytes / Sector`: how clean the output path is
- top stalls: where the machine is actually waiting

In this project, higher occupancy by itself is not the goal. The stronger pattern is:

- better kernels raise `Issued`
- better kernels lower `No Eligible`
- better kernels usually do that by increasing operand reuse, widening memory transactions, or shortening `load -> use` dependency chains
- the best kernels usually also reduce wasted traffic or shorten dependency chains

## 1. Baseline To Shared Tiling

### `naive_row` -> `smem_tiling`

**Code change**

```cpp
smem_a[thread_m][thread_n] = a_ptr[...];
smem_b[thread_m][thread_n] = b_ptr[...];
__syncthreads();

for (int k = 0; k < kBlockSize; ++k) {
  accum += smem_a[thread_m][k] * smem_b[k][thread_n];
}
```

**Metric delta**

| Kernel | TFLOPS | Issued | No Eligible | Top Stall |
| --- | ---: | ---: | ---: | --- |
| `naive_row` | 1.98 | 0.27 | 73.08% | `LG Throttle` |
| `smem_tiling` | 2.67 | 0.20 | 79.74% | `MIO Throttle` |

**Why it got faster**

- The same `A` and `B` values are fetched once per CTA tile and then reused from shared memory instead of being reloaded from DRAM for every output.
- That reduces global-memory traffic per FMA, so the kernel is no longer limited purely by raw DRAM demand.
- The gain is still small because the saved DRAM work is replaced by shared-memory traffic and barrier overhead.

**What is still lacking**

- Each thread still computes only one output.
- The tile is not reused deeply enough to amortize the staging cost.
- The bottleneck moves from global memory to shared-memory pressure and barriers.

This step matters conceptually, but not yet enough numerically.

## 2. Shared Tiling To 1D Register Tiling

### `smem_tiling` -> `blocktiling_1d_v2_128x128x16_64`

**Code change**

```cpp
float accum[ThreadM] = {0.f};

for (int k = 0; k < ThreadblockK; ++k) {
  float elem_b = smem_b[...];
  for (int m = 0; m < ThreadM; ++m) {
    float elem_a = smem_a[...];
    accum[m] += elem_a * elem_b;
  }
}
```

**Metric delta**

| Kernel | TFLOPS | Issued | No Eligible | Regs | Top Stall |
| --- | ---: | ---: | ---: | ---: | --- |
| `smem_tiling` | 2.67 | 0.20 | 79.74% | 38 | `MIO Throttle` |
| `blocktiling_1d_v2_128x128x16_64` | 12.24 | 0.50 | 49.77% | 110 | `MIO Throttle` |

**Why it got faster**

- One shared-memory operand now feeds a strip of register accumulators instead of a single scalar result.
- Each operand fetch buys many more `FFMA`s, so arithmetic intensity rises sharply.
- That is why `Issued` jumps from `0.20` to `0.50`: the scheduler sees much longer compute runs between memory operations.

**What is still lacking**

- Reuse is mostly one-dimensional.
- The kernel still does not form a full outer product.
- Shared-memory and scoreboard pressure remain visible.

This is the first major throughput jump, but it is still not enough.

## 3. 1D Tiling To 2D Outer Product

### `blocktiling_1d_v2_128x128x16_64` -> `blocktiling_2d_128x128x16_8x8`

**Code change**

```cpp
float accum[ThreadM][ThreadN] = {0.f};

for (int k = 0; k < ThreadblockK; ++k) {
  for (int m = 0; m < ThreadM; ++m) {
    float elem_a = smem_a[...];
    for (int n = 0; n < ThreadN; ++n) {
      float elem_b = smem_b[...];
      accum[m][n] += elem_a * elem_b;
    }
  }
}
```

**Metric delta**

| Kernel | TFLOPS | Issued | No Eligible | Shared Excessive | Uncoalesced Global |
| --- | ---: | ---: | ---: | ---: | ---: |
| `blocktiling_1d_v2_128x128x16_64` | 12.24 | 0.50 | 49.77% | 1% | low |
| `blocktiling_2d_128x128x16_8x8` | 19.29 | 0.67 | 32.78% | 40% | 10% |

**Why it got faster**

- Both staged operands now participate in a true outer product, so each `A` or `B` value is reused across far more FMAs.
- The kernel spends a larger fraction of time executing dense `FFMA` groups and a smaller fraction reloading operands.
- That is why `Issued` rises to `0.67` and `No Eligible` falls so much: more math is ready before the next dependency wall appears.

**What is still lacking**

- Shared-memory access is still messy.
- Shared bank conflicts and excessive wavefronts become obvious.
- Global store path is still not clean.

This is the moment the arithmetic side gets strong enough to expose the memory path clearly.

## 4. 2D Outer Product To Vec4 Global Path

### `blocktiling_2d_128x128x16_8x8` -> `blocktiling_2d_vec4_v1_128x128x16_8x8`

**Code change**

```cpp
using ThreadMapB =
    utils::PitchLinearStripminedThreadMap<ThreadblockN, ThreadblockK,
                                          kThreads, 4>;

for (int n = 0; n < ThreadN; n += 4) {
  float4 value = utils::make_float4_from_array(&accum[m][n]);
  utils::store_float4(&c_ptr[...], value);
}
```

**Metric delta**

| Kernel | TFLOPS | Issued | No Eligible | L2 Excessive | Uncoalesced Global |
| --- | ---: | ---: | ---: | ---: | ---: |
| `blocktiling_2d_128x128x16_8x8` | 19.29 | 0.67 | 32.78% | 14.68 MB | 10% |
| `blocktiling_2d_vec4_v1_128x128x16_8x8` | 20.26 | 0.68 | 31.78% | 2.10 MB | 2% |

**Why it got faster**

- Vec4 turns many narrow global transactions into fewer wide ones, which reduces both memory-instruction count and sector waste.
- Better coalescing means more useful bytes per request and much less excessive L2 traffic.
- The math core is unchanged, but the pipeline spends less time doing memory housekeeping, which is enough to lift throughput.

**What is still lacking**

- Shared bank conflicts are still there.
- Shared excessive wavefronts are still high.

This is a strong reminder that global-path cleanup can pay off even when the shared path is not fixed yet.

## 5. Warptiling Starts: Fix The Path Before Going Deeper

### `warptiling_v0_128x128x16_32x64x16` -> `warptiling_v1_128x128x8_32x64x8`

**Code change**

```cpp
int row_major = lane_id / Traits::kLaneStride;
int residual = lane_id - row_major * Traits::kLaneStride;
int lane_n_idx = residual / Traits::kLaneLayout;
int row_minor = residual - lane_n_idx * Traits::kLaneLayout;
int lane_m_idx = row_major * Traits::kLaneLayout + row_minor;
```

**Metric delta**

| Kernel | TFLOPS | Issued | No Eligible | L2 Excessive | Store Bytes / Sector | Shared Excessive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `warptiling_v0_128x128x16_32x64x16` | 19.42 | 0.69 | 31.37% | 14.68 MB | 4 | 38% |
| `warptiling_v1_128x128x8_32x64x8` | 19.54 | 0.69 | 31.39% | 6.29 MB | 8 | 0 |

**Why it got faster**

- The lane remap lines warp accesses up with shared-memory layout and output-store layout, so fewer accesses fight the mapping.
- That removes hidden structural waste: shared excessive wavefronts disappear and stores become less fragmented.
- The benchmark gain is small only because pipeline depth is unchanged, but this cleanup removes a tax that later versions no longer have to carry.

**What is still lacking**

- The pipeline is still shallow.
- Global ingress is still scalar.

The lesson here is important: the path had to be cleaned before deeper scheduling could really pay off.

## 6. Clean 2-Stage And Wide Ingress

### `warptiling_v1` -> `warptiling_v2` -> `warptiling_v2_vec4`

**Code change: clean 2-stage**

```cpp
threadblock_tile_iterator_a.load(tb_frag_a);
threadblock_tile_iterator_b.load(tb_frag_b);
store_fragments_to_stage<Traits>(..., tb_frag_a, tb_frag_b);
__syncthreads();

for (; gemm_k_iterations > 0; --gemm_k_iterations) {
  if (gemm_k_iterations > 1) {
    threadblock_tile_iterator_a.load(tb_frag_a);
    threadblock_tile_iterator_b.load(tb_frag_b);
  }
  ...
}
```

**Code change: vec4 ingress**

```cpp
using ThreadMapA =
    utils::PitchLinearStripminedThreadMap<kThreadblockK, kThreadblockM,
                                          kThreads, 4>;
using ThreadMapB =
    utils::PitchLinearStripminedThreadMap<kThreadblockN, kThreadblockK,
                                          kThreads, 4>;
```

**Metric delta**

| Kernel | TFLOPS | Issued | No Eligible | Key Change |
| --- | ---: | ---: | ---: | --- |
| `warptiling_v1` | 19.54 | 0.69 | 31.39% | clean path baseline |
| `warptiling_v2` | 20.70 | 0.74 | 26.06% | clean 2-stage overlap |
| `warptiling_v2_vec4` | 23.01 | 0.81 | 19.39% | same pipeline, much wider ingress |

**Why it got faster**

- `v2` starts real overlap: while one stage is being consumed by warp MMAs, the next stage is already being loaded and staged.
- That shortens the exposed memory-to-compute gap, so the scheduler finds ready warps more often.
- `v2_vec4` goes much further because the same overlap is now fed by wider loads, so each stage fill needs fewer instructions and leaves far fewer long-scoreboard bubbles.

**What is still lacking**

- Even `v2_vec4` still does not have the direct global-to-shared path of `torch`.
- The output path is still not CUTLASS-grade yet.

This is the point where warptiling becomes clearly competitive.

## 7. `L2::128B` Helper Loads And Predication

### `warptiling_v2` -> `warptiling_v3` / `warptiling_v4`
### `warptiling_v2_vec4` -> `warptiling_v3_vec4`

**Code change: helper loads**

```cpp
threadblock_tile_iterator_a.load_ptx(frag_a);
threadblock_tile_iterator_b.load_ptx(frag_b);
```

**Code change: predicated loads**

```cpp
threadblock_tile_iterator_a.load(frag_a, gemm_k_iterations > 1);
threadblock_tile_iterator_b.load(frag_b, gemm_k_iterations > 1);
```

**Metric delta**

| Kernel | TFLOPS | Issued | No Eligible | L2 Hit |
| --- | ---: | ---: | ---: | ---: |
| `warptiling_v2` | 20.70 | 0.74 | 26.06% | 80.35% |
| `warptiling_v3` | 20.96 | 0.75 | 24.60% | 82.03% |
| `warptiling_v4` | 21.66 | 0.78 | 21.51% | 81.72% |
| `warptiling_v2_vec4` | 23.01 | 0.81 | 19.39% | 79.76% |
| `warptiling_v3_vec4` | 23.75 | 0.84 | 15.52% | 81.73% |

**Why it got faster**

- `load_ptx()` pushes codegen toward `LDG.E.LTC128B`-class loads, which improves how requests are packaged into the cache hierarchy.
- Better line utilization improves L2 residency and makes the next tile more likely to arrive warm.
- Vec4 gains more because wide coalesced accesses can actually exploit that cache hint well, and predication keeps the steady-state loop cleaner by removing tail irregularity.

**What is still lacking**

- This is still not direct copy.
- It improves ingress quality, but does not solve the stage-boundary seam.

This is where the vec4 family effectively reaches its plateau.

## 8. Deeper CUTLASS-Style Pipeline: Helps Scalar, Hurts Vec4

### `warptiling_v4 -> v5 -> v6`
### `warptiling_v4_vec4 -> v5_vec4 -> v6_vec4`

**Code change**

```cpp
typename Traits::WarpTileIteratorA::Fragment warp_frag_a[2];
typename Traits::WarpTileIteratorB::Fragment warp_frag_b[2];

warp_tile_iterator_a.load(warp_frag_a[0]);
warp_tile_iterator_b.load(warp_frag_b[0]);
warp_tile_iterator_a.load(warp_frag_a[(warp_mma_k + 1) % 2]);
warp_tile_iterator_b.load(warp_frag_b[(warp_mma_k + 1) % 2]);
```

**Metric delta**

| Kernel | TFLOPS | Issued | No Eligible | Regs | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `warptiling_v4` | 21.66 | 0.78 | 21.51% | 128 | scalar baseline before deep pipeline |
| `warptiling_v5` | 21.79 | 0.80 | 19.71% | 125 | partial deepening helps |
| `warptiling_v6` | 21.85 | 0.82 | 18.49% | 120 | full deepening still helps scalar |
| `warptiling_v4_vec4` | 23.70 | 0.84 | 15.55% | 127 | vec4 baseline before deep pipeline |
| `warptiling_v5_vec4` | 23.57 | 0.84 | 16.31% | 127 | slightly worse |
| `warptiling_v6_vec4` | 22.83 | 0.79 | 20.60% | 129 | too deep, drops to `1 block/SM` |

**Why it got faster**

- On the scalar path, the deeper pipeline pulls the next warp fragments earlier, so fewer `FFMA` groups stall on a just-in-time load.
- In practice, more of the `LDG/STS/LDS -> FFMA` latency chain is hidden behind already-running compute.
- That is why scalar `Issued` keeps rising even though the algorithm itself is not fundamentally different.

**What is still lacking**

- The extra overlap comes with more control cost and more data pressure.
- In vec4 form, the same idea is simply too deep: live ranges grow, occupancy collapses, and performance falls.

This is one of the strongest results in the entire study. The deepest pipeline is not automatically the best pipeline.

## 9. CUTLASS-Style Epilogue Reorder And Explicit Choreography

### `warptiling_v6 -> v7 -> v8`
### `warptiling_v6_vec4 -> v7_vec4 -> v8_vec4`

**Code change: epilogue reorder**

```cpp
warp_store_iter.store(accum_frag);
__syncthreads();
shared_load_iter.load(output_frag);
out_iter.store(output_frag);
__syncthreads();
```

**What changed first (`v7`)**

- scalar family reaches:
  - `Global Load / Store Avg Bytes / Sector = 32 / 32`
  - `L2 Excessive = 0`
- vec4 family also reaches `32 / 32` while preserving wide ingress

**Code change: explicit choreography cleanup (`v8`)**

- keep the same epilogue
- rewrite the mainloop into explicit `prologue -> steady-state -> drain`

**Metric delta**

| Kernel | TFLOPS | Issued | No Eligible | Store Bytes / Sector |
| --- | ---: | ---: | ---: | ---: |
| `warptiling_v6` | 21.85 | 0.82 | 18.49% | 8 |
| `warptiling_v7` | 22.01 | 0.82 | 17.83% | 32 |
| `warptiling_v8` | 22.21 | 0.83 | 17.24% | 32 |
| `warptiling_v6_vec4` | 22.83 | 0.79 | 20.60% | 8 |
| `warptiling_v7_vec4` | 23.76 | 0.85 | 14.79% | 32 |
| `warptiling_v8_vec4` | 23.70 | 0.85 | 14.87% | 32 |

**Why it got faster**

- The epilogue reorder repacks accumulator export into clean `32B` sectors, so the output path stops leaking bandwidth through partial stores.
- That removes unnecessary write traffic and shrinks the tail cost after the MMA mainloop.
- `v8` then tightens the `prologue -> steady-state -> drain` choreography, so the kernel reaches the same clean epilogue with fewer transition bubbles.

**What is still lacking**

- Even after the epilogue is fixed, explicit CUTLASS still leads scalar warptiling on load-side L2 hit and issue density.
- That means the problem is no longer the output path. It is the ingress path and the stage seam.

## 10. Scalar Endgame: `v9` Timing, `v10` Codegen, `v11` Swizzle

### `warptiling_v9 -> v10 -> v11`

**Code change in `v10`: same logic, different symbol**

```diff
- __global__ void sgemm_warptiling(...)
+ __global__ void cutlass_sgemm_warptiling(...)
```

**Code change in `v11`: same scalar family, different CTA order**

```cpp
int cta_m_idx = 0;
int cta_n_idx = 0;
utils::GroupedThreadblockSwizzle<8>::get_tile_offset(cta_m_idx, cta_n_idx);
```

**Metric delta**

| Kernel | ms | TFLOPS | Issued | No Eligible | L2 Hit | DRAM BW | Wait | Short/Long Scoreboard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `warptiling_v9` | 6.170 | 22.27 | 0.82 | 17.70% | 81.37% | 141.24 GB/s | 0.17 | `0.09 / 0.07` |
| `warptiling_v10` | 6.041 | 22.74 | 0.84 | 15.93% | 81.38% | 152.66 GB/s | 0.07 | `0.02 / 0.01` |
| `warptiling_v11` | 5.991 | 22.94 | 0.84 | 15.94% | 91.07% | 95.49 GB/s | 0.15 | `0.07 / 0.06` |

**Why it got faster**

- `v10` is the backend-local win. The mainloop is still `v9`, but the `cutlass`-named symbol gets tighter address-generation and seam-local scheduling, so `LDG -> STS -> LDS -> FFMA` is packed with fewer bookkeeping gaps. That is why `Wait` and scoreboard stalls drop while `L2 Hit` stays flat.
- `v11` is the cache-residency win. It keeps the plain symbol, applies grouped threadblock swizzling, and changes which neighboring CTAs touch the same tiles back-to-back. The mainloop seam is not cleaner than `v10`, but the cache sees far more cross-CTA reuse.
- In practice that means `v10` and `v11` improve for different reasons: `v10` by local instruction packing, `v11` by better tile reuse across CTAs.

**What the swizzle is doing**

- The default launch order walks tiles roughly as `(m0,n0) -> (m0,n1) -> (m0,n2) ...`, which naturally reuses `A` across neighboring CTAs.
- `GroupedThreadblockSwizzle<8>` changes that to `(m0,n0) -> (m1,n0) -> ... -> (m7,n0) -> (m0,n1) ...`, which instead keeps the same `N` tile hot across a group of CTAs.
- In this kernel family that trade is favorable. `v11` raises `L2 Hit` from `81.37%` to `91.07%` and cuts DRAM bandwidth from `141.24` to `95.49 GB/s`, which is enough to outweigh its slightly worse seam-local waits.

**Important caveat**

- The earlier `cutlass` naming trick helped `v10`, but it did not help the swizzled kernel. The faster `v11` is the plain-symbol version, so the improvement here should be read as a swizzle effect, not another codegen-name effect.

## Why `v7_vec4` Still Trails `torch`

The current best study kernel is `warptiling_v7_vec4_128x128x8_32x64x8` at `5.784 ms / 23.76 TFLOPS`. `v8_vec4` is only narrowly behind at `23.70 TFLOPS`, so the vec4 story is still a plateau, not a new direction.

It beats both CUTLASS references because it combines:

- a clean warptiling lane map
- a clean 2-stage schedule
- vec4 ingress
- `L2::128B` helper loads
- a fully cleaned epilogue
- no generic-template overhead

But it still trails `torch_matmul` by a small margin because `torch` still keeps two structural advantages:

1. `LDGSTS` direct global-to-shared copy
2. a larger `256x128x8` tile, which gives it more arithmetic density per copy wave

That is why the remaining gap now looks narrow but stubborn.

## Best-Kernel Comparison

This is the shortest useful comparison view: one overall ceiling, one scalar reference, one vec4 study winner, and one scalar study winner.

| Kernel | Role | TFLOPS | Issued | No Eligible | L2 Hit | DRAM BW | Regs | Ach. Occ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `torch_matmul` | overall reference | 23.87 | 0.88 | 12.26% | 91.25% | 69.95 GB/s | 202 | 16.67% |
| `cutlass_simt_128x128x8_32x64x8_2stage` | scalar reference | 22.64 | 0.88 | 12.02% | 93.11% | 155.13 GB/s | 122 | 32.41% |
| `warptiling_v7_vec4_128x128x8_32x64x8` | vec4 best | 23.76 | 0.85 | 14.79% | 81.35% | 152.15 GB/s | 127 | 32.46% |
| `warptiling_v11_128x128x8_32x64x8` | scalar best | 22.94 | 0.84 | 15.94% | 91.07% | 95.49 GB/s | 127 | 32.34% |

What this says at a glance:

- `v7_vec4` almost reaches `torch` on benchmark result, but it does not do it by matching `torch`'s cache behavior. Its `L2 Hit` is much lower and its DRAM traffic is much higher, which points back to the missing `LDGSTS` path and smaller arithmetic density.
- `v7_vec4` still beats the scalar CUTLASS reference end-to-end because the fixed-shape vec4 ingress is strong enough to offset weaker scheduler and cache metrics.
- `v11` beats CUTLASS on benchmark by a different route. It does not match CUTLASS on issue density, but grouped swizzling pulls `L2 Hit` up to `91.07%` and cuts DRAM pressure hard enough to win overall.
- `v11` is also the cleanest scalar comparison against `torch`: cache residency is now close, but the remaining gap is still visible in `Issued`, `No Eligible`, and the fact that `torch` gets more compute density from the same copy wave.

## Final Takeaways

- The biggest structural wins came from 2D outer product, lane remap, clean 2-stage overlap, vec4 ingress, and `L2::128B` helper loads.
- The vec4 family reaches the best overall result with a moderate-depth pipeline, not the deepest one. The current best study kernel is `v7_vec4` at `23.76 TFLOPS`.
- The scalar family now peaks at `v11` with `22.94 TFLOPS`, but the last two gains come from different sources:
  - `v10`: denser backend codegen around the stage seam
  - `v11`: better inter-CTA cache residency from grouped swizzling
- The study gets faster for three repeatable reasons:
  - each staged operand feeds more `FFMA`s
  - each memory instruction carries more useful bytes
  - each stage boundary or CTA schedule leaves fewer wasted cycles before compute can continue
- The output path is no longer the main problem. The remaining gap is about ingress quality, stage-boundary compression, L2 residency, and arithmetic density.

In short:

- fixed-shape warptiling already beats both CUTLASS references
- `v7_vec4` is the current best study kernel
- `v11` is the current best scalar kernel
- the final target is still the `torch` combination of direct global-to-shared copy and larger tile density
