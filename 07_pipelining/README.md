# CUDA Kernel Study: Pipelining

This document compares and analyzes two software pipelining techniques for hiding Global Memory -> Shared Memory transfer latency in CUDA kernels.

1. **Ping-Pong (Double Buffering)**: A classic technique that works on all architectures
2. **`cp.async` Multi-Stage Pipeline**: A hardware asynchronous pipeline available on Ampere (SM80) and later

---

## Table of Contents

- [Background: Why Pipelining Matters](#background-why-pipelining-matters)
- [Case 1: Ping-Pong Double Buffering](#case-1-ping-pong-double-buffering)
- [Case 2: cp.async Multi-Stage Pipeline](#case-2-cpasync-multi-stage-pipeline)
- [Key Differences Between the Two Techniques](#key-differences-between-the-two-techniques)
- [Example Code](#example-code)
- [Benchmark Results](#benchmark-results)
- [NCU Profiling Comparison](#ncu-profiling-comparison)
- [References](#references)

---

## Background: Why Pipelining Matters

Consider the simplest tiled kernel structure:

```cuda
for (int k = 0; k < NUM_TILES; k++) {
    // Phase 1: Global -> Shared
    smem[tid] = global[k * TILE + tid];
    __syncthreads();

    // Phase 2: Compute
    accumulate(smem, ...);
    __syncthreads();
}
```

In this structure, **compute cannot start until the load phase has fully completed**. The timeline looks like this:

```
         iter 0              iter 1              iter 2
     ┌──────┬──────┐     ┌──────┬──────┐     ┌──────┬──────┐
     │ Load │ Comp │     │ Load │ Comp │     │ Load │ Comp │
     └──────┴──────┘     └──────┴──────┘     └──────┴──────┘
      ◄── stall ──►       ◄── stall ──►
```

Global memory load latency is on the order of hundreds of cycles. During that time, the SM's compute units sit idle. **The goal of pipelining is to overlap the compute of iteration N with the load of iteration N+1 to remove this stall.**

---

## Case 1: Ping-Pong Double Buffering

### Overview

Two shared memory buffers, `buf[0]` and `buf[1]`, are used alternately. While one buffer is being consumed by compute, the other buffer is filled with the next tile so that the two phases overlap.

### Data Path

```
Global Memory ──LDG──▶ Register File ──STS──▶ Shared Memory
                 ▲                       ▲
                 │                       │
          hundreds of cycles       executes after register arrival
```

In the ping-pong approach, the Global -> Shared transfer must pass through registers:
1. `LDG` (Load Global -> Register): load from global memory into registers, with hundreds of cycles of latency.
2. `STS` (Store to Shared): write the register value into shared memory. This cannot execute until `LDG` completes.

### Timeline

```
                iter 0       iter 1       iter 2       iter 3
Load (G→R→S):   [buf0]       [buf1]       [buf0]       [buf1]
Compute:          ×          [buf0]       [buf1]       [buf0]
                             ◄─overlap──▶ ◄─overlap──▶
```

The first iteration performs only the load because there is no tile ready to compute yet (prologue). After that, the main loop overlaps load and compute on different buffers. The final iteration performs only the compute because there is no next tile to load (epilogue).

### Implementation Pattern

```cuda
__shared__ float smem[2][TILE_K][TILE_N];  // two buffers

// Prologue: load the first tile
load_tile(smem[0], global, /*tile=*/0);
__syncthreads();

// Main loop
for (int k = 1; k < NUM_TILES; k++) {
    int curr = (k - 1) & 1;   // buffer used for compute
    int next = k & 1;         // buffer used for load

    // Load the next tile into the next buffer
    load_tile(smem[next], global, k);

    // Compute from the current buffer
    compute_tile(smem[curr], ...);

    __syncthreads();  // ensures both load and compute are complete
}

// Epilogue: compute the last tile
compute_tile(smem[(NUM_TILES - 1) & 1], ...);
```

Here, `load_tile` looks simple in CUDA C++, but it is actually lowered into separate `LDG` and `STS` instructions:

```cuda
__device__ void load_tile(float smem[][TILE_N], const float* global, int tile_idx) {
    // This single line becomes LDG + STS under the hood
    smem[tid_y][tid_x] = global[tile_idx * TILE_K * N + tid_y * N + tid_x];
}
```

### Limitations and Caveats

**1. Dependence on compiler scheduling**

`smem[next][...] = global[...]` is one line in source code, but it is split into `LDG` and `STS`. To achieve ideal overlap, the compiler should schedule `LDG` near the beginning of the loop and `STS` later. In practice, the compiler does not always produce the ideal schedule.

```
Ideal scheduling:                    Worst-case scheduling:
┌─ LDG (next)  ◀ issued here        ┌─ LDG (next)  ◀ issued here
│  compute(curr) ◀ while LDG runs   │  ... stall (waiting for LDG) ...
│  ...                              │  STS (next)   ◀ after LDG completes
└─ STS (next)  ◀ after LDG          └─ compute(curr)
```

**2. Register pressure**

Data loaded by `LDG` remains in registers until it is written into shared memory by `STS`. As tile sizes grow, this intermediate data can consume a meaningful number of registers and reduce occupancy.

**3. Synchronization granularity**

Because a single `__syncthreads()` must fence both load and compute, extending this approach beyond depth 2 makes the meaning of the barrier less clear and the pipeline harder to manage.

---

## Case 2: cp.async Multi-Stage Pipeline

### Overview

Introduced in Ampere (SM80), the `cp.async` instruction performs **direct asynchronous Global -> Shared copies without explicitly staging data through registers**. Its `commit_group` and `wait_group<N>` mechanism gives fine-grained control over multiple in-flight transfers, making deeper pipelines such as 3-stage or 4-stage natural to express.

The measurements in this repository were collected with the Makefile default of `STAGES=2`. In other words, the performance comparison below is a comparison between ping-pong and `cp.async` at the **same 2-stage pipeline depth**, isolating the effect of the data path and synchronization model.

### Data Path

```
                  ┌─── Ping-Pong path (LDG+STS) ───┐
                  │                                │
Global Memory ────┤                                ├──▶ Shared Memory
                  │                                │
                  └─── cp.async path (HW direct) ──┘
                       register bypass, asynchronous transfer
```

### Core Primitives

#### `cp.async.ca.shared.global`

This instruction **issues** an asynchronous Global -> Shared copy. The transfer begins at that point, but completion is not yet guaranteed. The copy itself does not use an explicit intermediate register, but that does **not** imply that total `register/thread` for the entire kernel must decrease. Overall register usage still depends on address calculation, stage bookkeeping, loop unrolling, and compiler scheduling.

```cuda
// PTX-level
cp.async.ca.shared.global [smem_ptr], [gmem_ptr], 16;  // copy 16 bytes

// CUDA C++ wrapper (__cuda_pipeline or cuda::memcpy_async)
__pipeline_memcpy_async(&smem[stage][tid], &global[offset], sizeof(float4));
```

#### `cp.async.commit_group()`

This groups the `cp.async` instructions issued so far into a single **group** whose completion can be tracked together.

```
cp.async(A)  ─┐
cp.async(B)   ├── commit_group() -> [Group 0]
cp.async(C)  ─┘
cp.async(D)  ─┐
cp.async(E)   ├── commit_group() -> [Group 1]
```

#### `cp.async.wait_group<N>()`

This waits until **all groups except the most recent N groups have completed**.

That is the core idea behind a multi-stage pipeline:

```
commit order:  [G0] [G1] [G2] [G3] [G4]
                oldest ─────────▶ newest

wait_group<2>:  G0 ✓   G1 ✓   G2 ✓  │ G3 ?   G4 ?
                 ──── guaranteed ─── │ ─ in-flight allowed ─
                                     │
                                     └ the most recent 2 groups may still be in flight
```

So the larger N is, the more groups can remain in flight, increasing the opportunity for latency hiding. The trade-off is that shared memory usage grows as `(N+1) × tile_size`.

### Timeline (Generic 3-Stage Example)

```
           k=0    k=1    k=2    k=3    k=4    k=5    k=6    k=7
Load:     [s0]   [s1]   [s2]   [s0]   [s1]   [s2]   [s0]
Compute:                       [s0]   [s1]   [s2]   [s0]   [s1]   [s2]
                                │                     │
                         prologue drain          epilogue drain

In-flight:  1      2      3      3      3      3      2      1      0
```

With three loads kept in flight, compute begins as soon as the oldest completed tile is ready. Compared with a 2-stage pipeline, a deeper pipeline can hide global memory latency more aggressively.

### Implementation Pattern

```cuda
constexpr int STAGES = 3;
__shared__ float smem[STAGES][TILE_K][TILE_N];

// Prologue: fill the pipeline
#pragma unroll
for (int s = 0; s < STAGES; s++) {
    __pipeline_memcpy_async(
        &smem[s][tid_y][tid_x],
        &global[s * TILE_K * N + tid_y * N + tid_x],
        sizeof(float)
    );
    __pipeline_commit();
}

// Main loop
for (int k = 0; k < NUM_TILES - STAGES; k++) {
    int stage = k % STAGES;

    // 1. Wait for the oldest group to complete
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();

    // 2. Compute from the ready buffer
    compute_tile(smem[stage], ...);

    // 3. Reuse this slot for the next tile
    __syncthreads();
    __pipeline_memcpy_async(
        &smem[stage][tid_y][tid_x],
        &global[(k + STAGES) * TILE_K * N + tid_y * N + tid_x],
        sizeof(float)
    );
    __pipeline_commit();
}

// Epilogue: drain the pipeline
#pragma unroll
for (int s = 0; s < STAGES; s++) {
    int stage = (NUM_TILES - STAGES + s) % STAGES;

    __pipeline_wait_prior(STAGES - 1 - s);
    __syncthreads();

    compute_tile(smem[stage], ...);
}
```

### Stage Selection Guide

| Stages | In-Flight Groups | SMEM Usage | Latency Hiding |
|--------|------------------|------------|----------------|
| 2 | 1 | 2 × tile | Similar depth to ping-pong, but with a different data path |
| 3 | 2 | 3 × tile | Usually a good balance |
| 4 | 3 | 4 × tile | Useful when memory latency pressure is higher |
| 5+ | 4+ | 5+ × tile | Risk of occupancy loss due to shared memory pressure |

In practice, **3-stage** is often the sweet spot between latency hiding and shared memory cost. Production kernels such as FlashAttention 2 and CUTLASS 3.x commonly use 3 stages by default.

That said, the artifacts in this repository were intentionally collected with **`STAGES=2`** to isolate the effect of the mechanism itself. The benchmark and profiling results below should therefore be interpreted as a **2-stage ping-pong vs 2-stage `cp.async`** comparison, not as the result of deep multi-stage tuning.

---

## Key Differences Between the Two Techniques

| Perspective | Ping-Pong | `cp.async` Multi-Stage |
|---|---|---|
| **Data path** | Global -> **Register** -> Shared (`LDG` + `STS`) | Global -> Shared **directly** (HW bypass) |
| **Register pressure** | Loaded data passes through registers | No explicit intermediate register in the copy path |
| **Pipeline depth** | Usually 2 (3+ is awkward) | 3, 4, 5+ are straightforward |
| **Synchronization model** | Single `__syncthreads()` barrier | Fine-grained `commit_group` + `wait_group<N>` |
| **Latency-hiding guarantee** | Depends on compiler instruction scheduling | Structurally supported by hardware async copy and multi-stage control |
| **SMEM usage** | 2 × tile | `STAGES × tile` |
| **Hardware requirement** | All architectures (Kepler and later) | SM80 (Ampere) and later |
| **Code complexity** | Lower | Moderate (requires prologue/epilogue management) |

### When to Use Which

- **Ping-Pong**: when compatibility with pre-Ampere GPUs matters, shared memory is extremely limited, or the tile size is small enough that a 2-stage pipeline already hides most of the latency.
- **`cp.async`**: when targeting Ampere or newer hardware and pursuing maximum performance, especially in large-tile, high-arithmetic-intensity kernels.

---

## Example Code

All three kernels implement the same tiled SGEMM (`C = A × B`, FP32). The only difference is the pipelining strategy.

### File Layout

```
.
├── common.cuh                  # common utilities (init, verify, benchmark harness)
├── baseline_no_pipeline.cu     # baseline with no pipelining
├── ping_pong.cu                # double buffering (ping-pong)
├── cp_async.cu                 # cp.async N-stage pipeline
├── Makefile                    # build / run / profiling targets
├── README.kr.md                # Korean documentation
└── README.md                   # English documentation
```

### Kernel Parameters

Shared configuration defined in `common.cuh`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| M, N, K | 2048 | Matrix size |
| Block tile | 128×128×8 | `BLOCK_TILE_M × BLOCK_TILE_N × BLOCK_TILE_K` |
| Thread block | 16×16 (256 threads) | `BLOCK_DIM_X × BLOCK_DIM_Y` |
| Thread tile | 8×8 | Output region computed by each thread |
| K tiles | 256 | `K / BLOCK_TILE_K` |

### Build

```bash
# Build everything with the Makefile
make all

# Or build each binary individually
nvcc -O3 -arch=sm_80 -std=c++17 --expt-relaxed-constexpr -o baseline baseline_no_pipeline.cu
nvcc -O3 -arch=sm_80 -std=c++17 --expt-relaxed-constexpr -o ping_pong ping_pong.cu
nvcc -O3 -arch=sm_80 -std=c++17 --expt-relaxed-constexpr -DSTAGES=2 -o cp_async cp_async.cu
```

> `sm_80` or later is required for `cp.async`. On other Ampere/Hopper GPUs, use `sm_86`, `sm_89`, `sm_90`, and so on.

### Run

```bash
# Run the full benchmark suite
make run

# Or run each binary directly
./artifacts/baseline
./artifacts/ping_pong
./artifacts/cp_async
```

Each binary computes a CPU reference, runs the kernel benchmark (5 warm-up runs + 10 extra warm-up runs + 20 measured runs), and then verifies correctness.

---

## Benchmark Results

Measurement setup:

- GPU: NVIDIA RTX A6000 (CC 8.6)
- Problem size: `M=N=K=2048`
- Tile shape: block `128x128x8`, thread `8x8`
- `cp.async` configuration: `STAGES=2`
- Correctness: all three kernels report `Correct: YES`

| Kernel | Avg Time (ms) | GFLOPS | Speedup vs Baseline | Interpretation |
|--------|---------------|--------|---------------------|----------------|
| Baseline (No Pipeline) | 2.296 | 7484.0 | 1.00x | Load and compute are fully serialized |
| Ping-Pong (Double Buffering) | 2.189 | 7846.8 | 1.049x | Slight gain from load/compute overlap |
| `cp.async` 2-Stage | 1.908 | 9003.3 | 1.203x | Direct async copy provides the strongest latency hiding |

Key takeaways:

- Ping-pong is **about 4.8% faster** than the baseline. It introduces overlap, but does not fully hide global load latency.
- `cp.async` 2-stage is **about 20.3% faster** than the baseline and **about 14.7% faster** than ping-pong.
- The important point here is that `cp.async` wins convincingly over ping-pong **even without moving to 3+ stages**.

---

## NCU Profiling Comparison

### Profiling Commands

```bash
# Run the full profiling workflow from the Makefile
make profile

# Dump PTX / SASS
make ptx
make sass
```

### Key Metric Comparison

| Metric | Baseline (No Pipeline) | Ping-Pong | `cp.async` 2-Stage |
|--------|------------------------|-----------|--------------------|
| **Kernel Duration (ms)** | 2.29 | 2.19 | 1.90 |
| **Compute (SM) Throughput (%)** | 28.71 | 30.85 | 35.60 |
| **Memory Throughput (%)** | 37.30 | 38.64 | 43.87 |
| **DRAM Throughput (%)** | 6.28 | 6.69 | 6.62 |
| **DRAM Bandwidth (GB/s)** | 45.77 | 48.73 | 48.28 |
| **L1/TEX Cache Throughput (%)** | 48.57 | 50.61 | 57.12 |
| **L2 Cache Throughput (%)** | 14.28 | 14.58 | 17.19 |
| **Issued Warp / Scheduler** | 0.37 | 0.40 | 0.46 |
| **Eligible Warps / Scheduler** | 0.42 | 0.46 | 0.53 |
| **Warp Cycles / Issued Instruction** | 5.33 | 4.94 | 4.30 |

| Metric | Baseline (No Pipeline) | Ping-Pong | `cp.async` 2-Stage |
|--------|------------------------|-----------|--------------------|
| **Registers / Thread** | 146 | 162 | 254 |
| **Static Shared Memory / Block (KB)** | 8.19 | 16.38 | 16.38 |
| **Achieved Occupancy (%)** | 16.54 | 16.66 | 16.62 |
| **L1/TEX Hit Rate (%)** | 16.46 | 14.02 | 14.63 |
| **Stall Wait (inst)** | 1.64 | 1.47 | 1.28 |
| **Stall Long Scoreboard (inst)** | 0.95 | 0.93 | 0.19 |
| **Stall Barrier (inst)** | 0.21 | 0.29 | 0.16 |
| **L1 Wavefronts Shared Excessive** | 33,554,432 | 33,554,432 | 39,845,888 |
| **Uncoalesced Shared Access Ratio** | 38% | 38% | 42% |
| **Global Excessive Sectors Ratio** | 18% | 18% | 18% |

### Detailed Analysis

**1. Why the ping-pong gain is modest**

Ping-pong is 4.8% faster than the baseline in elapsed time, but `Stall Long Scoreboard` only changes from `0.95` to `0.93`. In other words, double buffering does not meaningfully eliminate the underlying long-latency global load dependency in this kernel, and the overlap between `LDG` and `STS` only helps to a limited extent.

At the same time, `Stall Barrier` increases from `0.21` to `0.29`. Ping-pong does overlap load and compute, but a single `__syncthreads()` still fences both phases together, so part of the overlap benefit is given back to barrier cost.

**2. Why `cp.async` 2-stage is the fastest**

The measured `cp.async` kernel here is **2-stage**, not 3-stage. That makes this result less about deeper pipelines and more about the fact that **`cp.async` is structurally better than ping-pong even at the same depth**.

The most important metric is `Stall Long Scoreboard`:

- Baseline: `0.95`
- Ping-pong: `0.93`
- `cp.async`: `0.19`

That means `cp.async` reduces long scoreboard stall by **about 80%** relative to the baseline and **about 79.6%** relative to ping-pong. The stall sampling counters tell the same story:

- `stall_long_sb` samples: `18,109 -> 17,986 -> 3,654`

Scheduler readiness also improves:

- `Issued Warp / Scheduler`: `0.37 -> 0.40 -> 0.46`
- `Eligible Warps / Scheduler`: `0.42 -> 0.46 -> 0.53`
- `No Eligible`: `62.56% -> 59.59% -> 53.54%`
- `Warp Cycles / Issued Instruction`: `5.33 -> 4.94 -> 4.30`

So the gain from `cp.async` does not come from increased occupancy. It comes from making the already resident warps reach an issue-ready state more often.

**3. This kernel is latency- and scheduling-bound, not DRAM-bandwidth-bound**

All three kernels stay at roughly `6.3~6.7%` DRAM throughput and `46~49 GB/s` of DRAM bandwidth. In contrast, `Compute (SM) Throughput` rises from `28.71%` to `35.60%`, and `Memory Throughput` rises from `37.30%` to `43.87%`.

That tells us this SGEMM is **not slow because it saturates HBM bandwidth**. It is slow because of load latency and warp issue inefficiency. Pipelining helps not by increasing raw bandwidth, but by moving data-transfer time behind useful compute so the SM stays busy more consistently.

**4. Occupancy is effectively unchanged**

All three kernels sit at about `16.6%` achieved occupancy. Both `Block Limit Registers = 1` and `Block Limit Shared Mem = 1` are active, so the current tile shape effectively limits the kernel to **one block per SM**.

That means these measurements should not be interpreted as "more blocks or more warps won." They should be interpreted as "with roughly the same occupancy, how much did the pipeline design reduce warp stall?"

**5. `cp.async` does not fix every bottleneck**

The profiling results also show persistent shared-memory issues:

- all three kernels have `18%` global excessive sectors
- baseline and ping-pong both show `38%` excessive shared wavefronts
- `cp.async` actually increases excessive shared wavefronts to `42%`

So the performance gain from `cp.async` does **not** come from a better shared-memory access pattern. Those shared-side inefficiencies remain, and some get worse. The reason the overall kernel is still faster is that the structural reduction in staging latency is worth more than the extra shared-side cost.

The next optimization step is therefore probably **not** "increase the number of stages first." A more direct next step would be to reduce shared-memory bank conflicts through padding, swizzling, or reworking the vectorized load/store layout.

**6. "Register bypass" is not the same as "lower register/thread"**

Conceptually, `cp.async` replaces an `LDG -> register -> STS` copy path with a direct Global -> Shared transfer, so it removes the explicit intermediate register dependence of the copy itself.

But the actual compiled kernel shows:

- Baseline: `146` registers/thread
- Ping-pong: `162`
- `cp.async`: `254`

So in this implementation, `cp.async` ends up with the largest register footprint. That tells us the total register count is dominated by stage indexing, address calculation, prologue/epilogue logic, aggressive unrolling, and compiler scheduling, not just by the copy path.

So the correct statement is:

- `cp.async` reduces the temporary register dependence in the copy path itself
- total `register/thread` is still a separate property that must be checked in the final compiled kernel

### SASS-Level Analysis

```bash
rg -n "ld\\.global|st\\.shared|bar\\.sync" artifacts/baseline.ptx artifacts/ping_pong.ptx
rg -n "cp\\.async|commit_group|wait_group|bar\\.sync" artifacts/cp_async.ptx
rg -n "LDG|STS|BAR\\.SYNC" artifacts/baseline.sass artifacts/ping_pong.sass
rg -n "LDGSTS|LDGDEPBAR|DEPBAR|BAR\\.SYNC" artifacts/cp_async.sass
```

- In `baseline.ptx` and `ping_pong.ptx`, `ld.global.nc.f32` is repeatedly followed by `st.shared.f32`, with `bar.sync 0` at tile boundaries.
- In `cp_async.ptx`, `cp.async.ca.shared.global`, `cp.async.commit_group`, `cp.async.wait_group 1`, and `bar.sync 0` appear explicitly.

The SASS view makes the distinction even clearer:

- Baseline / Ping-pong: `LDG.E.CONSTANT` + `STS` + `BAR.SYNC.DEFER_BLOCKING`
- `cp.async`: `LDGSTS.E` + `LDGDEPBAR` + `DEPBAR.LE SB0, ...` + `BAR.SYNC.DEFER_BLOCKING`

So on Ampere, PTX-level `cp.async` does not necessarily appear as literal `CP.ASYNC` text in SASS. Instead, it lowers into **`LDGSTS`-family instructions plus dependency barrier sequences**.

For the artifacts in this repository, the interpretation is:

- ping-pong still uses a **separate global load + shared store**
- `cp.async` uses a **fused/asynchronous global-to-shared staging path**

That means the advantage of the `cp.async` kernel is not merely source-level code motion. It comes from using a **different instruction-level data movement mechanism**.

### Overall Conclusion

The results in this repository can be summarized in one sentence:

> **At the same 2-stage depth, `cp.async` hides global memory latency much more effectively than ping-pong.**

Ping-pong is better than the baseline, but because `LDG` and `STS` are still separate and coupled to barriers, it does not fundamentally reduce long scoreboard stall in this kernel. In contrast, `cp.async` improves issue readiness for resident warps through direct async copy plus commit/wait control, and both compute and memory throughput rise as a result.

The remaining bottlenecks are clear:

1. shared-memory bank conflicts and excessive wavefronts
2. low occupancy (effectively 1 block/SM at the current tile size)
3. high `register/thread` in the `cp.async` implementation

So the next experiment should probably **not** be "just raise `STAGES=3`." A more defensible sequence is to first improve the shared-memory layout and rein in register footprint, then sweep the pipeline stage count again.

---

## References

- [CUDA Programming Guide - Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [CUDA Programming Guide - Asynchronous Data Copies (legacy)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- [CUTLASS Tutorial: Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
