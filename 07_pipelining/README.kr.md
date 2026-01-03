# CUDA Kernel Study: Pipelining

Global Memory → Shared Memory 전송의 latency를 숨기기 위한 두 가지 소프트웨어 파이프라이닝 기법을 비교 분석한다.

1. **Ping-Pong (Double Buffering)**: 모든 아키텍처에서 동작하는 고전적 기법
2. **`cp.async` Multi-Stage Pipeline**: Ampere(SM80) 이상에서 사용 가능한 하드웨어 비동기 파이프라인

---

## 목차

- [Background: 왜 파이프라이닝이 필요한가](#background-왜-파이프라이닝이-필요한가)
- [Case 1: Ping-Pong Double Buffering](#case-1-ping-pong-double-buffering)
- [Case 2: cp.async Multi-Stage Pipeline](#case-2-cpasync-multi-stage-pipeline)
- [두 기법의 핵심 차이](#두-기법의-핵심-차이)
- [예시 코드](#예시-코드)
- [벤치마크 결과](#벤치마크-결과)
- [NCU Profiling 결과 비교](#ncu-profiling-결과-비교)
- [References](#references)

---

## Background: 왜 파이프라이닝이 필요한가

가장 단순한 tiled 커널을 생각해 보자.

```cuda
for (int k = 0; k < NUM_TILES; k++) {
    // Phase 1: Global → Shared
    smem[tid] = global[k * TILE + tid];
    __syncthreads();

    // Phase 2: Compute
    accumulate(smem, ...);
    __syncthreads();
}
```

이 구조에서는 **Load가 완전히 끝난 후에야 Compute가 시작**된다. 타임라인으로 그리면:

```
         iter 0              iter 1              iter 2
     ┌──────┬──────┐     ┌──────┬──────┐     ┌──────┬──────┐
     │ Load │ Comp │     │ Load │ Comp │     │ Load │ Comp │
     └──────┴──────┘     └──────┴──────┘     └──────┴──────┘
      ◄── stall ──►       ◄── stall ──►
```

Global memory load latency는 수백 사이클이다. 이 시간 동안 SM의 compute unit은 idle 상태로 낭비된다. **파이프라이닝의 목표는 iteration N의 compute와 iteration N+1의 load를 오버랩시켜 이 stall을 제거하는 것이다.**

---

## Case 1: Ping-Pong Double Buffering

### 개요

두 개의 shared memory 버퍼(`buf[0]`, `buf[1]`)를 번갈아 사용한다. 한쪽 버퍼로 compute하는 동안 다른 쪽 버퍼에 다음 타일을 로드하여 두 phase를 오버랩시킨다.

### 데이터 경로

```
Global Memory ──LDG──▶ Register File ──STS──▶ Shared Memory
                 ▲                       ▲
                 │                       │
          수백 사이클 latency       register 도착 후 실행
```

Ping-pong 방식에서 global → shared 전송은 반드시 **register를 경유**한다:
1. `LDG` (Load Global → Register): global memory에서 register로 로드. 수백 사이클 latency.
2. `STS` (Store to Shared): register의 값을 shared memory에 기록. LDG가 완료되어야 실행 가능.

### 타임라인

```
                iter 0       iter 1       iter 2       iter 3
Load (G→R→S):   [buf0]       [buf1]       [buf0]       [buf1]
Compute:          ×          [buf0]       [buf1]       [buf0]
                             ◄─overlap──▶ ◄─overlap──▶
```

첫 iteration은 compute할 데이터가 없으므로 load만 수행한다 (prologue). 이후 main loop에서는 load와 compute가 서로 다른 버퍼를 대상으로 동시에 진행된다. 마지막 iteration은 load할 데이터가 없으므로 compute만 수행한다 (epilogue).

### 구현 패턴

```cuda
__shared__ float smem[2][TILE_K][TILE_N];  // 두 개의 버퍼

// ── Prologue: 첫 타일 로드 ──
load_tile(smem[0], global, /*tile=*/0);
__syncthreads();

// ── Main Loop ──
for (int k = 1; k < NUM_TILES; k++) {
    int curr = (k - 1) & 1;   // compute 대상 버퍼
    int next = k & 1;          // load 대상 버퍼

    // 다음 타일을 next 버퍼에 로드 (LDG 발행)
    load_tile(smem[next], global, k);

    // curr 버퍼의 데이터로 연산
    compute_tile(smem[curr], ...);

    __syncthreads();  // load 완료 + compute 완료를 동시에 보장
}

// ── Epilogue: 마지막 타일 compute ──
compute_tile(smem[(NUM_TILES - 1) & 1], ...);
```

여기서 `load_tile`은 실제로는 아래와 같이 풀어진다:

```cuda
__device__ void load_tile(float smem[][TILE_N], const float* global, int tile_idx) {
    // 이 한 줄이 실제로는 LDG + STS 두 명령어로 분리됨
    smem[tid_y][tid_x] = global[tile_idx * TILE_K * N + tid_y * N + tid_x];
}
```

### 한계와 주의점

**1. 컴파일러 스케줄링 의존성**

`smem[next][...] = global[...]` 은 코드 한 줄이지만 LDG와 STS로 분리된다. 컴파일러가 LDG를 루프 초반에, STS를 루프 후반에 배치해야 이상적인 오버랩이 만들어진다. 하지만 컴파일러가 항상 최적 스케줄링을 보장하지는 않는다.

```
이상적 스케줄링:                    최악 스케줄링:
┌─ LDG (next)  ◀ 여기서 발행       ┌─ LDG (next)  ◀ 여기서 발행
│  compute(curr) ◀ LDG 진행 중     │  ... stall (LDG 대기) ...
│  ...                             │  STS (next)   ◀ LDG 완료 후
└─ STS (next)  ◀ LDG 완료 후       └─ compute(curr)
```

**2. Register pressure**

LDG로 로드된 데이터는 STS로 shared에 쓰일 때까지 register에 머문다. 타일 크기가 커지면 이 중간 데이터가 register를 많이 점유하여 occupancy가 떨어질 수 있다.

**3. 동기화 granularity**

`__syncthreads()` 하나로 load와 compute를 동시에 fence해야 하므로, 파이프라인 depth를 3 이상으로 늘리면 동기화 지점의 의미가 모호해지고 관리가 복잡해진다.

---

## Case 2: cp.async Multi-Stage Pipeline

### 개요

Ampere(SM80)에서 도입된 `cp.async` 명령은 **register를 우회하여 Global → Shared를 직접 비동기 전송**한다. `commit_group` / `wait_group<N>` 메커니즘으로 다수의 in-flight 전송을 세밀하게 제어할 수 있어, 3-stage, 4-stage 등 깊은 파이프라인을 자연스럽게 구성할 수 있다.

이 저장소의 실제 측정은 `Makefile` 기본값인 `STAGES=2`로 수행했다. 즉, 아래 성능 비교는 **동일한 2-stage depth에서 ping-pong과 `cp.async`의 데이터 경로/동기화 방식 차이**를 본 결과다.

### 데이터 경로

```
                  ┌─── Ping-Pong 경로 (LDG+STS) ───┐
                  │                                │
Global Memory ────┤                                ├──▶ Shared Memory
                  │                                │
                  └─── cp.async 경로 (HW direct) ──┘
                       register bypass, 비동기 전송
```

### 핵심 primitive

#### `cp.async.ca.shared.global`

Global → Shared 비동기 복사를 **발행**한다. 이 시점에서 전송은 시작만 되고, 완료 여부는 알 수 없다. 복사 자체는 명시적인 중간 register를 거치지 않는다. 다만 이는 **커널 전체 register/thread가 반드시 줄어든다**는 뜻은 아니다. 실제 register 사용량은 주소 계산, stage bookkeeping, loop unrolling, 컴파일러 스케줄링의 영향을 함께 받는다.

```cuda
// PTX 수준
cp.async.ca.shared.global [smem_ptr], [gmem_ptr], 16;  // 16 bytes 복사

// CUDA C++ wrapper (__cuda_pipeline 또는 cuda::memcpy_async)
__pipeline_memcpy_async(&smem[stage][tid], &global[offset], sizeof(float4));
```

#### `cp.async.commit_group()`

현재까지 발행된 `cp.async` 명령들을 하나의 **그룹**으로 묶는다. 그룹 단위로 완료 여부를 추적할 수 있다.

```
cp.async(A)  ─┐
cp.async(B)   ├── commit_group() → [Group 0]
cp.async(C)  ─┘
cp.async(D)  ─┐
cp.async(E)   ├── commit_group() → [Group 1]
```

#### `cp.async.wait_group<N>()`

**"가장 최근 N개 그룹을 제외한 나머지 그룹이 모두 완료될 때까지 대기"**한다.

이것이 multi-stage 파이프라인의 핵심이다:

```
commit 순서:  [G0] [G1] [G2] [G3] [G4]
               oldest ─────────▶ newest

wait_group<2>:  G0 ✓   G1 ✓   G2 ✓  │ G3 ?   G4 ?
                ──── 완료 보장 ──── │ ── in-flight 허용 ──
                                    │
                                    └ 최근 2개는 아직 진행 중이어도 OK
```

즉, `wait_group<N>`에서 N이 클수록 더 많은 그룹이 in-flight 상태로 유지되어 latency hiding 여지가 커진다. 반면 shared memory 사용량은 `(N+1) × tile_size`로 증가한다.

### 타임라인 (3-Stage 일반 예시)

```
           k=0    k=1    k=2    k=3    k=4    k=5    k=6    k=7
Load:     [s0]   [s1]   [s2]   [s0]   [s1]   [s2]   [s0]
Compute:                       [s0]   [s1]   [s2]   [s0]   [s1]   [s2]
                                │                     │
                         prologue drain          epilogue drain

In-flight:  1      2      3      3      3      3      2      1      0
```

3개의 로드가 항상 in-flight 상태를 유지하면서, 가장 오래된 것이 완료되면 compute에 사용된다. 2-stage(ping-pong)보다 더 깊은 파이프라인으로 global memory latency를 효과적으로 숨긴다.

### 구현 패턴

```cuda
constexpr int STAGES = 3;
__shared__ float smem[STAGES][TILE_K][TILE_N];

// ══════════════════════════════════════════
//  Prologue: 파이프라인 채우기
// ══════════════════════════════════════════
#pragma unroll
for (int s = 0; s < STAGES; s++) {
    __pipeline_memcpy_async(
        &smem[s][tid_y][tid_x],
        &global[s * TILE_K * N + tid_y * N + tid_x],
        sizeof(float)
    );
    __pipeline_commit();
}

// ══════════════════════════════════════════
//  Main Loop
// ══════════════════════════════════════════
for (int k = 0; k < NUM_TILES - STAGES; k++) {
    int stage = k % STAGES;

    // 1. 가장 오래된 그룹 완료 대기
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();

    // 2. 완료된 버퍼로 연산 (데이터 소비)
    compute_tile(smem[stage], ...);

    // 3. 소비 완료 — 이 슬롯을 다음 타일로 재사용
    __syncthreads();
    __pipeline_memcpy_async(
        &smem[stage][tid_y][tid_x],
        &global[(k + STAGES) * TILE_K * N + tid_y * N + tid_x],
        sizeof(float)
    );
    __pipeline_commit();
}

// ══════════════════════════════════════════
//  Epilogue: 파이프라인 drain
// ══════════════════════════════════════════
#pragma unroll
for (int s = 0; s < STAGES; s++) {
    int stage = (NUM_TILES - STAGES + s) % STAGES;

    __pipeline_wait_prior(STAGES - 1 - s);
    __syncthreads();

    compute_tile(smem[stage], ...);
}
```

### Stage 수 선택 가이드

| Stages | In-flight 그룹 | SMEM 사용량 | Latency Hiding |
|--------|----------------|-------------|----------------|
| 2 | 1 | 2 × tile | ping-pong과 동일 수준 (경로만 다름) |
| 3 | 2 | 3 × tile | 대부분의 경우 충분 |
| 4 | 3 | 4 × tile | HBM bandwidth가 매우 높은 경우 |
| 5+ | 4+ | 5+ × tile | shared memory 부족으로 occupancy 하락 위험 |

일반적으로 **3-stage**가 latency hiding과 shared memory 사용량의 균형점이다. Flash Attention 2, CUTLASS 3.x 등 실전 커널에서도 3-stage를 기본으로 사용한다.

다만 이 저장소의 artifact는 **기법 자체의 차이를 분리해서 보기 위해 `STAGES=2`**로 고정되어 있다. 따라서 아래 벤치마크/프로파일링 결과는 "깊은 multi-stage tuning 결과"가 아니라, **2-stage ping-pong vs 2-stage `cp.async`** 비교로 해석해야 한다.

---

## 두 기법의 핵심 차이

| 관점 | Ping-Pong | `cp.async` Multi-Stage |
|---|---|---|
| **데이터 경로** | Global → **Register** → Shared (LDG+STS) | Global → Shared **직접** (HW bypass) |
| **Register pressure** | 로드 데이터가 레지스터를 경유 | 복사 경로상 explicit intermediate register 없음 |
| **파이프라인 depth** | 통상 2 (3+ 어려움) | 3, 4, 5+ 자유롭게 설정 |
| **동기화 모델** | `__syncthreads()` 단일 barrier | `commit_group` + `wait_group<N>` 세밀 제어 |
| **Latency hiding 보장** | 컴파일러 instruction scheduling에 의존 | HW 비동기 + multi-stage로 **구조적 보장** |
| **SMEM 사용량** | 2 × tile | STAGES × tile |
| **HW 요구** | 모든 아키텍처 (Kepler~) | SM80 (Ampere) 이상 |
| **코드 복잡도** | 낮음 | 중간 (prologue/epilogue 관리 필요) |

### 언제 어떤 것을 사용하는가

- **Ping-Pong**: Volta 이하 호환이 필요하거나, shared memory가 극도로 부족한 경우, 또는 타일이 작아서 2-stage로도 latency hiding이 충분한 경우.
- **`cp.async`**: Ampere 이상에서 최대 성능을 추구하는 경우. 특히 large tile + high arithmetic intensity 커널에서 효과가 극대화된다.

---

## 예시 코드

세 커널 모두 동일한 tiled SGEMM (C = A × B, FP32)을 수행하며 파이프라이닝 전략만 다르다.

### 파일 구성

```
.
├── common.cuh                  # 공통 유틸 (init, verify, benchmark harness)
├── baseline_no_pipeline.cu     # 파이프라이닝 없는 baseline
├── ping_pong.cu                # Double buffering (ping-pong)
├── cp_async.cu                 # cp.async N-stage pipeline
├── Makefile                    # 빌드 / 실행 / 프로파일링 타겟
└── README.md
```

### 커널 파라미터

`common.cuh`에 정의된 공통 설정:

| Parameter | Value | 비고 |
|-----------|-------|------|
| M, N, K | 2048 | 행렬 크기 |
| Block tile | 128×128×8 | BLOCK_TILE_M × BLOCK_TILE_N × BLOCK_TILE_K |
| Thread block | 16×16 (256 threads) | BLOCK_DIM_X × BLOCK_DIM_Y |
| Thread tile | 8×8 | 각 스레드가 담당하는 출력 크기 |
| K tiles | 256 | K / BLOCK_TILE_K |

### 빌드

```bash
# 전체 빌드 (Makefile 사용)
make all

# 또는 개별 빌드
nvcc -O3 -arch=sm_80 -std=c++17 --expt-relaxed-constexpr -o baseline baseline_no_pipeline.cu
nvcc -O3 -arch=sm_80 -std=c++17 --expt-relaxed-constexpr -o ping_pong ping_pong.cu
nvcc -O3 -arch=sm_80 -std=c++17 --expt-relaxed-constexpr -DSTAGES=2 -o cp_async cp_async.cu
```

> `sm_80` 이상 필수 (`cp.async` 지원). 다른 Ampere/Hopper GPU의 경우 `sm_86`, `sm_89`, `sm_90` 등으로 변경.

### 실행

```bash
# 전체 벤치마크
make run

# 또는 개별 실행
./artifacts/baseline
./artifacts/ping_pong
./artifacts/cp_async
```

각 바이너리는 CPU reference 계산 → 커널 벤치마크 (warmup 5회 + 추가 warmup 10회 + 측정 20회) → 정합성 검증 순으로 실행된다.

---

## 벤치마크 결과

측정 환경:

- GPU: NVIDIA RTX A6000 (CC 8.6)
- 문제 크기: `M=N=K=2048`
- 타일: block `128x128x8`, thread `8x8`
- `cp.async` 설정: `STAGES=2`
- 정확성: 세 커널 모두 `Correct: YES`

| Kernel | Avg Time (ms) | GFLOPS | Speedup vs Baseline | 해석 |
|--------|---------------|--------|---------------------|------|
| Baseline (No Pipeline) | 2.296 | 7484.0 | 1.00x | load와 compute가 완전히 직렬화됨 |
| Ping-Pong (Double Buffering) | 2.189 | 7846.8 | 1.049x | load/compute overlap으로 소폭 개선 |
| `cp.async` 2-Stage | 1.908 | 9003.3 | 1.203x | direct async copy로 latency hiding 효과가 가장 큼 |

핵심 포인트:

- Ping-pong은 baseline 대비 **약 4.8%** 빨라졌다. 파이프라인을 도입했지만, global load latency를 완전히 가리지는 못했다.
- `cp.async` 2-stage는 baseline 대비 **약 20.3%**, ping-pong 대비 **약 14.7%** 빨랐다.
- 여기서 중요한 점은 `cp.async`가 **3-stage 이상의 깊은 파이프라인 없이도**, 동일한 2-stage depth에서 ping-pong을 확실히 앞섰다는 것이다.

---

## NCU Profiling 결과 비교

### 프로파일링 명령

```bash
# Makefile 타겟으로 전체 프로파일링
make profile

# PTX/SASS 덤프
make ptx
make sass
```

### 주요 메트릭 비교

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

### 상세 분석

**1. Ping-Pong의 개선폭이 작은 이유**

Ping-pong은 baseline 대비 time 기준으로는 4.8% 빨라졌지만, `Stall Long Scoreboard`는 `0.95 → 0.93`으로 거의 줄지 않았다. 즉, double buffering을 넣어도 **실제 long-latency global load dependency가 크게 해소되지는 않았고**, 컴파일러가 배치한 `LDG` / `STS` overlap이 제한적으로만 작동했다는 뜻이다.

또한 `Stall Barrier`가 `0.21 → 0.29`로 증가했다. Ping-pong은 load와 compute를 겹치려 하지만, 여전히 `__syncthreads()` 하나로 두 phase를 함께 fence하기 때문에 overlap으로 얻은 이득 일부를 barrier 비용이 다시 가져간다.

**2. `cp.async` 2-stage가 가장 빠른 핵심 이유**

이번 측정의 `cp.async`는 3-stage가 아니라 **2-stage**다. 따라서 이 결과는 "더 깊은 파이프라인이라서 빠르다"라기보다, **같은 2-stage depth에서도 `cp.async` 경로가 ping-pong보다 구조적으로 더 낫다**는 증거에 가깝다.

가장 결정적인 지표는 `Stall Long Scoreboard`다:

- Baseline: `0.95`
- Ping-pong: `0.93`
- `cp.async`: `0.19`

즉 `cp.async`는 baseline 대비 **약 80%**, ping-pong 대비 **약 79.6%** long scoreboard stall을 줄였다. Source Counters의 stall sampling도 같은 결론을 보여준다:

- `stall_long_sb` samples: `18,109 → 17,986 → 3,654`

이와 함께 scheduler 관점의 readiness도 좋아진다:

- `Issued Warp / Scheduler`: `0.37 → 0.40 → 0.46`
- `Eligible Warps / Scheduler`: `0.42 → 0.46 → 0.53`
- `No Eligible`: `62.56% → 59.59% → 53.54%`
- `Warp Cycles / Issued Instruction`: `5.33 → 4.94 → 4.30`

즉 `cp.async`의 이득은 occupancy 증가가 아니라, **기존에 resident 하던 warp들이 더 자주 "발행 가능한 상태"가 되도록 만든 것**에서 나온다.

**3. 이 커널은 DRAM bandwidth-bound라기보다 latency/scheduling-bound다**

세 커널 모두 `DRAM Throughput`은 `6.3~6.7%`, `DRAM Bandwidth`는 `46~49 GB/s` 수준에 머문다. 반면 `Compute (SM) Throughput`은 `28.71% → 35.60%`, `Memory Throughput`은 `37.30% → 43.87%`로 함께 올라간다.

즉 이 SGEMM은 **HBM 대역폭을 꽉 채워서 느린 상태가 아니라**, load latency와 warp issue inefficiency 때문에 느린 상태다. 파이프라이닝이 효과적인 이유도 raw bandwidth를 늘려서가 아니라, **이미 존재하는 데이터 이동 시간을 compute 뒤로 숨겨서 SM을 더 꾸준히 바쁘게 만들기 때문**이다.

**4. Occupancy는 사실상 동일하다**

세 커널의 achieved occupancy는 모두 약 `16.6%`로 거의 같다. `Block Limit Registers = 1`, `Block Limit Shared Mem = 1`이 함께 잡혀 있어, 현재 타일 크기에서는 사실상 **SM당 1 block**만 상주하는 형태다.

그래서 이 README의 측정 결과는 "더 많은 block/warp를 올려서 이긴 것"이 아니라 "같은 occupancy에서도 파이프라인 설계가 warp stall을 얼마나 줄였는가"를 보여주는 데이터로 읽는 것이 맞다.

**5. `cp.async`가 모든 병목을 해결한 것은 아니다**

프로파일링 결과를 보면, 공통으로 남아 있는 문제도 분명하다.

- 세 커널 모두 global access의 excessive sectors 비율이 `18%`
- baseline / ping-pong의 excessive shared wavefronts 비율이 `38%`
- `cp.async`는 오히려 excessive shared wavefronts 비율이 `42%`로 증가

즉 `cp.async`의 성능 향상은 **shared-memory access pattern이 좋아져서** 나온 것이 아니다. 오히려 shared-side 비효율은 그대로 남아 있고, 일부는 더 악화되었다. 그럼에도 전체 성능이 오른 이유는, global-to-shared staging latency를 구조적으로 줄인 효과가 그 손실보다 컸기 때문이다.

다음 최적화 단계는 파이프라인 depth를 더 늘리는 것보다 먼저, **shared-memory bank conflict를 줄이기 위한 padding / swizzle / vectorized load-store 재배치**를 검토하는 쪽이 더 직접적이다.

**6. "register bypass"와 "register/thread 감소"는 같은 말이 아니다**

개념적으로 `cp.async`는 `LDG → register → STS` 경유 복사가 아니라 Global → Shared direct copy이므로, **복사 그 자체에 대한 explicit intermediate register 의존성**은 제거된다.

하지만 실제 컴파일 결과에서 `Registers / Thread`는 다음과 같다:

- Baseline: `146`
- Ping-pong: `162`
- `cp.async`: `254`

즉 이 구현에서는 `cp.async`가 오히려 가장 큰 register footprint를 가진다. 이는 stage 인덱싱, 주소 계산, prologue/epilogue 관리, aggressive unrolling, 컴파일러 스케줄링이 전체 register 사용량을 지배하고 있음을 보여준다.

따라서 "`cp.async`는 register bypass라서 occupancy가 무조건 좋아진다"라고 일반화하면 틀리다. 정확한 표현은:

- `cp.async`는 **copy 경로 상의 임시 register 의존성을 줄인다**
- 그러나 **커널 전체 register/thread는 별개의 문제**이며, 실제 구현/컴파일 결과를 반드시 확인해야 한다

이다.

### SASS 수준 분석

```bash
rg -n "ld\\.global|st\\.shared|bar\\.sync" artifacts/baseline.ptx artifacts/ping_pong.ptx
rg -n "cp\\.async|commit_group|wait_group|bar\\.sync" artifacts/cp_async.ptx
rg -n "LDG|STS|BAR\\.SYNC" artifacts/baseline.sass artifacts/ping_pong.sass
rg -n "LDGSTS|LDGDEPBAR|DEPBAR|BAR\\.SYNC" artifacts/cp_async.sass
```

- `baseline.ptx`, `ping_pong.ptx`에서는 반복적으로 `ld.global.nc.f32` 다음에 `st.shared.f32`가 나오고, tile 경계마다 `bar.sync 0`가 등장한다.
- `cp_async.ptx`에서는 `cp.async.ca.shared.global`, `cp.async.commit_group`, `cp.async.wait_group 1`, `bar.sync 0`가 명확히 보인다.

SASS에서는 차이가 더 선명하다:

- Baseline / Ping-pong: `LDG.E.CONSTANT` + `STS` + `BAR.SYNC.DEFER_BLOCKING`
- `cp.async`: `LDGSTS.E` + `LDGDEPBAR` + `DEPBAR.LE SB0, ...` + `BAR.SYNC.DEFER_BLOCKING`

즉 Ampere에서 PTX의 `cp.async`는 SASS에서 literal `CP.ASYNC` 텍스트가 아니라 **`LDGSTS` 계열 명령과 dependency barrier 시퀀스**로 내려간다. 현재 artifact 기준으로는 다음과 같이 해석하면 된다:

- Ping-pong은 여전히 **분리된 global load + shared store**
- `cp.async`는 **fused/asynchronous global-to-shared staging**

따라서 `cp.async` 커널의 우위는 단순한 소스 코드 재배치가 아니라, **명령어 수준에서 다른 데이터 이동 메커니즘**을 사용한 결과라고 볼 수 있다.

### 종합 결론

이 저장소의 측정 결과는 다음 한 줄로 요약할 수 있다:

> **동일한 2-stage depth에서는, ping-pong보다 `cp.async`가 훨씬 더 강하게 global-memory latency를 숨긴다.**

Ping-pong은 분명 baseline보다 낫지만, `LDG`와 `STS`가 여전히 분리되어 있고 barrier에 묶여 있어 long scoreboard stall을 근본적으로 줄이지 못했다. 반면 `cp.async`는 direct async copy + commit/wait 모델 덕분에 resident warp의 issue readiness를 실제로 개선했고, 그 결과 throughput이 compute/memory 양쪽에서 함께 상승했다.

남은 병목은 분명하다:

1. shared-memory bank conflict와 excessive wavefronts
2. 낮은 occupancy (현재 타일에서는 사실상 1 block/SM)
3. `cp.async` 구현의 높은 register/thread

즉 다음 실험은 `STAGES=3`만 올려보는 것보다, **shared layout 개선과 register footprint 제어를 먼저 수행한 뒤 stage 수를 다시 스윕**하는 순서가 더 타당하다.

---

## References

- [CUDA Programming Guide — Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [CUDA Programming Guide — Asynchronous Data Copies (legacy)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- [CUTLASS Tutorial: Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
