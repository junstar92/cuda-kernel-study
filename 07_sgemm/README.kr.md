# CUDA Kernel Study: SGEMM Optimization

이 README는 Nsight Compute 프로파일링과 benchmark 측정값을 바탕으로, [`how_to_optimize_sgemm.kr.md`](./how_to_optimize_sgemm.kr.md)에서 제시한 최적화 흐름에 맞춰 각 SGEMM 커널이 어떻게 진화하는지 정리한다. 핵심 관심사는 커널별 절대 수치 자체보다, 어떤 구조 변화가 큰 성능 점프를 만들었고 병목이 어떻게 이동했는지 추적하는 데 있다.

현재 최고점은 `sgemm_warptiling_v4<128, 128, 16, 32, 64, 16>`의 `20.99 TFLOPS`다. 이후 `v5`에서는 CUTLASS-style epilogue까지 실험했지만, store path는 좋아졌어도 shared-memory residency 비용 때문에 이 최고점을 넘지 못했다. 아래 내용은 naive 커널에서 시작해 shared tiling, register tiling, 2D outer product, vec4 ingest, warptiling refinement, 그리고 epilogue 실험까지 이어지는 과정을 순서대로 정리한 것이다.

분석 대상 커널은 아래 순서를 따른다.

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

또한 `v5` 섹션에서는 default 구현과 별도로 epilogue address-generation 패턴 변경 실험을 추가 비교한다.

## 분석 기준

- GPU: NVIDIA RTX A6000, SM86
- 문제 크기: `M=N=K=4096`
- 프로파일링: `ncu --set full` 기반 Nsight Compute 측정
- 커널 구현 위치
  - [`01_naive.cuh`](./01_naive.cuh)
  - [`02_smem_tiling.cuh`](./02_smem_tiling.cuh)
  - [`03_blocktiling_1d.cuh`](./03_blocktiling_1d.cuh)
  - [`04_blocktiling_2d.cuh`](./04_blocktiling_2d.cuh)
  - [`05_blocktiling_2d_vec4.cuh`](./05_blocktiling_2d_vec4.cuh)
  - [`06_warptiling.cuh`](./06_warptiling.cuh)
  - 보조 load/store helper: [`utils.cuh`](./utils.cuh)
- 해석 기준 문서: [`how_to_optimize_sgemm.kr.md`](./how_to_optimize_sgemm.kr.md)

`how_to_optimize_sgemm.kr.md`의 핵심 흐름은 크게 다음과 같다.

1. global memory만 쓰는 naive 커널을 만든다.
2. shared memory tiling으로 global reload를 줄인다.
3. register tiling으로 thread당 output reuse를 늘린다.
4. 1D accumulate를 2D outer-product 형태로 바꿔 ILP를 늘린다.
5. `BLOCK_K`를 키워 barrier/loop overhead를 줄인다.
6. padding과 vectorized load/store로 memory path를 정리한다.
7. warp tiling으로 `threadblock -> warp -> thread` 계층을 명시적으로 만든다.
8. global->shared staging을 thread-private register fragment로 분리해 load/store scheduling을 더 명시적으로 만든다.

주의할 점은, 이번에 분석하는 `blocktiling_*` 커널들은 이미 template 마지막 인자로 `4`를 넣고 있어 `A` shared-memory layout padding 효과가 포함되어 있다는 점이다. 즉 `blocktiling_1d`, `blocktiling_2d`의 성능 향상은 순수한 register/block tiling만의 결과가 아니라, layout 정리까지 함께 반영된 결과다.

## 한눈에 보는 성능 흐름

| Kernel | 핵심 아이디어 | Elapsed (ms) | TFLOPS | 대표 병목 |
| --- | --- | ---: | ---: | --- |
| `sgemm_naive_row` | global-only baseline | 67.32 | 2.04 | `LG Throttle`, low issue rate |
| `sgemm_smem_tiling` | shared memory tiling | 46.86 | 2.93 | `MIO Throttle`, barrier, shared-store conflict |
| `blocktiling_1d<64,64,8,8,4>` | 1D register tiling 시작 | 22.04 | 6.23 | long scoreboard, MIO, barrier |
| `blocktiling_1d<64,64,8,16,4>` | thread tile 확대 | 19.36 | 7.10 | scoreboard dependency |
| `blocktiling_1d<128,128,8,64,4>` | CTA tile 확대 + 큰 register tile | 11.84 | 11.61 | short scoreboard, MIO |
| `blocktiling_1d<128,128,16,64,4>` | `BLOCK_K=16` | 11.46 | 11.99 | short scoreboard, 일부 shared overhead |
| `blocktiling_2d<128,128,8,8,8,4>` | 2D outer product | 8.56 | 16.05 | shared wavefront excess |
| `blocktiling_2d<128,128,16,8,8,4>` | 2D + `BLOCK_K=16` | 7.90 | 17.39 | shared wavefront excess, not selected |
| `blocktiling_2d_vec4<128,128,8,8,8,4>` | vec4 global path | 8.15 | 16.86 | shared wavefront excess |
| `blocktiling_2d_vec4<128,128,16,8,8,4>` | vec4 + `BLOCK_K=16` | 7.55 | 18.20 | shared wavefront excess |
| `warptiling_v0<128,128,8,32,64,8>` | 첫 warp tiling, wide-N warp tile | 8.50 | 16.16 | shared bank conflict, uncoalesced store |
| `warptiling_v0<128,128,8,64,32,8>` | 첫 warp tiling, wide-M warp tile | 8.15 | 16.86 | uncoalesced store, register-limited occupancy |
| `warptiling_v1<128,128,8,32,64,8>` | refined lane mapping + `4x4` island MMA/store | 7.95 | 17.28 | register-limited occupancy, scheduler/not selected |
| `warptiling_v1<128,128,8,64,32,8>` | refined lane mapping + `4x4` island MMA/store | 7.91 | 17.37 | register-limited occupancy, scheduler/not selected |
| `warptiling_v2<128,128,8,32,64,8>` | explicit register-staged global->shared | 7.53 | 18.25 | register-limited occupancy, scheduler/not selected |
| `warptiling_v2<128,128,8,64,32,8>` | explicit register-staged global->shared | 7.69 | 17.86 | register-limited occupancy, scheduler/not selected |
| `warptiling_v2<128,128,16,32,64,16>` | register-staged global->shared + `BLOCK_K=16` | 7.18 | 19.13 | scheduler/not selected, uncoalesced global store |
| `warptiling_v3<128,128,8,32,64,8>` | 1-stage register prefetch | 7.34 | 18.72 | scheduler/not selected, residual scoreboard |
| `warptiling_v3<128,128,8,64,32,8>` | 1-stage register prefetch | 7.29 | 18.84 | scheduler/not selected, residual scoreboard |
| `warptiling_v3<128,128,16,32,64,16>` | prefetch + `BLOCK_K=16` | 8.01 | 17.15 | register cliff, scheduler starvation |
| `warptiling_v4<128,128,8,32,64,8>` | vec4 global ingest + prefetch skeleton 유지 | 6.79 | 20.22 | scheduler/not selected, residual scalar epilogue |
| `warptiling_v4<128,128,8,64,32,8>` | vec4 global ingest + prefetch skeleton 유지 | 6.86 | 20.04 | scheduler/not selected, residual scalar epilogue |
| `warptiling_v4<128,128,16,32,64,16>` | vec4 global ingest + `BLOCK_K=16` | 6.55 | 20.99 | scheduler/not selected, residual scalar epilogue |
| `warptiling_v5<128,128,8,32,64,8>` | CUTLASS-style epilogue shared staging + cooperative vec4 store | 7.50 | 18.33 | shared-memory cliff, reduced issue rate |
| `warptiling_v5<128,128,8,64,32,8>` | CUTLASS-style epilogue shared staging + cooperative vec4 store | 7.51 | 18.29 | shared-memory cliff, reduced issue rate |
| `warptiling_v5<128,128,16,32,64,16>` | CUTLASS-style epilogue shared staging + `BLOCK_K=16` | 6.91 | 19.88 | shared-memory cliff, reduced issue rate |

전체 흐름을 한 줄로 요약하면 다음과 같다.

- 병목 이동: `LG Throttle -> MIO Throttle + barrier -> scoreboard -> shared wavefront excess -> scheduler/not selected + residual scoreboard -> BK16 register cliff(v3) -> vec4 ingest로 회복된 BK16 + residual scalar epilogue -> store path는 정리되지만 shared-memory cliff를 밟은 CUTLASS-style epilogue(v5)`
- 최종 성능: `2.04 -> 20.99 TFLOPS`, 약 `10.3x`
- 현재 최고점: `warptiling_v4<128x128x16, 32x64, BK=16>`의 `20.99 TFLOPS`, `6.55 ms`

> 큰 성능 점프는 shared tiling 자체보다, shared reuse를 register reuse로 확장한 시점과 1D accumulate를 2D outer-product로 재구성한 시점에서 나왔다.
>
> `v5`는 scalar epilogue inefficiency가 실제로 존재함을 확인해 주지만, full CTA epilogue staging은 현재 `128x128` family에서 residency 비용이 더 컸다.

## 1. `sgemm_naive_row`

### 구조

`01_naive.cuh`의 `sgemm_naive_row`는 thread 하나가 `C[row, col]` 하나를 끝까지 계산하는 가장 단순한 형태다.

- 각 loop iteration마다 `A[row, k]`와 `B[k, col]`를 global memory에서 직접 읽는다.
- thread는 output accumulator 하나만 가진다.
- `K` 방향 reuse가 전혀 없으므로, 같은 `A` row와 `B` col이 여러 thread에 의해 반복적으로 DRAM/L2/L1 경유로 다시 읽힌다.

### NCU 핵심 수치

- Duration: `78.30 ms`
- Throughput: `2.04 TFLOPS`
- Achieved occupancy: `66.66%`
- Eligible warps per scheduler: `1.16`
- Issued warps per scheduler: `0.27`
- No eligible: `73.08%`
- 대표 stall: `Stall LG Throttle 18.32`, `Long Scoreboard 3.02`

### 해석

이 단계는 `occupancy`가 아주 낮아서 느린 것이 아니다. 실제로 occupancy는 66% 수준이다. 문제는 scheduler가 issue할 warp를 자주 찾지 못한다는 점이다.

- `Issued Warp 0.27`은 cycle당 instruction 발행이 거의 안 된다는 뜻이다.
- `No Eligible 73.08%`는 대부분의 warp가 메모리 응답을 기다리거나 다음 명령을 바로 못 내는 상태라는 의미다.
- `LG Throttle`이 가장 크다는 것은 global memory/L1TEX load instruction 경로가 과도하게 압박받고 있음을 보여준다.

즉 출발점의 병목은 "연산량이 적어서"가 아니라, `A`와 `B`를 너무 자주 global에서 다시 읽는 접근 구조 자체다. `how_to_optimize_sgemm.kr.md`의 출발점과 정확히 일치한다.

## 2. `sgemm_smem_tiling`

### 구조

`02_smem_tiling.cuh`는 `BM x BK`, `BK x BN` 타일을 shared memory에 올린 뒤, block 안의 thread들이 이를 재사용한다.

- 한 번 global에서 읽은 `A`, `B` tile을 CTA 내부에서 여러 thread가 공유한다.
- `K`를 `BK` 단위로 쪼개며 `load -> sync -> compute -> sync` 흐름을 반복한다.
- global reload는 줄지만, 이제 shared memory load/store와 `__syncthreads()`가 새 hot path가 된다.

### NCU 핵심 수치

- Duration: `60.20 ms`
- Throughput: `2.93 TFLOPS`
- Achieved occupancy: `66.66%`
- Eligible warps per scheduler: `0.89`
- Issued warps per scheduler: `0.20`
- No eligible: `79.79%`
- 대표 stall: `Stall MIO Throttle 23.77`, `Long Scoreboard 5.97`, `Barrier 2.76`
- Nsight 경고: shared store 평균 `1.2-way` bank conflict, bank conflicts `29,216,048`

### 해석

shared memory tiling은 분명히 맞는 방향이지만, 이 구현은 아직 큰 block과 높은 synchronization 비용을 동반한다.

- `2.04 -> 2.93 TFLOPS`로 개선은 있지만 폭이 크지 않다.
- `LG Throttle` 중심 병목은 줄었지만, 대신 `MIO Throttle`이 크게 튀어나온다.
- 이는 shared-memory instruction 파이프가 병목으로 전환되었다는 뜻이다.
- `Barrier 2.76`은 각 `BK` 타일마다 필요한 sync가 실질 비용으로 보인다는 증거다.
- shared-store bank conflict 경고는 메모리 reuse를 얻는 대신 shared path 효율이 아직 깨끗하지 않음을 보여준다.

즉 이 단계는 "global 병목을 shared 병목으로 옮겨 놓은 단계"다. 다음 단계에서 필요한 것은 shared tile을 더 많이 재사용하도록 thread당 output 수를 늘리는 것이다.

## 3. `sgemm_blocktiling_1d<64, 64, 8, 8, 4>`

### 구조

`03_blocktiling_1d.cuh`의 1D block tiling은 thread가 output 하나가 아니라 `THREAD_M`개를 register에 들고 간다.

- CTA는 `64 x 64` C tile을 담당한다.
- `BLOCK_K=8`씩 shared memory에 올린다.
- 각 thread는 `THREAD_M=8`개의 output row strip을 계산한다.
- `A`는 shared memory에서 transpose된 형태로 다루고, padding `4`가 들어간다.

### NCU 핵심 수치

- Duration: `22.47 ms`
- Throughput: `6.23 TFLOPS`
- Registers per thread: `41`
- Threads per block: `512`
- Achieved occupancy: `66.31%`
- Issued warps per scheduler: `0.32`
- No eligible: `68.21%`
- 대표 stall: `Long Scoreboard 7.49`, `MIO Throttle 7.30`, `Barrier 6.34`
- `L1 Wavefronts Shared Excessive = 0`

### 해석

여기서 첫 번째 큰 점프가 나온다. `2.93 -> 6.23 TFLOPS`다.

핵심 이유는 다음 두 가지다.

1. shared에서 읽은 `B[k, col]` 값을 한 thread가 여러 output 계산에 재사용한다.
2. accumulator를 register에 여러 개 두면서 shared-load당 FMA 수가 크게 늘어난다.

즉 `sgemm_smem_tiling`이 "shared로 옮겼지만 reuse가 아직 thread 수준에서 작았던 구조"였다면, 이 단계는 reuse를 register level까지 끌고 들어온 단계다.

또 하나 중요한 점은 padding 효과다.

- 이번 분석 대상인 `pad4` 버전은 `L1 Wavefronts Shared Excessive = 0`이다.
- 비교한 no-pad 변형에서는 excessive wavefront가 크게 나타난다.

즉 이 커널의 개선은 register tiling 효과뿐 아니라, shared-memory layout 정리 효과도 포함하고 있다.

남은 병목은 `Long Scoreboard`, `MIO Throttle`, `Barrier`다. shared에서 읽은 뒤 곧바로 scalar accumulation을 이어가는 구조라 load-use dependency가 아직 길다.

## 4. `sgemm_blocktiling_1d<64, 64, 8, 16, 4>`

### 구조

이 버전은 CTA tile과 `BLOCK_K`는 그대로 두고, `THREAD_M`만 `8 -> 16`으로 늘린다.

- thread당 accumulator 수가 증가한다.
- 같은 output tile을 계산하는 데 필요한 thread 수는 줄어든다.
- shared에서 읽은 operand를 thread 안에서 더 많이 소비한다.

### NCU 핵심 수치

- Duration: `19.35 ms`
- Throughput: `7.10 TFLOPS`
- Registers per thread: `55`
- Threads per block: `256`
- Achieved occupancy: `65.68%`
- Issued warps per scheduler: `0.35`
- No eligible: `65.20%`
- 대표 stall: `Long Scoreboard 8.62`, `Barrier 6.86`, `Short Scoreboard 2.47`, `MIO Throttle 2.38`
- `L1 Wavefronts Shared Excessive = 0`

### 해석

이 단계는 "더 큰 thread tile이 항상 유리한가"를 보여주는 작은 실험이다.

- 성능은 `6.23 -> 7.10 TFLOPS`로 개선된다.
- `MIO Throttle`이 `7.30 -> 2.38`로 크게 줄어 shared instruction 압박이 완화된다.
- 반면 register 수가 `41 -> 55`로 증가하고, `Long Scoreboard`는 오히려 `8.62`로 더 커진다.

즉 thread당 reuse는 늘었지만, dependency chain도 길어졌다. 여기서 보이는 패턴은 다음과 같다.

- 작은 tile: instruction 수는 많고 reuse가 부족하다.
- 큰 tile: reuse는 좋지만 register dependency가 길어진다.

이 커널은 아직 긍정적인 쪽이 더 크다. 하지만 이제 다음 개선은 단순히 `THREAD_M`만 계속 키우는 방향이 아니라, CTA tile 자체를 키워 block-level reuse를 더 키우는 쪽으로 가야 한다.

## 5. `sgemm_blocktiling_1d<128, 128, 8, 64, 4>`

### 구조

여기서는 CTA tile을 `64x64`에서 `128x128`로 키우고, thread당 `64`개 accumulator를 사용한다.

- 한 CTA가 더 큰 `C` tile을 계산한다.
- shared에 올린 operand tile이 더 많은 output에 기여한다.
- thread당 work가 커지면서 occupancy는 줄어들지만 ILP와 reuse는 커진다.

### NCU 핵심 수치

- Duration: `13.39 ms`
- Throughput: `11.61 TFLOPS`
- Registers per thread: `116`
- Threads per block: `256`
- Achieved occupancy: `32.32%`
- Eligible warps per scheduler: `0.83`
- Issued warps per scheduler: `0.49`
- No eligible: `51.04%`
- 대표 stall: `MIO Throttle 2.28`, `Short Scoreboard 2.30`, `Long Scoreboard 0.71`, `Barrier 0.34`

### 해석

겉으로 보면 occupancy가 `65% -> 32%`로 크게 떨어지므로 불리해 보인다. 하지만 실제 성능은 `7.10 -> 11.61 TFLOPS`로 더 오른다.

이 결과는 `how_to_optimize_sgemm.kr.md`의 중요한 메시지를 잘 보여준다.

- 높은 occupancy가 항상 더 빠른 것은 아니다.
- 충분한 reuse와 ILP를 만들면, warp 수가 줄어도 scheduler efficiency는 오를 수 있다.

실제로 이 커널은 다음 점에서 훨씬 건강하다.

- `Issued Warp 0.49`로 발행 효율이 크게 오른다.
- `Warp Cycles Per Issued Instruction`도 크게 줄어든다.
- `Long Scoreboard 0.71`은 이전 64x64 계열보다 훨씬 낮다.
- `Barrier 0.34`로 sync 오버헤드도 줄어든다.

즉 큰 CTA tile이 shared tile reuse를 충분히 끌어올려, 낮은 occupancy 손해를 상쇄하고도 남는 단계다.

## 6. `sgemm_blocktiling_1d<128, 128, 16, 64, 4>`

### 구조

이 버전은 앞선 128x128 1D 구조를 유지한 채 `BLOCK_K`를 `8 -> 16`으로 늘린다.

- loop iteration 수가 줄어든다.
- `__syncthreads()` 호출 횟수도 줄어든다.
- 한 번의 타일 로드 이후 더 많은 compute를 진행한다.

### NCU 핵심 수치

- Duration: `13.04 ms`
- Throughput: `11.99 TFLOPS`
- Registers per thread: `116`
- Achieved occupancy: `32.36%`
- Issued warps per scheduler: `0.50`
- No eligible: `50.18%`
- 대표 stall: `Short Scoreboard 2.20`, `MIO Throttle 2.54`, `Long Scoreboard 0.46`, `Barrier 0.23`

### 해석

개선 폭은 `11.61 -> 11.99 TFLOPS`로 크지 않지만, 방향은 명확하다.

- `Barrier 0.34 -> 0.23`으로 감소한다.
- `Long Scoreboard 0.71 -> 0.46`로 더 줄어든다.
- `MIO Throttle`도 완화된다.

즉 `BLOCK_K` 확대는 이미 reuse 구조가 잘 잡힌 상태에서 barrier와 loop overhead를 줄이는 보정으로 작동한다. 반면 shared-memory footprint는 커지므로, 초기 단계에서 무작정 `BLOCK_K`부터 키우는 것은 정답이 아니었을 것이다.

이 단계는 "큰 구조 최적화 이후의 미세 조정"에 가깝다.

## 7. `sgemm_blocktiling_2d<128, 128, 8, 8, 8, 4>`

### 구조

`04_blocktiling_2d.cuh`는 1D strip accumulate에서 2D outer-product 형태로 넘어간다.

- thread는 `THREAD_M x THREAD_N = 8 x 8` 크기의 output tile을 register에 갖는다.
- inner loop에서 `A` fragment와 `B` fragment를 각각 row/col fragment로 읽어 outer-product를 만든다.
- 총 accumulator 수는 64개로 1D `THREAD_M=64`와 같지만, operand 하나가 열어 주는 즉시 실행 가능한 FMA 묶음의 모양이 달라진다.

### NCU 핵심 수치

- Duration: `8.98 ms`
- Throughput: `16.05 TFLOPS`
- Registers per thread: `104`
- Achieved occupancy: `32.34%`
- Eligible warps per scheduler: `1.60`
- Issued warps per scheduler: `0.62`
- No eligible: `38.16%`
- 대표 stall: `Not Selected 1.59`, `Short Scoreboard 0.87`, `Long Scoreboard 0.84`, `Barrier 0.62`
- `L1 Wavefronts Shared: 704,643,072`
- Ideal shared wavefronts: `436,207,616`
- Shared excessive wavefronts: `268,435,456`

### 해석

이 단계가 두 번째 큰 전환점이다. `11.99 -> 16.05 TFLOPS`다.

1D register tiling은 reuse를 늘렸지만, accumulate가 한 방향 strip에 치우쳐 있었다. 2D outer-product로 바뀌면 같은 64개 accumulator를 더 scheduler-friendly한 형태로 갱신할 수 있다.

#### 왜 accumulator가 64개로 같아도 2D가 더 빠른가

이 지점은 `1d<128,128,16,64,4>`와 `2d<128,128,16,8,8,4>`를 비교해서 보는 것이 가장 명확하다.

- 둘 다 CTA tile은 `128 x 128`이다.
- 둘 다 `BLOCK_K=16`이다.
- 둘 다 thread당 accumulator 수는 `64`개다.
- 둘 다 block당 thread 수는 `256`개다.

즉 차이는 "연산 개수"보다 "같은 64개 결과를 만들 때 어떤 입력 조각을 준비해야 하느냐"에 가깝다.

1D kernel의 inner loop는 개념적으로 아래와 비슷하다.

```cpp
b = B[k, col]
for m in 0..63:
  a = A[k, row + m]
  acc[m] += a * b
```

이 구조에서는 `b` 하나를 64번 재사용한다는 장점이 있다. 하지만 같은 64개 accumulator update를 만들기 위해 thread는 `B` 1개와 `A` 64개를 순서대로 써야 한다. 즉 `B` reuse는 아주 좋지만, `A`는 거의 재사용되지 않는다. 계산 모양도 "B 하나에 A를 계속 갈아 끼우는 긴 1줄짜리 dot-product"에 가깝다.

2D kernel의 inner loop는 개념적으로 아래와 비슷하다.

```cpp
for m in 0..7:
  a = A[k, row + m]
  for n in 0..7:
    b = B[k, col + n]
    acc[m][n] += a * b
```

여기서는 같은 64개 accumulator update를 만들기 위해 `A` 8개와 `B` 8개만 준비하면 된다. 즉 `8 + 8 = 16`개의 operand 조각으로 `8 x 8 = 64`개의 FMA를 만든다. 계산 모양도 "긴 한 줄"이 아니라 "작은 곱셈표"에 가깝다.

같은 내용을 더 간단히 표로 쓰면 아래와 같다.

| 형태 | 결과 수 | 필요한 A 조각 | 필요한 B 조각 | 합계 |
| --- | ---: | ---: | ---: | ---: |
| `64 x 1` 1D | `64` | `64` | `1` | `65` |
| `8 x 8` 2D | `64` | `8` | `8` | `16` |

핵심은 2D가 "A도 재사용하고 B도 재사용한다"는 점이다. 1D는 한쪽(`B`) reuse에 치우친 구조이고, 2D는 양쪽 operand reuse가 동시에 생긴다. 그래서 operand가 한 번 준비되면 그 조합으로 연속해서 여러 FMA를 issue하기가 더 쉬워진다.

이 관점에서 보면 2D의 이점은 "accumulator 수가 더 많아서"가 아니라, "같은 64개 결과를 훨씬 작은 operand 묶음으로 만들 수 있어서"라고 이해하는 편이 더 정확하다.

NCU 수치도 이 해석을 지지한다. `1d<128,128,16,64,4>`에서 `2d<128,128,16,8,8,4>`로 바꾸면:

- Throughput: `11.99 -> 17.39 TFLOPS`
- Eligible warps per scheduler: `0.90 -> 1.90`
- Issued warps per scheduler: `0.50 -> 0.66`
- No eligible: `50.18% -> 34.02%`
- Short scoreboard: `2.20 -> 0.63`

특히 `Short Scoreboard`가 크게 줄어든 것이 중요하다. 이건 2D가 메모리 access 자체를 더 완벽하게 만든다는 뜻이 아니라, load 뒤에 바로 막히는 dependency 패턴을 많이 완화했다는 뜻이다.

이 효과는 scheduler 지표에 그대로 드러난다.

- `Eligible Warp 1.60`은 이전 1D보다 훨씬 높다.
- `Issued Warp 0.62`도 크게 상승한다.
- `No Eligible 38.16%`로 줄어든다.

즉 scheduler가 "할 일이 없는 상태"에서 많이 벗어난다.

다만 새로운 병목도 분명하다.

- shared wavefront가 ideal보다 크게 많다.
- excessive wavefront가 `268,435,456`으로 크다.

즉 compute 구조는 많이 좋아졌지만, shared-memory load access pattern은 아직 완전히 정돈되지 않았다. 이 시점부터 핵심 병목은 compute 자체보다 shared-memory path의 비효율 쪽으로 이동한다.

## 8. `sgemm_blocktiling_2d<128, 128, 16, 8, 8, 4>`

### 구조

2D outer-product 구조 위에서 `BLOCK_K=16`을 적용한 버전이다.

- 2D register blocking의 장점은 유지한다.
- loop와 barrier 횟수는 줄인다.
- 한 번 shared에 올린 타일을 더 길게 사용한다.

### NCU 핵심 수치

- Duration: `8.25 ms`
- Throughput: `17.39 TFLOPS`
- Registers per thread: `108`
- Achieved occupancy: `32.35%`
- Eligible warps per scheduler: `1.90`
- Issued warps per scheduler: `0.66`
- No eligible: `34.02%`
- 대표 stall: `Not Selected 1.89`, `MIO Throttle 0.67`, `Long Scoreboard 0.41`, `Short Scoreboard 0.63`
- Shared excessive wavefronts: `285,212,672`

### 해석

이 커널은 요청한 scalar 계열 중 최고 성능이다.

- `16.05 -> 17.39 TFLOPS`
- `Eligible Warp 1.60 -> 1.90`
- `Issued Warp 0.62 -> 0.66`

즉 `BLOCK_K` 증가 효과가 2D 구조와 결합되면서 더 잘 드러난다. shared tile 하나를 쓰는 동안 compute가 더 길게 이어지므로, loop overhead 감소 이득이 명확해진다.

눈에 띄는 점은 대표 stall이 `Not Selected`로 올라온다는 것이다.

- 이건 "warp가 없어서 못 돈다"가 아니라,
- "돌릴 warp는 충분한데 scheduler가 그중 무엇을 택할지 고민한다"는 상태에 가깝다.

즉 이 시점에서는 kernel 구조가 이미 상당히 정돈되어 있고, 남은 병목은 memory path 세부 최적화나 더 깊은 pipeline 최적화다.

## 9. `sgemm_blocktiling_2d_vec4<128, 128, 8, 8, 8, 4>`

### 구조

`05_blocktiling_2d_vec4.cuh`는 2D kernel의 compute 구조는 유지하면서 global load/store 경로를 `float4` 기반으로 바꾼다.

- `A`, `B` tile preload에서 vectorized transaction을 사용한다.
- `C` store도 vectorized path를 사용한다.
- compute 자체보다 memory movement 효율을 개선하는 단계다.

### NCU 핵심 수치

- Duration: `8.41 ms`
- Throughput: `16.86 TFLOPS`
- Registers per thread: `111`
- Achieved occupancy: `32.37%`
- Eligible warps per scheduler: `1.54`
- Issued warps per scheduler: `0.64`
- No eligible: `36.05%`
- 대표 stall: `Not Selected 1.42`, `Long Scoreboard 1.09`, `Short Scoreboard 0.96`, `MIO Throttle 0.39`
- Shared excessive wavefronts: `268,435,456`
- `L2 Theoretical Sectors Global Excessive`: `2.10 MB`

### 해석

이 결과는 반드시 같은 `BLOCK_K`의 scalar 2D 버전과 비교해야 한다.

- scalar `2d<..., BK=8>`: `8.56 ms`, `16.05 TFLOPS`
- vec4 `2d_vec4<..., BK=8>`: `8.15 ms`, `16.86 TFLOPS`

즉 vec4는 분명히 이득이 있다. 다만 요청한 커널 순서상 직전 kernel이 `scalar BK=16` 버전이기 때문에, 순서대로만 읽으면 개선 폭이 작거나 오히려 후퇴해 보일 수 있다.

핵심은 아래다.

- vec4는 global memory transaction 효율을 개선한다.
- 실제로 `L2 Theoretical Sectors Global Excessive`가 크게 줄어든다.
- 반면 shared excessive wavefront는 거의 그대로다.

즉 vec4는 shared-memory access pattern 문제를 해결하는 최적화가 아니다. 이 단계의 이득은 global path 정리에서 나오며, 병목의 최종 중심이 shared path 쪽으로 더 선명하게 남게 된다.

## 10. `sgemm_blocktiling_2d_vec4<128, 128, 16, 8, 8, 4>`

### 구조

이 커널은 blocktiling 계열의 종착점이다.

- 2D outer-product
- `BLOCK_K=16`
- padding된 shared layout
- vec4 global load/store

즉 지금까지 쌓아 온 주요 최적화 축이 모두 결합된 상태다.

### NCU 핵심 수치

- Duration: `7.82 ms`
- Throughput: `18.20 TFLOPS`
- Registers per thread: `103`
- Achieved occupancy: `32.35%`
- Eligible warps per scheduler: `1.92`
- Issued warps per scheduler: `0.68`
- No eligible: `32.16%`
- 대표 stall: `Not Selected 1.83`, `Short Scoreboard 0.73`, `MIO Throttle 0.54`, `Long Scoreboard 0.56`
- Shared excessive wavefronts: `285,212,672`
- `L2 Theoretical Sectors Global Excessive`: `2.10 MB`

### 해석

blocktiling 계열 최고점은 naive baseline 대비 매우 크다.

- `2.04 -> 18.20 TFLOPS`
- 시간은 `67.32 -> 7.55 ms`
- 약 9배 speedup

이 kernel이 빠른 이유는 각 최적화가 서로 다른 병목을 순차적으로 제거했기 때문이다.

- shared tiling: global reload 감소
- register tiling: shared operand reuse 증가
- 2D outer-product: compute density와 ILP 증가
- `BLOCK_K=16`: barrier/loop overhead 감소
- vec4: global transaction 효율 개선

하지만 아직 완전히 끝난 것은 아니다.

- shared excessive wavefront가 여전히 크다.
- 대표 stall이 완전히 사라진 것이 아니라 `shared-memory path`와 `scheduler selection` 형태로 남아 있다.

즉 현재 남은 headroom은 global path가 아니라, shared-memory access pattern과 deeper software pipeline 쪽에 있다.

warptiling 계열에서 아래 성능 표는 각 커널을 독립 benchmark로 다시 측정한 값이고, 뒤의 4096 NCU 표는 대표 Nsight Compute 프로파일링 결과다. 따라서 같은 섹션 안에서도 benchmark 표의 Duration/TFLOPS와 NCU 표의 Duration/TFLOPS가 완전히 일치하지 않을 수 있다.

## 11. `sgemm_warptiling_v0<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v0<128, 128, 8, 64, 32, 8>`

### 구조

`06_warptiling.cuh`의 `v0`는 이번 흐름에서 처음으로 `threadblock tile -> warp tile -> thread tile` 계층을 코드에 명시적으로 넣은 버전이다.

- CTA tile은 그대로 `128x128x8`
- block 안의 warp는 `8`개
- 각 thread는 여전히 `8x8 = 64` accumulator를 가진다.
- 차이는 warp 하나가 맡는 tile이 `32x64`인지 `64x32`인지다.

즉 두 kernel의 차이는 "thread당 accumulator 수"가 아니라, 같은 `128x128` CTA tile을 warp가 어떤 방향으로 나눠 가지는가에 있다.

이 `v0`는 의도적으로 단순한 첫 버전이다.

- global -> shared preload는 여전히 scalar helper를 쓴다.
- `warp_mma()`도 CUTLASS-like `4x4 x 4` fragment가 아니라, thread가 자기 `8x8` micro-tile을 직관적으로 갱신한다.
- epilogue store도 thread가 자기 `8x8` tile을 global에 바로 scalar store한다.
- register prefetch, software pipeline, vec4 path는 아직 없다.

즉 이 단계는 "warp hierarchy를 도입하면 무엇이 좋아지고, 무엇이 아직 남는가"를 보기 위한 기준점에 가깝다.

### 성능 측정 결과

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

4096 기준으로 보면 `64x32` warp tile이 전 구간에서 더 빠르고, 최종적으로도 `8.15 ms`, `16.86 TFLOPS`까지 올라간다. 이건 같은 `BLOCK_K=8`, non-vec4 기준의 `blocktiling_2d<128,128,8,8,8,4>` (`8.56 ms`, `16.05 TFLOPS`)보다 분명히 낫다. 반면 `32x64` warp tile은 `8.50 ms`, `16.16 TFLOPS`에 머물러 `64x32` 대비 약 `4.2%` 낮다.

즉 warptiling 자체는 방향이 맞지만, warp tile의 방향과 lane mapping이 맞지 않으면 이득이 거의 사라진다.

### 4096 NCU 핵심 수치

`warptiling_v0<128,128,8,32,64,8>`:

- Duration: `9.01 ms`
- Throughput: `15.93 TFLOPS`
- Registers per thread: `125`
- Achieved occupancy: `32.29%`
- Eligible warps per scheduler: `1.71`
- Issued warps per scheduler: `0.61`
- No eligible: `38.76%`
- 대표 stall: `Not Selected 1.79`, `Long Scoreboard 0.79`, `Barrier 0.68`, `MIO Throttle 0.67`, `Short Scoreboard 0.44`
- Shared bank conflicts: `268,435,456`
- Shared excessive wavefronts: `268,435,456`
- `L2 Theoretical Sectors Global Excessive`: `14.68 MB`
- Global store 평균 사용량: `4 byte / sector`

`warptiling_v0<128,128,8,64,32,8>`:

- Duration: `8.55 ms`
- Throughput: `16.65 TFLOPS`
- Registers per thread: `125`
- Achieved occupancy: `32.22%`
- Eligible warps per scheduler: `1.92`
- Issued warps per scheduler: `0.65`
- No eligible: `35.13%`
- 대표 stall: `Not Selected 1.95`, `Long Scoreboard 0.79`, `Barrier 0.68`, `MIO Throttle 0.32`, `Short Scoreboard 0.23`
- Shared bank conflicts: `0`
- Shared excessive wavefronts: `0`
- `L2 Theoretical Sectors Global Excessive`: `14.68 MB`
- Global store 평균 사용량: `4 byte / sector`

### 해석

이 결과는 warptiling의 핵심을 잘 보여준다. warp-level hierarchy를 넣는다고 자동으로 빨라지는 것이 아니라, warp가 맡는 tile의 방향과 warp 내부 lane 배치가 shared access pattern과 맞아야 한다.

두 kernel은 아래 조건이 거의 같다.

- 같은 CTA tile: `128x128x8`
- 같은 block size: `256`
- 같은 thread tile: `8x8`
- 같은 accumulator 수: `64`
- 같은 register pressure: `125 regs/thread`

그런데도 결과는 꽤 다르다.

- `32x64`: `16.16 TFLOPS`
- `64x32`: `16.86 TFLOPS`

즉 차이는 "얼마나 많은 output을 계산하는가"가 아니라, "같은 output을 warp 안에서 어떤 모양으로 나눠 계산하는가"다.

가장 큰 차이는 shared-memory 경로에서 바로 보인다.

- `32x64`: shared bank conflicts `268,435,456`, excessive wavefront `268,435,456`
- `64x32`: 둘 다 `0`

이건 현재 `v0`의 단순한 row-major lane mapping에서는 `32x64` warp tile이 특히 `B` 쪽 shared load pattern과 잘 맞지 않는다는 뜻으로 읽는 것이 자연스럽다. 반대로 `64x32`는 warp의 N 폭이 좁아지면서 shared load가 훨씬 깔끔해지고, 그 결과 `MIO Throttle`과 `Short Scoreboard`도 같이 줄어든다.

- `MIO Throttle`: `0.67 -> 0.32`
- `Short Scoreboard`: `0.44 -> 0.23`
- `Eligible Warp`: `1.71 -> 1.92`
- `Issued Warp`: `0.61 -> 0.65`

즉 `64x32` warp tile의 이득은 "연산량 증가"가 아니라, shared consumption path가 더 깔끔해진 데서 나온다.

하지만 `v0`가 최종 best point를 넘지 못하는 이유도 분명하다.

첫째, register pressure가 이미 높다.

- 두 kernel 모두 `125 regs/thread`
- occupancy는 둘 다 약 `32%`

둘째, global store path가 아직 매우 거칠다.

- 두 kernel 모두 `L2 Theoretical Sectors Global Excessive = 14.68 MB`
- global store 평균 사용량이 `4 byte / sector`

즉 epilogue가 thread의 `8x8` tile을 scalar store하는 현재 구조에서는, warp가 global memory에 결과를 쓰는 방식이 여전히 비효율적이다. 이 때문에 shared path를 정리해도, 최종 vec4 kernel처럼 깔끔한 global path까지는 못 간다.

셋째, `v0`는 아직 CUTLASS-like lane fragment layout을 쓰지 않는다.

현재 `v0`는 thread가 자기 `8x8`을 "붙어 있는 dense tile"처럼 다룬다. 이번 결과에서 `32x64`가 shared path를 크게 망친 것은, 왜 다음 단계에서 `RowMajorInterleaved<2>`와 `4x4 x 4` fragment 배치가 중요해지는지를 잘 보여준다.

즉 `warptiling v0`의 의미는 다음처럼 정리할 수 있다.

- 좋은 점: `64x32` 쪽은 `2D BK=8` baseline보다 확실히 빨라진다.
- 한계: warp tile 방향을 잘못 잡으면 같은 구조에서도 shared path가 바로 깨진다.
- 교훈: warptiling의 진짜 핵심은 "warp 계층을 넣는 것" 자체보다, "warp tile과 lane fragment를 shared/global path에 맞게 배치하는 것"이다.

## 12. `sgemm_warptiling_v1<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v1<128, 128, 8, 64, 32, 8>`

### 구조

`06_warptiling.cuh`의 `v1`은 `v0`와 같은 CTA/block/thread 규모를 유지하면서, warp 내부 lane 배치와 epilogue를 다듬은 버전이다.

- CTA tile: 그대로 `128x128x8`
- warp 수: 그대로 `8`
- thread tile: 그대로 `8x8`
- accumulator 수: 그대로 `64`
- register pressure: 여전히 `125 regs/thread`

즉 `v1`의 핵심은 "더 많은 계산을 한다"가 아니라, 같은 계산을 warp 안에서 더 좋은 layout으로 수행하는 것이다.

구체적으로는 아래 세 가지가 바뀐다.

1. lane mapping이 단순 row-major에서 `RowMajorInterleaved<2>`에 가까운 형태로 바뀐다.
2. `warp_mma()`가 thread의 논리적 `8x8` tile을 그대로 dense하게 다루지 않고, `4x4` island 네 개로 나눠 소비한다.
3. epilogue store도 thread의 `8x8` tile을 row-major scalar store하지 않고, `4x4` group 단위로 정리해 쓴다.

이 변화의 목적은 명확하다.

- `v0`에서 warp shape에 따라 갈리던 shared-memory consumption path를 안정화한다.
- thread tile 내부 lane fragment 배치를 shared bank 구조에 더 잘 맞춘다.
- scalar epilogue의 비효율을 줄여 global store sector 활용도를 개선한다.

즉 `v1`은 "warp hierarchy 도입" 자체를 증명한 `v0` 다음 단계로서, CUTLASS SIMT 스타일의 lane fragment 정리를 부분적으로 가져온 첫 refinement라고 볼 수 있다.

### 성능 측정 결과

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

4096 기준으로 보면 두 kernel은 사실상 같은 급의 성능에 도달한다.

- `32x64`: `7.95 ms`, `17.28 TFLOPS`
- `64x32`: `7.91 ms`, `17.37 TFLOPS`

즉 `v0`에서 뚜렷했던 warp shape 민감도가 거의 사라졌고, 두 변형이 사실상 같은 급으로 수렴한다. `64x32`가 benchmark상 아주 근소하게 앞서지만 차이는 작다. 이 수치는 `warptiling v0` 최고점 `16.86 TFLOPS`를 넘어서는 동시에, `blocktiling_2d_vec4<128,128,8,8,8,4>`의 `16.86 TFLOPS`보다도 높다. 다만 blocktiling 계열 최고점인 `blocktiling_2d_vec4<128,128,16,8,8,4>`의 `18.20 TFLOPS`에는 아직 못 미친다.

### 4096 NCU 핵심 수치

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

두 kernel의 공통점은 더 중요하다.

- shared bank conflict가 둘 다 `0`
- shared excessive wavefront도 둘 다 `0`
- top stall이 `Not Selected`, `Long Scoreboard`, `Barrier` 쪽으로 모인다.
- `MIO Throttle`은 둘 다 약 `0.20` 수준으로 매우 낮다.
- global store 평균 사용량이 `8 byte / sector`까지 올라간다.

즉 `v1`에서는 shared-memory path가 더 이상 주된 문제로 보이지 않는다. 병목은 이제 "warp가 shared에서 얼마나 예쁘게 읽느냐"보다는, register pressure가 높은 상태에서 scheduler가 warp를 얼마나 끊김 없이 발행하느냐 쪽으로 옮겨간다.

### v0 -> v1 비교 해석

`v1`의 의미는 단순한 소폭 개선이 아니라, `v0`에서 드러난 두 가지 약점을 정확히 찌른 refinement라는 데 있다.

첫째, `32x64` warp tile의 병적인 shape 민감도가 사라졌다.

- `TFLOPS`: `16.16 -> 17.28`
- `Duration`: `8.50 -> 7.95 ms`
- shared bank conflicts: `268,435,456 -> 0`
- shared excessive wavefronts: `268,435,456 -> 0`
- `MIO Throttle`: `0.67 -> 0.20`
- `Short Scoreboard`: `0.44 -> 0.09`

즉 `v0 32x64`의 문제는 warp tile shape 자체보다, 그 shape를 받쳐 주는 lane mapping과 fragment/store layout이 맞지 않았던 데 있었다. `v1`은 바로 그 부분을 고치면서 `32x64`를 `64x32`와 거의 동급까지 끌어올린다.

둘째, 이미 shared path가 깨끗했던 `64x32`도 global/store 쪽이 더 정리된다.

- `TFLOPS`: `16.86 -> 17.37`
- `Duration`: `8.15 -> 7.91 ms`
- global store 평균 사용량: `4 -> 8 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: `14.68 -> 6.29 MB`

즉 `64x32`는 `v0`에서도 구조적으로 나쁘지 않았지만, `v1`에서 epilogue와 lane fragment가 더 정리되면서 추가 이득을 얻는다.

셋째, 남은 병목의 성격이 더 분명해졌다.

- register pressure는 여전히 `125 regs/thread`
- occupancy는 여전히 `32%` 안팎
- 대표 stall은 `Not Selected`, `Long Scoreboard`, `Barrier`

즉 `v1` 이후의 headroom은 shared bank conflict를 더 없애는 문제가 아니라, 낮은 occupancy 환경에서 load-to-use 거리를 더 늘리고 scheduler idle을 줄이는 문제에 가깝다. 이건 다음 단계가 register prefetch, software pipeline, 그리고 더 나은 global/store path 정리 쪽으로 가야 함을 의미한다.

결론적으로 `warptiling v1`은 아래처럼 요약할 수 있다.

- `v0`가 증명한 warp hierarchy의 가능성을 실제 성능 개선으로 연결했다.
- warp tile shape 민감도를 크게 줄였다.
- shared path 병목을 대부분 제거하고, 병목을 scheduler/scoreboard 쪽으로 옮겼다.
- 그러나 `vec4 + BK16` 최종점에 도달하려면 아직 pipeline과 memory path 측면의 refinement가 더 필요하다.

## 13. `sgemm_warptiling_v2<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v2<128, 128, 8, 64, 32, 8>` vs `sgemm_warptiling_v2<128, 128, 16, 32, 64, 16>`

### 구조

`06_warptiling.cuh`의 `v2`는 `v1`의 warp MMA, lane mapping, epilogue 구조는 유지하고, CTA tile load path만 바꾼 버전이다.

- `v1`은 `utils::load_a_threadblock_tile()`, `utils::load_b_threadblock_tile()` helper 안에서 thread가 맡은 global element를 곧바로 shared layout으로 옮긴다.
- `v2`는 `load_a_thread_fragment()`, `load_b_thread_fragment()`로 먼저 thread-private register array `tb_frag_a`, `tb_frag_b`에 읽는다.
- 그 다음 `store_a_thread_fragment_to_smem()`, `store_b_thread_fragment_to_smem()`로 shared memory에 배치한다.
- 즉 source-level data path가 `global -> register fragment -> shared -> warp_mma`로 명시적으로 분해된다.

중요한 점은 이게 `cp.async`나 hardware direct global->shared copy는 아니라는 점이다.

- 코드상으로도 thread-private fragment를 거쳐 shared에 쓰는 manual staging이다.
- NCU에서도 `smsp__inst_executed_op_ldgsts.sum = 0`이라서, 이번 개선은 `LDGSTS`/`cp.async` 도입이 아니라 regular global load + shared store의 scheduling을 더 명시적으로 만든 결과로 봐야 한다.

또 하나의 포인트는 `BLOCK_K=16` 변형이다.

- 같은 register-staged load path를 유지하면서 outer loop trip count를 절반으로 줄인다.
- 따라서 barrier 빈도와 loop overhead를 함께 낮출 여지가 생긴다.

즉 `v2`의 핵심은 "warp layout을 또 바꿨다"가 아니라, 이미 정리된 `v1` 위에서 global->shared ingest path를 explicit software staging으로 바꿨다는 데 있다.

### 성능 측정 결과

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

4096 기준으로 보면 `v2`는 세 가지를 보여준다.

- `BK=8, 32x64`: `17.28 -> 18.25 TFLOPS`, `7.95 -> 7.53 ms`
- `BK=8, 64x32`: `17.37 -> 17.86 TFLOPS`, `7.91 -> 7.69 ms`
- `BK=16, 32x64`: `19.13 TFLOPS`, `7.18 ms`로 당시 전체 커널 중 새 최고점

즉 explicit register staging은 `v1` 대비 BK8 계열 둘 다 확실한 이득을 만들고, 여기에 `BLOCK_K=16`을 겹치면 처음으로 `19 TFLOPS`를 넘긴다.

### 4096 NCU 핵심 수치

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

공통된 메시지는 분명하다.

- shared bank conflict는 이미 `v1`에서 `0`이었고, `v2`에서도 그대로 `0`
- shared excessive wavefront도 이미 `0`이었고, `v2`에서도 그대로 `0`
- global store 평균 사용량도 여전히 `8 byte / sector`

즉 `v2`의 성능 향상은 shared layout 문제를 새로 해결해서 생긴 것이 아니라, global->shared staging과 inner-loop scheduling을 더 명시적으로 만든 데서 나온 것이다.

### v1 -> v2 비교 해석

첫째, `v2`의 이득은 "shared path를 더 깨끗하게 만들었다"라기보다, 이미 깨끗한 shared path 위에서 load staging을 더 잘 제어했다는 데 있다.

`v1`에서도 bank conflict와 excessive wavefront는 사실상 정리돼 있었다. 그런데도 `v2`에서 BK8 두 변형이 모두 더 빨라진다.

- `32x64`: `17.28 -> 18.25 TFLOPS`
- `64x32`: `17.37 -> 17.86 TFLOPS`
- occupancy는 둘 다 여전히 약 `32%`
- global store 평균 사용량과 excessive sector도 그대로 `8 byte / sector`, `6.29 MB`

즉 개선 원인은 store path나 occupancy가 아니다. 이번 변경은 source code에서 register fragment lifetime을 드러내면서 global load, shared store, 다음 FMA 사이의 scheduling 여지를 더 만든 쪽으로 해석하는 것이 맞다.

둘째, `v2`는 warp shape 민감도를 다시 키우지 않고, `32x64`를 계속 최상위 변형으로 유지한다.

`v0`에서는 `32x64`가 shared path 문제로 크게 흔들렸고, `v1`에서 그 민감도가 거의 사라졌다. `v2`에서는 그 상태를 유지한 채 `32x64`가 가장 높은 성능을 낸다.

- `32x64 BK8`: `18.25 TFLOPS`
- `64x32 BK8`: `17.86 TFLOPS`

즉 explicit register staging은 warp shape의 불안정을 다시 만들지 않았고, 오히려 `v1`이 만든 안정된 warp layout 위에서 추가 이득을 얹는 역할을 한다.

셋째, `BLOCK_K=16`이 결합된 `v2`는 이번 세대의 핵심 결과다.

- `19.13 TFLOPS`, `7.18 ms`
- Compute Throughput `68.28%`
- Issued Warps / Scheduler `0.73`
- Eligible Warps / Scheduler `2.39`
- `Stall Barrier 0.48`, `Stall Long Scoreboard 0.48`

특히 `v2 BK=8 32x64`와 비교하면 아래 변화가 눈에 띈다.

- Compute Throughput: `65.37 -> 68.28%`
- Issued Warp: `0.70 -> 0.73`
- `Stall Barrier`: `0.62 -> 0.48`
- `Stall Long Scoreboard`: `0.72 -> 0.48`

이는 `BLOCK_K=16`으로 outer loop와 barrier 노출이 줄고, explicit register staging과 결합되면서 load-to-use distance도 조금 더 유리해진 결과로 읽을 수 있다.

넷째, 남은 병목도 더 분명해졌다.

- 세 변형 모두 theoretical occupancy가 약 `33%` 수준이고, achieved occupancy도 `32%` 전후
- top stall은 여전히 `Not Selected`
- NCU는 여전히 global store의 `8 byte / sector` 사용량과 `6.29 MB` excessive sectors를 지적한다

즉 `v2` 이후의 headroom은 shared bank conflict 제거가 아니라, 더 깊은 prefetch/pipeline과 epilogue store 정리에 있다.

보조 실험 메모: 아래 표는 최신 mainline benchmark와 별도로 rerun한 실험 결과라, 절대값보다 helper 변경에 따른 상대 변화에 의미가 있다.

### load/store fragment helper 3-way 비교

같은 `v2` 커널 안에서도 `load_*_thread_fragment()`와 `store_*_thread_fragment_to_smem()`의 작성 방식에 따라 성능 차이가 뚜렷하게 난다.

- 버전 1: load/store 모두 original `elem_idx -> row/col` 방식
- 버전 2: load만 `offset += advance_offset` 방식으로 변경
- 버전 3: 버전 2에서 store도 같은 방식으로 변경

4096 기준 TFLOPS를 요약하면 아래와 같다.

| Kernel | 버전 1 | 버전 2 | 버전 3 | v1 -> v2 | v2 -> v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2<128,128,8,32,64,8>` | 17.82 | 18.16 | 17.45 | `+1.96%` | `-3.94%` |
| `v2<128,128,8,64,32,8>` | 17.53 | 18.03 | 17.71 | `+2.81%` | `-1.74%` |
| `v2<128,128,16,32,64,16>` | 15.72 | 19.31 | 19.58 | `+22.84%` | `+1.40%` |

4096 기준 NCU 핵심 수치도 같은 경향을 보여 준다.

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

세 프로파일 모두 `L2 Theoretical Sectors Global Excessive = 6.29 MB`로 동일하다. 즉 이번 helper 실험은 epilogue global-store inefficiency를 개선한 것이 아니라, mainloop 안의 주소 계산과 그에 따른 codegen/scheduling을 바꾼 실험으로 보는 게 맞다.

이 결과에서 중요한 건 `%`와 `/`를 없앴느냐 자체가 아니다. 현재 shape에서는 `BK=8/16`, `N=128`이라 divisor가 전부 compile-time power-of-two이므로, compiler는 old 버전의 `/`, `%`를 이미 shift/mask 수준으로 strength reduction 할 수 있다.

차이를 만든 핵심은 load와 store의 "최종 주소 식"이 다르다는 점이다.

- `A load` old:
  - `a_ptr[(tid / BK + iter * (256 / BK)) * lda + tile_k + (tid % BK)]`
  - 즉 iteration마다 runtime stride인 `lda`가 곱해진다.
- `B load` old:
  - `b_ptr[(tid / 128 + 2 * iter + tile_k) * ldb + (tid % 128)]`
  - 마찬가지로 iteration마다 runtime stride인 `ldb`가 들어간다.
- `A store` old:
  - `smem_a[(tid % BK) * kSmemStrideA + (tid / BK) + iter * (256 / BK)]`
  - shared stride와 increment가 모두 compile-time 상수다.
- `B store` old:
  - `smem_b[(tid / 128 + 2 * iter) * 128 + (tid % 128)]`
  - 현재 shape에서는 사실상 `smem_b[tid + iter * 256]`으로 단순화된다.

즉 load 쪽 old 버전은 loop body 안에 runtime `lda/ldb` 기반 address generation이 남아 있지만, store 쪽 old 버전은 이미 `base + const * iter`에 가까운 형태다.

그래서 버전 2의 load 최적화는 "표현 변경"이 아니라 실제 최적화가 된다.

- `offset += advance_offset`로 바꾸면 per-iteration dynamic address-gen이 줄어든다.
- 예를 들어 `load_a_thread_fragment()`의 old 버전은 매 iteration마다 사실상 아래 주소를 다시 계산한다.

```cpp
int elem_idx = tid + iter * 256;
int row = elem_idx / BK;
int col = elem_idx % BK;
addr = row * lda + tile_k + col;
tb_frag_a[iter] = a_ptr[addr];
```

반면 new 버전은 첫 주소와 다음 주소까지의 간격만 먼저 계산해 둔 뒤, loop 안에서는 그 간격만 더한다.

```cpp
int row = tid / BK;
int col = tid % BK;
int offset = row * lda + tile_k + col;
int advance_offset = (256 / BK) * lda + (256 % BK);

for (...) {
  tb_frag_a[iter] = a_ptr[offset];
  offset += advance_offset;
}
```

즉 old 버전은 "매번 주소를 처음부터 다시 계산"하고, new 버전은 "첫 주소를 구한 뒤 일정한 간격만 더해 가며 접근"하는 형태다.

- `BK=8` 두 커널은 `v1 -> v2`에서 각각 `+1.96%`, `+2.81%`의 완만한 개선이 나온다.
- 반면 `BK=16`은 `15.72 -> 19.31 TFLOPS (+22.84%)`로 가장 크게 튄다.
- NCU에서도 `Registers/Thread 141 -> 127`, `Achieved Occupancy 16.66% -> 32.23%`, `No Eligible 41.04% -> 27.24%`, `Compute 55.29% -> 68.29%`로 같이 좋아져, 이 변화가 helper 내부 산술 감소를 넘어 커널 전체 codegen과 latency hiding을 유리하게 바꿨음을 보여 준다.

반면 버전 3의 store 변경은 access pattern을 바꾸는 것이 아니라, 이미 충분히 단순하던 shared store address 계산을 다른 형태로 다시 쓰는 것에 가깝다.

- 예를 들어 현재 shape의 `store_b_thread_fragment_to_smem()` old 버전은 아래처럼 보이지만,

```cpp
int elem_idx = tid + iter * 256;
int row = elem_idx / 128;
int col = elem_idx % 128;
smem_b[row * 128 + col] = tb_frag_b[iter];
```

현재는 `N=128`이므로 이 식은 사실상 아래와 같다.

```cpp
smem_b[tid + iter * 256] = tb_frag_b[iter];
```

즉 old 버전도 이미 "현재 thread가 맡은 base 위치에서 iteration마다 256칸씩 떨어진 곳에 저장"하는 단순한 형태로 정리된다.

new 버전은 같은 주소열을 아래처럼 `offset` 누적 방식으로 다시 표현한 것이다.

```cpp
int offset = tid;
int advance_offset = 256;

for (...) {
  smem_b[offset] = tb_frag_b[iter];
  offset += advance_offset;
}
```

즉 old와 new는 "어디에 저장하느냐"가 다르지 않다. 둘 다 같은 shared-memory 위치에 쓴다. 차이는 old가 compiler 입장에서 이미 `base + imm` 형태로 충분히 단순할 수 있었던 반면, new는 그 같은 주소열을 `offset`을 계속 갱신하는 형태로 다시 적은 것에 가깝다는 점이다.

- 얻는 이득은 작다.
- 대신 `offset`의 loop-carried dependency가 생긴다.
- 즉 compiler가 old store에서 만들 수 있던 독립적인 `STS [base + imm]` 형태 대신, 새 버전에서는 `offset`을 갱신해 가며 store하는 codegen이 나올 수 있다.

이 trade-off가 shape마다 다르게 보인 결과가 위 수치다.

- `BK=8`은 `v2 -> v3`에서 각각 `-3.94%`, `-1.74%`로 소폭 후퇴했다.
- profile상 occupancy tier는 그대로인데 `Compute`가 `65.33% -> 63.19%`, `63.80% -> 63.02%`로 내려가고 `No Eligible`도 `30.28% -> 32.56%`, `32.48% -> 32.98%`로 나빠져, access pattern 개선보다는 codegen/scheduling 부담이 더 크게 남은 쪽으로 해석하는 편이 맞다.
- `BK=16`은 `19.31 -> 19.58 TFLOPS (+1.40%)`로 소폭 개선했다. 이때도 occupancy는 같은 tier지만 `Registers/Thread 127 -> 125`, `Compute 68.29% -> 69.86%`, `No Eligible 27.24% -> 25.56%`, `DRAM BW 125.76 -> 128.46 GB/s`로 조금씩 좋아진다.

따라서 이번 실험의 해석은 아래처럼 정리하는 것이 맞다.

- load recurrence 변환은 실제 성능 최적화다.
- store recurrence 변환은 현재 shape에서는 memory access 개선이 아니라 codegen/scheduling trade-off다.
- 그래서 `v1 -> v2`는 특히 `BK=16`에서 큰 실효를 냈고, `v2 -> v3`는 BK8에서는 음수, BK16에서는 작은 양수의 2차 효과로만 남는다.

여기서 `epilogue store inefficiency`를 남은 병목으로 보는 근거는 코드와 NCU 양쪽에서 동시에 나온다.

첫째, 코드상 epilogue는 아직 scalar store 중심이다.

- `06_warptiling.cuh`의 `store_accum_to_gmem()`는 `row_group -> col_group -> m -> n` 4중 loop로 accumulator를 global memory에 쓴다.
- 각 thread는 자기 `8x8` 결과 tile을 `4x4` island 네 개로 나눠 쓰지만, 결국 `dst[n] = accum[...]` 형태의 scalar store를 반복한다.
- 즉 compute 단계의 lane mapping은 정리됐어도, 최종 writeback은 여전히 dedicated vector store epilogue가 아니라 per-thread scalar write에 가깝다.

둘째, NCU가 바로 그 global store path를 별도로 경고한다.

- `Average Bytes Per Sector For Global Stores = 8`
- `Maximum Bytes Per Sector For Global Stores = 32`
- `L2 Theoretical Sectors Global Excessive = 6.29 MB`

이 조합은 store sector 하나가 실을 수 있는 32 byte 중 평균 8 byte만 유효하게 쓰이고 있음을 뜻한다. 즉 store 자체가 완전히 흩어져 있다는 뜻은 아니지만, warp 전체 관점에서는 여전히 sector utilization이 낮다.

셋째, BK16 변형의 NCU 메시지도 같은 방향을 가리킨다.

- "The memory access pattern for global stores to L1TEX might not be optimal"
- "On average, only 8.0 of the 32 bytes transmitted per sector are utilized"

즉 이건 단순한 추측이 아니라, NCU가 남은 memory inefficiency의 위치를 global store 쪽으로 직접 지목한 경우다.

넷째, 이 문제는 "주 병목"이라기보다 "shared path를 정리한 뒤에도 남아 있는 잔여 병목"으로 보는 것이 맞다.

- shared bank conflict: 이미 `0`
- shared excessive wavefront: 이미 `0`
- compute throughput: 이미 `65~68%`
- 그런데 store sector utilization은 더 좋아지지 않고 `8 byte / sector`에서 멈춰 있다

즉 `v2`에서 성능이 올라간 뒤에도 memory side에서 가장 설명력이 남아 있는 비효율이 epilogue store path라는 뜻이다. 그래서 문서에서는 이를 "확정된 단일 병목"이 아니라, 다음 headroom을 설명하는 가장 강한 residual inefficiency로 해석하는 것이 적절하다.

결론적으로 `warptiling v2`는 아래처럼 정리할 수 있다.

- `v1`이 정리한 lane mapping과 shared consumption path를 유지한다.
- global->shared를 explicit register staging으로 바꾸면서 BK8에서도 확실한 추가 이득을 만든다.
- `BLOCK_K=16`과 결합하면 기존 blocktiling 최고점 `18.20 TFLOPS`를 넘어 `19.13 TFLOPS`에 도달한다.
- 하지만 여전히 `cp.async`는 아니며, occupancy와 epilogue store inefficiency가 다음 병목으로 남는다.

## 14. `sgemm_warptiling_v3<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v3<128, 128, 8, 64, 32, 8>` vs `sgemm_warptiling_v3<128, 128, 16, 32, 64, 16>`

### 구조

`06_warptiling.cuh`의 `v3`는 `v2`의 explicit register staging에 "1 stage register prefetch"를 얹은 버전이다.

- loop 밖에서 `tile 0`의 `tb_frag_a`, `tb_frag_b`를 먼저 읽어 둔다.
- loop 안에서는 현재 register fragment를 shared에 쓰고, 현재 tile을 계산한 뒤, 다음 tile fragment를 register에 미리 읽는다.
- steady-state 관점에서 보면 `v2`의 "다음 iteration 시작점에 있던 global load"를 "현재 iteration 끝"으로 당겨 온 형태다.
- 하지만 shared stage는 여전히 한 벌뿐이라서, next tile fragment를 다른 shared buffer에 즉시 써 두지는 못한다. `register -> shared`는 여전히 다음 iteration head에서 수행된다.

즉 `v3`는 `cp.async`도 아니고 CTA-level double buffering도 아니다. ordinary global load를 더 일찍 발행해서 다음 tile의 load-to-use distance를 늘리는 partial prefetch에 가깝다.

중요한 건 "현재 tile의 FMA와 next tile global load가 완전히 겹친다"가 아니라는 점이다. next tile load는 여전히 현재 `warp_mma()`가 끝난 뒤에 실행된다. 다만 `v2`에서보다 한 iteration boundary만큼 앞당겨져서, 다음 tile 입장에서는 적어도 한 번의 `__syncthreads()`와 shared store 구간만큼 latency hiding window가 늘어난다.

### 성능 측정 결과

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

4096 기준으로 보면 `v3`는 매우 비대칭적인 결과를 만든다.

- `BK=8, 32x64`: `18.25 -> 18.72 TFLOPS`, `7.53 -> 7.34 ms`
- `BK=8, 64x32`: `17.86 -> 18.84 TFLOPS`, `7.69 -> 7.29 ms`
- `BK=16, 32x64`: `19.13 -> 17.15 TFLOPS`, `7.18 -> 8.01 ms`

즉 같은 1-stage prefetch라도 BK8에서는 유효하지만, BK16에서는 오히려 큰 퇴행을 만든다.

### 4096 NCU 핵심 수치

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

stall 관점에서도 BK8과 BK16의 차이가 분명하다.

- `v3 BK8 32x64`: `Long Scoreboard 0.09`, `Barrier 0.67`, `Dispatch Stall 0.48`
- `v3 BK8 64x32`: `Long Scoreboard 0.03`, `Barrier 0.74`, `Short Scoreboard 0.29`
- `v3 BK16 32x64`: `Long Scoreboard 0.45`, `Dispatch Stall 0.26`, `Wait 0.19`

즉 BK8에서는 long-scoreboard가 사실상 꺼지지만, BK16에서는 그렇지 않다.

### v2 -> v3 비교 해석

첫째, BK8에서의 이득은 epilogue나 shared layout이 아니라 mainloop scheduling에서 나온다.

`v2 -> v3`로 올라가도 아래 항목은 사실상 바뀌지 않는다.

- global store 평균 사용량: 계속 `8 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: 계속 `6.29 MB`
- shared bank conflict: 계속 `0`
- shared stage 수: 계속 `1`

그런데도 BK8 두 변형은 모두 빨라진다.

- `32x64`: `18.25 -> 18.72 TFLOPS` (`+2.56%`)
- `64x32`: `17.86 -> 18.84 TFLOPS` (`+5.51%`)

즉 이번 이득은 epilogue 개선이 아니라, next tile global load를 더 앞당겨 load-to-use distance를 늘린 데서 나온다.

특히 `BK8 32x64`는 `v2`와 비교할 때 변화 방향이 아주 깔끔하다.

- Compute Throughput: `65.37 -> 71.85%`
- Eligible Warps / Scheduler: `2.08 -> 2.43`
- Issued Warps / Scheduler: `0.70 -> 0.76`
- No Eligible: `30.34 -> 23.49%`
- `Stall Long Scoreboard`: `0.72 -> 0.09`

occupancy와 store 효율은 그대로인데 scoreboard와 scheduler 지표만 좋아졌으므로, `v3`의 BK8 이득은 거의 그대로 partial prefetch 효과로 읽을 수 있다.

둘째, `v3`에서는 BK8 내부의 우세 shape도 바뀐다.

`v2`에서는 BK8 안에서 `32x64`가 더 빨랐지만, `v3`에서는 `64x32`가 오히려 최고점이 된다.

- `v3 BK8 32x64`: `18.72 TFLOPS`
- `v3 BK8 64x32`: `18.84 TFLOPS`

이 차이는 shared path나 epilogue 차이라기보다, 같은 prefetch hoist가 `64x32`에서 scheduler와 memory path에 더 잘 맞았기 때문으로 보는 편이 자연스럽다.

- Compute Throughput: `71.85 -> 73.71%`
- DRAM BW: `129.20 -> 146.61 GB/s`
- Issued Warps / Scheduler: `0.77 -> 0.78`

즉 `64x32`는 `128 regs/thread`에 걸치면서도 아직 `2 blocks/SM` tier를 유지한 채, prefetch 이득을 가장 크게 흡수한 변형이 된다.

셋째, BK16의 퇴행은 "prefetch가 나쁘다"라기보다 "register cliff를 넘었다"로 보는 것이 맞다.

`v2 BK16`과 `v3 BK16`의 차이는 아래처럼 요약된다.

- TFLOPS: `19.13 -> 17.15` (`-10.37%`)
- Duration: `7.18 -> 8.01 ms` (`+11.57%`)
- Registers / thread: `127 -> 152`
- Achieved Occupancy: `32.23 -> 16.67%`
- Eligible Warps / Scheduler: `2.39 -> 1.16`
- Issued Warps / Scheduler: `0.73 -> 0.63`
- No Eligible: `27.26 -> 36.92%`

여기서 핵심은 `256 threads/block`에서 `128 regs/thread`가 사실상 `2 blocks/SM`의 경계라는 점이다.

- `v2 BK16`의 `127 regs/thread`는 겨우 `2 blocks/SM`에 남아 있었다.
- `v3 BK16`의 `152 regs/thread`는 그 경계를 넘어서 `1 block/SM`으로 떨어진다.

즉 BK16에서는 prefetch로 늘어난 fragment live range가 가져온 register pressure가, load-to-use distance 증가 이득보다 더 비쌌다는 뜻이다.

이때 shared 쪽에 작은 새 신호도 보인다.

- shared bank conflict: `0 -> 469,497`

하지만 이 값은 전체 shared wavefront 규모에 비해 매우 작고, shared memory footprint도 `v2 BK16`과 본질적으로 같기 때문에 주된 퇴행 원인으로 보기는 어렵다. 또한 source-level shared store 경로 자체는 `v2`와 `v3`가 동일하므로, 이를 "prefetch 때문에 shared layout이 갑자기 깨졌다"로 보기는 어렵다. 오히려 BK16에서 prefetch가 register live range를 늘리면서 codegen과 issue pattern이 조금 흔들렸고, 그 부산물로 작은 shared-store conflict 신호가 관측된 것으로 해석하는 편이 더 자연스럽다. 결정적인 변화는 register 수 증가와 그에 따른 occupancy tier 붕괴다.

넷째, `v3`는 "1-stage register prefetch는 공짜가 아니다"라는 점을 아주 명확하게 보여준다.

- 같은 occupancy tier를 유지한 BK8에서는 성공한다.
- `128 regs/thread` 경계를 넘은 BK16에서는 실패한다.
- epilogue store inefficiency는 여전히 그대로 남아 있다.

따라서 이번 결과의 핵심 교훈은 "prefetch를 더 넣자"가 아니라, "prefetch가 늘리는 live range를 어떤 occupancy tier 안에서 감당할 수 있느냐"다.

보조 실험 메모: 아래 표는 최신 mainline benchmark와 별도로 rerun한 실험 결과라, 절대값보다 load 위치 이동에 따른 상대 변화에 의미가 있다.

### fragment load를 `warp_mma()` 직전으로 옮기면

같은 `v3` 아이디어 안에서, next tile fragment load를 현재처럼 `warp_mma()` 직후에 두지 않고 직전으로 옮겨 보면 최신 독립 측정 기준에서는 BK8 둘은 사실상 비슷하거나 소폭 느려지고, BK16만 분명히 개선된다. 즉 이 변화의 실질적인 수혜자는 register cliff를 밟고 있던 BK16 쪽이다.

4096 기준 요약:

| Kernel | 기존 `v3` | load 위치 이동 | 변화 |
| --- | ---: | ---: | ---: |
| `BK8 32x64` | `18.72 TFLOPS` | `18.71 TFLOPS` | `-0.05%` |
| `BK8 64x32` | `18.84 TFLOPS` | `18.66 TFLOPS` | `-0.96%` |
| `BK16 32x64` | `17.15 TFLOPS` | `18.20 TFLOPS` | `+6.14%` |

4096 기준 프로파일 비교:

| Kernel | Compute Throughput | Registers / thread | Achieved Occupancy | Eligible / Issued | No Eligible | DRAM BW | 특이점 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `BK8 32x64` | `71.85 -> 70.79%` | `127 -> 122` | `32.36 -> 32.45%` | `2.43 / 0.77 -> 2.36 / 0.75` | `23.49 -> 24.61%` | `129.20 -> 139.55 GB/s` | global store `8 byte/sector`, excessive `6.29 MB`는 그대로 |
| `BK8 64x32` | `73.71 -> 70.94%` | `128 -> 126` | `32.35 -> 32.52%` | `2.35 / 0.78 -> 2.33 / 0.75` | `21.59 -> 24.65%` | `146.61 -> 158.93 GB/s` | global store `8 byte/sector`, excessive `6.29 MB`는 그대로 |
| `BK16 32x64` | `59.17 -> 66.50%` | `152 -> 128` | `16.67 -> 32.37%` | `1.16 / 0.63 -> 2.35 / 0.71` | `36.92 -> 29.13%` | `110.87 -> 125.95 GB/s` | shared bank conflict `469,497 -> 0`, excessive `6.29 MB`는 그대로 |

주요 stall 비교:

| Kernel | 주요 stall 변화 |
| --- | --- |
| `BK8 32x64` | `Long Scoreboard 0.09 -> 0.04`, `Barrier 0.67 -> 0.77`, `Dispatch Stall 0.48 -> 0.58` |
| `BK8 64x32` | `Long Scoreboard 0.03 -> 0.03`, `Barrier 0.74 -> 0.77`, `Short Scoreboard 0.29 -> 0.16` |
| `BK16 32x64` | `Long Scoreboard 0.45 -> 0.29`, `Dispatch Stall 0.26 -> 0.72`, `Wait 0.19 -> 0.20` |

해석은 세 줄로 정리된다.

- benchmark 기준으로는 BK8 두 변형이 각각 `-0.05%`, `-0.96%`로 사실상 비슷하거나 소폭 느려진다. profile에서도 `Compute Throughput`, `Eligible / Issued`, `No Eligible`가 오히려 조금 나빠진다. 즉 BK8에서는 robust한 latency-hiding 개선이 확인되지 않고, load 위치 이동의 효과가 크지 않은 민감한 구간으로 보는 편이 더 안전하다.
- BK16은 반대로 개선 이유가 분명하다. load 위치를 바꾸자 fragment live range가 짧아지면서 register가 `152 -> 128`로 내려가고 occupancy가 `16.67 -> 32.37%`로 복귀해, 원래 `v3 BK16`의 register cliff를 상당 부분 되돌렸다.
- 세 프로파일 모두 global store 평균 사용량 `8 byte / sector`와 `L2 Theoretical Sectors Global Excessive = 6.29 MB`는 그대로다. 즉 이번 변화도 epilogue나 global store path를 고친 것이 아니라, mainloop 내부의 scheduling/register allocation을 흔든 실험이다.

즉 fragment load 위치 변경은 "load를 더 늦게 배치해서 use 지점에 가깝게 만들면 live range가 줄어든다"는 효과를 노린 실험으로 보는 편이 맞다. 이번 독립 측정 기준에서는 BK8은 benchmark 이득이 사실상 없고 profile 신호도 섞여 있어 민감한 영역으로 남는다. 반면 BK16은 register cliff 회피 효과가 분명해서, 이 변경의 실질적 수혜자가 된다.

## 15. `sgemm_warptiling_v4<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v4<128, 128, 8, 64, 32, 8>` vs `sgemm_warptiling_v4<128, 128, 16, 32, 64, 16>`

### 구조

`06_warptiling.cuh`의 `v4`는 `v3`의 loop skeleton을 유지한 채, CTA tile ingest helper를 `float4` 기반으로 다시 쓴 버전이다.

- `load_a_thread_fragment_vec4()`, `load_b_thread_fragment_vec4()`가 global `A/B`를 `float4`로 읽는다.
- shared store는 대칭적이지 않다. `A`는 shared에서 transpose layout을 유지해야 하므로 `float4` fragment를 네 개의 scalar store로 풀어 쓴다.
- `B`는 shared row-major 배치가 연속적이어서 `float4` store를 그대로 유지한다.
- epilogue는 아직 `v1::store_accum_to_gmem()`를 그대로 호출한다. `store_accum_to_gmem_vec4()` helper는 정의돼 있지만 현재 benchmark 경로에서는 주석 처리되어 있다.

즉 이번 `v4`는 "final C store까지 전부 vec4" 실험이 아니라, "global load를 vec4로 바꾸고, shared path는 layout이 허용하는 범위 안에서만 vectorize"한 버전으로 보는 편이 정확하다.

### 성능 측정 결과

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

이번에는 그림이 꽤 분명하다.

- `BK8 32x64`: `18.72 -> 20.22 TFLOPS`, `7.34 -> 6.79 ms`
- `BK8 64x32`: `18.84 -> 20.04 TFLOPS`, `7.29 -> 6.86 ms`
- `BK16 32x64`: `17.15 -> 20.99 TFLOPS`, `8.01 -> 6.55 ms`

특히 독립 benchmark 기준으로 `BK16 32x64`는 `v3` 대비 `+22.39%`, `v2 BK16` 대비로도 `19.13 -> 20.99 TFLOPS`로 `+9.70%`다. 즉 `v4`는 `v3`에서 무너졌던 BK16을 복구하는 수준을 넘어, 현재까지의 전체 최고점으로 끌어올린다. 다만 작은 문제 크기인 `128/256/512`에서는 `BK8 32x64`가 더 빠르고, `1024`부터 `4096`까지는 `BK16 32x64`가 세 변형 중 최고 성능이다.

### 4096 NCU 핵심 수치

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

stall 관점에서 보면 `v4`의 BK16이 특히 눈에 띈다.

- `v4 BK8 32x64`: `Not Selected 2.05`, `Barrier 0.80`, `Dispatch Stall 0.44`, `Short Scoreboard 0.20`, `Long Scoreboard 0.12`
- `v4 BK8 64x32`: `Not Selected 2.05`, `Barrier 0.81`, `Dispatch Stall 0.44`, `Short Scoreboard 0.20`, `Long Scoreboard 0.12`
- `v4 BK16 32x64`: `Not Selected 2.27`, `Barrier 0.52`, `Dispatch Stall 0.43`, `Short Scoreboard 0.16`, `Long Scoreboard 0.01`

즉 `BK16`은 단순히 bandwidth만 오른 게 아니라, `v3`에서 문제였던 long-scoreboard와 scheduler starvation도 거의 같이 정리된다.

### v3 -> v4 비교 해석

첫째, `v4`는 "`BK16 + prefetch`는 안 된다"는 해석을 뒤집는다.

`v3 BK16`과 `v4 BK16`의 차이는 대표 Nsight Compute 프로파일링 기준으로 아래처럼 요약된다.

- TFLOPS: `16.73 -> 20.34` (`+21.58%`)
- Duration: `9.00 -> 6.53 ms` (`-27.44%`)
- Registers / thread: `152 -> 128`
- Achieved Occupancy: `16.67 -> 32.27%`
- Eligible Warps / Scheduler: `1.16 -> 2.70`
- Issued Warps / Scheduler: `0.63 -> 0.83`
- No Eligible: `36.92 -> 17.24%`
- DRAM BW: `110.87 -> 147.43 GB/s`
- L2 BW: `477.56 -> 657.49 GB/s`
- `Stall Long Scoreboard`: `0.45 -> 0.01`

즉 `v4`의 핵심은 BK16이 다시 `128 regs/thread` 근처로 내려오면서 `2 blocks/SM` tier를 회복했다는 점이다. 이 결과는 `v3 BK16`의 실패를 "BK16과 prefetch 조합 자체의 한계"보다, "helper/codegen이 register cliff를 넘겼던 케이스"로 읽는 편이 더 자연스럽게 만든다.

둘째, BK8의 이득은 여전히 "epilogue 개선"이 아니라 load-side ingest 개선으로 보는 편이 맞다.

`v3 -> v4`로 가도 아래 항목은 그대로 남아 있다.

- global store 평균 사용량: 계속 `8 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: 계속 `6.29 MB`
- epilogue는 계속 scalar store
- shared bank conflict: 계속 `0`

그런데도 BK8은 둘 다 `20 TFLOPS`를 넘는다.

- `BK8 32x64`: `18.72 -> 20.22 TFLOPS`
- `BK8 64x32`: `18.84 -> 20.04 TFLOPS`

즉 `v4`의 BK8 성능 상승은 최종 C store가 아니라, global load instruction/transaction 경로를 더 효율적으로 만든 데서 나온 것으로 보는 편이 타당하다.

셋째, BK8에서는 "stall이 전부 개선됐다"기보다 "같은 occupancy tier에서 ingest cost가 더 낮아졌다"는 쪽이 더 가깝다.

`BK8 32x64`는 `v3` 대비 아래처럼 좋아진다.

- Compute Throughput: `71.85 -> 74.48%`
- DRAM BW: `129.20 -> 141.74 GB/s`
- L2 BW: `575.68 -> 633.80 GB/s`
- Issued Warps / Scheduler: `0.77 -> 0.79`
- No Eligible: `23.49 -> 21.32%`

하지만 stall mix를 보면 `Barrier`와 `Long Scoreboard`가 일괄적으로 줄어든 형태는 아니다. 즉 BK8의 이득은 "v3보다 stall이 더 깔끔해졌다"라기보다, vec4 load helper가 같은 register/occupancy tier 안에서 load-side overhead를 더 잘 정리한 결과로 읽는 편이 안전하다.

넷째, shared path 관점에서는 "무조건 전부 vec4"보다 "layout이 허용하는 곳만 vec4"가 더 현실적이라는 점이 드러난다.

- `A`는 shared transpose 때문에 `float4` fragment를 결국 scalar scatter로 풀어 써야 한다.
- `B`는 row-major 연속 배치라 vec4 store를 유지할 수 있다.
- 그 상태에서도 shared bank conflict는 세 변형 모두 `0`이다.

즉 `v4`는 vec4의 실질적 이득이 "모든 store를 억지로 vectorize하는 것"보다, global ingest와 contiguous path에 먼저 적용하는 데서 더 크게 나온다는 점을 보여준다.

보조 실험 메모: 아래 표는 최신 mainline benchmark와 별도로 rerun한 실험 결과라, 절대값보다 epilogue 변경의 상대 변화에 의미가 있다.

### `store_accum_to_gmem_vec4()`를 켜면

같은 `v4` 코드에서 마지막 epilogue만 `v1::store_accum_to_gmem()` 대신 `store_accum_to_gmem_vec4()`로 바꾸면, 결과는 "store 효율은 좋아지지만 kernel 전체는 오히려 느려진다"로 정리된다.

4096 기준 요약:

| Kernel | 현재 `v4` scalar epilogue | vec4 epilogue | 변화 |
| --- | ---: | ---: | ---: |
| `BK8 32x64` | `20.22 TFLOPS` | `19.46 TFLOPS` | `-3.78%` |
| `BK8 64x32` | `20.04 TFLOPS` | `19.74 TFLOPS` | `-1.48%` |
| `BK16 32x64` | `20.99 TFLOPS` | `19.91 TFLOPS` | `-5.11%` |

4096 기준 프로파일 비교:

| Kernel | Compute Throughput | Registers / thread | Achieved Occupancy | Eligible / Issued | DRAM BW | store path 변화 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `BK8 32x64` | `74.48 -> 72.79%` | `128 -> 121` | `32.16 -> 32.36%` | `2.40 / 0.79 -> 2.21 / 0.78` | `141.74 -> 139.82 GB/s` | global store `8 -> 32 byte/sector`, excessive `6.29 MB -> 0` |
| `BK8 64x32` | `73.50 -> 72.80%` | `128 -> 121` | `32.19 -> 32.39%` | `2.38 / 0.78 -> 2.22 / 0.78` | `139.73 -> 139.99 GB/s` | global store `8 -> 32 byte/sector`, excessive `6.29 MB -> 0` |
| `BK16 32x64` | `77.70 -> 70.36%` | `128 -> 147` | `32.27 -> 16.64%` | `2.70 / 0.83 -> 1.39 / 0.75` | `147.43 -> 138.15 GB/s` | global store `8 -> 32 byte/sector`, excessive `6.29 MB -> 0`, shared bank conflict `0 -> 864,274` |

해석은 네 줄로 정리된다.

- vec4 epilogue 자체는 분명히 성공했다. global store 평균 사용량이 `8 -> 32 byte/sector`로 올라가고, `L2 Theoretical Sectors Global Excessive`도 `6.29 MB -> 0`으로 사라졌다. 즉 scalar epilogue가 비효율적이었다는 관찰 자체는 맞다.
- 하지만 BK8 두 변형에서는 이 이득이 kernel tail에서 한 번 나타나는 개선에 그친다. register 수는 `128 -> 121`로 줄지만 여전히 같은 `2 blocks/SM` tier에 묶여 있어 residency 이득은 없고, 대신 compute throughput과 eligible warps가 조금씩 내려간다. DRAM bandwidth는 거의 비슷한 수준이라, 4096 benchmark 후퇴는 순수한 store-bandwidth 문제가 아니라 kernel 전체 codegen trade-off에 더 가깝다.
- stall도 BK8에서 일괄 개선으로 보이지 않는다. `32x64`는 `Barrier 0.80 -> 1.23`, `Dispatch Stall 0.44 -> 0.55`로 악화되고, `64x32`도 `Barrier 0.81 -> 1.22`, `Dispatch Stall 0.44 -> 0.55`로 비슷하다. 일부 `Not Selected`나 scoreboard 계열은 줄어도 issue quality 전체가 더 좋아지지는 않는다. 즉 epilogue store는 좋아졌지만 mainloop 쪽 scheduling quality가 같이 흔들려 전체적으로는 손해가 된다.
- BK16에서는 퇴행 원인이 더 분명하다. vec4 epilogue를 켜자 register가 `128 -> 147`로 다시 늘면서 occupancy가 `32.27 -> 16.64%`로 반 토막 나고, eligible warps도 `2.70 -> 1.39`로 급감한다. 여기에 shared bank conflict가 `0 -> 864,274`로 새로 생기고 `No Eligible`도 `17.24 -> 25.37%`로 나빠져, store-path 이득보다 register cliff와 codegen 교란 비용이 더 크게 작용한다.

즉 이 실험은 "epilogue를 vec4로 바꾸면 무조건 이득"이 아니라, "epilogue helper 하나가 전체 codegen과 register allocation을 흔들 수 있다"는 점을 보여준다. BK8에서는 occupancy tier가 유지되어 손실이 작고, BK16에서는 `128 regs/thread` 경계를 다시 넘어가면서 손실이 크게 증폭된다. 이유는 CUDA 컴파일러가 epilogue를 별도 함수처럼 다루지 않고, `__forceinline__`된 커널 전체에 대해 register allocation과 instruction scheduling을 다시 잡기 때문이다. 따라서 vec4 epilogue가 추가하는 `float4` pack과 주소 계산 임시값은 tail에만 머물지 않고, 커널 전체의 peak live set과 SASS 배치를 바꿀 수 있다. BK8에서는 이 재배치 비용이 작지만, BK16에서는 그 변화가 `128 regs/thread` 경계를 넘기면서 mainloop residency와 scheduler quality까지 같이 무너뜨린다. 따라서 현재 `v4`의 최고점은 vec4 global ingest를 유지하되, final C store는 아직 scalar epilogue로 두는 쪽이 더 낫다.

## 16. `sgemm_warptiling_v5<128, 128, 8, 32, 64, 8>` vs `sgemm_warptiling_v5<128, 128, 8, 64, 32, 8>` vs `sgemm_warptiling_v5<128, 128, 16, 32, 64, 16>`

### 구조

`06_warptiling.cuh`의 `v5`는 `v4`의 mainloop와 vec4 ingest path를 유지한 채, epilogue를 CUTLASS식 shared-memory staging 구조로 바꾼 버전이다.

- mainloop는 계속 `v4::load_*_thread_fragment_vec4()`와 `v4::store_*_thread_fragment_to_smem_vec4()`를 사용한다.
- 차이는 마지막 writeback이다. thread가 가진 accumulator를 곧바로 global에 scalar store하지 않고, 먼저 `store_accum_to_epilogue_smem()`으로 CTA-wide epilogue shared tile에 재배치한다.
- 그 다음 `__syncthreads()` 뒤에 `store_epilogue_smem_to_gmem()`가 CTA 전체 tile을 cooperative `float4` global store로 내보낸다.
- `KernelTraits`에는 이를 위해 `kEpilogueSmemStride = BLOCK_N + 4`, `kEpilogueSmemNumElems = BLOCK_M * kEpilogueSmemStride`, `kV5SmemBytes = max(kTotalSmemElems, kEpilogueSmemNumElems)`가 추가된다.

즉 `v5`의 목적은 명확하다. `v4`에서 남아 있던 scalar epilogue의 낮은 store sector utilization을 shared-memory reordering으로 정리하는 것이다.

> `v5`는 `v4`의 residual scalar epilogue를 바로 겨냥한 첫 mainline 시도지만, 동시에 CTA-wide epilogue scratch라는 새 비용을 도입한다.

default `v5`는 `store_epilogue_smem_to_gmem()` 안에서 매 iteration마다 `vec_idx`, `row`, `vec_col`을 계산하는 첫 번째 구현을 기준으로 한다. `smem_offset/gmem_offset += advance` 형태로 바꾼 버전은 이 섹션 뒤의 추가 실험으로 분리해 다룬다.

### 성능 측정 결과

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

4096 기준으로 보면 `v5`는 세 변형 모두 `v4`를 넘지 못한다.

- `BK8 32x64`: `20.22 -> 18.33 TFLOPS`
- `BK8 64x32`: `20.04 -> 18.29 TFLOPS`
- `BK16 32x64`: `20.99 -> 19.88 TFLOPS`

즉 CUTLASS-style epilogue staging은 분명한 목적을 갖고 들어왔지만, mainline 결과만 보면 store-path 이득보다 residency 손해가 더 크게 나타난다.

### 4096 NCU 핵심 수치

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

이 표에서 가장 먼저 읽어야 할 것은 store path와 residency가 동시에 극단으로 움직였다는 점이다.

- `v4`의 global store 평균 사용량 `8 byte / sector`, `L2 Theoretical Sectors Global Excessive = 6.29 MB`는 `v5`에서 `32 byte / sector`, `0`으로 정리된다.
- 반면 dynamic shared memory는 `v4 BK8`의 `8.32 KB`, `v4 BK16`의 `16.51 KB`에서 `v5`는 세 변형 모두 `67.58 KB`로 급증한다.
- 그 결과 `Block Limit Shared Mem = 1`, theoretical occupancy `16.67%`가 되고, `eligible warp`와 `issued warp`가 `v4`보다 눈에 띄게 줄어든다.

### `v4 -> v5` 비교 해석

첫째, `v5`는 scalar epilogue inefficiency가 실제 문제였음을 다시 확인해 준다.

- global store 평균 사용량: `8 -> 32 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: `6.29 MB -> 0`

즉 CUTLASS-style epilogue staging의 방향 자체는 맞다. `v4`의 residual inefficiency가 정말 global store path에 있었다는 점은 `v5`가 더 깔끔하게 보여 준다.

둘째, 문제는 그 대가가 너무 컸다는 점이다. 이번에는 register cliff가 아니라 shared-memory cliff를 밟는다.

- `BK8 32x64`: `Eligible Warp 2.40 -> 1.19`, `Issued Warp 0.79 -> 0.68`, `No Eligible 21.32 -> 31.81%`
- `BK8 64x32`: `Eligible Warp 2.38 -> 1.20`, `Issued Warp 0.78 -> 0.68`, `No Eligible 22.14 -> 31.62%`
- `BK16 32x64`: `Eligible Warp 2.70 -> 1.35`, `Issued Warp 0.83 -> 0.74`, `No Eligible 17.24 -> 25.58%`

`v5`는 shared path 자체를 크게 망치지 않는다. `L1 Wavefronts Shared Excessive = 0`이고, shared conflict 신호도 주 병목으로 보기 어려운 수준이다. 하지만 epilogue scratch가 CTA당 `67.58 KB`가 되면서 residency가 반 토막 나고, scheduler가 고를 warp pool 자체가 작아진다. 결국 store path에서 번 이득보다 latency hiding 손실이 더 크게 남는다.

셋째, `Stall Not Selected`가 `v4`보다 줄어든 것은 이번에는 개선 신호로 읽기 어렵다. `1 block/SM`에 가까운 환경에서는 scheduler가 고를 warp 수 자체가 적으므로, `Not Selected` 감소는 "고를 warp가 줄었다"의 부산물일 수 있다. 이번 단계에서 더 설명력이 큰 지표는 occupancy, eligible warps, issued warps, `No Eligible`이다.

> `8 byte/sector -> 32 byte/sector` 자체는 맞는 개선이지만, `67.58 KB`짜리 CTA epilogue scratch가 현재 `128x128` family에서는 더 비싼 대가였다.

즉 `v5`는 앞선 `store_accum_to_gmem_vec4()` 실험이 보여 준 "store path를 고치면 sector efficiency는 좋아진다"는 관찰을 더 정돈된 형태로 재현하지만, 이번에는 register cliff 대신 shared-memory cliff로 성능이 막힌다.

### 추가 실험: epilogue address-generation 패턴 변경

default `v5`는 loop 안에서 `vec_idx / kEpilogueVectorsPerRow`, `vec_idx % kEpilogueVectorsPerRow`를 계산해 row/column 위치를 정하는 구현이다. 추가 실험에서는 이 부분을 loop 밖에서 `smem_offset`, `gmem_offset`, `smem_advance`, `gmem_advance`를 미리 계산해 두고, loop 안에서는 `offset += advance`만 수행하도록 바꿨다.

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

4096 기준 profile 비교:

| Kernel | Duration (ms) | Compute Throughput | Registers / thread | Eligible / Issued | No Eligible | 주요 stall 변화 |
| --- | --- | --- | --- | --- | --- | --- |
| `BK8 32x64` | `7.91 -> 10.55` | `63.90 -> 49.36%` | `126 -> 121` | `1.19 / 0.68 -> 0.87 / 0.52` | `31.81 -> 47.78%` | `Barrier 0.38 -> 1.23`, `Long Scoreboard 0.07 -> 0.56` |
| `BK8 64x32` | `7.90 -> 9.97` | `64.00 -> 52.96%` | `126 -> 121` | `1.20 / 0.68 -> 0.92 / 0.55` | `31.65 -> 44.95%` | `Barrier 0.37 -> 1.24`, `Long Scoreboard 0.07 -> 0.49` |
| `BK16 32x64` | `7.28 -> 7.27` | `69.86 -> 69.46%` | `139 -> 147` | `1.35 / 0.74 -> 1.39 / 0.75` | `25.63 -> 25.01%` | `Barrier 0.18 -> 0.19`, `Long Scoreboard 0.07 -> 0.10` |

이 실험에서 중요한 점은, store path와 residency의 큰 틀은 전혀 바뀌지 않는다는 것이다.

- global store 평균 사용량: 계속 `32 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: 계속 `0`
- dynamic shared memory: 계속 `67.58 KB`
- `Block Limit Shared Mem`: 계속 `1`

즉 여기서 달라진 것은 memory access pattern 자체보다, epilogue store loop의 주소 계산이 만들어 내는 codegen과 scheduling이다.

BK8 두 변형은 크게 무너진다.

- benchmark 기준으로 `18.33 -> 13.63 TFLOPS`, `18.29 -> 13.17 TFLOPS`
- profile 기준으로 `Eligible Warp`, `Issued Warp`, `Compute Throughput`가 함께 내려간다.
- 대신 `Barrier`와 `Long Scoreboard`가 크게 늘어난다.

반면 BK16은 소폭 좋아진다.

- benchmark 기준 `19.88 -> 20.09 TFLOPS`
- profile 기준 `Eligible Warp 1.35 -> 1.39`, `Issued Warp 0.74 -> 0.75`
- 하지만 여전히 `v4 BK16`의 `20.99 TFLOPS`에는 못 미친다.

즉 이번 추가 실험의 해석은 "`/`, `%`를 없애면 더 빠르다"가 아니다. 같은 주소열을 `offset` 누적 방식으로 표현했을 때 BK8에서는 loop-carried dependency 또는 그에 따른 불리한 codegen이 더 크게 드러났고, BK16에서는 그 비용이 상대적으로 작거나 일부 상쇄된 것이다. 특히 BK8에서 register 수가 `126 -> 121`로 줄어도 shared memory가 이미 `1 block/SM`을 강제하고 있으므로, 그 작은 register 절감은 residency를 복구하지 못한다.

> loop 안에서 `%`와 `/`를 없앴다는 사실보다, 같은 주소열을 어떤 의존성 형태로 표현했는지가 더 중요했다.

## 핵심 해석 정리

### 1. 가장 큰 구조적 성능 점프는 두 번 나온다

- `sgemm_smem_tiling -> blocktiling_1d<64,64,8,8,4>`
  - `2.93 -> 6.23 TFLOPS`
  - shared reuse를 register reuse로 확장한 효과
- `blocktiling_1d<128,128,16,64,4> -> blocktiling_2d<128,128,8,8,8,4>`
  - `11.99 -> 16.05 TFLOPS`
  - 1D accumulate를 2D outer-product로 바꾼 효과

즉 이 데이터셋에서는 단순 shared tiling보다 register/blocking 구조 변화가 훨씬 큰 영향을 준다.

### 2. occupancy는 중간 지표일 뿐 목표가 아니다

빠른 커널일수록 occupancy가 반드시 높은 것은 아니다.

- naive/smem 단계: occupancy 약 `66%`
- 빠른 128x128 계열: occupancy 약 `32%`

그런데도 후자가 훨씬 빠르다. 이는 reuse와 ILP가 좋아지면 낮은 occupancy를 충분히 상쇄할 수 있음을 보여준다.

더 중요한 점은, 현재 warptiling 계열의 낮은 occupancy가 상당 부분 "피할 수 없는 trade-off"라는 것이다.

- `v2` 변형들은 `119~127 regs/thread` 범위에 있다.
- thread당 `8x8` output tile accumulator만 해도 `64`개의 float register가 필요하다.
- 여기에 operand fragment, 주소 계산, staging용 임시 register까지 더해지면 register pressure가 높아지는 것이 자연스럽다.

즉 현재의 `32%` 안팎 occupancy는 구현 실수라기보다, reuse와 ILP를 사기 위해 지불한 비용에 가깝다. 실제로 현재 최고점인 `warptiling_v4<128,128,16,32,64,16>`도 `128 regs/thread`, `32.27%` occupancy에서 나온다.

다만 `v3`는 여기서 한 가지 중요한 예외를 추가로 보여준다. 낮은 occupancy 자체는 문제가 아니지만, register 증가로 occupancy tier를 한 단계 더 잃는 순간 손해가 급격히 커진다.

- `v2 BK16`: `127 regs/thread`, `32.23%` occupancy, `19.13 TFLOPS`
- `v3 BK8`: `127~128 regs/thread`, `~32%` occupancy, `18.72~18.84 TFLOPS`
- `v3 BK16`: `152 regs/thread`, `16.67%` occupancy, `17.15 TFLOPS`

또한 이 구간에서는 "조금 낮춘 register 수"가 실제 occupancy 단계 자체를 바꾸지 못할 가능성이 크다.

- 현재 block size는 `256 threads`
- `119 -> 128 regs/thread` 범위에서는 여전히 register limit 때문에 `2 blocks/SM`, 약 `33%` theoretical occupancy에 묶인다
- `129 regs/thread`를 넘기면 바로 `1 block/SM`, 약 `16.7%` theoretical occupancy로 떨어질 수 있다
- 다음 occupancy 단계로 실제로 올라가려면 register 수를 대략 `85 regs/thread` 이하 수준으로 크게 낮춰 `3 blocks/SM`를 허용해야 한다

따라서 occupancy를 분석할 때는 아래처럼 보는 것이 맞다.

- 낮은 occupancy 자체를 1순위 병목으로 보지 않는다.
- 그 낮은 occupancy 아래에서 scheduler stall, scoreboard, barrier, epilogue inefficiency가 얼마나 큰지를 먼저 본다.
- occupancy 최적화는 "정말로 다음 tier로 넘어갈 수 있는 큰 register 절감"이 가능할 때, 혹은 반대로 `v3 BK16`처럼 하위 tier로 추락하는 것을 막아야 할 때만 우선순위가 된다.

`v5`는 같은 원칙이 shared memory에도 그대로 적용된다는 점을 추가로 보여준다.

- `v4 BK8`의 dynamic shared memory는 `8.32 KB`, `v4 BK16`은 `16.51 KB`였다.
- `v5`는 CUTLASS-style epilogue scratch 때문에 세 변형 모두 `67.58 KB`를 사용한다.
- 그 결과 `Block Limit Shared Mem = 1`, theoretical occupancy `16.67%`로 떨어진다.

즉 낮은 occupancy 자체보다 중요한 것은, register든 shared memory든 다음 residency tier로 떨어지는 순간이다.

### 3. 병목은 단계별로 이동한다

이번 흐름에서 병목은 다음처럼 이동한다.

1. naive: `LG Throttle`
2. smem: `MIO Throttle + Barrier + shared-store conflict`
3. 1D block tiling: `Scoreboard dependency`
4. 2D block tiling 이후: `shared wavefront excess`
5. refined warptiling(`v1`): `Not Selected + residual scoreboard + barrier`
6. explicit register staging(`v2`): `Not Selected + residual long scoreboard + epilogue store inefficiency`
7. 1-stage register prefetch(`v3`): BK8에서는 scoreboard가 더 줄지만, BK16에서는 register cliff로 `No Eligible`이 급증한다
8. vec4 global ingest(`v4`): BK16 register cliff는 복구되지만, 남은 병목은 `Not Selected`와 residual scalar epilogue inefficiency에 가깝다
9. CUTLASS-style epilogue staging(`v5`): store path는 정리되지만 shared-memory cliff 때문에 eligible warp와 issue rate가 다시 줄어든다

즉 최적화는 "하나의 병목을 제거하면 다음 병목이 드러나는 과정"으로 보는 것이 맞다.

### 4. vec4는 "전부 vectorize"보다 load-side ingest에 먼저 쓰는 게 맞다

`v4`는 vec4의 유효 범위를 더 구체적으로 보여준다.

- `A/B` global load는 vec4로 바꾼다.
- `A`의 transposed shared store는 scalar scatter로 푼다.
- `B`의 contiguous shared store만 vec4를 유지한다.
- final C epilogue는 여전히 scalar store다.

그런데도 `v4 BK16`은 `20.99 TFLOPS`까지 올라간다. 반면 profile에서는 global store 평균 사용량 `8 byte / sector`와 `6.29 MB` excessive sectors가 그대로 남아 있다.

즉 이번 결과의 교훈은 간단하다. vec4의 큰 이득은 먼저 global ingest 쪽에서 나오며, final store나 transposed scatter까지 한 번에 억지로 vectorize해야만 성능이 나는 것은 아니다.

### 5. warptiling의 첫 성패는 warp tile 방향과 lane mapping이 함께 결정한다

이번 `warptiling_v0` 두 변형은 그 점을 아주 직접적으로 보여준다.

- 둘 다 thread당 `8x8` output tile
- 둘 다 `64` accumulators
- 둘 다 `125` registers/thread

그런데 `32x64`는 shared excessive wavefront와 bank conflict가 크게 남고, `64x32`는 같은 구조에서 그것이 사라진다. 즉 warptiling의 첫 성패는 "warp를 하나 더 넣었다"가 아니라, "그 warp가 shared/global path와 맞는 shape와 lane mapping을 가졌는가"에 달려 있다.

### 6. refined warptiling은 shape 민감도를 줄이고 병목을 scheduler/scoreboard 쪽으로 옮긴다

`warptiling_v1`은 바로 위의 교훈을 실제 수정으로 이어 붙인 결과다.

- `32x64`: `16.16 -> 17.28 TFLOPS`
- `64x32`: `16.86 -> 17.37 TFLOPS`
- 두 변형 모두 shared bank conflict `0`, excessive wavefront `0`
- global store 평균 사용량: `4 -> 8 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: `14.68 -> 6.29 MB`

즉 `v1`의 효과는 단순히 조금 더 빠르다는 데 있지 않다. shared-memory consumption path와 epilogue store를 함께 정리하면서, warp shape에 따른 편차를 거의 없애고 병목을 `Not Selected`, `Long Scoreboard`, `Barrier` 같은 더 상위의 scheduling/pipeline 문제로 이동시킨다.

### 7. explicit register-staged global->shared는 shared 병목이 아니라 load scheduling과 loop overhead를 다듬는다

`warptiling_v2`는 `v1`의 lane mapping과 warp MMA를 그대로 둔 채, CTA tile ingest path를 source-level register staging으로 분해한 버전이다.

- `32x64 BK8`: `17.28 -> 18.25 TFLOPS`
- `64x32 BK8`: `17.37 -> 17.86 TFLOPS`
- `32x64 BK16`: `19.13 TFLOPS`, 당시 전체 최고점

중요한 건 무엇이 바뀌지 않았는가다.

- shared bank conflict: 계속 `0`
- shared excessive wavefront: 계속 `0`
- global store 평균 사용량: 계속 `8 byte / sector`
- `smsp__inst_executed_op_ldgsts.sum`: 계속 `0`

즉 `v2`의 이득은 async copy 같은 새 hardware path에서 온 것이 아니다. manual register fragment staging으로 global load와 shared store의 scheduling을 더 명시적으로 만들고, 여기에 `BLOCK_K=16`으로 barrier/loop overhead를 줄인 결과다. 이제 남은 큰 headroom은 shared layout보다 prefetch/pipeline과 epilogue 쪽에 있다.

### 8. 1-stage register prefetch는 "같은 occupancy tier 안"에서만 강하게 먹힌다

`warptiling_v3`는 `v2`의 register staging 위에 next-tile global load를 한 iteration 앞당긴 버전이다.

- `BK8 32x64`: `18.25 -> 18.72 TFLOPS`
- `BK8 64x32`: `17.86 -> 18.84 TFLOPS`
- `BK16 32x64`: `19.13 -> 17.15 TFLOPS`

이 차이는 prefetch라는 아이디어 자체보다, 그 prefetch가 늘린 live range를 현재 kernel shape가 감당할 수 있었는지에 의해 갈린다.

- BK8은 `127~128 regs/thread`에서 여전히 `2 blocks/SM`를 유지한다.
- BK16은 `152 regs/thread`로 올라가며 `1 block/SM`으로 떨어진다.

즉 `v3`의 교훈은 단순하다. register prefetch는 scoreboarding을 줄이는 데 실제로 유효하지만, 그 대가로 occupancy tier를 잃으면 이득이 바로 상쇄된다.

다만 `v4`는 여기서 한 가지 중요한 보정을 추가한다. 같은 prefetch skeleton이라도 load helper와 staging codegen을 바꾸면 BK16을 다시 `128 regs/thread`로 끌어내릴 수 있다. 즉 더 정확한 교훈은 "`BK16 + prefetch`는 안 된다"가 아니라, "`128 regs/thread` 경계를 넘기는 prefetch는 안 된다"에 가깝다.

### 9. `v4`는 BK16의 register cliff를 되돌리면서 새 최고점을 만든다

`warptiling_v4`는 `v3`의 partial prefetch를 버린 버전이 아니다. 오히려 그 skeleton을 유지한 채, global ingest helper를 vec4 기반으로 다시 써서 codegen과 register pressure를 정리한 버전이다.

- `v4 BK8 32x64`: `20.22 TFLOPS`
- `v4 BK8 64x32`: `20.04 TFLOPS`
- `v4 BK16 32x64`: `20.99 TFLOPS`, 현재 최고점

특히 `BK16`에서는 아래 변화가 결정적이다.

- Registers / thread: `152 -> 128`
- Achieved Occupancy: `16.67 -> 32.27%`
- Eligible Warps / Scheduler: `1.16 -> 2.70`
- `Stall Long Scoreboard`: `0.45 -> 0.01`

즉 `v4`는 "prefetch의 한계"보다 "register cliff의 한계"가 더 본질적이었다는 점을 확인해 준다. 현재 최적 조합은 "refined warptiling + explicit register staging/prefetch + vec4 global ingest + BK16" 쪽에 더 가깝다.

### 10. CUTLASS-style epilogue는 store path를 고치지만, CTA-wide staging 자체가 공짜는 아니다

`v5`는 `v4`에서 남아 있던 scalar epilogue inefficiency를 정면으로 겨냥한 첫 mainline 버전이다.

- global store 평균 사용량: `8 -> 32 byte / sector`
- `L2 Theoretical Sectors Global Excessive`: `6.29 MB -> 0`

즉 store path를 고친다는 목적 자체는 성공했다. 하지만 동시에 CTA-wide epilogue scratch가 `67.58 KB`까지 커지며 `Block Limit Shared Mem = 1`, theoretical occupancy `16.67%`가 된다.

- `BK8 32x64`: `20.22 -> 18.33 TFLOPS`
- `BK8 64x32`: `20.04 -> 18.29 TFLOPS`
- `BK16 32x64`: `20.99 -> 19.88 TFLOPS`

즉 이번에는 register cliff가 아니라 shared-memory cliff가 전체 성능을 막는다. 추가로 epilogue address-generation 패턴을 `offset += advance` 형태로 바꾼 실험은, 같은 store path와 같은 shared-memory footprint 아래에서도 BK8을 `13 TFLOPS`대까지 떨어뜨릴 수 있음을 보여 준다. 이는 epilogue 최적화에서 중요한 것이 단순한 loop 산술 개수보다, 그 산술이 만들어 내는 의존성 구조와 codegen/scheduling이라는 점을 뜻한다.

결국 CUTLASS-style epilogue의 방향성은 맞지만, 현재 `128x128` family에서는 helper 하나를 옮기는 것만으로는 충분하지 않다. epilogue, residency, pipeline이 함께 균형을 맞춰야 library-grade 결과로 이어진다.

## `torch.matmul` / CUTLASS와의 비교

현재 우리 쪽 최고점은 `warptiling_v4<128,128,16,32,64,16>`의 `20.99 TFLOPS`다. 4096 독립 benchmark 기준으로 라이브러리 kernel들과 나란히 놓으면 아래처럼 정리된다.

| Kernel | 4096 TFLOPS | `v4 BK16` 대비 |
| --- | ---: | ---: |
| `warptiling_v4<128,128,16,32,64,16>` | 20.99 | 기준 |
| `cutlass_simt_128x128x8_32x64x8_2stage` | 21.02 | `+0.15%` |
| `cutlass_simt_128x128x8_64x32x8_2stage` | 21.05 | `+0.29%` |
| `cutlass_universal_simt_128x256_8x4` | 21.64 | `+3.10%` |
| `torch.matmul` | 22.72 | `+8.27%` |

여기서 중요한 건 비교 기준 kernel의 성격이다.

- `torch.matmul`은 대표 Nsight Compute 프로파일링에서 `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`를 호출한다. 즉 이번 fp32 baseline은 tensor core kernel이 아니라 라이브러리 수준의 SIMT kernel이다.
- 가장 직접적인 동형 비교 기준은 `cutlass_simt_128x128x8_*_2stage` 두 변형이다. 같은 `128x128x8` 계열에서 우리 `v4 BK16`보다 아주 조금 앞선다.
- `cutlass_universal_simt_128x256_8x4`와 `torch.matmul`은 `128x256`, `256x128`처럼 더 큰 CTA tile을 쓰며, `2048/4096` 같은 큰 문제에서 더 높은 상한을 보여준다.

대표 Nsight Compute 프로파일링 기준 핵심 수치는 아래와 같다.

| Kernel | Compute Throughput | Registers / thread | Achieved Occupancy | Eligible / Issued | DRAM BW | Global Store Avg Bytes / Sector | L2 Excessive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `torch.matmul` | 76.76% | 202 | 16.67% | `1.59 / 0.88` | 69.74 GB/s | 32 | 0 |
| `cutlass_simt_128x128x8_32x64x8_2stage` | 82.52% | 122 | 32.41% | `2.47 / 0.88` | 155.01 GB/s | 32 | 0 |
| `cutlass_simt_128x128x8_64x32x8_2stage` | 82.46% | 122 | 32.40% | `2.47 / 0.88` | 141.13 GB/s | 32 | 0 |
| `cutlass_universal_simt_128x256_8x4` | 76.20% | 254 | 16.67% | `1.59 / 0.87` | 67.06 GB/s | 32 | 0 |
| `warptiling_v4<128,128,16,32,64,16>` | 77.70% | 128 | 32.27% | `2.70 / 0.83` | 147.43 GB/s | 8 | 6.29 MB |

이 비교에서 읽을 수 있는 포인트는 여섯 가지다.

- 첫째, 가장 가까운 목표는 CUTLASS SIMT 2-stage다. 같은 `128x128x8` family에서 benchmark gap이 `0.15~0.29%`밖에 안 나는데도, CUTLASS는 `Issued Warp / Scheduler = 0.88`, `Global Store Avg Bytes / Sector = 32`, `L2 Excessive = 0`을 이미 달성한다. 즉 지금 남은 차이는 "새로운 대원칙"보다 true 2-stage overlap과 더 나은 epilogue/store path에 더 가깝다.
- 둘째, epilogue/store path는 이제 분명한 1차 격차다. reference kernel들은 모두 global store가 `32 byte/sector`이고 `L2 Theoretical Sectors Global Excessive = 0`인데, 현재 `v4 BK16`만 `8 byte/sector`, `6.29 MB` excessive sectors가 남아 있다.
- 셋째, occupancy는 여전히 목표가 아니라 제약 조건이다. `torch.matmul`과 `cutlass_universal`은 `16.67%` occupancy, `202/254 regs/thread`인데도 더 높은 benchmark를 낸다. 즉 "무조건 occupancy를 더 올리는 것"보다, 더 큰 CTA tile과 더 좋은 schedule/pipeline이 더 중요할 수 있다.
- 넷째, 그렇다고 곧바로 `128x256`이나 `256x128`으로 넘어가는 것도 순서가 아니다. `cutlass_universal`은 `1024`에서는 `10.76 TFLOPS`로 `v4 BK16`보다 한참 느리고, `2048/4096`에서만 상한이 높다. 따라서 먼저 현재 `128x128` family에서 CUTLASS SIMT 2-stage 수준을 맞춘 뒤, 그 다음에 큰 CTA tile을 탐색하는 편이 자연스럽다.
- 다섯째, shared bank conflict는 이제 1차 목표가 아니다. reference kernel 중에는 shared bank conflict counter가 크거나 0이 아닌 경우도 있지만 여전히 빠르다. 즉 앞으로 shared layout은 "문제가 생기면 고치는 진단 항목"이지, 기본 우선순위의 맨 앞에 둘 항목은 아니다.
- 여섯째, `v5`는 왜 CUTLASS식 epilogue를 그대로 가져오는 것만으로는 충분하지 않은지도 보여 준다. `32 byte/sector`, `L2 Excessive = 0`은 달성했지만 CTA당 `67.58 KB` epilogue scratch 때문에 `1 block/SM`에 묶여 `v4`를 넘지 못했다. 즉 CUTLASS와의 차이는 epilogue helper 유무가 아니라, epilogue를 포함한 전체 pipeline과 residency balance에 더 가깝다.

## 다음 단계 제안

위 비교를 기준으로 보면 방향 자체는 맞지만, 우선순위는 조금 바꾸는 편이 낫다. 핵심은 "CUTLASS SIMT 2-stage와의 작은 격차를 먼저 닫고, 그 다음에 torch/cuBLAS가 보여주는 더 큰 CTA tile 영역으로 넘어간다"는 순서다.

1. 2-stage software pipeline / double buffering
2. global store / epilogue 추가 정리
3. CTA tile shape / swizzle 탐색
4. register / shared-memory budget 제어
5. deeper shared-memory layout refinement

### 1. 2-stage software pipeline / double buffering: 가장 가까운 CUTLASS 기준을 먼저 맞추는 단계

가장 직접적인 reference는 `cutlass_simt_128x128x8_*_2stage`다. 같은 `128x128x8` family에서 이미 `21 TFLOPS`를 넘고, `Issued Warp / Scheduler`도 `0.88`까지 올라가 있다. 즉 지금 우리 구현에서 가장 먼저 메워야 할 구조적 차이는 true 2-stage overlap이다.

- shared memory를 두 벌 두고 `ping-pong`한다.
- next tile을 register에 오래 들고 있지 말고, 가능한 빨리 alternate shared buffer로 넘긴다.
- `global -> shared(next) -> compute(curr)`를 steady-state 안에서 겹치게 만든다.

여기서 목표는 추상적인 "더 깊은 prefetch"가 아니라, CUTLASS 2-stage처럼 issue rate와 latency hiding을 동시에 올리는 것이다.

### 2. global store / epilogue 추가 정리: `v5`의 store-path 개선은 살리고, CTA-wide scratch 비용은 줄이는 단계

reference kernel들은 전부 `Global Store Avg Bytes / Sector = 32`, `L2 Excessive = 0`이다. 반면 현재 `v4 BK16`만 store path가 `8 byte/sector`, `6.29 MB` excessive sectors에 머물러 있다. 즉 epilogue는 이제 "나중에 보면 좋은 항목"이 아니라, CUTLASS SIMT와의 작은 격차를 줄이기 위한 직접 과제다.

`v5`는 이 방향의 가능성과 제약을 동시에 보여 줬다.

- global store 평균 사용량 `32 byte/sector`, `L2 Excessive = 0` 자체는 달성할 수 있다.
- 하지만 full CTA epilogue staging은 현재 `128x128` family에서 `67.58 KB` shared scratch를 요구해 `1 block/SM`으로 떨어진다.

- output lane mapping을 store-friendly하게 다시 맞춘다.
- 가능하면 vectorized output store를 도입하되, `store_accum_to_gmem_vec4()`나 `v5`처럼 codegen 전체를 흔들거나 CTA-wide scratch를 과도하게 키우지 않는 방식으로 한다.
- 필요하면 accumulator layout 자체를 epilogue 기준으로 한 번 더 재배치한다.
- warp-local staging, 더 작은 epilogue tile, cooperative strip-mined store처럼 shared-memory footprint를 줄이는 방향을 우선 본다.

핵심 제약은 간단하다. store path는 CUTLASS처럼 만들되, BK16이 다시 `129+ regs/thread`로 넘어가거나 `v5`처럼 `67 KB`대 shared-memory cliff를 밟게 해서는 안 된다.

### 3. CTA tile shape / swizzle 탐색: `128x128` family를 정리한 뒤 큰 타일 상한을 보는 단계

`torch.matmul`은 `256x128_8x4`, `cutlass_universal`은 `128x256_8x4`를 쓴다. 둘 다 `2048/4096`에서 현재 `v4`보다 높은 상한을 보여준다. 즉 다음 상위 단계는 `128x128` 안에서만 최적화하는 것이 아니라, CTA shape 자체를 넓혀 보는 것이다.

다만 이건 3순위가 맞다.

- `cutlass_universal`은 `1024`에서 `v4 BK16`보다 훨씬 느리다.
- 즉 큰 CTA tile은 "항상 더 낫다"가 아니라, 큰 문제에서 상한을 높여 주는 선택지다.
- 따라서 먼저 `128x128` family에서 2-stage + epilogue를 정리한 뒤, 그 다음 `256x128`, `128x256`, swizzle을 비교하는 편이 맞다.

### 4. register / shared-memory budget 제어: 목표가 아니라 전 단계 전체의 guardrail

`v3 BK16`의 실패와 `v4 BK16`의 복구를 보면 register budget은 여전히 핵심 guardrail이다. `v5`는 여기에 shared-memory budget도 같은 급의 guardrail이라는 점을 추가한다. 하지만 reference kernel 비교까지 놓고 보면, register pressure 제어를 독립적인 최우선 목표로 둘 필요는 없다. 더 정확한 목표는 "새 구조를 넣되 필요한 residency와 issue rate를 깨지 않는 수준으로 관리한다"이다.

- `128x128 BK16` family에서는 가능하면 `128 regs/thread` 근처를 지킨다.
- 같은 family에서 dynamic shared memory가 `67 KB`대까지 커지면 `1 block/SM`으로 떨어질 수 있으므로, epilogue/pipeline용 shared scratch도 같이 감시한다.
- 큰 CTA tile에서는 occupancy가 `16.67%`여도 benchmark가 더 좋을 수 있으니, occupancy 수치만 보고 실패로 판단하지 않는다.
- 대신 `Issued Warp / Scheduler`, `No Eligible`, epilogue store efficiency를 함께 본다.

즉 register pressure와 shared-memory footprint는 1~3번 단계를 추진할 때 계속 감시해야 할 제약이지, 단독 목표는 아니다.

### 5. deeper shared-memory layout refinement: 문제 발생 시에만 들어가는 단계

현재 데이터 기준으로 shared path는 이미 1차 병목에서 내려왔다. reference kernel들 중에는 shared bank conflict counter가 작지 않은데도 더 빠른 경우가 있다. 즉 shared layout refinement는 "항상 다음 단계"가 아니라, 새 pipeline이나 새 CTA shape가 실제로 shared 병목을 다시 만들었을 때 들어가면 된다.

- 새 설계에서 bank conflict나 wavefront excess가 다시 커질 때 대응한다.
- `A/B` layout, `4x4` island 소비 순서, swizzle은 그때 다시 본다.
- 현 시점에서는 pipeline, epilogue, CTA shape보다 우선순위가 낮다.

## 결론

여기서 정리한 프로파일링 결과는 `how_to_optimize_sgemm.kr.md`의 단계적 접근이 실제 NCU 수치로도 잘 검증된다는 점을 보여준다.

- naive에서 시작하면 global load path가 병목이다.
- shared tiling만으로는 부족하고, register blocking이 들어가야 성능이 크게 오른다.
- 진짜 큰 전환점은 2D outer-product다.
- 이후 `BLOCK_K`와 vec4는 이미 좋아진 구조를 다듬는 역할을 한다.
- warptiling v0는 warp hierarchy의 가능성을 보여줬고, warptiling v1은 lane mapping과 `4x4` island store refinement가 실제로 유효함을 증명했다.
- warptiling v2는 explicit register-staged global->shared가 BK8에서도 추가 이득을 만들고, `BLOCK_K=16`과 결합하면 새 최고점을 만든다는 점을 보여준다.
- warptiling v3는 1-stage register prefetch가 BK8에서는 실제로 잘 먹히지만, BK16에서는 register pressure를 통해 바로 역효과가 날 수 있음을 보여준다.
- warptiling v4는 같은 prefetch skeleton에서도 vec4 global ingest와 더 나은 codegen으로 BK16의 register cliff를 되돌릴 수 있음을 보여준다.
- warptiling v5는 CUTLASS-style epilogue staging이 store path를 실제로 정리할 수 있음을 보여주지만, full CTA epilogue scratch는 현재 `128x128` family에서 shared-memory cliff를 만들어 `v4`를 넘지 못한다.
- `v5`의 address-generation 추가 실험은 epilogue loop의 겉보기 산술을 줄이는 것보다, 그 표현이 만들어 내는 의존성 구조와 codegen/scheduling이 더 중요하다는 점을 보여준다.

최종적으로 현재 최고점은 `sgemm_warptiling_v4<128, 128, 16, 32, 64, 16>`의 `20.99 TFLOPS`다. 즉 현재 mainline에서 가장 잘 결합된 상태는 여전히 "refined warptiling + explicit register staging/prefetch + vec4 global ingest + BK16"이라고 보는 편이 맞다.

이제 `BK16`도 더 이상 `v2`의 약점 구간이 아니다.

- `v2 BK16`: `19.13 TFLOPS`
- `v3 BK16`: `17.15 TFLOPS`
- `v4 BK16`: `20.99 TFLOPS`

즉 다음 도약은 prefetch를 더 깊게 넣는 것 자체가 아니라, `v4`가 회복한 register budget을 지키면서 진짜 2-stage pipeline과 더 나은 epilogue store를 결합하는 데 달려 있다. `v5`가 보여 준 것처럼 store path를 고치는 방향 자체는 맞지만, 그 구현이 shared-memory residency를 무너뜨려서는 안 된다. occupancy는 여전히 목표 그 자체가 아니라 trade-off의 결과이지만, 이번 결과는 그 trade-off의 핵심 경계가 "prefetch 사용 여부"보다 "`128 regs/thread` 경계를 넘느냐", 그리고 CTA당 shared-memory footprint가 다음 residency tier를 깎아 먹느냐에 더 가깝다는 점을 분명하게 보여준다.
