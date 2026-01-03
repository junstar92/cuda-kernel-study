# CUDA Kernel Study: SGEMM Optimization

이 문서는 리팩토링 이후의 SGEMM 커널을 [`artifacts/benchmarks/results.csv`](./artifacts/benchmarks/results.csv)와 [`artifacts/profiles`](./artifacts/profiles) 기준으로 정리한 README다. 먼저 `torch`와 `cutlass`를 기준선으로 놓고, 이후 `naive -> shared tiling -> block tiling -> vec4 -> warptiling` 순서로 병목이 어떻게 이동하는지를 추적한다.

최신 benchmark 기준 최고 reference는 `torch_matmul`의 `23.87 TFLOPS`, explicit reference 중 최고는 `cutlass_simt_128x128x8_32x64x8_2stage`의 `22.64 TFLOPS`다. study kernel 최고점은 `warptiling_v7_vec4_128x128x8_32x64x8`의 `23.76 TFLOPS`이며, 그다음은 `warptiling_v3_vec4_128x128x8_32x64x8`의 `23.75 TFLOPS`, `warptiling_v8_vec4_128x128x8_32x64x8`의 `23.70 TFLOPS`, `warptiling_v4_vec4_128x128x8_32x64x8`의 `23.70 TFLOPS`다. scalar warptiling 최고점은 이번에 추가된 `warptiling_v11_128x128x8_32x64x8`의 `22.94 TFLOPS`다. 즉 현재 최고 study kernel은 `torch` 대비 약 `99.52%`까지 올라왔고, vec4 family는 여전히 explicit / universal CUTLASS를 모두 앞선다. scalar family도 이제 `v11` 기준으로 explicit / universal CUTLASS를 모두 넘는다. 다만 `v10`의 이득이 이름 기반 codegen heuristic에 더 가까웠다면, `v11`의 이득은 threadblock swizzling으로 inter-CTA L2 residency를 끌어올린 쪽에 가깝다.

이번 문서는 `warptiling_v11`까지 포함한 현재 구현 범위를 기준으로 정리한다.

## 분석 범위

- benchmark / NCU profile: [`test.py`](./test.py)의 기본 shape인 `M=N=K=4096`
- 기준 문서: [`how_to_optimize_sgemm.kr.md`](./how_to_optimize_sgemm.kr.md)
- 구현 파일
  - reference: [`00_cutlass.cuh`](./00_cutlass.cuh)
  - naive: [`01_naive.cuh`](./01_naive.cuh)
  - shared tiling: [`02_smem_tiling.cuh`](./02_smem_tiling.cuh)
  - 1D block tiling: [`03_blocktiling_1d.cuh`](./03_blocktiling_1d.cuh)
  - 2D block tiling: [`04_blocktiling_2d.cuh`](./04_blocktiling_2d.cuh)
  - 2D vec4: [`05_blocktiling_2d_vec4.cuh`](./05_blocktiling_2d_vec4.cuh)
  - warptiling: [`06_warptiling_v0.cuh`](./06_warptiling_v0.cuh), [`06_warptiling_v1.cuh`](./06_warptiling_v1.cuh), [`06_warptiling_v2.cuh`](./06_warptiling_v2.cuh), [`06_warptiling_v2_vec4.cuh`](./06_warptiling_v2_vec4.cuh), [`06_warptiling_v3.cuh`](./06_warptiling_v3.cuh), [`06_warptiling_v3_vec4.cuh`](./06_warptiling_v3_vec4.cuh), [`06_warptiling_v4.cuh`](./06_warptiling_v4.cuh), [`06_warptiling_v4_vec4.cuh`](./06_warptiling_v4_vec4.cuh), [`06_warptiling_v5.cuh`](./06_warptiling_v5.cuh), [`06_warptiling_v5_vec4.cuh`](./06_warptiling_v5_vec4.cuh), [`06_warptiling_v6.cuh`](./06_warptiling_v6.cuh), [`06_warptiling_v6_vec4.cuh`](./06_warptiling_v6_vec4.cuh), [`06_warptiling_v7.cuh`](./06_warptiling_v7.cuh), [`06_warptiling_v7_vec4.cuh`](./06_warptiling_v7_vec4.cuh), [`06_warptiling_v8.cuh`](./06_warptiling_v8.cuh), [`06_warptiling_v8_vec4.cuh`](./06_warptiling_v8_vec4.cuh), [`06_warptiling_v9.cuh`](./06_warptiling_v9.cuh), [`06_warptiling_v10.cuh`](./06_warptiling_v10.cuh), [`06_warptiling_v11.cuh`](./06_warptiling_v11.cuh)
  - helper / iterator: [`utils.cuh`](./utils.cuh)

## 이번 문서에서 보는 주요 메트릭

SGEMM 최적화에서는 TFLOPS만 보면 "얼마나 빠른가"만 보이고, "왜 빨라졌는가"는 잘 보이지 않는다. 그래서 이 문서에서는 성능 숫자, scheduler 상태, memory path 상태를 같이 읽는다. benchmark의 `Elapsed Time`과 `TFLOPS`는 최종 사용자 입장에서 가장 중요한 결과이며, 어떤 커널이 실제로 더 빠른지는 결국 이 값으로 판단한다. 다만 같은 TFLOPS라도 전혀 다른 이유로 만들어질 수 있어서, benchmark만으로는 병목의 위치를 설명할 수 없다.

`Registers Per Thread`, `Achieved Occupancy`, `Waves Per SM`은 커널이 SM 안의 자원을 어떤 방식으로 쓰는지 보여 준다. register 수가 크다는 것은 accumulator나 temporary를 많이 들고 간다는 뜻이라서 reuse와 ILP에는 유리하지만, 동시에 resident warp 수를 줄여 occupancy cliff를 만들 수 있다. `Achieved Occupancy`는 실제로 얼마나 많은 warp가 동시에 살아 있었는지 보여 주지만, SGEMM에서는 이 값이 높다고 반드시 빠른 것은 아니다. 오히려 큰 tile과 많은 accumulator를 쓰는 빠른 커널이 낮은 occupancy를 감수하는 경우가 많다. `Waves Per SM`은 현재처럼 `4096` 고정 비교에서도 grid가 SM 위를 어떤 residency 형태로 통과하는지 보는 데 도움이 된다.

`Eligible Warps Per Scheduler`, `Issued Warp Per Scheduler`, `No Eligible`은 이 문서에서 가장 중요하게 보는 scheduler 계열 지표다. `Eligible`은 scheduler가 지금 당장 선택할 수 있는 ready warp 수이고, `Issued`는 그 scheduler가 실제로 얼마나 자주 warp instruction을 발행했는지를 보여 준다. 반대로 `No Eligible`은 issue할 warp를 찾지 못한 cycle 비율이다. 이 세 값은 함께 봐야 의미가 있다. 예를 들어 occupancy가 높아도 `Issued`가 낮고 `No Eligible`이 높으면, warp 수는 많아 보여도 실제로는 모두 memory 응답이나 dependency를 기다리고 있다는 뜻이다. 이번 문서에서 reference kernel이 빠른 이유를 설명할 때도 occupancy보다 `Issued`와 `No Eligible`을 더 중요하게 본다.

`Compute (SM) Throughput`, `Memory Throughput`, `DRAM Bandwidth`는 커널이 어느 자원을 더 강하게 밀고 있는지를 보여 준다. compute throughput이 높으면 산술 파이프가 바쁘다는 뜻이고, memory throughput이나 DRAM bandwidth가 높으면 on-chip/off-chip memory path에 더 큰 압력이 걸리고 있다는 뜻이다. 같은 TFLOPS라도 DRAM bandwidth를 덜 쓰면서 달성했다면 operand reuse가 좋아졌다고 해석할 수 있고, 반대로 compute throughput이 높게 보여도 `Issued`가 낮다면 실제 병목은 scheduler 쪽일 수 있다. 그래서 이 값들은 단독으로 보기보다 scheduler 지표와 같이 읽어야 한다.

top stall reason은 현재 병목이 어떤 종류인지 가장 직접적으로 보여 주는 신호다. `LG Throttle`은 global load path 압박, `MIO Throttle`은 shared-memory나 special math/load-store instruction 경로 압박, `Long Scoreboard`와 `Short Scoreboard`는 dependency 때문에 다음 instruction을 못 내는 상태, `Barrier`는 동기화 비용, `Not Selected`는 ready warp는 있지만 scheduler slot 경쟁이 심한 상태를 뜻한다. 이 문서에서 "병목이 global에서 shared로 옮겨 갔다", "scoreboard가 줄고 scheduler 경쟁이 늘었다" 같은 해석은 대부분 이 stall reason 변화에서 출발한다.

마지막으로 `L1 Wavefronts Shared Excessive`, shared-memory bank conflict 경고, uncoalesced global access 경고는 memory path가 얼마나 깔끔한지 보여 준다. `L1 Wavefronts Shared Excessive`가 크면 shared access가 이상적인 wavefront보다 더 잘게 쪼개지고 있다는 뜻이고, bank conflict 경고가 크면 여러 lane이 같은 shared-memory bank를 동시에 두드려 serialization이 생긴다는 뜻이다. uncoalesced global access 경고는 warp의 주소가 sector 단위로 잘 묶이지 않아 extra sector를 낭비하고 있다는 뜻이다. 이 세 지표는 2D block tiling과 초기 warptiling에서 왜 계산 구조는 좋은데도 성능이 덜 나오는지, 그리고 vec4나 lane remap이 왜 필요한지를 설명할 때 핵심 역할을 한다.

이 프로젝트에서 자주 보는 `Global Store Avg Bytes / Sector`, `L2 Theoretical Sectors Global Excessive`, `L1/TEX Hit Rate`, `L2 Hit Rate`도 같은 맥락에서 중요하다. `Global Store Avg Bytes / Sector`는 store transaction이 sector를 얼마나 꽉 채우는지 보여 주므로 epilogue가 scalar store인지 vec4 store인지, 또는 warp store mapping이 정리되어 있는지를 바로 드러낸다. `L2 Theoretical Sectors Global Excessive`는 이상적인 sector 수보다 얼마나 더 많은 global sector를 쓰는지를 보여 주기 때문에, 단순한 uncoalesced warning보다 더 정량적으로 store/load path 낭비를 설명할 수 있다. `L1/TEX Hit Rate`, `L2 Hit Rate`는 cache가 이 access pattern을 얼마나 잘 흡수하는지를 보는 보조 지표다. SGEMM에서는 cache hit rate 자체보다, 그 hit rate가 높거나 낮은데도 왜 TFLOPS가 그대로이거나 달라지는지를 같이 읽는 편이 더 중요하다.

보조적으로 `Threads per Block`, `Dynamic Smem / Block`, `Block Limit Registers`, `Block Limit Shared Mem`, `Warp Cycles Per Issued Instruction` 같은 지표도 본다. `Threads per Block`과 `Dynamic Smem / Block`은 같은 kernel shape라도 residency tier가 왜 달라지는지 설명할 때 필요하고, `Block Limit Registers`와 `Block Limit Shared Mem`은 실제 occupancy를 무엇이 막고 있는지 더 직접적으로 보여 준다. `Warp Cycles Per Issued Instruction`은 `Issued Warp`의 역방향 표현처럼 읽으면 된다. 값이 크면 issue 간격이 길다는 뜻이고, 같은 kernel family 안에서 이 값이 줄어들면 scheduler가 더 촘촘하게 명령을 내보내고 있다고 해석할 수 있다.

## 현재 스냅샷

- reference ceiling
  - `torch_matmul`: `23.87 TFLOPS`
  - `cutlass_simt_128x128x8_32x64x8_2stage`: `22.64 TFLOPS`
  - `cutlass_universal_simt_256x128_8x4`: `22.28 TFLOPS`
- study best
  - `warptiling_v7_vec4_128x128x8_32x64x8`: `23.76 TFLOPS`
  - `warptiling_v3_vec4_128x128x8_32x64x8`: `23.75 TFLOPS`
  - `warptiling_v8_vec4_128x128x8_32x64x8`: `23.70 TFLOPS`
  - `warptiling_v4_vec4_128x128x8_32x64x8`: `23.70 TFLOPS`
  - scalar best: `warptiling_v11_128x128x8_32x64x8`: `22.94 TFLOPS`
- 큰 흐름
- `naive_row 1.98 -> smem_tiling 2.64 -> blocktiling_1d(best) 12.16 -> blocktiling_2d(best) 19.10 -> blocktiling_2d_vec4(best) 20.13 -> current warptiling(best) 23.76 TFLOPS`
- 병목 이동
  - `LG Throttle`
  - `MIO Throttle + Barrier`
  - `MIO/Scoreboard`
  - `shared wavefront excess + uncoalesced global store`
  - `warptiling v1에서 shared/global path 정리`
  - `warptiling v2에서 2-stage overlap로 scheduler 개선`
  - `warptiling v3/v3_vec4에서 PTX 기반 L2::128B global load 도입`
  - `warptiling v4/v4_vec4에서 predicated global load로 tail 정리`
  - `warptiling v5/v5_vec4에서 partial CUTLASS-style pipeline 실험`
  - `warptiling v6/v6_vec4에서 full CUTLASS-style pipeline 실험`
  - `warptiling v7/v7_vec4에서 CUTLASS-style epilogue reorder 도입`
  - `warptiling v8/v8_vec4에서 explicit prologue -> steady-state -> drain choreography 정리`
  - `warptiling v9에서 next-tile global load 시점 재배치`
  - `warptiling v10에서 symbol name만 바꿔 backend codegen 압축`
  - `warptiling v11에서 grouped threadblock swizzle로 inter-CTA L2 residency 개선`

현재 기준으로 보면, 가장 빠른 study kernel은 `warptiling_v7_vec4`이며 `v3_vec4`, `v8_vec4`, `v4_vec4`가 모두 아주 근소한 차이로 뒤따른다. 즉 vec4 계열은 이미 넓은 ingress, `L2::128B` helper, predication만으로 거의 천장에 도달했고, 이후 epilogue/mainloop 실험은 주로 정체 구간 안에서의 트레이드오프로 읽는 편이 맞다. 반면 scalar 계열에서는 `v7`의 epilogue reorder 이후 `v8`의 choreography 정리, `v9`의 load timing 재배치, `v10`의 이름 기반 codegen 변화, 그리고 `v11`의 threadblock swizzling이 차례로 더해지며 `22.94 TFLOPS`까지 올라갔다. 여기서 `v10`은 source는 그대로인데 backend가 보조 명령열을 더 압축한 경우이고, `v11`은 반대로 symbol 이름 실험을 버리고 CTA 방문 순서만 바꿔 cache residency를 끌어올린 경우다. 즉 둘 다 빨라졌지만 이유는 다르다. 이제 남은 과제는 vec4 계열 최고점과 `v7~v11`의 정돈된 output path를 유지한 채 `torch_matmul`이 쓰는 직접적인 global->shared 복사 경로와 더 큰 tile density가 만드는 마지막 gap을 줄이는 일이다.

## 1. Reference Group: `torch` / `cutlass`

reference는 “이 GPU에서 SIMT FP32 SGEMM이 어느 수준까지 갈 수 있는가”를 보여 주는 기준선이다. 특히 이번 환경에서 `torch_matmul`을 프로파일해 보면 커널 이름이 `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`로 나온다. 즉 이번 비교는 “직접 작성한 study kernel 대 정체를 알 수 없는 벤더 내부 구현”이 아니라, 사실상 같은 SIMT 계열 커널과의 비교다.

| Kernel | 4096 TFLOPS | Regs | Occ | Issued | No Eligible | 특징 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `torch_matmul` | 23.87 | 202 | 16.67% | 0.88 | 12.26% | 큰 tile, 높은 ILP, `LDGSTS` direct copy |
| `cutlass_simt_128x128x8_32x64x8_2stage` | 22.64 | 122 | 32.41% | 0.88 | 12.02% | 2-stage mainloop, clean shared path |
| `cutlass_universal_simt_256x128_8x4` | 22.28 | 255 | 16.67% | 0.85 | 14.56% | 동일 tile, 더 무거운 shared/internal traffic |

instruction 수준에서 보면 reference 셋의 차이는 “수학 자체”보다 “tile을 shared로 어떻게 밀어 넣고, warp가 그것을 어떤 폭으로 읽어 오는가”에 더 가깝다. 세 커널 모두 warp operand fetch에서는 `LDS.128`을 사용하지만, global ingress와 stage commit 방식은 분명히 다르다.

| Kernel | Global ingress | Shared stage commit | Warp operand read | Epilogue store | 128B 경로 요약 |
| --- | --- | --- | --- | --- | --- |
| `torch_matmul` | `LDGSTS.E.LTC128B` + `LDGDEPBAR/DEPBAR` | direct global->shared copy 중심 | `LDS.128` | scalar `STG.E` | global ingress 128B O, shared read 128B O, global store 128B X |
| `cutlass_simt_128x128x8_32x64x8_2stage` | predicated `LDG.E.LTC128B` | register-staged scalar `STS` | `LDS.128` | scalar `STG.E` | global load 128B O, direct global->shared 128B X |
| `cutlass_universal_simt_256x128_8x4` | `LDGSTS.E.LTC128B` | direct global->shared copy + heavier remap | `LDS.128` | scalar `STG.E` | hardware path는 torch와 유사하지만 address/predicate plumbing이 더 무겁다 |

### `torch_matmul`

#### 구조

`torch_matmul`은 이번 환경에서 런타임이 고른 `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` kernel이다. 즉 opaque vendor kernel이라기보다, 큰 `256x128x8` CTA tile과 `8x4` thread-level MMA 성격을 가진 CUTLASS SIMT family의 대표 구현으로 보는 편이 맞다. 구조적으로는 큰 tile, 낮은 occupancy, 높은 register budget을 감수하는 대신 thread tile 안에서 accumulator를 두껍게 들고 가며 ILP와 reuse를 극대화하는 쪽이다. SASS에서도 `FFMA` 본문 밀도가 높고, 그 사이에 `LDGSTS.E.LTC128B`, `LDGDEPBAR`, `DEPBAR.LE`, `LDS.128`, scalar `STG.E`가 배치되어 있어 global ingress와 warp operand feed가 모두 넓다.

대표적인 mainloop / epilogue 패턴은 다음과 같다.

```sass
LDGSTS.E.LTC128B P1, [R184+0x8400], [R4.64]
LDGSTS.E.LTC128B P2, [R184+0x8800], [R10.64]
...
LDGDEPBAR
DEPBAR.LE SB0, 0x2
BAR.SYNC.DEFER_BLOCKING 0x0
LDS.128 R132, [R156.X16]
FFMA R127, R132, R136, R127
...
@P0 STG.E [R170], R67
@P0 STG.E [R170+0x80], R5
```

즉 `torch_matmul`은 global에서 shared로 들어오는 경로에 128B direct copy를 쓰고, shared에서 register로 올라오는 경로도 `LDS.128`로 유지한다. 다만 epilogue store는 128B vector store가 아니라 scalar `STG.E`다. 이 kernel의 강점은 store 폭이 아니라, 큰 `256x128x8` tile 안에서 copy wave 하나당 더 많은 `FFMA`를 끼워 넣을 수 있다는 점이다.

#### NCU 핵심 수치

- Benchmark (4096): `5.76 ms / 23.87 TFLOPS`
- Profile duration: `6.60 ms`
- Compute / Memory / DRAM: `76.80% / 40.35% / 69.95 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `0.00% / 91.25% / 0.00`
- Registers / thread: `202`
- Achieved Occupancy: `16.67%`
- Block limit Registers / Shared Mem: `1 / 1`
- Eligible / Issued / No Eligible: `1.59 / 0.88 / 12.26%`
- Top stall: `Not Selected 0.81`, `Dispatch Stall 0.16`, `Barrier 0.10`, `Short Scoreboard 0.09`, `Long Scoreboard 0.02`, `MIO Throttle 0.02`
- Shared path: `L1 Wavefronts Shared Excessive 83,886,240 (27%)`

#### 해석

핵심 해석은 단순하다. 이 커널은 occupancy를 희생하고도 `Issued 0.88`, `No Eligible 12%`대 초반을 만드는 데 성공한 kernel이다. 즉 resident warp 수를 많이 두는 방식으로 빠른 것이 아니라, 큰 tile과 높은 ILP로 적은 warp를 매우 잘 굴린다. shared excessive `27%`가 보이는데도 최고 성능이 나오는 것은 이 값이 곧바로 병목을 뜻하지 않음을 보여 준다. 이 수치는 CUTLASS-style shared epilogue remap의 흔적으로 보는 편이 자연스럽고, 실제로 더 중요한 것은 그 상태에서도 scheduler가 거의 쉬지 않는다는 점이다.

현재 최고 study kernel인 `warptiling_v3_vec4`와 instruction 수준에서 비교하면 남은 gap이 더 선명하다. `v3_vec4`는 이미 `LDG.E.LTC128B -> STS.128 -> LDS.128 -> FFMA -> STG.E`까지 올라와 있어 explicit CUTLASS와 같은 계열의 ingress를 갖는다. 그런데도 `torch_matmul`은 ingress부터 `LDGSTS.E.LTC128B` direct copy를 쓰고 tile도 `256x128x8`이라, copy/control overhead를 더 큰 `FFMA` 본문 위에 얹는다. 이 차이가 대략 `L2 Hit 91% vs 82%`, `Issued 0.88 vs 0.84`, `No Eligible 12.26% vs 15.52%`로 이어진다. 즉 이제 남은 차이는 “vec4를 안 써서”가 아니라, `LDGSTS` 계열 direct copy와 더 큰 tile이 만드는 arithmetic density 차이에 가깝다. 이번 프로젝트에서 reference가 제시하는 목표치는 결국 `Issued ~0.87~0.88`, `No Eligible ~12~13%`다.

### `cutlass_simt_128x128x8_32x64x8_2stage`

#### 구조

이 kernel은 CUTLASS의 explicit SIMT `Gemm` 인스턴스로, threadblock shape `128x128x8`, warp shape `32x64x8`, 그리고 이름 그대로 `2-stage` mainloop를 사용한다. 즉 `torch_matmul`보다 한 단계 작은 CTA tile을 쓰고, shared staging을 두 단계로 굴리며 steady-state를 만드는 쪽이다. 구조적으로는 register budget을 과도하게 쓰지 않으면서 shared path를 아주 깨끗하게 유지하는 균형형 설계에 가깝다. 실제로 `L1 Wavefronts Shared Excessive = 0`이라서 reference 셋 중 shared path 정돈도는 가장 좋다. instruction 수준에서는 `LDG.E.LTC128B`로 global에서 먼저 register에 받고, 그것을 scalar `STS`로 shared에 풀어 쓰는 전형적인 register-staged 2-stage SIMT mainloop다.

대표적인 패턴은 다음과 같다.

```sass
@P0 LDG.E.LTC128B R4, [R18.64]
@P6 LDG.E.LTC128B R6, [R10.64]
...
STS [R109.X4], R6
STS [R109.X4+0x80], R4
...
LDS.128 R92, [R66+0xca0]
FFMA R63, R84, R88, R63
...
@P0 STG.E [R96.64], R63
```

여기서 중요한 점은 128B global load는 분명히 존재하지만, `torch_matmul`처럼 `LDGSTS`로 바로 shared에 꽂지는 않는다는 것이다. 그 대신 predicated `LDG.E.LTC128B`와 정돈된 `STS`/`LDS.128` 조합으로 stage를 운영한다.

#### NCU 핵심 수치

- Benchmark (4096): `6.07 ms / 22.64 TFLOPS`
- Profile duration: `6.77 ms`
- Compute / Memory / L2 / DRAM: `82.50% / 50.67% / 93.11% / 155.13 GB/s`
- L1/TEX Hit / L2 Excessive: `0.02% / 0`
- Registers / thread: `122`
- Achieved Occupancy: `32.41%`
- Block limit Registers / Shared Mem: `2 / 3`
- Eligible / Issued / No Eligible: `2.47 / 0.88 / 12.02%`
- Top stall: `Not Selected 1.80`, `Barrier 0.72`, `Short Scoreboard 0.34`, `Dispatch Stall 0.23`, `MIO Throttle 0.08`, `Wait 0.07`
- Shared path: `L1 Wavefronts Shared Excessive 0`

#### 해석

첫 reference 비교 기준은 당연히 `torch_matmul`이다. 두 kernel 모두 `Issued 0.88` 수준으로 scheduler를 거의 쉬지 않게 만들지만, 이 kernel은 더 작은 `128x128x8` tile과 더 잦은 stage 전환 때문에 같은 GEMM을 더 memory-active한 방식으로 푼다. 실제로 `Memory Throughput 50.66%`, `DRAM 141.37 GB/s`, `Barrier 0.72`, `Not Selected 1.80`은 `torch_matmul`보다 모두 무겁다. 즉 compute를 못 굴려서 느린 것이 아니라, shared path 정돈도를 얻는 대신 reuse와 ILP에서 한 단계 양보한 결과다.

이제는 `warptiling_v2`보다 `warptiling_v3_vec4`와의 비교가 더 직접적이다. 두 커널은 모두 `128x128x8 / 32x64x8 / 2-stage` 계열이고, study kernel도 넓은 ingress 계열까지 올라왔다. 즉 남은 차이는 단순 access width보다 predication, cache hint, stage orchestration, 그리고 fixed-shape kernel이냐 generic template path냐의 차이로 좁혀졌다. 실제로 CUTLASS는 scheduler 지표가 더 깨끗하다. `Issued 0.88`, `No Eligible 12.02%`, `Long Scoreboard 0.00`는 여전히 reference 급이다. 다만 benchmark는 현재 artifacts에서도 `warptiling_v3_vec4 23.75 TFLOPS`가 이 kernel의 `22.64 TFLOPS`를 더 분명하게 앞선다. 즉 현재 fixed-shape study kernel은 더 이상 “CUTLASS보다 한 단계 앞선 scalar ingress 버전”이 아니고, 넓은 ingress를 확보한 뒤에는 일반성 비용이 적은 특화 kernel로 explicit CUTLASS와 직접 경쟁하는 단계를 넘어서는 쪽으로 가고 있다고 보는 편이 맞다.

### `cutlass_universal_simt_256x128_8x4`

#### 구조

이 kernel은 [`00_cutlass.cuh`](./00_cutlass.cuh)에서 고정한 `GemmUniversal<256, 128, 8, 64, 64, 8, 4, 8>` 인스턴스다. 중요한 점은 지금 이 kernel이 `torch_matmul`과 같은 `256x128_8x4` family라는 것이다. 즉 이전처럼 tile 방향 차이로 설명할 수 있는 상황이 아니고, 남는 차이는 동일한 tile family 안에서 specialized path와 universal path가 얼마나 다르게 codegen되는가에 있다. 구조적으로는 큰 CTA tile과 높은 register budget을 써서 arithmetic reuse를 확보하지만, universal 경로답게 iterator, shared remap, epilogue 쪽 generic overhead를 더 많이 안고 간다. instruction 수준에서도 `LDGSTS.E.LTC128B`와 `LDS.128` 자체는 `torch_matmul`과 유사하지만, 그 앞뒤로 더 많은 predicate/address plumbing이 붙는다.

대표적으로 다음 패턴이 보인다.

```sass
@P1   LDGSTS.E.LTC128B [R224+0x200][R206.64]
@!PT  LDS RZ, [RZ]
...
LDS.128 R136, [R130+0x1400]
FFMA R39, R148, R132, R39
...
@P6   LDGSTS.E.LTC128B [R224+0x300][R244.64]
```

즉 hardware primitive만 놓고 보면 `torch_matmul`과 꽤 가깝다. 하지만 `LDGSTS` 주변의 `LOP3`, `LEA`, `IADD3`, predicate 갱신, masked no-op load가 더 많이 끼어 있어 동일한 tile family 안에서도 codegen이 한층 무거워진다.

#### NCU 핵심 수치

- Benchmark (4096): `6.17 ms / 22.28 TFLOPS`
- Profile duration: `7.07 ms`
- Compute / Memory / L1/TEX / DRAM: `74.89% / 52.95% / 0% / 72.37 GB/s`
- L2 Hit / L2 Excessive: `92.92% / 0`
- Registers / thread: `255`
- Achieved Occupancy: `16.67%`
- Block limit Registers / Shared Mem: `1 / 1`
- Eligible / Issued / No Eligible: `1.55 / 0.85 / 14.56%`
- Top stall: `Not Selected 0.82`, `MIO Throttle 0.16`, `Dispatch Stall 0.14`, `Barrier 0.09`, `Short Scoreboard 0.03`, `Wait 0.01`
- Shared path: `L1 Wavefronts Shared Excessive 167,772,490 (42%)`

#### 해석

이 절에서는 직전 kernel인 `cutlass_simt_128x128x8_32x64x8_2stage`와 `torch_matmul` 둘 다가 비교 대상이다. `torch_matmul`과 같은 `256x128_8x4` family인데도 `Issued 0.85`, `No Eligible 14.55%`, `Shared Excessive 42%`로 뒤처진다는 점이 핵심이다. DRAM bandwidth는 `69.85 -> 72.30 GB/s`로 거의 비슷한데 memory throughput은 더 높고 `MIO Throttle`도 더 크므로, off-chip bandwidth 부족이 아니라 동일한 tile family 안에서 universal path가 더 무거운 shared/internal traffic을 만든다고 보는 해석이 자연스럽다.

instruction 수준에서 보면 이 커널은 이제 `v2`보다 `v3_vec4`와 더 좋은 대비를 만든다. 둘 다 epilogue는 scalar `STG.E` 중심이고, study kernel도 넓은 global ingress를 회복했다. 그런데 universal CUTLASS는 여전히 `shared excessive 42%`를 안고 있고, `v3_vec4`는 `shared excessive 0`을 유지한다. benchmark가 `22.28 TFLOPS` 대 `23.75 TFLOPS`로 뒤집힌 이유도 여기서 읽힌다. 즉 현재 fixed-shape warptiling은 universal CUTLASS보다 내부 traffic이 훨씬 깨끗하고, explicit CUTLASS보다도 절대 benchmark가 더 높다. 이제 남은 reference gap은 사실상 `torch_matmul` 하나라고 보는 편이 자연스럽다.

### Reference 정리

reference 세 커널을 같이 놓고 보면 다음 결론이 나온다.

- `high occupancy`는 목표가 아니다. `torch_matmul`과 `cutlass_universal`은 `16%` occupancy로도 `22~24 TFLOPS`를 낸다.
- 진짜 목표는 `Issued Warp/Scheduler`를 `0.88` 근처까지 밀어 올리고 `No Eligible`을 `12%`대 초반으로 유지하는 것이다.
- `shared excessive = 0`이 반드시 최고 성능을 뜻하지는 않는다. `cutlass_simt_2stage`는 shared path가 가장 깨끗하지만 absolute best는 아니다.
- 반대로 shared remap 오버헤드가 있더라도, 높은 ILP와 좋은 scheduler 상태가 유지되면 충분히 최고점이 나올 수 있다.
- `torch_matmul`이 explicit CUTLASS보다 빠른 이유는 신비한 별도 알고리즘 때문이 아니라, `LDGSTS.E.LTC128B` 기반 ingress와 더 큰 `256x128` tile이 같은 SIMT family 안에서도 copy/control 오버헤드를 더 잘 amortize하기 때문이다.
- `cutlass_simt_128x128x8_32x64x8_2stage`는 `LDG.E.LTC128B -> STS -> LDS.128`라는 보수적인 2-stage path를 쓰지만 shared path가 매우 깨끗하고, `cutlass_universal_simt_256x128_8x4`는 `LDGSTS.E.LTC128B`까지 포함한 hardware path는 강하지만 universal/generality 비용 때문에 shared/internal traffic이 더 무거워진다.
- 현재 `warptiling_v3_vec4`는 두 CUTLASS reference를 모두 넘어서고 있으므로, 이제 reference 비교의 핵심은 “CUTLASS 수준의 ingress width를 회복했는가”가 아니라 “`torch`가 쓰는 direct global->shared copy와 더 큰 tile이 만드는 마지막 gap을 얼마나 줄였는가”다.
- 세 reference 모두 epilogue는 scalar `STG.E` 중심이라, 지금 gap의 핵심은 “128B global store가 없어서”가 아니라 “global ingress 폭과 그 주변 address/predicate code가 어떻게 정리되어 있는가”에 있다.

study kernel이 앞으로 reference를 더 따라잡으려면, 단순히 shared conflict 하나를 줄이는 것보다 `issue efficiency`를 reference 수준으로 끌어올리는 쪽이 더 중요하다.

## 2. Naive Group

구현 파일: [`01_naive.cuh`](./01_naive.cuh)

| Kernel | 4096 TFLOPS | Regs | Occ | Issued | No Eligible | Top Stall | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `naive_row` | 1.98 | 40 | 66.66% | 0.27 | 73.08% | `LG Throttle` | baseline |
| `naive_col` | 0.28 | 40 | 65.54% | 0.03 | 96.52% | `LG Throttle` | uncoalesced global access 85% |

### `naive_row`

#### 구조

`naive_row`는 가장 단순한 baseline이다. `4096`에서 `69.24 ms`, `1.98 TFLOPS`이며, thread 하나가 `C[row, col]` 하나를 끝까지 계산하고, 매 `k` iteration마다 `A[row, k]`, `B[k, col]`를 global memory에서 바로 읽는다.

핵심 루프는 정말로 아래 형태다. output 하나를 끝까지 혼자 계산하고, `k`가 한 칸 움직일 때마다 global memory를 다시 본다.

```cpp
float accum = 0.f;
for (int k = 0; k < K; ++k) {
  accum += a_ptr[row * K + k] * b_ptr[k * N + col];
}
c_ptr[row * N + col] = accum;
```

#### NCU 핵심 수치

- Benchmark (4096): `69.38 ms / 1.98 TFLOPS`
- Profile duration: `78.28 ms`
- Registers / thread: `40`
- Achieved Occupancy: `66.66%`
- Eligible / Issued / No Eligible: `1.16 / 0.27 / 73.08%`
- Top stall: `LG Throttle 18.32`, `Not Selected 3.30`, `Long Scoreboard 3.01`, `Wait 2.03`, `Dispatch Stall 1.72`
- Memory warning: 특이 경고 없음

#### 해석

해석은 단순하다. occupancy가 낮아서 느린 것이 아니라, warp가 살아 있어도 대부분이 메모리 응답을 기다리느라 ready 상태로 못 올라온다. 같은 데이터를 너무 자주 global에서 다시 읽고, 그 latency를 숨길 구조가 없기 때문에 scheduler가 issue할 warp를 거의 찾지 못한다. 이 커널은 이후 모든 최적화가 왜 필요한지를 보여 주는 출발점이다.

### `naive_col`

#### 구조

`naive_col`은 산술 자체는 `naive_row`와 완전히 같고, 달라지는 것은 thread-to-output mapping뿐이다. `threadIdx.x`가 row를, `threadIdx.y`가 col을 잡으면서 warp 관점에서 행/열 접근 패턴이 뒤집히고, 그 결과 같은 inner product를 풀어도 global access 정렬성이 거의 무너진다. 다시 말해 이 kernel은 algorithm이 아니라 mapping 실험이라고 보는 편이 맞다.

#### NCU 핵심 수치

- Benchmark (4096): `492.56 ms / 0.28 TFLOPS`
- Profile duration: `607.09 ms`
- Registers / thread: `40`
- Achieved Occupancy: `65.54%`
- Eligible / Issued / No Eligible: `0.13 / 0.03 / 96.52%`
- Top stall: `LG Throttle 211.62`, `Long Scoreboard 8.68`, `Not Selected 2.85`, `Wait 1.72`, `Dispatch Stall 0.11`
- L1/TEX Hit / L2 Hit / L2 Excessive: `99.08% / 59.01% / 60.14 GB`
- Memory warning: uncoalesced global access `85%`

#### 해석

같은 naive group 안에서 `naive_row`와 비교하면 교훈이 더 선명하다. occupancy와 register 수는 거의 같은데도 `Issued 0.27 -> 0.03`, `No Eligible 73.08% -> 96.52%`, `LG Throttle 18.32 -> 211.63`으로 무너진다. SGEMM에서는 산술이 같아도 warp 단위 access pattern만 나빠지면 성능이 한 자릿수 배수로 무너질 수 있다는 뜻이다. 이후 커널들이 lane mapping과 store mapping을 집요하게 다루는 이유가 여기서 이미 드러난다.

### Naive 정리

naive group은 다음을 확인시켜 준다.

- baseline 병목은 `global memory path`
- 좋은 mapping이 나쁜 mapping보다 훨씬 중요함
- occupancy보다 `Issued Warp/Scheduler`와 `No Eligible`이 더 직접적인 설명 변수라는 점

## 3. Shared Tiling Group

구현 파일: [`02_smem_tiling.cuh`](./02_smem_tiling.cuh)

| Kernel | 4096 TFLOPS | Regs | Occ | Issued | No Eligible | Top Stall | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `smem_tiling` | 2.67 | 38 | 66.60% | 0.20 | 79.74% | `MIO Throttle` | shared store bank conflict |

### `smem_tiling`

#### 구조

`smem_tiling`은 `32x32` tile을 shared memory에 올린 뒤 CTA 내부에서 재사용하는 첫 단계다. `4096`에서 `51.31 ms`, `2.67 TFLOPS`로 이전 그룹의 best이자 마지막 kernel인 `naive_row` 대비 약 `1.35x` 개선된다. 분명 올바른 방향이지만, 개선 폭은 아직 작다.

구조적으로는 아래처럼 바뀐다. 먼저 A/B sub-tile을 shared에 올리고, CTA 전체가 같은 tile을 소비한 뒤 다음 tile로 넘어간다.

```cpp
smem_a[thread_tile_m_idx][thread_tile_n_idx] =
    a_ptr[thread_tile_m_idx * K + thread_tile_n_idx];
smem_b[thread_tile_m_idx][thread_tile_n_idx] =
    b_ptr[thread_tile_m_idx * N + thread_tile_n_idx];
__syncthreads();

for (int k = 0; k < kBlockSize; ++k) {
  partial_accum +=
      smem_a[thread_tile_m_idx][k] * smem_b[k][thread_tile_n_idx];
}
accum += partial_accum;
__syncthreads();
```

instruction 수준에서도 이 단계의 한계가 그대로 보인다. mainloop는 대체로 `LDG.E.CONSTANT -> STS -> BAR.SYNC -> LDS/LDS.128 -> FFMA -> BAR.SYNC` 패턴으로 반복된다. 즉 global memory에서 scalar로 읽어 shared에 scalar store를 하고, tile을 소비할 때도 scalar `LDS`와 일부 `LDS.128`을 섞어 쓰다가 barrier로 다시 끊긴다. shared tiling은 들어왔지만, tile 하나를 staging하고 소비하는 과정이 아직 너무 무겁다.

```sass
LDG.E.CONSTANT R19, [R2.64]
LDG.E.CONSTANT R25, [R22.64]
STS [R20], R19
STS [R20+0x1000], R25
BAR.SYNC.DEFER_BLOCKING 0x0
LDS R17, [R4.X4+0x1000]
LDS.128 R12, [R21+0x60]
FFMA R12, R17, R12, RZ
...
BAR.SYNC.DEFER_BLOCKING 0x0
```

#### NCU 핵심 수치

- Benchmark (4096): `51.96 ms / 2.64 TFLOPS`
- Profile duration: `59.90 ms`
- Registers / thread: `38`
- Achieved Occupancy: `66.60%`
- Eligible / Issued / No Eligible: `0.89 / 0.20 / 79.74%`
- Top stall: `MIO Throttle 23.77`, `Long Scoreboard 5.96`, `Not Selected 3.41`, `Barrier 2.70`, `Wait 1.84`, `LG Throttle 0.42`
- Shared warning: shared store avg `1.2-way` bank conflict, total bank conflicts `32,859,875`

#### 해석

이전 그룹 best인 `naive_row`와 비교하면 병목 이동이 분명하다. `LG Throttle` 중심 병목은 줄었지만 대신 `MIO Throttle 23.77`, `Barrier 2.67`, shared bank conflict가 새로 드러난다. 즉 shared tiling은 분명 DRAM reload를 줄였지만, thread당 output이 아직 하나라서 CTA가 가져온 tile을 충분히 오래 소비하지 못한다. 그래서 global 병목을 해결했다기보다 shared 병목으로 옮긴 단계에 가깝다. `Issued`가 `naive_row`보다도 낮은 이유가 바로 여기 있고, 다음 단계에서 필요한 것은 shared tile 하나당 더 많은 FMA를 만들어 낼 register reuse다.

## 4. Blocktiling 1D Group

구현 파일: [`03_blocktiling_1d.cuh`](./03_blocktiling_1d.cuh)

| Kernel | 4096 TFLOPS | Regs | Occ | Issued | No Eligible | Top Stall | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `blocktiling_1d_v0_64x64x8_8` | 8.31 | 41 | 65.98% | 0.38 | 62.42% | `MIO Throttle` | manual loader |
| `blocktiling_1d_v1_64x64x8_8` | 8.21 | 41 | 66.02% | 0.38 | 62.07% | `MIO Throttle` | iterator v0 |
| `blocktiling_1d_v2_64x64x8_8` | 8.91 | 40 | 98.09% | 0.43 | 56.89% | `MIO Throttle` | iterator 개선 |
| `blocktiling_1d_v3_64x64x8_8` | 8.96 | 40 | 98.10% | 0.43 | 56.89% | `MIO Throttle` | 1-tile preload |
| `blocktiling_1d_v2_64x64x8_16` | 10.56 | 48 | 80.53% | 0.47 | 53.28% | `MIO Throttle` | thread tile 확대 |
| `blocktiling_1d_v2_128x128x8_64` | 12.01 | 106 | 32.31% | 0.49 | 51.15% | `Short Scoreboard` | large CTA/tile |
| `blocktiling_1d_v2_128x128x16_64` | 12.24 | 110 | 32.35% | 0.50 | 49.77% | `MIO Throttle` | shared excessive 1% |

### `blocktiling_1d_v0_64x64x8_8`

#### 구조

이 커널에서 첫 번째 큰 점프가 나온다. `4096` 기준 `8.31 TFLOPS`로 이전 그룹의 best이자 마지막 kernel인 `smem_tiling`의 `2.67 TFLOPS`에서 한 번에 `3x` 이상 뛴다. 이유는 단순하다. 이제 thread가 output 하나가 아니라 `ThreadM=8`개의 output strip을 register에 들고 가며, shared에서 읽은 `B` 값을 여러 output에 재사용한다.

코드로 보면 변화가 더 분명하다. `elem_b` 하나를 읽고 `accum[ThreadM]` 전체에 뿌리는 구조가 1D register tiling의 핵심이다.

```cpp
float accum[ThreadM] = {0.f};

for (int k = 0; k < ThreadblockK; ++k) {
  float const elem_b = smem_b[ThreadblockN * k + thread_tile_n_idx];

  for (int thread_m_idx = 0; thread_m_idx < ThreadM; ++thread_m_idx) {
    float const elem_a = smem_a[kSmemStrideA * k +
                                ThreadM * thread_tile_m_idx + thread_m_idx];
    accum[thread_m_idx] += elem_a * elem_b;
  }
}
```

#### NCU 핵심 수치

- Benchmark (4096): `16.88 ms / 8.14 TFLOPS`
- Profile duration: `19.00 ms`
- Registers / thread: `41`
- Threads / block: `512`
- Achieved Occupancy: `65.98%`
- Eligible / Issued / No Eligible: `1.01 / 0.38 / 62.42%`
- Top stall: `MIO Throttle 9.43`, `Long Scoreboard 4.85`, `Barrier 2.49`, `Not Selected 1.68`, `Wait 0.71`, `Short Scoreboard 0.60`
- Shared path: `L1 Wavefronts Shared Excessive 0`

#### 해석

`smem_tiling`과 비교하면 `Issued 0.20 -> 0.38`, `No Eligible 79.73% -> 62.41%`로 scheduler 상태가 크게 좋아진다. 즉 shared tile을 가져온 뒤 thread가 그 tile에서 꺼낸 `B` 값을 register accumulator 여러 개에 다시 쓰기 시작하면서, shared load 하나당 실제 FMA 수가 크게 늘어난다. 다만 top stall은 여전히 `MIO Throttle 9.44`, `Long Scoreboard 4.85`, `Barrier 2.47`다. shared load 직후 바로 accumulation을 이어가는 구조라 load-use dependency가 아직 길고, 이 단계의 본질은 “shared reuse를 register reuse까지 끌어왔다”는 데 있다.

### `blocktiling_1d_v1_64x64x8_8`

#### 구조

`v1`은 수치상 거의 `v0`와 같다. `4096`에서 `8.21 TFLOPS`로 사실상 횡보다. 코드상으로는 수동 load helper 대신 iterator 계층을 도입한 버전인데, 이 변경만으로는 성능 이득이 거의 없다.

#### NCU 핵심 수치

- Benchmark (4096): `16.91 ms / 8.13 TFLOPS`
- Profile duration: `19.18 ms`
- Registers / thread: `41`
- Threads / block: `512`
- Achieved Occupancy: `66.02%`
- Eligible / Issued / No Eligible: `1.04 / 0.38 / 62.07%`
- Top stall: `MIO Throttle 9.09`, `Long Scoreboard 4.77`, `Barrier 2.71`, `Not Selected 1.74`, `Wait 0.75`, `Short Scoreboard 0.53`
- Shared path: `L1 Wavefronts Shared Excessive 0`

#### 해석

직전 커널인 `v0`와 비교하면 성능과 profile이 거의 복사 수준이다. `Issued`는 그대로 `0.38`이고 `No Eligible`도 `62.41% -> 62.07%` 수준이다. stall 분포 역시 `MIO Throttle`, `Long Scoreboard`, `Barrier` 축이 그대로라서 병목 성격이 바뀌지 않는다. 즉 이 단계는 구현 구조를 정리하는 의미는 있지만, address generation 추상화만으로는 kernel의 성능 성격이 바뀌지 않는다는 것을 보여 준다. 이 결과는 이후 `v2`에서 “iterator를 바꾼다”보다 “iterator가 만드는 실제 access pattern과 codegen을 바꾼다”가 중요하다는 배경이 된다.

### `blocktiling_1d_v2_64x64x8_8`

#### 구조

같은 `64x64x8, ThreadM=8`인데 `v2`에서 `8.91 TFLOPS`로 눈에 띄게 오른다. `v1` 대비 `4096`에서 약 `8.5%` 개선이다. 코드상으로 보면 `ThreadblockTileIterator`/`SmemTileIterator` 계열이 `v1`의 구형 iterator보다 더 잘 정리된 경로를 사용한다.

#### NCU 핵심 수치

- Benchmark (4096): `15.34 ms / 8.96 TFLOPS`
- Profile duration: `17.38 ms`
- Registers / thread: `40`
- Threads / block: `512`
- Achieved Occupancy: `98.09%`
- Eligible / Issued / No Eligible: `1.77 / 0.43 / 56.89%`
- Top stall: `MIO Throttle 13.26`, `Long Scoreboard 4.29`, `Barrier 3.61`, `Not Selected 3.11`, `Wait 0.73`, `Short Scoreboard 0.60`
- Shared path: `L1 Wavefronts Shared Excessive 0`

#### 해석

직전 커널 `v1`과 비교하면 핵심은 scheduler 상태다. `Regs 41 -> 40`, `Eligible 1.04 -> 1.77`, `Issued 0.38 -> 0.43`, `No Eligible 62.07% -> 56.90%`로 같이 좋아진다. 즉 reuse 구조는 그대로 두고도 load/store path의 codegen을 다듬으면 ready warp pool 자체가 두꺼워질 수 있다는 뜻이다. 다만 stall top은 여전히 `MIO Throttle 13.26`이라서 shared-memory instruction pressure는 남아 있다. 아직 큰 구조 변화가 아니라, 같은 구조를 더 효율적으로 실행한 단계다.

### `blocktiling_1d_v3_64x64x8_8`

#### 구조

`v3`는 loop 바깥에서 한 tile을 먼저 load한 뒤, loop 안에서 다음 tile을 뒤늦게 당겨오는 1-tile preload 형태다. 하지만 결과는 거의 `v2`와 같다. `4096`에서 `8.96 TFLOPS`, `Issued 0.43`, `No Eligible 56.90%`로 차이가 사실상 없다.

#### NCU 핵심 수치

- Benchmark (4096): `15.45 ms / 8.89 TFLOPS`
- Profile duration: `17.38 ms`
- Registers / thread: `40`
- Threads / block: `512`
- Achieved Occupancy: `98.10%`
- Eligible / Issued / No Eligible: `1.75 / 0.43 / 56.89%`
- Top stall: `MIO Throttle 13.43`, `Long Scoreboard 4.30`, `Barrier 3.61`, `Not Selected 3.06`, `Wait 0.72`, `Short Scoreboard 0.60`
- Shared path: `L1 Wavefronts Shared Excessive 0`

#### 해석

직전 커널 `v2`와 비교하면 benchmark와 profile이 사실상 동일하다. `Eligible 1.77 -> 1.75`, `Issued 0.43 -> 0.43`, `No Eligible 56.90% -> 56.90%`로 차이가 거의 없고 stall 분포도 거의 같다. 이건 중요한 신호다. 현재 이 shape에서는 단순한 1-stage lookahead만으로는 load-use distance를 충분히 벌리지 못한다. 이미 병목이 shared path와 barrier 쪽에 많이 남아 있어서, preload 한 벌을 앞당긴 것만으로는 구조적 한계를 넘지 못한다. 따라서 이후에는 단순 preload보다 thread tile, CTA tile, accumulate 구조 자체를 바꾸는 것이 더 큰 레버리지다.

### `blocktiling_1d_v2_64x64x8_16`

#### 구조

`ThreadM`을 `8 -> 16`으로 키우면 `4096` 기준 `10.56 TFLOPS`까지 오른다. 같은 shared operand를 thread 안에서 더 오래 소비하므로 register reuse가 좋아진 결과다.

#### NCU 핵심 수치

- Benchmark (4096): `12.88 ms / 10.66 TFLOPS`
- Profile duration: `14.74 ms`
- Registers / thread: `48`
- Threads / block: `256`
- Achieved Occupancy: `80.53%`
- Eligible / Issued / No Eligible: `2.00 / 0.47 / 53.28%`
- Top stall: `MIO Throttle 8.30`, `Not Selected 3.27`, `Barrier 2.87`, `Long Scoreboard 2.27`, `Short Scoreboard 1.48`, `Wait 0.83`
- Shared path: `L1 Wavefronts Shared Excessive 0`

#### 해석

직전 커널 `v3`와 비교하면 thread-level reuse 증가 효과가 분명하다. `Issued 0.43 -> 0.47`, `No Eligible 56.90% -> 53.28%`, `Long Scoreboard 4.31 -> 2.27`로 dependency가 눈에 띄게 줄었다. 반면 register는 `40 -> 48`로 늘고 `Not Selected 3.27`이 올라간다. 즉 shared path pressure는 줄었지만, warp끼리 scheduler slot을 두고 경쟁하는 흔적이 더 강해진다. 좋아진 점은 thread-level reuse가 확실히 늘었다는 것이고, 안 좋아진 점은 register pressure가 증가해 이후 large tile로 넘어갈수록 occupancy cliff에 더 가까워진다는 것이다.

### `blocktiling_1d_v2_128x128x8_64`

#### 구조

여기서 `64x64` family를 벗어나 `128x128` CTA tile과 `ThreadM=64`로 크게 점프한다. `4096`에서는 `12.01 TFLOPS`로 가장 빠른 1D kernel이 된다.

#### NCU 핵심 수치

- Benchmark (4096): `11.54 ms / 11.91 TFLOPS`
- Profile duration: `13.33 ms`
- DRAM Bandwidth: `71.58 GB/s`
- Registers / thread: `106`
- Threads / block: `256`
- Achieved Occupancy: `32.31%`
- Eligible / Issued / No Eligible: `0.82 / 0.49 / 51.15%`
- Top stall: `Short Scoreboard 2.35`, `MIO Throttle 2.30`, `Long Scoreboard 0.74`, `Not Selected 0.68`, `Wait 0.44`, `Barrier 0.34`

#### 해석

직전 커널 `64x64x8_16`과 비교하면 CTA tile 확대의 효과가 분명하다. `4096 TFLOPS`는 `10.56 -> 12.01`로 오르고, profile에서는 `Regs 48 -> 106`, `Occ 80.53% -> 32.31%`로 occupancy가 크게 내려가는데도 `Issued 0.47 -> 0.49`, `No Eligible 53.28% -> 51.15%`로 scheduler 상태는 유지된다. `DRAM 71.58 GB/s`까지 내려가는데 TFLOPS가 오르는 점도 중요하다. 같은 성능을 더 적은 off-chip traffic으로 내기 시작했다는 뜻이다. 좋아진 점은 block-level reuse가 커졌다는 것이고, 안 좋아진 점은 register budget이 커져 occupancy 여유가 줄었다는 것이다.

### `blocktiling_1d_v2_128x128x16_64`

#### 구조

이 버전은 위 구조에 `BLOCK_K=16`을 넣는다. `4096`에서 `12.24 TFLOPS`로 소폭 개선되고, `Issued 0.50`, `No Eligible 49.76%`로 scheduler 지표도 조금 더 나아진다.

SASS를 보면 이 단계의 성격이 더 분명해진다. global ingress는 아직 전부 `LDG.E.CONSTANT` scalar load이고, compute 쪽은 `LDS.128`으로 shared operand를 끌어온 뒤 `FFMA` strip을 길게 이어 붙이는 형태다. 즉 1D blocktiling의 핵심 이득은 vectorized memory path가 아니라, scalar로 가져온 operand를 thread 내부 register strip에서 오래 쓰도록 만들어 `FFMA` 본문을 늘린 데 있다.

```sass
LDG.E.CONSTANT R27, [R22.64]
LDG.E.CONSTANT R32, [R4.64]
...
LDS.128 R4, [R3+0x40]
FFMA R59, R45, R4, R59
FFMA R56, R45, R5, R56
FFMA R61, R45, R6, R61
FFMA R58, R45, R7, R58
```

#### NCU 핵심 수치

- Benchmark (4096): `11.30 ms / 12.16 TFLOPS`
- Profile duration: `12.98 ms`
- Registers / thread: `110`
- Threads / block: `256`
- Achieved Occupancy: `32.35%`
- Eligible / Issued / No Eligible: `0.86 / 0.50 / 49.77%`
- Top stall: `MIO Throttle 2.79`, `Short Scoreboard 2.28`, `Not Selected 0.72`, `Wait 0.34`, `Barrier 0.27`, `Long Scoreboard 0.25`
- Shared warning: shared store `1.5-way` bank conflict, `L1 Wavefronts Shared Excessive 16,777,216 (1%)`

#### 해석

직전 커널 `128x128x8_64`와 비교하면 `BLOCK_K` 증가 효과는 작지만 방향은 분명하다. `4096 TFLOPS`는 `12.01 -> 12.24`, `Issued`는 `0.49 -> 0.50`, `No Eligible`은 `51.15% -> 49.77%`로 조금씩 좋아진다. `BLOCK_K`를 키우면 같은 CTA tile을 계산하는 동안 barrier와 loop overhead를 더 적게 치르기 때문이다. 하지만 부작용도 생긴다. Nsight는 shared store `1.5-way bank conflict`, `shared excessive 1%`를 보고한다. 크지는 않지만, tile을 키우면서 shared staging의 정돈도가 완전히 공짜는 아니라는 뜻이다. 좋아진 점은 arithmetic intensity와 steady-state efficiency이고, 안 좋아진 점은 shared path에 작은 균열이 생기기 시작했다는 것이다. 이후 2D outer product가 필요한 이유가 여기 있다.

### Blocktiling 1D 정리

1D block tiling은 이 프로젝트에서 가장 큰 “초기 성능 점프”를 만든 구간이다. 다만 마지막 상태를 보면 한계도 분명하다.

- `64x64` family는 구현이 단순하고 안정적이지만 ceiling이 낮다.
- `128x128` family는 4096 기준 최고 1D 성능을 만들지만 register budget이 무겁다.
- 병목은 `LG Throttle`에서 `MIO/Scoreboard`로 옮겨졌지만, shared tile 하나당 만들 수 있는 FMA 수는 아직 충분하지 않다.

다음 단계는 같은 operand를 strip이 아니라 `ThreadM x ThreadN` outer product로 더 많이 재사용하는 것이다.

## 5. Blocktiling 2D Group

구현 파일: [`04_blocktiling_2d.cuh`](./04_blocktiling_2d.cuh)

| Kernel | 4096 TFLOPS | Regs | Occ | Issued | No Eligible | Top Stall | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `blocktiling_2d_128x128x8_8x8` | 17.53 | 103 | 32.31% | 0.62 | 38.21% | `Not Selected` | uncoalesced global 10%, shared excessive 38% |
| `blocktiling_2d_128x128x16_8x8` | 19.29 | 108 | 32.33% | 0.67 | 32.78% | `Not Selected` | uncoalesced global 10%, shared excessive 40% |

### `blocktiling_2d_128x128x8_8x8`

#### 구조

이 커널은 thread가 `8x8` output tile을 register에 들고 가는 2D outer-product 구조다. 바로 이전 그룹의 best이자 마지막 kernel인 `blocktiling_1d_v2_128x128x16_64`의 `12.24 TFLOPS`에서 `17.53 TFLOPS`로 점프한다. 이번 최적화 전체에서 가장 중요한 구조 변화 중 하나다.

1D strip과의 차이는 아래 inner loop에서 바로 보인다. 이제 `elem_a`와 `elem_b`를 한 축이 아니라 `ThreadM x ThreadN` accumulator tile 전체로 확장해서 쓴다.

```cpp
float accum[ThreadM][ThreadN] = {0.f};

for (int k = 0; k < ThreadblockK; ++k) {
  for (int m = 0; m < ThreadM; ++m) {
    float const elem_a =
        smem_a[kSmemStrideA * k + ThreadM * thread_tile_m_idx + m];

    for (int n = 0; n < ThreadN; ++n) {
      float const elem_b =
          smem_b[ThreadblockN * k + ThreadN * thread_tile_n_idx + n];
      accum[m][n] += elem_a * elem_b;
    }
  }
}
```

#### NCU 핵심 수치

- Benchmark (4096): `7.86 ms / 17.48 TFLOPS`
- Profile duration: `8.95 ms`
- Registers / thread: `103`
- Achieved Occupancy: `32.31%`
- L1/TEX Hit / L2 Hit / L2 Excessive: `8.75% / 81.67% / 14.68 MB`
- Block limit Registers / Shared Mem: `2 / 7`
- Eligible / Issued / No Eligible: `1.63 / 0.62 / 38.21%`
- Top stall: `Not Selected 1.64`, `Short Scoreboard 0.96`, `Long Scoreboard 0.83`, `Barrier 0.60`, `MIO Throttle 0.51`, `Dispatch Stall 0.29`
- Shared / global warning: shared load `5.0-way` bank conflict, `L1 Wavefronts Shared Excessive 268,435,456 (38%)`, uncoalesced global access `10%`

#### 해석

왜 이렇게 크게 뛰는지는 구현을 보면 명확하다. inner loop에서 `A` scalar 하나와 `B` scalar 하나를 읽으면 이제 `64`개의 accumulator update로 이어진다. 1D strip에서는 shared operand가 한 축으로만 재사용됐지만, 여기서는 `M`과 `N` 양쪽으로 동시에 재사용된다. 그래서 이전 그룹 best 대비 `Issued 0.50 -> 0.62`, `No Eligible 49.76% -> 38.16%`로 확 좋아진다. 다만 이 커널은 memory path가 아직 지저분하다. Nsight는 shared load `5.0-way bank conflict`, `shared excessive 38%`, uncoalesced global store `10%`, `L2 Excessive 14.68 MB`를 보고한다. 즉 계산 구조는 아주 좋아졌는데, load/store path가 그 개선을 다 따라오지 못한다. 그럼에도 큰 속도 향상이 나온 이유는 계산 밀도 증가의 효과가 그만큼 컸기 때문이다.

### `blocktiling_2d_128x128x16_8x8`

#### 구조

여기에 `BLOCK_K=16`을 적용하면 `19.29 TFLOPS`까지 오른다. `Issued 0.67`, `No Eligible 32.79%`는 reference에 조금 더 가까워진 수치다.

instruction 관점에서도 이 단계는 “memory path를 정리했다”기보다 “compute burst를 더 길게 만들었다”는 쪽에 가깝다. SASS는 여전히 `LDG.E.CONSTANT` scalar ingress를 쓰고, 그 뒤 긴 `FFMA` 묶음이 이어진다. 즉 scalar load 자체는 그대로지만, 2D outer product가 만들어 내는 register accumulator 수가 훨씬 많아져 같은 load 당 더 많은 `FFMA`를 밀어 넣을 수 있다.

```sass
LDG.E.CONSTANT R87, [R30.64]
LDG.E.CONSTANT R29, [R18.64]
...
FFMA R64, R21, R24, R64
FFMA R57, R22, R24, R57
FFMA R76, R23, R24, R76
FFMA R71, R28, R25, R71
```

#### NCU 핵심 수치

- Benchmark (4096): `7.19 ms / 19.10 TFLOPS`
- Profile duration: `8.20 ms`
- Registers / thread: `108`
- Achieved Occupancy: `32.33%`
- L1/TEX Hit / L2 Hit / L2 Excessive: `8.69% / 81.57% / 14.68 MB`
- Block limit Registers / Shared Mem: `2 / 3`
- Eligible / Issued / No Eligible: `1.94 / 0.67 / 32.78%`
- Top stall: `Not Selected 1.89`, `Short Scoreboard 0.69`, `MIO Throttle 0.67`, `Barrier 0.41`, `Long Scoreboard 0.38`, `Dispatch Stall 0.30`
- Shared / global warning: shared load `5.0-way` bank conflict, shared store `1.5-way` bank conflict, `L1 Wavefronts Shared Excessive 285,212,672 (40%)`, uncoalesced global access `10%`

#### 해석

직전 커널 `BK=8`과 비교하면 개선 이유는 1D 때와 같다. `BLOCK_K`가 커지면서 loop/barrier overhead가 줄고, 같은 thread tile이 더 긴 steady-state를 유지한다. 실제로 `Issued 0.62 -> 0.67`, `No Eligible 38.16% -> 32.79%`로 scheduler 상태가 더 좋아진다. 하지만 부작용도 같이 커진다. shared excessive는 `38% -> 40%`, shared store bank conflict도 새로 잡힌다. 즉 arithmetic intensity를 더 키우는 대신, shared path의 병목은 오히려 더 또렷해진다. 좋아진 점은 absolute TFLOPS와 scheduler 상태이고, 안 좋아진 점은 memory path의 낭비가 더 구조적으로 고착된다는 점이다. 그래서 다음 단계에서는 단순 `BK` 증가가 아니라 global/store 경로를 vectorized하게 정리해야 한다.

### Blocktiling 2D 정리

2D outer product는 현재 커널군 전체에서 가장 큰 구조적 성능 향상을 만든 단계다. 하지만 동시에 “계산 구조는 충분히 좋아졌는데 memory path가 뒤처진다”는 새로운 문제를 드러낸다. 그 다음 단계가 `vec4`다.

## 6. Blocktiling 2D Vec4 Group

구현 파일: [`05_blocktiling_2d_vec4.cuh`](./05_blocktiling_2d_vec4.cuh)

| Kernel | 4096 TFLOPS | Regs | Occ | Issued | No Eligible | Top Stall | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `blocktiling_2d_vec4_v0_128x128x8_8x8` | 18.02 | 113 | 32.39% | 0.61 | 38.69% | `Not Selected` | uncoalesced global 2%, shared excessive 38% |
| `blocktiling_2d_vec4_v0_128x128x16_8x8` | 20.11 | 103 | 32.34% | 0.69 | 31.29% | `Not Selected` | uncoalesced global 2%, shared excessive 40% |
| `blocktiling_2d_vec4_v1_128x128x8_8x8` | 19.11 | 111 | 32.34% | 0.65 | 35.25% | `Not Selected` | uncoalesced global 2%, shared excessive 38% |
| `blocktiling_2d_vec4_v1_128x128x16_8x8` | 20.26 | 103 | 32.32% | 0.68 | 31.78% | `Not Selected` | uncoalesced global 2%, shared excessive 40% |

### `blocktiling_2d_vec4_v0_128x128x8_8x8`

#### 구조

`v0`는 B tile load와 C store를 `vec4`로 바꾼 버전이고, A path는 아직 scalar transpose helper를 쓴다. 사용자 요청한 비교 기준대로 이전 그룹의 best이자 마지막 kernel인 `blocktiling_2d_128x128x16_8x8`과 먼저 비교하면 `4096` 기준 절대 성능은 `19.29 -> 18.02 TFLOPS`로 낮다. 다만 `BK`가 `16 -> 8`로 달라 직접 일대일 비교만으로 보면 왜곡되므로, 같은 `BK=8` scalar 2D baseline과 나란히 보면 vec4 도입 효과가 더 잘 보인다.

이 단계에서 중요한 코드는 아래 두 부분이다. B ingest는 vec4 iterator로 당기고, C store도 `float4` 단위로 묶어서 내보낸다. 그래서 global sector 낭비가 줄어든다.

```cpp
using ThreadMapB =
    utils::PitchLinearStripminedThreadMap<ThreadblockN, ThreadblockK,
                                          kThreads, kVecWidth>;

threadblock_tile_iterator_b.load(frag_b);
smem_tile_iterator_b.store(frag_b);

for (int n = 0; n < ThreadN; n += kVecWidth) {
  float4 const value = utils::make_float4_from_array(&accum[m][n]);
  utils::store_float4(&c_ptr[c_row * N + c_col_base + n], value);
}
```

#### NCU 핵심 수치

- Benchmark (4096): `7.66 ms / 17.93 TFLOPS`
- Profile duration: `8.80 ms`
- Registers / thread: `113`
- Achieved Occupancy: `32.39%`
- L1/TEX Hit / L2 Hit / L2 Excessive: `1.42% / 80.17% / 2.10 MB`
- Block limit Registers / Shared Mem: `2 / 7`
- Eligible / Issued / No Eligible: `1.58 / 0.61 / 38.69%`
- Top stall: `Not Selected 1.58`, `Short Scoreboard 1.01`, `Long Scoreboard 0.85`, `Barrier 0.68`, `MIO Throttle 0.50`, `Dispatch Stall 0.41`
- Shared / global warning: shared load `5.0-way` bank conflict, `L1 Wavefronts Shared Excessive 38%`, uncoalesced global access `2%`

#### 해석

이전 그룹 best인 `blocktiling_2d_128x128x16_8x8`과 비교하면 절대 성능은 낮지만, 같은 BK의 scalar 2D baseline과 비교하면 vec4 효과는 분명하다. 가장 눈에 띄는 변화는 Nsight 경고다. uncoalesced global access가 `10% -> 2%`로 줄고, `L2 Excessive`도 같은 BK의 scalar 기준 `14.68 MB -> 2.10 MB`로 크게 줄어든다. 즉 vec4 store가 output path를 실제로 정리해 준다. 반면 shared excessive `38%`와 shared load `5.0-way bank conflict`는 그대로 남는다. 그래서 absolute 성능 개선 폭이 아주 크지는 않다. 좋아진 점은 global store path이고, 아쉬운 점은 register가 `113`으로 늘었는데도 shared path는 거의 못 건드렸다는 것이다. 결국 이 버전은 “global은 정리되지만 shared는 그대로인 상태”다.

### `blocktiling_2d_vec4_v0_128x128x16_8x8`

#### 구조

이 버전은 `BK=16`까지 합쳐져 `20.11 TFLOPS`에 도달한다. 처음으로 `20 TFLOPS`를 넘는 study kernel이다. `Issued 0.69`, `No Eligible 31.28%`로 지금까지 중 가장 reference에 근접한 scheduler 상태를 보여 준다.

#### NCU 핵심 수치

- Benchmark (4096): `6.88 ms / 19.96 TFLOPS`
- Profile duration: `7.87 ms`
- Registers / thread: `103`
- Achieved Occupancy: `32.34%`
- L1/TEX Hit / L2 Hit / L2 Excessive: `1.41% / 80.07% / 2.10 MB`
- Block limit Registers / Shared Mem: `2 / 3`
- Eligible / Issued / No Eligible: `1.96 / 0.69 / 31.29%`
- Top stall: `Not Selected 1.86`, `Short Scoreboard 0.68`, `Long Scoreboard 0.50`, `MIO Throttle 0.45`, `Barrier 0.43`, `Dispatch Stall 0.41`
- Shared / global warning: shared load `5.0-way` bank conflict, shared store `2.4-way` bank conflict, `L1 Wavefronts Shared Excessive 40%`, uncoalesced global access `2%`

#### 해석

직전 커널 `v0 BK8`과 비교하면 이전 커널의 장점이 그대로 확대된다. vec4 store로 global sector 낭비를 줄였고 `BK=16`으로 mainloop overhead를 줄였기 때문에 `4096 TFLOPS`가 `18.02 -> 20.11`, `Issued`가 `0.61 -> 0.69`, `No Eligible`이 `38.69% -> 31.29%`로 좋아진다. 다만 shared path는 여전히 깨끗하지 않다. shared excessive `40%`, shared store `2.4-way bank conflict`가 그대로 잡힌다. 즉 이 커널은 “global path 정리 + BK16”의 성공 사례다. 하지만 동시에 “shared path를 그대로 둔 채로도 20 TFLOPS까지는 갈 수 있다”는 것을 보여 준다.

### `blocktiling_2d_vec4_v1_128x128x8_8x8`

#### 구조

`v1`은 A side도 iterator 기반 vec4 transpose/store 경로로 바꾼 버전이다. `4096`에서 `19.11 TFLOPS`로 `v0 BK8` 대비 약 `1.09 TFLOPS` 오른다.

#### NCU 핵심 수치

- Benchmark (4096): `7.21 ms / 19.05 TFLOPS`
- Profile duration: `8.27 ms`
- Compute / Memory / DRAM: `60.54% / 80.98% / 114.20 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `1.56% / 80.06% / 2.10 MB`
- Registers / thread: `111`
- Achieved Occupancy: `32.34%`
- Block limit Registers / Shared Mem: `2 / 7`
- Eligible / Issued / No Eligible: `1.62 / 0.65 / 35.25%`
- Top stall: `Not Selected 1.50`, `Short Scoreboard 1.01`, `Long Scoreboard 0.85`, `Barrier 0.67`, `MIO Throttle 0.43`, `Dispatch Stall 0.33`
- Shared / global warning: shared load `5.0-way` bank conflict, `L1 Wavefronts Shared Excessive 38%`, uncoalesced global access `2%`

#### 해석

직전 커널 `v0 BK8`과 비교하면 shared excessive 자체는 거의 그대로인데 성능이 오른다는 점이 중요하다. `Issued 0.61 -> 0.65`, `No Eligible 38.69% -> 35.25%`, `4096 TFLOPS 18.02 -> 19.11`이 같이 좋아지기 때문이다. 즉 A path vectorization이 shared path 낭비 숫자를 완전히 없애진 못해도, 실제 instruction mix와 issue efficiency를 개선한다. 좋아진 점은 A ingest까지 정리되었다는 것이고, 안 좋아진 점은 shared load bank conflict라는 더 깊은 구조 문제는 여전히 해결되지 않았다는 점이다.

### `blocktiling_2d_vec4_v1_128x128x16_8x8`

#### 구조

이 group 최고점이다. `4096`에서 `20.26 TFLOPS`, `6.76 ms`이며 torch 대비 약 `84.9%`, best CUTLASS 대비 약 `89.4%`다.

이 버전은 vec4화가 실제 instruction 폭으로도 내려간다는 점이 중요하다. SASS를 보면 ingest 쪽에 `LDG.E.128.CONSTANT`, stage commit 쪽에 `STS.128`, epilogue 쪽에 `STG.E.128`이 분명히 나타난다. 즉 `scalar 2D`에서 보이던 `32-bit global load/store + 긴 address path` 일부를, 여기서는 실제 128-bit ingress/egress로 치환한다. `L2 Excessive 2.10 MB`, uncoalesced global access `2%`가 나오는 이유가 여기 있다.

```sass
LDG.E.128.CONSTANT R68, [R68.64]
LDG.E.128.CONSTANT R72, [R72.64]
STS.128 [R97.X4+0x2100], R76
LDS.128 R68, [R72]
LDS.128 R84, [R98+0x2100]
FFMA R60, R68, R84, R60
...
STG.E.128 [R2.64], R60
STG.E.128 [R2.64+0x10], R52
```

#### NCU 핵심 수치

- Benchmark (4096): `6.83 ms / 20.13 TFLOPS`
- Profile duration: `7.81 ms`
- Compute / Memory / DRAM: `63.91% / 83.80% / 121.80 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `1.43% / 79.97% / 2.10 MB`
- Registers / thread: `103`
- Achieved Occupancy: `32.32%`
- Block limit Registers / Shared Mem: `2 / 3`
- Eligible / Issued / No Eligible: `1.97 / 0.68 / 31.78%`
- Top stall: `Not Selected 1.89`, `Short Scoreboard 0.67`, `Long Scoreboard 0.54`, `MIO Throttle 0.51`, `Barrier 0.44`, `Dispatch Stall 0.40`
- Shared / global warning: shared load `5.0-way` bank conflict, shared store `2.4-way` bank conflict, `L1 Wavefronts Shared Excessive 40%`, uncoalesced global access `2%`

#### 해석

직전 커널 `v1 BK8`과 비교하면 세 가지가 겹친다. 2D outer product로 높은 arithmetic reuse를 확보한 상태에서 `BK=16`으로 steady-state를 길게 만들고, vec4 A/B/C path로 global access 낭비를 `2%` 수준까지 줄였다. 실제로 `Issued 0.65 -> 0.68`, `No Eligible 35.25% -> 31.78%`, `4096 TFLOPS 19.11 -> 20.26`이 같이 좋아진다. SASS 기준으로 보면 이 커널의 핵심은 `LDG.E.128`과 `STG.E.128`이 실제로 mainloop ingress와 epilogue egress에 들어왔다는 점이다. 하지만 아직 끝은 아니다. shared excessive는 여전히 `40%`, shared load `5.0-way bank conflict`, shared store `2.4-way bank conflict`가 남아 있다. 즉 이 group 최고점은 “shared path가 완전히 정리된 커널”이 아니라, “global path를 먼저 정리해 성능을 최대한 뽑아낸 커널”이다.

### Blocktiling 2D Vec4 정리

이 group은 명확한 결론을 준다.

- vec4는 global path를 실제로 개선한다.
- 특히 uncoalesced global store가 `10% -> 2%`까지 줄어드는 효과가 크다.
- 그러나 shared load bank conflict와 excessive wavefront는 거의 그대로 남는다.

즉 vec4만으로는 충분하지 않다. 이제는 warp-level lane mapping과 shared read pattern 자체를 바꿔야 한다.

## 7. Warptiling Group

구현 파일: [`06_warptiling_v0.cuh`](./06_warptiling_v0.cuh), [`06_warptiling_v1.cuh`](./06_warptiling_v1.cuh), [`06_warptiling_v2.cuh`](./06_warptiling_v2.cuh), [`06_warptiling_v2_vec4.cuh`](./06_warptiling_v2_vec4.cuh), [`06_warptiling_v3.cuh`](./06_warptiling_v3.cuh), [`06_warptiling_v3_vec4.cuh`](./06_warptiling_v3_vec4.cuh), [`06_warptiling_v4.cuh`](./06_warptiling_v4.cuh), [`06_warptiling_v4_vec4.cuh`](./06_warptiling_v4_vec4.cuh), [`06_warptiling_v5.cuh`](./06_warptiling_v5.cuh), [`06_warptiling_v5_vec4.cuh`](./06_warptiling_v5_vec4.cuh), [`06_warptiling_v6.cuh`](./06_warptiling_v6.cuh), [`06_warptiling_v6_vec4.cuh`](./06_warptiling_v6_vec4.cuh), [`06_warptiling_v7.cuh`](./06_warptiling_v7.cuh), [`06_warptiling_v7_vec4.cuh`](./06_warptiling_v7_vec4.cuh), [`06_warptiling_v8.cuh`](./06_warptiling_v8.cuh), [`06_warptiling_v8_vec4.cuh`](./06_warptiling_v8_vec4.cuh), [`06_warptiling_v9.cuh`](./06_warptiling_v9.cuh), [`06_warptiling_v10.cuh`](./06_warptiling_v10.cuh), [`06_warptiling_v11.cuh`](./06_warptiling_v11.cuh)

| Kernel | 4096 TFLOPS | Regs | Occ | Issued | No Eligible | Top Stall | 핵심 변화 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `warptiling_v0_128x128x8_32x64x8` | 17.83 | 121 | 32.32% | 0.62 | 37.77% | `Not Selected` | 첫 warp hierarchy, scalar ingress |
| `warptiling_v0_128x128x16_32x64x16` | 19.42 | 121 | 32.39% | 0.69 | 31.37% | `Not Selected` | `BK16`로 steady-state 연장 |
| `warptiling_v1_128x128x8_32x64x8` | 19.54 | 109 | 32.43% | 0.69 | 31.39% | `Not Selected` | lane remap, clean shared path |
| `warptiling_v2_128x128x8_32x64x8` | 20.70 | 127 | 32.24% | 0.74 | 26.06% | `Not Selected` | 정리된 경로 + 2-stage |
| `warptiling_v2_vec4_128x128x8_32x64x8` | 23.01 | 127 | 32.42% | 0.81 | 19.39% | `Not Selected` | 2-stage + vec4 ingress |
| `warptiling_v3_128x128x8_32x64x8` | 20.96 | 127 | 32.23% | 0.75 | 24.60% | `Not Selected` | `v2 + PTX L2::128B load` |
| `warptiling_v3_vec4_128x128x8_32x64x8` | 23.75 | 127 | 32.43% | 0.84 | 15.52% | `Not Selected` | vec4 family도 `LTC128B` helper path 진입 |
| `warptiling_v4_128x128x8_32x64x8` | 21.66 | 128 | 32.30% | 0.78 | 21.51% | `Not Selected` | `v3 + predicated global load` |
| `warptiling_v4_vec4_128x128x8_32x64x8` | 23.70 | 127 | 32.41% | 0.84 | 15.55% | `Not Selected` | predicated vec4 path |
| `warptiling_v5_128x128x8_32x64x8` | 21.79 | 125 | 32.38% | 0.80 | 19.71% | `Not Selected` | `v4 + partial CUTLASS-style pipeline` |
| `warptiling_v5_vec4_128x128x8_32x64x8` | 23.57 | 127 | 32.44% | 0.84 | 16.31% | `Not Selected` | vec4 + partial CUTLASS-style pipeline |
| `warptiling_v6_128x128x8_32x64x8` | 21.85 | 120 | 32.40% | 0.82 | 18.49% | `Not Selected` | `v4 + full CUTLASS-style pipeline` |
| `warptiling_v6_vec4_128x128x8_32x64x8` | 22.83 | 129 | 16.66% | 0.79 | 20.60% | `Not Selected` | full CUTLASS-style + vec4, 1 block/SM |
| `warptiling_v7_128x128x8_32x64x8` | 22.01 | 120 | 32.39% | 0.82 | 17.83% | `Not Selected` | `v6 + CUTLASS-style epilogue reorder` |
| `warptiling_v7_vec4_128x128x8_32x64x8` | 23.76 | 127 | 32.46% | 0.85 | 14.79% | `Not Selected` | vec4 + CUTLASS-style epilogue reorder, current study best |
| `warptiling_v8_128x128x8_32x64x8` | 22.21 | 127 | 32.38% | 0.83 | 17.24% | `Not Selected` | `v7 + explicit prologue/steady-state/drain` |
| `warptiling_v8_vec4_128x128x8_32x64x8` | 23.70 | 128 | 32.35% | 0.85 | 14.87% | `Not Selected` | vec4 + explicit prologue/steady-state/drain |
| `warptiling_v9_128x128x8_32x64x8` | 22.27 | 127 | 32.36% | 0.82 | 17.70% | `Not Selected` | next-tile global load 시점 재배치 |
| `warptiling_v10_128x128x8_32x64x8` | 22.74 | 126 | 32.44% | 0.84 | 15.93% | `Not Selected` | `v9`와 동일 로직 + `cutlass` 이름으로 codegen 변화 |
| `warptiling_v11_128x128x8_32x64x8` | 22.94 | 127 | 32.34% | 0.84 | 15.94% | `Not Selected` | `v9 + grouped threadblock swizzle`, plain symbol 유지 |

이번 artifacts의 warptiling은 `v3`를 기점으로 성격이 한 번 바뀐다. `v0`부터 `v2`까지는 warp hierarchy, lane remap, 2-stage처럼 구조 자체를 만들어 가는 단계였다면, `v3`부터는 그 구조를 크게 바꾸지 않은 채 load primitive와 pipeline 방식을 다듬는 단계로 넘어간다. `v3`는 `v2`의 계산 구조를 유지한 채 `ThreadblockTileIterator::load_ptx()`로 `ld.global.L2::128B` helper를 직접 쓰는 버전이고, `v4`는 그 위에 predicated `load(frag, pred)`를 얹어 tail load를 정리한 버전이다. `v5`는 CUTLASS-style stage iterator와 write-stage ring을 부분적으로 도입한 버전이고, `v6`는 warp fragment ping-pong과 inner-loop prefetch까지 넣어 mainloop를 거의 CUTLASS의 steady-state 모양으로 가져간 버전이다. `v7`/`v7_vec4`는 이 mainloop를 유지한 채 epilogue만 CUTLASS-style `shared reorder -> global store` 구조로 교체한 버전이고, `v8`/`v8_vec4`는 같은 epilogue를 유지하면서 mainloop를 explicit `prologue -> steady-state -> drain` 형태로 다시 정리한 버전이다. `v9`는 여기에 새 primitive를 더한 것이 아니라, 같은 epilogue와 같은 mainloop 골격을 유지한 채 next-tile global load 시점을 outer loop 초반으로 당겨 dependency 모양만 조정한 버전이다. `v10`은 반대로 source-level algorithm 변경이 전혀 없다. [`06_warptiling_v9.cuh`](./06_warptiling_v9.cuh)와 [`06_warptiling_v10.cuh`](./06_warptiling_v10.cuh)의 차이는 namespace와 kernel symbol을 `cutlass_sgemm_warptiling`으로 바꾼 것뿐이다. 이번에 추가된 `v11`은 다시 다른 방향이다. [`06_warptiling_v11.cuh`](./06_warptiling_v11.cuh)는 `v9`의 mainloop를 유지한 채 `GroupedThreadblockSwizzle<8>`로 CTA 방문 순서만 바꾼 버전이다. 즉 `v3` 이후의 핵심 비교 축은 “더 깊은 hierarchy를 추가했는가”가 아니라 “같은 hierarchy를 어떤 load primitive와 어떤 stage orchestration으로 굴리느냐”, 그리고 마지막에는 “같은 mainloop 위에서 output path를 얼마나 CUTLASS처럼 정리하느냐”, “backend가 보조 명령열을 어떻게 고르느냐”, “CTA를 어떤 순서로 배치해 cache reuse를 만들 것이냐”로 넓어졌다. 여기서 `L2::128B` helper의 의미는 thread map 자체를 바꾸는 것이 아니라 plain C++ global load를 inline PTX로 고정해 compiler-dependent `LDG` 대신 `LDG.E.LTC128B` 계열 codegen을 더 안정적으로 얻는 데 있다. 다시 말해 `v3` 계열은 vec4처럼 per-thread access 폭을 넓히는 최적화라기보다, 같은 access 폭을 유지한 채 L2 fetch hint와 cache-friendly sector ingest를 명시적으로 주는 최적화다.

instruction 수준에서 보면 family가 더 선명하게 갈린다. `v2`는 여전히 `LDG.E.CONSTANT x16`, `LDS.128 x32`, `IADD3 x64`인 scalar 2-stage baseline이다. `v2_vec4`는 `LDG.E.128.CONSTANT x4`, `STS.128 x2`, `LDS.128 x32`, `IADD3 x39`로 올라가면서 첫 번째 넓은-ingress 기준점이 된다. `v3`와 `v4`는 scalar family에서도 `LDG.E.LTC128B x16`로 바뀌고, `v3`는 `IADD3 x64`, `v4`는 `IADD3 x61`이다. vec4 쪽은 `v3_vec4`와 `v4_vec4`가 모두 `LDG.E.LTC128B x4`, `STS.128 x2`, `LDS.128 x32`, `IADD3 x37`로 내려간다. `v5`는 partial CUTLASS-style stage iterator 때문에 scalar `IADD3 x76`, vec4 `IADD3 x50`으로 control footprint가 커지고, `v6`는 full CUTLASS-style ping-pong 때문에 scalar `LDS.128 x36 / IADD3 x80`, vec4 `LDS.128 x36 / IADD3 x54`까지 올라간다. `v7`/`v7_vec4`는 여기에 epilogue shared round-trip을 추가해 global store를 `32B/sector`까지 정리하고, `v8`/`v8_vec4`는 같은 store path를 유지한 채 mainloop choreography를 명시적인 `prologue -> steady-state -> drain`으로 단순화한다. `v9`는 instruction family를 새로 바꾸지 않고, 같은 scalar `32B/sector` path에서 next-tile global load를 warp MMA inner loop 바깥으로 끌어내 dependency 타이밍만 바꾼다. `v10`은 여기서 다시 흥미로운 분기다. core loop family는 `v9`와 완전히 같아서 `LDG.E.LTC128B x16`, `LDS.128 x32`, `FFMA x512`, `STS x80`, `STG.E x64`, `BAR.SYNC x18`이 그대로 유지된다. 바뀌는 것은 보조 명령열이다. `IADD3 58 -> 38`, `IADD3.X 18 -> 11`, `LEA 46 -> 38`, `MOV 43 -> 12`로 줄고, 반대로 `IMAD 23 -> 62`, `ULDC 3 -> 5`, `ULDC.64 1 -> 3`, `CS2R 48 -> 62`가 늘어난다. 더 세부적으로는 `v9`에는 없던 `IMAD.MOV.U32 x7`, `IMAD.SHL.U32 x6`, `IMAD.IADD x11`이 `v10`에 생긴다. 즉 cache hint나 primitive가 바뀐 것이 아니라, backend가 주소 생성과 constant materialization을 더 IMAD-centric하게 다시 고르면서 hot strip으로 들어가기 전 scalar scaffolding을 압축한 셈이다. 이 때문에 scalar family는 `v8`에서 L2 hit과 DRAM pressure가 더 좋아지고, `v9`에서는 store 품질을 유지한 채 dependency timing이 정리되며, `v10`에서는 마지막으로 codegen 보조 명령열이 더 압축된다. vec4 family는 이미 높은 issue efficiency 때문에 거의 같은 천장 안에서만 움직인다.

### `warptiling_v0_128x128x8_32x64x8`

`v0`는 CTA tile을 warp tile로 쪼개고, warp 안의 lane을 `warp_threads_m x warp_threads_n` grid로 배치하는 첫 버전이다. 구조만 보면 이전 그룹 best인 `blocktiling_2d_vec4_v1_128x128x16_8x8`보다 한 단계 더 정교해 보이지만, 성능은 `20.26 -> 17.83 TFLOPS`로 오히려 낮다. 이유는 간단하다. warp hierarchy는 들어왔지만, memory path는 아직 전혀 정리되지 않았다.

SASS를 보면 mainloop는 여전히 `LDG.E.CONSTANT -> STS -> BAR.SYNC -> LDS.128 -> FFMA`다. 즉 warp가 `LDS.128`으로 shared operand를 읽어 오는 구간만 넓고, global ingress와 shared commit은 모두 scalar다. 실제로 `Global Store Avg. Bytes / Sector 4`, `L2 Excessive 14.68 MB`, uncoalesced global access `10%`, `L1 Wavefronts Shared Excessive 38%`가 한꺼번에 남는다.

- Benchmark (4096): `7.78 ms / 17.67 TFLOPS`
- Profile duration: `8.89 ms`
- Registers / thread: `121`
- Achieved Occupancy: `32.32%`
- L1/TEX Hit / L2 Hit / L2 Excessive: `8.56% / 81.68% / 14.68 MB`
- Global Store Avg. Bytes / Sector: `4`
- Eligible / Issued / No Eligible: `1.69 / 0.62 / 37.77%`
- Top stall: `Not Selected 1.71`, `Long Scoreboard 0.74`, `MIO Throttle 0.69`, `Barrier 0.69`, `Short Scoreboard 0.47`, `Dispatch Stall 0.47`
- Shared / global warning: shared load `5.0-way` bank conflict, `L1 Wavefronts Shared Excessive 38%`, uncoalesced global access `10%`

이 커널은 “warp hierarchy만 넣는 것”으로는 충분하지 않다는 점을 보여 준다. shared와 global path가 blocktiling 시절과 거의 같은 수준으로 거칠어서, arithmetic hierarchy가 좋아진 만큼을 memory path가 따라오지 못한다.

### `warptiling_v0_128x128x16_32x64x16`

`BK=16`을 넣으면 `v0`가 곧바로 `19.42 TFLOPS`까지 오른다. 같은 scalar ingress, 같은 거친 shared/global path를 유지한 채로도 `BK=16`만으로 `Issued 0.62 -> 0.69`, `No Eligible 37.77% -> 31.37%`가 되는 것을 보면, 이 단계의 이득은 instruction 종류 변화가 아니라 더 긴 steady-state에 있다.

SASS도 그대로 그 사실을 보여 준다. 여전히 `LDG.E.CONSTANT`, scalar `STS`, `LDS.128`, `FFMA` 조합이고 새로운 wide op는 생기지 않는다. 좋아진 것은 `Barrier 0.68 -> 0.37`, `Long Scoreboard 0.74 -> 0.38`처럼 loop/barrier overhead가 희석된 부분이다. 반면 `L2 Excessive 14.68 MB`, `Global Store Avg. Bytes / Sector 4`, uncoalesced global `10%`는 그대로 남는다.

- Benchmark (4096): `7.07 ms / 19.43 TFLOPS`
- Profile duration: `8.10 ms`
- Registers / thread: `121`
- Achieved Occupancy: `32.39%`
- L1/TEX Hit / L2 Hit / L2 Excessive: `8.58% / 81.48% / 14.68 MB`
- Global Store Avg. Bytes / Sector: `4`
- Eligible / Issued / No Eligible: `1.95 / 0.69 / 31.37%`
- Top stall: `Not Selected 1.84`, `Short Scoreboard 0.61`, `MIO Throttle 0.59`, `Dispatch Stall 0.44`, `Barrier 0.38`, `Long Scoreboard 0.37`
- Shared / global warning: shared load `4.4-way` bank conflict, `L1 Wavefronts Shared Excessive 38%`, uncoalesced global access `10%`

즉 `v0 BK16`은 “warptiling 구조는 맞는데 path가 지저분하다”는 사실을 더 분명하게 만든 버전이다. 여기서 다음 단계로 필요한 것은 더 긴 `BK`가 아니라 lane/store mapping 자체의 정리다.

### `warptiling_v1_128x128x8_32x64x8`

`v1`은 warptiling에서 가장 중요한 turning point다. lane id를 그대로 쓰지 않고 warp 안에서 `(lane_m_idx, lane_n_idx)`로 다시 remap해 shared consume과 output store mapping을 정리한다. 코드에서 핵심은 아래 remap이다.

```cpp
int const row_major = lane_id / Traits::kLaneStride;
int const residual = lane_id - row_major * Traits::kLaneStride;
int const lane_n_idx = residual / Traits::kLaneLayout;
int const row_minor = residual - lane_n_idx * Traits::kLaneLayout;
int const lane_m_idx = row_major * Traits::kLaneLayout + row_minor;
```

직전 커널 `v0 BK16`과 비교하면 benchmark는 `19.42 -> 19.54 TFLOPS`로 소폭 오르지만, metric의 성격이 더 중요하게 바뀐다. `L1 Wavefronts Shared Excessive`가 `38% -> 0`, uncoalesced global access가 `10% -> 4%`, `Global Store Avg. Bytes / Sector`가 `4 -> 8`, `L2 Excessive`가 `14.68 -> 6.29 MB`로 정리된다. 즉 이 커널의 본질은 absolute TFLOPS보다 경로 정리에 있다.

- Benchmark (4096): `7.07 ms / 19.44 TFLOPS`
- Profile duration: `8.11 ms`
- Compute / Memory / DRAM: `64.07% / 42.14% / 115.72 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.55% / 80.46% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `109`
- Achieved Occupancy: `32.43%`
- Eligible / Issued / No Eligible: `1.98 / 0.69 / 31.39%`
- Top stall: `Not Selected 1.88`, `Long Scoreboard 0.76`, `Short Scoreboard 0.60`, `Dispatch Stall 0.45`, `Barrier 0.45`, `Wait 0.23`
- Shared / global warning: `L1 Wavefronts Shared Excessive 0`, uncoalesced global access `4%`

SASS는 여전히 `LDG.E.CONSTANT -> STS -> LDS.128 -> FFMA`라서 pipeline 자체는 아직 단순하다. 그래도 이후 버전이 모두 `v1`의 정리된 경로를 계승한다는 점에서, warptiling에서는 사실상 기준선이 이 버전부터 다시 시작된다고 보는 편이 맞다.

### `warptiling_v2_128x128x8_32x64x8`

`v2`는 `v1`의 정리된 경로 위에 `2-stage`를 올린 버전이다. prologue에서 stage 0를 먼저 shared에 올리고, outer loop에서는 다음 global tile을 읽은 뒤 현재 stage 전체를 소비하고 다음 stage에 commit하는 전형적인 2-stage 구조다.

```cpp
threadblock_tile_iterator_a.load(tb_frag_a);
threadblock_tile_iterator_b.load(tb_frag_b);
++threadblock_tile_iterator_a;
++threadblock_tile_iterator_b;

store_fragments_to_stage<Traits>(smem_a, smem_b, 0, tid, tb_frag_a, tb_frag_b);
__syncthreads();

for (; gemm_k_iterations > 0; --gemm_k_iterations) {
  if (gemm_k_iterations > 1) {
    threadblock_tile_iterator_a.load(tb_frag_a);
    threadblock_tile_iterator_b.load(tb_frag_b);
    ++threadblock_tile_iterator_a;
    ++threadblock_tile_iterator_b;
  }
  ...
}
```

metric은 매우 일관되게 좋아진다. `shared excessive 0`, `L2 Excessive 6.29 MB`, `Global Store Avg. Bytes / Sector 8`은 유지되면서 `Issued 0.69 -> 0.74`, `Eligible 1.98 -> 2.15`, `No Eligible 31.31% -> 26.03%`로 개선된다. `Barrier`, `MIO`, `Short Scoreboard`도 같이 줄어든다. 즉 `v2`의 이득은 memory path를 더 깨끗하게 만들었다기보다, 같은 정리된 경로를 더 잘 겹치게 만든 데 있다.

- Benchmark (4096): `6.64 ms / 20.70 TFLOPS`
- Profile duration: `7.58 ms`
- Compute / Memory / DRAM: `69.30% / 45.04% / 124.60 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.67% / 80.35% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `127`
- Achieved Occupancy: `32.24%`
- Eligible / Issued / No Eligible: `2.15 / 0.74 / 26.06%`
- Top stall: `Not Selected 1.91`, `Long Scoreboard 0.71`, `Short Scoreboard 0.45`, `Dispatch Stall 0.42`, `Barrier 0.35`, `Wait 0.14`

SASS를 보면 아직 한계도 분명하다. hot path는 `LDG.E.CONSTANT x16`, `STS x16`, `LDS.128 x32`, `IADD3 x64`다. shared consume은 이미 깔끔하지만, stage를 채우는 global ingress는 여전히 plain scalar라 CUTLASS의 `LDG.E.LTC128B` path보다 가늘다.

### `warptiling_v2_vec4_128x128x8_32x64x8`

`v2_vec4`는 `v2`의 clean 2-stage를 그대로 두고 ingress 폭만 넓힌 버전이다. `ThreadMapA/B`를 `kVecWidth=4`로 바꾸고, A stage도 vec4 transpose store iterator를 쓰게 해서 hot path를 wide-ingress family로 올린다.

```cpp
using ThreadMapA =
    utils::PitchLinearStripminedThreadMap<kThreadblockK, kThreadblockM,
                                          kThreads, 4>;
using ThreadMapB =
    utils::PitchLinearStripminedThreadMap<kThreadblockN, kThreadblockK,
                                          kThreads, 4>;
```

효과는 즉각적이다. 경로 정돈도는 그대로인데 `LDG.E.128.CONSTANT x4`, `STS.128 x2`, `LDS.128 x32`, `IADD3 x39`로 바뀌면서 `Long Scoreboard`가 `0.71 -> 0.00`, `Issued`가 `0.74 -> 0.81`, `No Eligible`이 `26.03% -> 19.38%`로 좋아진다. benchmark도 `20.70 -> 23.01 TFLOPS`로 크게 오른다.

- Benchmark (4096): `5.97 ms / 23.01 TFLOPS`
- Profile duration: `6.66 ms`
- Compute / Memory / DRAM: `75.66% / 49.45% / 150.85 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.58% / 79.76% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `127`
- Achieved Occupancy: `32.42%`
- Eligible / Issued / No Eligible: `2.62 / 0.81 / 19.39%`
- Top stall: `Not Selected 2.25`, `Dispatch Stall 0.52`, `Barrier 0.46`, `MIO Throttle 0.18`, `Short Scoreboard 0.15`, `Wait 0.12`

즉 `v2_vec4`의 본질은 새로운 pipeline이 아니라 “같은 clean pipeline을 더 넓은 ingress로 채운 것”이다. 이 시점부터 warptiling은 explicit CUTLASS와 같은 급에서 비교할 수 있는 수준으로 올라간다.

### `warptiling_v3_128x128x8_32x64x8`

이번 코드 기준의 `v3`는 더 깊은 ping-pong이 아니라 `v2 + PTX L2::128B global load`다. source에서도 `load_ptx()`를 직접 호출한다.

```cpp
threadblock_tile_iterator_a.load_ptx(frag_a);
threadblock_tile_iterator_b.load_ptx(frag_b);
...
if (gemm_k_iterations > 1) {
  threadblock_tile_iterator_a.load_ptx(frag_a);
  threadblock_tile_iterator_b.load_ptx(frag_b);
}
```

SASS 변화는 정확히 그 부분에만 집중된다. `v2`의 `LDG.E.CONSTANT x16`이 `v3`에서는 `LDG.E.LTC128B x16`으로 바뀌고, `LDS.128 x32`와 `IADD3 x64`는 그대로 유지된다. 그래서 이 버전은 load primitive 효과를 가장 깨끗하게 분리해서 보여 준다. 여기서 중요한 점은 `L2::128B`가 per-thread logical access 수를 줄이는 최적화는 아니라는 것이다. 각 thread는 여전히 scalar access를 하지만, plain dereference 대신 PTX helper를 통해 “이 load는 128B sector locality를 강하게 활용하는 경로”라는 힌트를 L2 쪽에 명시적으로 전달한다. 그 결과 contiguous SGEMM ingress에서는 codegen이 더 안정적으로 `LDG.E.LTC128B`로 고정되고, cache hit와 scheduler friendliness가 소폭 개선된다. 반면 이것만으로 `LDGSTS` 같은 direct global-to-shared copy가 생기거나 instruction 수 자체가 줄어드는 것은 아니다.

- Benchmark (4096): `6.56 ms / 20.96 TFLOPS`
- Profile duration: `7.45 ms`
- Compute / Memory / DRAM: `70.65% / 45.85% / 127.16 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.45% / 82.03% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `127`
- Achieved Occupancy: `32.23%`
- Eligible / Issued / No Eligible: `2.10 / 0.75 / 24.60%`
- Top stall: `Not Selected 1.79`, `Long Scoreboard 0.77`, `Short Scoreboard 0.54`, `Dispatch Stall 0.36`, `Barrier 0.33`, `Wait 0.09`

비교 기준은 같은 scalar family의 `v2`다. `v3`는 `Issued 0.74 -> 0.75`, `No Eligible 26.06% -> 24.60%`, `L2 Hit 80.35% -> 82.03%`, benchmark `20.70 -> 20.96 TFLOPS`로 소폭이지만 분명히 오른다. 즉 `L2::128B` global load는 실제로 효과가 있다. 이 최적화의 장점은 구조를 거의 건드리지 않고도 global ingress codegen을 더 L2-friendly하게 고정할 수 있다는 점이다. 다만 `Long Scoreboard`가 `0.71 -> 0.78`로 오히려 조금 올라가므로, load primitive 하나만으로 CUTLASS 수준의 scheduler 정돈도가 바로 만들어지지는 않는다. 결국 `v3`는 “codegen과 cache hint를 바로잡는 단계”이지, 아직 “copy path 전체를 재설계하는 단계”는 아니다.

### `warptiling_v3_vec4_128x128x8_32x64x8`

`v3_vec4`는 vec4 family가 `LTC128B` helper path로 들어간 첫 버전이다. 현재 소스에서는 이미 `load(frag, pred)` helper를 쓰고 있고, SASS도 `LDG.E.LTC128B x4`, `STS.128 x2`, `LDS.128 x32`, `IADD3 x37`로 내려간다. 즉 `v2_vec4`의 `LDG.E.128.CONSTANT` path를 CUTLASS식 cache-hinted load로 바꾼 셈이다. 여기서도 포인트는 똑같다. `v2_vec4 -> v3_vec4`의 이득은 “vec4가 새로 들어갔다”가 아니라, 이미 넓어진 vec4 ingress를 더 L2-friendly한 PTX load helper 위에 올렸다는 데 있다. 즉 이 단계는 access width를 늘리는 단계가 아니라, 넓어진 access를 더 일관된 cache hint와 codegen 위에서 굴리는 단계다.

- Benchmark (4096): `5.79 ms / 23.75 TFLOPS`
- Profile duration: `6.41 ms`
- Compute / Memory / DRAM: `79.27% / 51.37% / 152.93 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.64% / 81.73% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `127`
- Achieved Occupancy: `32.43%`
- Eligible / Issued / No Eligible: `2.68 / 0.84 / 15.54%`
- Top stall: `Not Selected 2.17`, `Dispatch Stall 0.45`, `Short Scoreboard 0.36`, `Barrier 0.36`, `Wait 0.09`, `MIO Throttle 0.06`

비교 기준은 이전 vec4 baseline인 `v2_vec4`다. 이 버전은 occupancy를 잃지 않으면서 `Issued 0.81 -> 0.84`, `No Eligible 19.39% -> 15.52%`, `L2 Hit 79.76% -> 81.73%`를 만들고 benchmark를 `23.01 -> 23.75 TFLOPS`까지 끌어올린다. 즉 vec4 family에서는 `L2::128B` helper가 단순한 부가 최적화가 아니라 꽤 큰 전환점이다. scalar family보다 vec4 family에서 효과가 더 크게 보이는 이유는, 이미 `LDG.E.128` 수준까지 넓어진 ingress가 `LTC128B` helper와 결합되면서 cache hit와 scheduler issue efficiency를 동시에 밀어 올리기 때문이다. 다만 여기서도 바뀌는 것은 ingress quality이지 epilogue path는 아니므로, CUTLASS와의 마지막 차이는 여전히 direct copy와 output store 쪽에 남아 있다.

### `warptiling_v4_128x128x8_32x64x8`

`v4`는 `v3 + predicate`다. source에서는 `load_ptx()` 대신 predicated `load(frag, gemm_k_iterations > 1)`를 쓰고, 마지막 tile에서도 one-past-end load를 막는다.

```cpp
threadblock_tile_iterator_a.load(frag_a, true);
threadblock_tile_iterator_b.load(frag_b, true);
...
threadblock_tile_iterator_a.load(frag_a, gemm_k_iterations > 1);
threadblock_tile_iterator_b.load(frag_b, gemm_k_iterations > 1);
```

SASS 레벨에서는 `LDG.E.LTC128B x16`과 `LDS.128 x32`는 유지되면서 `IADD3`가 `64 -> 61`로 조금 줄고, metric은 더 깔끔하게 좋아진다.

- Benchmark (4096): `6.34 ms / 21.66 TFLOPS`
- Profile duration: `7.15 ms`
- Compute / Memory / DRAM: `73.71% / 47.77% / 134.49 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.40% / 81.72% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `128`
- Achieved Occupancy: `32.30%`
- Eligible / Issued / No Eligible: `2.62 / 0.78 / 21.51%`
- Top stall: `Not Selected 2.33`, `Dispatch Stall 0.72`, `Barrier 0.41`, `Wait 0.18`, `MIO Throttle 0.11`, `Long Scoreboard 0.05`

`v3`와 비교하면 scalar family에서 predication의 의미가 꽤 크다. `Issued 0.75 -> 0.78`, `Eligible 2.10 -> 2.62`, `No Eligible 24.60% -> 21.51%`, benchmark `20.96 -> 21.66 TFLOPS`가 모두 같은 방향으로 움직인다. `Long Scoreboard`와 `Short Scoreboard`가 동시에 줄어드는 점도 중요하다. 즉 `v4`는 “tail-safe load가 correctness만 위한 장치가 아니라 steady-state를 더 규칙적으로 만든다”는 것을 보여 준다.

### `warptiling_v4_vec4_128x128x8_32x64x8`

현재 최고점과 거의 같은 급의 vec4 상위 버전이다. 다만 중요한 점이 하나 있다. 현재 소스에서는 `v3_vec4`와 `v4_vec4`가 모두 predicated `load(frag, pred)` helper를 타고 있고, static hot path도 사실상 같다. 둘 다 `LDG.E.LTC128B x4`, `STS.128 x2`, `LDS.128 x32`, `IADD3 x37`이다. 그래서 이 둘은 “새 알고리즘 vs 구 알고리즘”보다 “같은 vec4/predicated path의 안정화된 결과”로 읽는 편이 맞다.

- Benchmark (4096): `5.80 ms / 23.70 TFLOPS`
- Profile duration: `6.41 ms`
- Compute / Memory / DRAM: `79.25% / 51.36% / 154.92 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.57% / 81.72% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `127`
- Achieved Occupancy: `32.41%`
- Eligible / Issued / No Eligible: `2.68 / 0.84 / 15.55%`
- Top stall: `Not Selected 2.17`, `Dispatch Stall 0.46`, `Short Scoreboard 0.36`, `Barrier 0.36`, `Wait 0.09`, `MIO Throttle 0.06`

`v3_vec4`와 비교하면 차이는 아주 작고 사실상 같은 정체 구간에 있다. benchmark가 `23.75 -> 23.70 TFLOPS`로 거의 같고, metric도 사실상 같은 정체 구간에 머문다. 오히려 여기서 더 중요한 비교는 reference다. `v4_vec4`도 여전히 `torch_matmul`의 약 `99%` 수준까지 올라와 있고, explicit CUTLASS와 universal CUTLASS는 모두 넘는다. 즉 현재 study kernel 최고점은 “full CUTLASS-style pipeline”이 아니라 “적당히 얕은 pipeline + 정리된 vec4 ingress + predication”에서 나온다.

### `warptiling_v5_128x128x8_32x64x8`

`v5`는 `v4 + partial CUTLASS-style pipeline`이다. 여기서부터 mainloop 모양이 다시 바뀐다. `smem_tile_iterator_a/b`, `warp_tile_iterator_a/b`, `smem_write_stage_idx`를 커널 전체에 들고 가며, stage ring을 `add_stage_offset()`으로 되감는다. 다만 full ping-pong은 아직 아니고 warp fragment는 single buffer다.

```cpp
++smem_tile_iterator_a;
++smem_tile_iterator_b;
smem_write_stage_idx ^= 1;
...
if (smem_write_stage_idx == 1) {
  smem_tile_iterator_a.add_stage_offset(-Traits::kStages);
  smem_tile_iterator_b.add_stage_offset(-Traits::kStages);
} else {
  warp_tile_iterator_a.add_tile_offset(-Traits::kStages *
                                       Traits::kWarpGemmIterations);
  warp_tile_iterator_b.add_tile_offset(-Traits::kStages *
                                       Traits::kWarpGemmIterations);
}
```

SASS에서는 `LDG.E.LTC128B`와 `LDS.128 x32`는 유지되지만 control cost가 커진다. scalar는 `IADD3 61 -> 76`, vec4는 `37 -> 50`까지 올라간다. 즉 이 버전의 트레이드오프는 아주 명확하다. overlap은 늘지만 pointer/rewrite 비용도 같이 늘어난다.

- Benchmark (4096): `6.31 ms / 21.79 TFLOPS`
- Profile duration: `7.06 ms`
- Compute / Memory / DRAM: `75.38% / 48.36% / 146.73 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.53% / 81.48% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `125`
- Achieved Occupancy: `32.38%`
- Eligible / Issued / No Eligible: `2.57 / 0.80 / 19.71%`
- Top stall: `Not Selected 2.20`, `Dispatch Stall 0.47`, `Barrier 0.44`, `MIO Throttle 0.23`, `Wait 0.18`, `Short Scoreboard 0.14`

비교 기준은 직전 scalar 버전 `v4`다. scalar family에서는 partial CUTLASS-style pipeline이 효과가 있다. `Issued 0.78 -> 0.80`, `No Eligible 21.51% -> 19.71%`, benchmark `21.66 -> 21.79 TFLOPS`가 모두 좋아진다. 다만 개선 폭이 크지 않은 이유도 동시에 보인다. `IADD3`와 `MIO`가 늘어났기 때문이다. 즉 partial pipeline은 이득이 있지만, control overhead가 그 이득 일부를 상쇄한다.

### `warptiling_v5_vec4_128x128x8_32x64x8`

`v5_vec4`는 같은 partial CUTLASS-style pipeline을 vec4 family에 적용한 버전이다. SASS는 `LDG.E.LTC128B x4`, `STS.128 x2`, `LDS.128 x32`를 유지하지만 `IADD3 x50`으로 커지고, `Long Scoreboard 0.22`가 새로 눈에 띄게 올라온다.

- Benchmark (4096): `5.85 ms / 23.47 TFLOPS`
- Profile duration: `6.52 ms`
- Compute / Memory / DRAM: `78.51% / 50.52% / 147.43 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.32% / 81.84% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `127`
- Achieved Occupancy: `32.44%`
- Eligible / Issued / No Eligible: `2.56 / 0.84 / 16.31%`
- Top stall: `Not Selected 2.05`, `Barrier 0.38`, `Dispatch Stall 0.37`, `Short Scoreboard 0.27`, `Long Scoreboard 0.22`, `Wait 0.12`

비교 기준은 이전 vec4 best인 `v4_vec4`다. 여기서는 방향이 반대로 간다. `Issued`는 `0.84`로 같지만 `Eligible 2.68 -> 2.56`, `No Eligible 15.55% -> 16.31%`, benchmark `23.70 -> 23.57 TFLOPS`로 소폭 후퇴한다. scalar family와 달리 vec4 family에서는 이미 ingress가 충분히 넓기 때문에, 추가 pipeline depth가 주소 계산과 dependency 비용만 늘리고 실익은 작다는 뜻이다.

### `warptiling_v6_128x128x8_32x64x8`

`v6`는 `v4 + full CUTLASS-style pipeline`이다. 여기서는 partial stage iterator만 쓰는 것이 아니라, `warp_frag_a[2] / warp_frag_b[2]` ping-pong, prologue의 첫 warp-frag load, inner-loop next-frag prefetch까지 모두 넣는다. mainloop 모양이 CUTLASS의 `mma_pipelined`에 가장 가깝다.

```cpp
typename Traits::WarpTileIteratorA::Fragment warp_frag_a[2];
typename Traits::WarpTileIteratorB::Fragment warp_frag_b[2];
...
warp_tile_iterator_a.load(warp_frag_a[0]);
warp_tile_iterator_b.load(warp_frag_b[0]);
...
warp_tile_iterator_a.load(warp_frag_a[(warp_mma_k + 1) % 2]);
warp_tile_iterator_b.load(warp_frag_b[(warp_mma_k + 1) % 2]);
...
warp_mma<Traits>(accum, warp_frag_a[warp_mma_k % 2],
                 warp_frag_b[warp_mma_k % 2]);
```

그 대가와 이득도 SASS에 함께 보인다. `LDS.128`이 `32 -> 36`, `IADD3`가 `76 -> 80`으로 늘지만, scalar family에서는 이 더 깊은 overlap이 실제 성능 개선으로 이어진다.

- Benchmark (4096): `6.29 ms / 21.85 TFLOPS`
- Profile duration: `7.00 ms`
- Compute / Memory / DRAM: `76.56% / 48.72% / 204.57 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.61% / 75.16% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `120`
- Achieved Occupancy: `32.40%`
- Eligible / Issued / No Eligible: `2.63 / 0.82 / 18.49%`
- Top stall: `Not Selected 2.23`, `Dispatch Stall 0.49`, `Barrier 0.42`, `Wait 0.18`, `MIO Throttle 0.18`, `Short Scoreboard 0.15`

비교 기준은 직전 scalar 버전 `v5`다. `v6`는 scalar family에서 full CUTLASS-style pipeline이 여전히 유효함을 보여 준다. `Issued 0.80 -> 0.82`, `No Eligible 19.71% -> 18.49%`, benchmark `21.79 -> 21.85 TFLOPS`로 다시 아주 조금 오른다. 흥미로운 점은 `DRAM 146.73 -> 204.57 GB/s`, `L2 Hit 81.48% -> 75.16%`로 바뀐다는 것이다. 즉 full CUTLASS-style pipeline은 scalar family에서 reuse를 더 좋게 만들었다기보다, 더 공격적으로 데이터를 끌어오며 latency를 더 겹치게 만든 버전에 가깝다.

### `warptiling_v6_vec4_128x128x8_32x64x8`

vec4 family에서 full CUTLASS-style pipeline을 그대로 밀어 넣으면 결과가 달라진다. SASS는 `LDG.E.LTC128B x4`, `STS.128 x2`, `LDS.128 x36`, `IADD3 x54`로 scalar보다 훨씬 넓은 ingress를 유지하지만, registers가 `129`까지 올라가면서 `1 block/SM` cliff를 밟는다.

- Benchmark (4096): `6.02 ms / 22.83 TFLOPS`
- Profile duration: `6.87 ms`
- Compute / Memory / DRAM: `74.45% / 48.00% / 145.09 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `2.53% / 80.68% / 6.29 MB`
- Global Store Avg. Bytes / Sector: `8`
- Registers / thread: `129`
- Achieved Occupancy: `16.66%`
- Eligible / Issued / No Eligible: `1.40 / 0.79 / 20.60%`
- Top stall: `Not Selected 0.77`, `Dispatch Stall 0.26`, `Short Scoreboard 0.14`, `Barrier 0.13`, `Wait 0.10`, `MIO Throttle 0.04`

비교 기준은 `v5_vec4`다. 결과는 분명한 후퇴다. benchmark가 `23.57 -> 22.83 TFLOPS`로 떨어지고, `Eligible 2.56 -> 1.40`, `Issued 0.84 -> 0.79`, `No Eligible 16.31% -> 20.60%`, occupancy `32.44% -> 16.66%`로 내려간다. 즉 full CUTLASS-style pipeline은 vec4 family에서는 지나치게 깊다. 넓은 ingress 자체는 이미 충분한데, 추가된 `LDS.128`, control path, 그리고 register live range 증가가 residency를 무너뜨린다.

### `warptiling_v7_128x128x8_32x64x8`

`v7`은 mainloop를 `v6` 그대로 두고 epilogue만 바꾼 버전이다. direct global store를 버리고, CUTLASS SIMT처럼 `warp accumulator -> shared compact strip -> CTA-wide global store` 구조를 넣는다. 현재 구현에서는 `16 x (128 + 5)` logical strip를 shared의 앞부분에 재해석해 쓰고, `AccumulatorFragmentIterator`, `WarpEpilogueStoreIterator`, `EpilogueSharedLoadIterator`, `EpilogueOutputStoreIterator`가 `8`번의 epilogue iteration을 나눠 처리한다. 핵심은 CTA output 전체를 `128x128` shared tile로 staging하지 않고, `16x128` compact strip를 반복 재사용하면서 global store만 CUTLASS-style thread map으로 정리한다는 점이다.

```cpp
#pragma unroll
for (int iter = 0; iter < Traits::kThreadTileM; ++iter) {
  accum_iter.load(accum_frag);
  ++accum_iter;

  warp_store_iter.store(accum_frag);
  __syncthreads();

  shared_load_iter.load(output_frag);
  out_iter.store(output_frag);
  __syncthreads();
  ++out_iter;
}
```

metric은 `v6`와 아주 선명하게 대비된다. `v6`는 mainloop 자체는 더 정교해졌지만 epilogue가 여전히 direct store라 `Global Store Avg. Bytes / Sector 8`, `L2 Excessive 6.29 MB`, `L2 Hit For Loads 74.15%`를 남겼다. `v7`은 benchmark를 `21.85 -> 22.01 TFLOPS`로 올리면서 `Global Store Avg. Bytes / Sector 32`, `L2 Excessive 0`, `L2 Hit For Loads 77.22%`까지 끌어올린다. 즉 `v7`의 성능 향상은 arithmetic이나 ingress 폭이 아니라 output path 정리에서 온다. scheduler 숫자도 `Issued 0.82 -> 0.82`, `No Eligible 18.49% -> 17.83%`로 조금 더 좋아진다.

- Benchmark (4096): `6.24 ms / 22.01 TFLOPS`
- Profile duration: `6.94 ms`
- Compute / Memory / DRAM: `77.22% / 49.37% / 176.50 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `0.08% / 77.57% / 0.00`
- Global Load / Store Avg. Bytes / Sector: `32 / 32`
- Registers / thread: `120`
- Achieved Occupancy: `32.39%`
- Eligible / Issued / No Eligible: `2.65 / 0.82 / 17.83%`
- Top stall: `Not Selected 2.23`, `Dispatch Stall 0.49`, `Barrier 0.42`, `Wait 0.19`, `MIO Throttle 0.19`, `Short Scoreboard 0.15`
- Shared / global warning: `L1 Conflicts Shared N-Way 216`, `L1 Wavefronts Shared Excessive 0`, `L2 Theoretical Sectors Global Excessive 0`

instruction 수준에서 보면 `v7`의 핵심은 “mainloop를 더 복잡하게 만드는 것”이 아니라 “epilogue shared round-trip을 감수하고 store 품질을 끝까지 끌어올리는 것”이다. SASS 기준으로 mainloop 쪽의 `LDG.E.LTC128B x16`, `LDS.128 x36`, `FFMA x512`는 `v6`와 같고, 차이는 epilogue에서의 extra barrier와 shared traffic이다. `BAR.SYNC`는 `2 -> 18`로 늘지만, 그 대신 global store path는 CUTLASS와 같은 `32B/sector`, `L2 Excessive 0`에 도달한다. 즉 `v7`은 이제 scalar warptiling에서 “epilogue 때문에 느린 버전”이 아니라, epilogue를 정리해 scalar family의 기준점을 한 단계 올린 버전이다.

CUTLASS와 비교하면 흥미로운 지점도 분명하다. `v7`과 explicit CUTLASS는 이제 store path가 거의 같다. 둘 다 `Global Store Avg. Bytes / Sector 32`, `L2 Excessive 0`, `shared conflicts 216`, `dynamic shared 16.64 KB`다. 그런데 benchmark는 `22.01` 대 `22.64`로 아직 뒤지고, scheduler 지표도 `Issued 0.82 vs 0.88`, `No Eligible 17.83% vs 12.02%`, `Warp Cycles / Issued 4.73 vs 4.42`로 더 거칠다. load-side 차이도 크다. `L2 Hit For Loads 77.22% vs 93.11%`, `DRAM 176.50 vs 155.13 GB/s`를 보면, `v7`은 epilogue는 맞췄지만 A/B ingress reuse와 mainloop issue density는 아직 CUTLASS만큼 정제되지 않았다는 뜻이다. 즉 `v7`의 성과는 “output path를 CUTLASS와 같은 급으로 끌어올렸다”는 데 있고, 남은 차이는 더 이상 epilogue가 아니라 global iterator와 mainloop steady-state 쪽에 있다.

### `warptiling_v7_vec4_128x128x8_32x64x8`

`v7_vec4`는 `v7`의 CUTLASS-style epilogue reorder를 vec4 family에 결합한 버전이다. mainloop ingress는 `v3_vec4`/`v4_vec4` 계열의 wide path를 유지하면서 output path만 `shared compact strip -> coalesced global store`로 교체한다. 즉 vec4 family에서도 `Global Load / Store Avg. Bytes / Sector = 32 / 32`, `L2 Excessive 0`를 달성하면서, scalar `v7`보다 훨씬 넓은 ingress가 주는 scheduler 이점을 거의 그대로 들고 간다.

- Benchmark (4096): `5.78 ms / 23.76 TFLOPS`
- Profile duration: `6.42 ms`
- Compute / Memory / DRAM: `79.90% / 51.46% / 150.49 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `0.10% / 81.40% / 0.00`
- Global Load / Store Avg. Bytes / Sector: `32 / 32`
- Registers / thread: `127`
- Achieved Occupancy: `32.46%`
- Eligible / Issued / No Eligible: `2.75 / 0.85 / 14.81%`
- Top stall: `Not Selected 2.23`, `Barrier 0.49`, `Dispatch Stall 0.45`, `Short Scoreboard 0.16`, `Wait 0.12`, `MIO Throttle 0.05`
- Shared / global warning: `L1 Conflicts Shared N-Way 408`, `L1 Wavefronts Shared Excessive 1,572,864 (1%)`, `L2 Theoretical Sectors Global Excessive 0`

직전 vec4 버전 `v6_vec4`와 비교하면 회복 폭이 아주 크다. occupancy가 `16.66% -> 32.46%`, `Eligible 1.40 -> 2.75`, `Issued 0.79 -> 0.85`, `No Eligible 20.60% -> 14.79%`로 바뀌면서 benchmark가 `22.83 -> 23.76 TFLOPS`로 돌아온다. 즉 vec4 family에서 진짜 문제는 epilogue 자체보다 `v6_vec4`가 만든 `1 block/SM` cliff였고, `v7_vec4`는 그 cliff를 피하면서 output path까지 CUTLASS식으로 정리한 버전이다. 이번 snapshot에서는 `v7_vec4`가 vec4 family 최고점이다. 다만 `v3_vec4 23.75 TFLOPS`, `v8_vec4 23.70 TFLOPS`, `v4_vec4 23.70 TFLOPS`와 차이가 매우 작으므로, 이 역시 새로운 구조적 전환이라기보다 plateau 안의 미세 우위로 읽는 편이 맞다.

### `warptiling_v8_128x128x8_32x64x8`

`v8`은 `v7`의 epilogue를 유지한 채 mainloop를 explicit `prologue -> steady-state -> drain` 형태로 다시 쪼갠 버전이다. 핵심은 새로운 primitive를 넣는 것이 아니라, stage-local iterator를 매 outer iteration에서 다시 구성하고 `has_next` predicate로 다음 tile load를 더 직접적으로 관리하는 데 있다. 즉 `v8`은 cp.async나 direct copy가 아니라, 같은 scalar wide-load family 안에서 mainloop choreography를 더 단순하고 명시적으로 정리한 버전이다.

- Benchmark (4096): `6.19 ms / 22.21 TFLOPS`
- Profile duration: `6.92 ms`
- Compute / Memory / DRAM: `77.63% / 49.54% / 138.08 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `0.11% / 81.39% / 0.00`
- Global Load / Store Avg. Bytes / Sector: `32 / 32`
- Registers / thread: `127`
- Achieved Occupancy: `32.38%`
- Eligible / Issued / No Eligible: `2.59 / 0.83 / 17.24%`
- Top stall: `Not Selected 2.13`, `Dispatch Stall 0.47`, `Barrier 0.35`, `Short Scoreboard 0.30`, `Wait 0.19`, `MIO Throttle 0.13`
- Shared / global warning: `L1 Conflicts Shared N-Way 208`, `L1 Wavefronts Shared Excessive 0`, `L2 Theoretical Sectors Global Excessive 0`

비교 기준은 직전 scalar best였던 `v7`이다. `v8`은 benchmark를 `22.01 -> 22.21 TFLOPS`로 더 끌어올리면서 `L2 Hit For Loads 77.22% -> 81.09%`, `DRAM 176.50 -> 138.08 GB/s`, `Issued 0.82 -> 0.83`, `No Eligible 17.83% -> 17.24%`를 만든다. 즉 같은 `32B/sector` epilogue를 유지한 채 load-side reuse와 scheduler 상태가 조금 더 좋아진 셈이다. 부작용도 있다. registers가 `120 -> 127`로 늘고 `Short Scoreboard 0.15 -> 0.30`으로 올라가므로 dependency 거리는 더 짧아진다. 그래도 `Barrier 0.42 -> 0.35`와 더 높은 L2 hit이 그 비용을 상쇄해, `v8`은 scalar family 최고점을 한 번 더 갱신한다. 다만 후속 `v9`가 이 값을 다시 소폭 넘고, `v10`과 `v11`은 서로 다른 방식으로 한 단계 더 오른다.

### `warptiling_v8_vec4_128x128x8_32x64x8`

`v8_vec4`는 같은 choreography 정리를 vec4 family에 적용한 버전이다. output path는 이미 `v7_vec4`에서 `32B/sector`까지 정리되어 있고, ingress도 `LDG.E.LTC128B x4` + `STS.128` + `LDS.128` 조합으로 충분히 넓다. 그래서 이 버전의 의미는 새로운 최고점을 만들었다기보다, vec4 family에서 mainloop 정리의 순효과가 얼마나 작은지를 확인하는 데 있다.

- Benchmark (4096): `5.80 ms / 23.70 TFLOPS`
- Profile duration: `6.42 ms`
- Compute / Memory / DRAM: `79.82% / 51.41% / 150.59 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `0.06% / 81.31% / 0.00`
- Global Load / Store Avg. Bytes / Sector: `32 / 32`
- Registers / thread: `128`
- Achieved Occupancy: `32.35%`
- Eligible / Issued / No Eligible: `2.68 / 0.85 / 14.87%`
- Top stall: `Not Selected 2.15`, `Dispatch Stall 0.49`, `Barrier 0.47`, `Short Scoreboard 0.19`, `Wait 0.08`, `MIO Throttle 0.04`
- Shared / global warning: `L1 Conflicts Shared N-Way 400`, `L1 Wavefronts Shared Excessive 1,572,864 (1%)`, `L2 Theoretical Sectors Global Excessive 0`

비교 기준은 `v7_vec4`다. 결과는 소폭 후퇴지만 여전히 정체 구간 안쪽이다. benchmark는 `23.76 -> 23.70 TFLOPS`로 약간 내려가고, `Issued`도 `0.85`로 그대로다. 다만 `Eligible 2.75 -> 2.68`, `No Eligible 14.79% -> 14.87%`, registers `127 -> 128`을 보면 vec4 family에서는 choreography 단순화의 이득이 control/predicate 비용과 거의 상쇄된다고 보는 편이 맞다. 더 넓게 보면 `v8_vec4`도 여전히 `v7_vec4`, `v3_vec4`, `v4_vec4`와 같은 plateau 안에 있으므로, vec4 family의 남은 과제는 epilogue나 단순 stage 정리가 아니라 `torch_matmul`/CUTLASS가 보여 주는 더 좋은 direct copy와 arithmetic density 쪽에 있다.

### `warptiling_v9_128x128x8_32x64x8`

`v9`는 `v8` 위에 새 primitive를 더한 버전이 아니다. 핵심 변화는 next-tile global load를 inner loop의 `warp_mma_k == 0` 분기 안에서 수행하던 방식을 버리고, outer loop 시작 직후로 끌어올려 load 시점만 재배치한 것이다.

```cpp
bool const has_next = gemm_k_iterations > 1;

threadblock_tile_iterator_a.load(tb_frag_a, has_next);
threadblock_tile_iterator_b.load(tb_frag_b, has_next);

++threadblock_tile_iterator_a;
++threadblock_tile_iterator_b;

auto warp_tile_iterator_a = make_warp_tile_iterator_a(read_stage);
auto warp_tile_iterator_b = make_warp_tile_iterator_b(read_stage);
```

`v8`과 비교하면 바뀌는 것은 dependency choreography다. output path는 그대로 `Global Load / Store Avg. Bytes / Sector = 32 / 32`, `L2 Excessive 0`, `shared excessive 0`을 유지하고, benchmark는 `22.21 -> 22.27 TFLOPS`로 scalar family 최고점을 다시 아주 조금 갱신한다. scheduler 지표는 `Eligible 2.59 -> 2.69`로 좋아지지만 `Issued 0.83 -> 0.82`, `No Eligible 17.24% -> 17.70%`로 완전히 깔끔해지지는 않는다. 대신 `Short Scoreboard 0.30 -> 0.09`가 크게 줄고 `Long Scoreboard 0.02 -> 0.07`이 조금 늘어, 즉시 의존성은 완화됐지만 더 긴 load-to-use 꼬리는 약간 남았다고 읽는 편이 자연스럽다.

instruction level에서 explicit CUTLASS와의 차이도 여기서 비교적 명확해진다. `v9`와 `cutlass_simt_128x128x8_32x64x8_2stage`는 둘 다 main ingress primitive가 `LDG.E.LTC128B`이고, warp operand fetch도 `LDS.128`, output도 scalar `STG.E` 계열이라 “우리 커널이 낮은 primitive를 써서 느리다”는 설명은 맞지 않는다. 실제 차이는 same `LDG -> STS -> LDS.128 -> FFMA` family를 stage boundary 근처에서 얼마나 짧게 압축해 놓느냐다. `v9`는 next-tile `LDG`를 outer loop 머리로 당겨 `Short Scoreboard`를 줄이는 대신, 그 payload와 주소 상태를 더 오래 들고 가야 한다. 반면 explicit CUTLASS는 같은 2-stage register-staged 구조 안에서도 `LDS.128 -> FFMA -> STS -> FFMA -> LDS.128`을 더 촘촘하게 braid해 live range와 hand-off 거리를 줄인다.

이 차이를 stage boundary만 떼어 놓고 보면 더 직관적이다.

```text
CUTLASS
next-tile ingress :        [LDG(next) already in flight]
current stage     : LDS(cur) -> FFMA(cur) -> STS(next) -> FFMA(cur) -> LDS(next-head) -> FFMA(next)
핵심              : producer(LDG/STS)와 consumer(LDS/FFMA)가 boundary 근처에서 짧게 맞물린다.

v9
next-tile ingress : LDG(next) ------------------------------------------------------------->
current stage     :            LDS(cur) -> FFMA(cur) -> FFMA(cur) -> STS(next) -> BAR -> LDS(next) -> FFMA(next)
핵심              : short-scoreboard는 줄지만, next tile payload와 주소 상태를 더 오래 들고 가므로
                    CUTLASS보다 L2 residency와 issue density가 불리해진다.
```

즉 “얼마나 빨리 읽었느냐”만이 아니라, 읽어 온 tile을 `STS/BAR/LDS` hand-off 바로 근처에서 얼마나 짧게 이어 붙였느냐가 남은 차이를 만든다.

여기서 말하는 CUTLASS-like의 정확한 뜻도 분명히 해 둘 필요가 있다. 핵심은 `LDG`, `STS`, `LDS.128`, `FFMA`가 다 보이느냐가 아니라, 마지막 slot에서 `STS(next) -> stage hand-off -> LDS(next-head)`를 만든 뒤 그 `next-head` operand가 다음 outer iteration 첫 `FFMA`로 바로 carry되느냐다. `v9`도 같은 instruction family와 ping-pong 구조를 갖고 있지만, outer iteration마다 `warp_frag[0]` bootstrap을 다시 열기 때문에 CUTLASS처럼 across-iteration seam이 닫히지는 않는다. 그래서 `v9`는 CUTLASS actor를 공유하는 kernel이지만, 아직 CUTLASS seam pattern 자체를 재현한 것은 아니다.

- Benchmark (4096): `6.17 ms / 22.27 TFLOPS`
- Profile duration: `6.88 ms`
- Compute / Memory / DRAM: `77.21% / 49.85% / 141.24 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `0.07% / 81.37% / 0.00`
- Global Load / Store Avg. Bytes / Sector: `32 / 32`
- Registers / thread: `127`
- Achieved Occupancy: `32.36%`
- Eligible / Issued / No Eligible: `2.69 / 0.82 / 17.70%`
- Top stall: `Not Selected 2.26`, `Dispatch Stall 0.47`, `Barrier 0.43`, `Wait 0.17`, `MIO Throttle 0.14`, `Short Scoreboard 0.09`, `Long Scoreboard 0.07`
- Shared / global warning: `L1 Conflicts Shared N-Way 208`, `L1 Wavefronts Shared Excessive 0`, `L2 Theoretical Sectors Global Excessive 0`

이 해석은 차분하게 볼 필요가 있다. `v9`는 `v7/v8`이 만들어 둔 정돈된 epilogue와 `32B/sector` store path를 유지한 상태에서, global load의 “언제”를 조정해 scalar path를 조금 더 다듬은 미세 조정이다. 그래서 `cutlass_universal`은 `22.28 TFLOPS`를 근소하게 넘지만 explicit CUTLASS `22.64 TFLOPS`와 `torch_matmul 23.87 TFLOPS`에는 아직 못 미친다. 특히 explicit CUTLASS와 비교하면 남은 차이는 epilogue가 아니라 mainloop steady-state에 있다. CUTLASS는 `Issued 0.88`, `No Eligible 12.02%`, `L2 Hit 93.11%`, `Dispatch Stall 0.23`, `Wait 0.07`인데 `v9`는 `0.82`, `17.70%`, `81.37%`, `0.47`, `0.17`에 머문다. 즉 `v9`는 short dependency 하나는 잘 눌렀지만, CUTLASS처럼 stage boundary를 짧게 압축하지는 못해서 ready warp density와 L2 residency가 여전히 부족하다. 다시 말해 `v9`의 성과는 “scalar family도 output path 정리 이후에는 timing-only tweak로 조금 더 올라간다”는 것을 보여 준 데 있고, 남은 gap의 본질은 여전히 ingress residency와 issue density 쪽이다.

### `warptiling_v10_128x128x8_32x64x8`

`v10`은 source-level algorithm 변경이 아니라, `v9`와 완전히 같은 로직을 별도 파일로 복사한 뒤 kernel symbol 이름만 `cutlass_sgemm_warptiling`으로 바꾼 버전이다. 실제 diff도 본질적으로 이 정도다.

```diff
- namespace v9 {
- __global__ void sgemm_warptiling(...)
+ namespace v10 {
+ __global__ void cutlass_sgemm_warptiling(...)
```

즉 이 버전은 “더 좋은 pipeline을 새로 구현했다”기보다, backend가 symbol name을 보고 어떤 codegen heuristic을 적용하는지 확인하는 실험용 커널이다. 중요한 점은 source가 아니라 SASS에서 변화가 나타난다는 것이다.

- Benchmark (4096): `6.04 ms / 22.74 TFLOPS`
- Profile duration: `6.64 ms`
- Compute / Memory / DRAM: `78.82% / 51.67% / 152.66 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `0.08% / 81.38% / 0.00`
- Global Load / Store Avg. Bytes / Sector: `32 / 32`
- Registers / thread: `126`
- Achieved Occupancy: `32.44%`
- Eligible / Issued / No Eligible: `2.53 / 0.84 / 15.93%`
- Top stall: `Not Selected 2.01`, `Barrier 0.90`, `Dispatch Stall 0.43`, `MIO Throttle 0.11`, `Wait 0.07`, `Short Scoreboard 0.02`, `Long Scoreboard 0.01`
- Shared / global warning: `L1 Conflicts Shared N-Way 208`, `L1 Wavefronts Shared Excessive 0`, `L2 Theoretical Sectors Global Excessive 0`

핵심은 “무엇이 바뀌지 않았는가”부터 보는 것이다. `v9`와 `v10`은 둘 다 core loop family가 동일하다. `LDG.E.LTC128B x16`, `LDS.128 x32`, `FFMA x512`, `STS x80`, `STG.E x64`, `BAR.SYNC x18`이 그대로라서 algorithm, mainloop class, epilogue class가 바뀐 것은 아니다. cache hint도 동일하게 `LDG.E.LTC128B`이고, `LDGSTS` 같은 direct global-to-shared copy가 새로 생기지도 않는다. 즉 이 개선을 “이제 CUTLASS-like pipeline을 구현했다”로 읽으면 틀린다.

바뀌는 것은 보조 명령열이다. SASS 파일 길이는 `1109 -> 1077` lines로 줄고, scalar 주소/상수 준비 구간이 더 공격적으로 재선택된다. 대표적으로 `IADD3 58 -> 38`, `IADD3.X 18 -> 11`, `LEA 46 -> 38`, `MOV 43 -> 12`로 줄고, 반대로 `IMAD 23 -> 62`, `ULDC 3 -> 5`, `ULDC.64 1 -> 3`, `CS2R 48 -> 62`가 늘어난다. 더 세부적으로는 `v9`에 없던 `IMAD.MOV.U32 x7`, `IMAD.SHL.U32 x6`, `IMAD.IADD x11`이 `v10`에서 새로 나타난다. 즉 backend가 constant materialization, 주소 생성, 간단한 증분 계산을 더 IMAD-centric하게 다시 짜면서 핫 루프에 들어가기 전 scalar scaffolding을 짧게 압축한 셈이다.

구간별로 보면 차이는 더 분명하다. `prologue / stage-entry`에서는 `v9`가 `MOV -> IADD3 -> LEA` 계열로 풀어 쓰던 주소 준비와 상수 materialization 일부를 `v10`이 `ULDC + IMAD` 계열로 더 짧게 접는다. 그래서 첫 `LDG.E.LTC128B` 묶음에 도달하기 전 준비 구간이 더 압축된다. 반면 `steady-state mainloop`는 사실상 그대로다. `LDG.E.LTC128B x16`, `LDS.128 x32`, `FFMA x512`, `STS x80`, `BAR.SYNC x18`이 같다는 것은, operand fetch 폭과 MMA body, 그리고 2-stage 골격 자체는 바뀌지 않았다는 뜻이다. `epilogue`도 동일하다. 둘 다 `shared compact strip -> scalar LDS -> STG.E` flush path를 유지하고 `Global Load / Store Avg. Bytes / Sector = 32 / 32`를 그대로 만든다. 즉 이름 효과가 건드린 것은 핫 루프의 알고리즘이 아니라, 핫 루프 전후를 둘러싼 address-gen / 보조 명령열의 압축도다.

이 변화는 NCU 지표에도 그대로 찍힌다. benchmark는 `22.27 -> 22.74 TFLOPS`, benchmark runtime은 `6.17 -> 6.04 ms`, profile duration은 `6.88 -> 6.64 ms`, `Issued`는 `0.82 -> 0.84`, `No Eligible`은 `17.70% -> 15.93%`, `Wait`는 `0.17 -> 0.07`, `Short/Long Scoreboard`는 `0.09/0.07 -> 0.02/0.01`, registers per thread는 `127 -> 126`으로 좋아진다. 반면 `L2 Hit Rate For Loads`는 `81.08% -> 81.09%`로 사실상 그대로다. 즉 이번 개선은 cache residency가 좋아진 것이 아니라, compiler가 보조 명령열을 더 압축하면서 local dependency chain과 issue conversion을 다듬은 결과로 읽는 편이 맞다. `Barrier`가 `0.43 -> 0.90`으로 오히려 커진 것도 같은 해석을 지지한다. barrier를 없앤 것이 아니라, barrier 바깥의 wait/scoreboard 잡음을 더 많이 깎아 내서 barrier가 더 또렷하게 드러난 것이다.

이름 실험을 CUTLASS와 직접 비교하면 한계도 분명하다. `v10`은 이미 benchmark에서 explicit CUTLASS `22.64 TFLOPS`를 소폭 넘고 `Issued 0.84`, `No Eligible 15.93%`까지 개선됐지만, CUTLASS의 핵심 강점은 그대로 남아 있다. `L2 Hit 81.38% vs 93.11%`, `Dispatch Stall 0.43 vs 0.23`, `Issued 0.84 vs 0.88`이 대표적이다. 즉 `cutlass` 이름이 backend-local scheduling과 보조 명령열 압축은 개선해 주지만, 우리가 문서 전체에서 강조해 온 stage-boundary seam, L2 residency, 직접적인 global-to-shared 복사 경로의 부재 같은 구조적 차이를 대신 해결해 주지는 않는다. 이 실험의 의미는 “이름만 바꿔도 빨라진다” 자체보다, 그 정도로 backend codegen choice가 현재 scalar kernel의 마지막 gap에 큰 영향을 주고 있다는 것을 보여 준 데 있다.

### `warptiling_v11_128x128x8_32x64x8`

`v11`은 `v9`의 scalar mainloop를 그대로 두고 threadblock swizzling만 얹은 버전이다. 중요한 점은 이 버전이 `v10`의 연장선이 아니라는 것이다. 실제 실험에서는 swizzled kernel에 `cutlass` 이름을 붙였을 때 오히려 더 느렸고, plain symbol로 되돌린 뒤 성능이 다시 좋아졌다. 따라서 `v11`은 “`v10` + swizzle”로 읽기보다, “`v9` + swizzle, 그리고 symbol 이름 효과는 제거”로 읽는 편이 정확하다.

코드 diff의 본질은 CTA 좌표를 직접 쓰지 않고 swizzle helper를 통해 다시 매핑한다는 데 있다.

```cpp
int cta_m_idx = 0;
int cta_n_idx = 0;
utils::GroupedThreadblockSwizzle<8>::get_tile_offset(cta_m_idx, cta_n_idx);

int const cta_row = cta_m_idx * Traits::kThreadblockM;
int const cta_col = cta_n_idx * Traits::kThreadblockN;
```

즉 CTA tile shape, warp tile shape, shared-memory layout, mainloop, epilogue는 그대로 두고 “어떤 CTA가 어떤 tile을 언제 맡느냐”만 바꾼다. 이 점이 중요하다. `v11`의 개선은 instruction 종류가 달라져서가 아니라, launch order가 달라지면서 L2가 보는 reuse pattern이 바뀐 데서 나온다.

- Benchmark (4096): `5.99 ms / 22.94 TFLOPS`
- Profile duration: `6.74 ms`
- Compute / Memory / DRAM: `78.84% / 50.89% / 95.49 GB/s`
- L1/TEX Hit / L2 Hit / L2 Excessive: `0.03% / 91.07% / 0.00`
- L2 Hit Rate For Loads / Stores: `90.93% / 99.93%`
- Global Load / Store Avg. Bytes / Sector: `32 / 32`
- Registers / thread: `127`
- Achieved Occupancy: `32.34%`
- Eligible / Issued / No Eligible: `2.86 / 0.84 / 15.94%`
- Top stall: `Not Selected 2.41`, `Dispatch Stall 0.51`, `Barrier 0.28`, `Wait 0.15`, `MIO Throttle 0.07`, `Short Scoreboard 0.07`, `Long Scoreboard 0.06`
- Shared / global warning: `L1 Conflicts Shared N-Way 208`, `L1 Wavefronts Shared Excessive 0`, `L2 Theoretical Sectors Global Excessive 0`

여기서 핵심은 swizzling이 정확히 무엇을 바꾸는지다. 기본 launch order에서는 linear CTA 순서가 대략 `(m0,n0) -> (m0,n1) -> (m0,n2) -> ...`처럼 진행된다. 즉 같은 `m` row tile이 연속으로 다시 등장하므로 inter-CTA 관점에서는 `A` tile 재사용에 더 유리하다. 반면 `GroupedThreadblockSwizzle<8>`은 한 그룹 안에서 CTA를 `(m0,n0) -> (m1,n0) -> ... -> (m7,n0) -> (m0,n1) -> ...` 순으로 재배치한다. 지금 `4096 / 128 = 32`이므로 grid는 `32 x 32` tile이고, `GroupRows=8`이면 `8 x 32 = 256`개의 CTA가 하나의 swizzle group을 이룬다. 이 그룹 안에서는 같은 `n` column tile이 연속으로 유지되므로 inter-CTA 관점에서는 `B` tile 재사용에 더 유리해진다.

이 kernel family에서는 그 교환이 실제로 이득으로 측정된다. `v9`와 비교하면 benchmark가 `22.27 -> 22.94 TFLOPS`, runtime이 `6.17 -> 5.99 ms`로 좋아지고, profile에서는 `L2 Hit 81.37% -> 91.07%`, `L2 Hit For Loads 81.08% -> 90.93%`, `DRAM Bandwidth 141.24 -> 95.49 GB/s`, `Issued 0.82 -> 0.84`, `No Eligible 17.70% -> 15.94%`로 개선된다. `Barrier`도 `0.43 -> 0.28`로 줄어든다. 즉 `v11`은 동일한 mainloop 골격을 유지하면서, CTA 간 tile reuse를 더 L2-friendly하게 만들어 off-chip pressure를 크게 낮춘 버전이라고 볼 수 있다.

다만 `v10`과는 해석이 완전히 다르다. `v10`은 `L2 Hit`가 거의 그대로인 대신 `Wait 0.07`, `Short/Long Scoreboard 0.02/0.01`까지 내려가면서 seam-local dependency를 더 짧게 만든 버전이다. 반면 `v11`은 `L2 Hit 91.07%`와 `DRAM 95.49 GB/s`가 말해 주듯 cache residency는 크게 좋아졌지만, `Wait 0.15`, `Short/Long Scoreboard 0.07/0.06`, `Dispatch Stall 0.51`은 `v10`보다 불리하다. 즉 `v11`은 stage boundary 자체를 CUTLASS처럼 더 짧게 압축한 버전이 아니다. 같은 CTA 내부의 `LDG -> STS -> LDS -> FFMA` seam은 `v10` 쪽이 더 촘촘하고, `v11`은 그 약점을 inter-CTA cache reuse로 상쇄해 이긴다.

이 차이를 더 쉽게 말하면 이렇다. `v10`이 “같은 CTA 내부에서 다음 stage operand를 더 빨리 ready 상태로 만든 버전”이라면, `v11`은 “다음 CTA들이 필요로 하는 tile을 L2에 더 오래 남겨 두는 버전”이다. 따라서 `v11`을 보고 곧바로 “이제 CUTLASS seam pattern을 재현했다”라고 해석하면 틀린다. `v11`의 성과는 mainloop primitive 변경이 아니라 CTA 순서 재배치, 즉 launch-order/cache-residency 최적화다.

#### Threadblock Swizzling을 조금 더 자세히 보면

threadblock swizzling은 각 CTA가 계산하는 수학 자체를 바꾸지 않는다. 각 CTA는 여전히 같은 `128x128x8` tile을 계산하고, warp도 같은 `32x64x8` 분할을 유지한다. 바뀌는 것은 오직 “grid 상의 어떤 CTA index가 어떤 `(tile_m, tile_n)` 쌍을 받느냐”이다. 그래서 swizzling 최적화는 보통 다음 네 가지 질문으로 나눠 읽는 편이 좋다.

1. 무엇이 그대로인가?
`mainloop`, `warp_mma`, shared-memory stage 수, epilogue, register budget은 거의 그대로다. 즉 single-CTA 내부의 instruction class는 바뀌지 않는다.

2. 무엇이 바뀌는가?
CTA 방문 순서가 바뀐다. 기본 순서는 `n`이 빠르게 변하고, swizzle 이후에는 한 그룹 안에서 `m`이 빠르게 변한다.

3. 그 결과 어떤 operand가 더 재사용되는가?
기본 순서는 같은 `A` tile이 연속 CTA에서 더 쉽게 다시 보이고, grouped swizzle은 같은 `B` tile이 연속 CTA에서 더 쉽게 다시 보이게 만든다.

4. 왜 이번 kernel에서는 그게 이득인가?
실측상 `B` 쪽 reuse를 더 살리는 쪽이 L2 residency를 더 크게 올렸기 때문이다. 이것이 `L2 Hit 91.07%`와 `DRAM 95.49 GB/s`로 바로 찍힌다.

즉 threadblock swizzling은 “메모리 명령을 더 넓게 바꾼다”거나 “FFMA body를 더 줄인다”는 종류의 최적화가 아니다. 이미 존재하던 load/store/mma 파이프라인 위에, 그 파이프라인이 받아먹는 tile의 시간적 배치를 바꾸는 최적화다. `v11`이 빨라진 이유를 문장 하나로 줄이면, “같은 코드 골격을 유지한 채 CTA 간 `B` tile의 L2 체류 시간을 늘려서 DRAM 왕복을 줄였기 때문”이라고 정리할 수 있다.

### Warptiling 정리

- `v0`와 `v0 BK16`은 warp hierarchy만으로는 충분하지 않다는 것을 보여 준다. arithmetic 구조는 좋아졌지만 memory path가 blocktiling 말기의 문제를 그대로 끌고 온다.
- `v1`은 benchmark보다 경로 정리가 핵심인 버전이다. `shared excessive 0`, `Global Store Avg. Bytes / Sector 8`, `L2 Excessive 6.29 MB`가 이후 모든 버전의 출발점이 된다.
- `v2`는 정리된 경로 위에 2-stage를 올려 scalar family를 한 단계 끌어올린다.
- `v2_vec4`는 같은 2-stage를 넓은 ingress로 채워 첫 reference-tier 점프를 만든다.
- `v3`와 `v3_vec4`는 `L2::128B` load helper가 실제로 큰 의미가 있음을 보여 준다. scalar는 `20.70 -> 20.96`, vec4는 `23.01 -> 23.75 TFLOPS`로 오른다.
- `v4`는 predication이 scalar family에서 꽤 큰 안정화 효과를 낸다는 것을 보여 준다. vec4 쪽은 이미 helper path가 정리돼 있어 `v3_vec4 -> v4_vec4` 차이가 작다.
- `v5`는 partial CUTLASS-style pipeline이 scalar family에는 유리하지만 vec4 family에는 이미 애매해지는 지점을 보여 준다.
- `v6`는 full CUTLASS-style pipeline이 scalar family를 더 밀어 올리지만, vec4 family에서는 `129 regs`와 `1 block/SM`으로 되돌아간다.
- `v7`은 scalar family에서 `32B/sector`, `L2 Excessive 0` epilogue를 처음 안정적으로 만든다. `v7_vec4`는 같은 epilogue를 vec4 family에 붙여 `v6_vec4`의 occupancy cliff를 사실상 회복한다.
- `v8`은 같은 epilogue를 유지한 채 mainloop choreography를 더 단순하게 정리해 scalar family를 `22.21 TFLOPS`까지 끌어올린다. 반면 `v8_vec4`는 거의 같은 천장에 머문다.
- `v9`는 같은 scalar 정리된 경로를 유지한 채 next-tile global load 시점만 조정해 `22.27 TFLOPS`까지 소폭 더 끌어올린다. output path 개선이 끝난 뒤에는 이런 timing-level 미세 조정도 의미가 있다는 점을 보여 준다.
- `v10`은 `v9`와 동일한 로직에 symbol name만 `cutlass` 계열로 바꿔 `22.74 TFLOPS`까지 오른다. 이 버전은 algorithm 개선이라기보다 backend codegen choice가 scalar kernel의 마지막 gap에 얼마나 큰 영향을 주는지 보여 준다.
- `v11`은 `v9`에 grouped threadblock swizzle을 적용해 `22.94 TFLOPS`까지 오른다. 이 버전은 `v10`처럼 seam-local codegen을 더 압축한 것이 아니라, CTA 방문 순서를 바꿔 L2 residency를 크게 끌어올린 버전이다.
- 현재 최고점은 `warptiling_v7_vec4_128x128x8_32x64x8`의 `23.76 TFLOPS`다. 지금 최고점은 “가장 깊은 pipeline”보다 “정리된 경로 + 넓은 ingress + `L2::128B` helper + 과하지 않은 pipeline depth”에서 나온다.

## 8. 중간 정리

이번 artifacts만 놓고 보면 결론은 꽤 분명하다.

- `torch`와 CUTLASS reference는 결국 더 좋은 issue efficiency와 더 좋은 ingress primitive로 빠르다.
- study kernel에서 가장 큰 구조 점프는 `2D outer product`, `v1`의 lane remap, `v2`의 2-stage, 그리고 `v2_vec4` 이후의 넓은 ingress에서 나왔다.
- scalar warptiling은 `v2 20.70 -> v3 20.96 -> v4 21.66 -> v5 21.79 -> v6 21.85 -> v7 22.01 -> v8 22.21 -> v9 22.27 TFLOPS`로 비교적 단조롭게 좋아진다. 즉 `L2::128B`, predication, CUTLASS-style pipeline, epilogue reorder, choreography 정리, 그리고 마지막 load timing 재배치까지 scalar family에서는 모두 유효했다. 여기에 `v10 22.74 TFLOPS`와 `v11 22.94 TFLOPS`가 붙지만, 둘의 의미는 다르다. `v10`은 codegen heuristic 실험이고, `v11`은 inter-CTA cache residency 실험이다.
- vec4 warptiling은 `v2_vec4 23.01 -> v3_vec4 23.75 -> v4_vec4 23.70 -> v5_vec4 23.47 -> v6_vec4 22.83 -> v7_vec4 23.76 -> v8_vec4 23.70 TFLOPS` 흐름이다. 여기서는 `v3/v4` 이후 deeper pipeline이나 choreography 실험이 대부분 정체 구간 내부의 이동으로 읽힌다.
- 현재 최고점 `v7_vec4`는 `torch_matmul`의 `99.52%`, explicit CUTLASS의 `104.93%`, universal CUTLASS의 `106.60%` 수준이다.
- `v7`/`v8`/`v9`는 scalar path의 store 품질을 CUTLASS와 같은 수준까지 올린 뒤, load-side reuse와 mainloop steady-state를 조금씩 개선하는 단계다. `v10`은 그 위에서 source 변경 없이도 compiler의 보조 명령열 압축만으로 성능이 더 오를 수 있음을 보여 주고, `v11`은 symbol 이름 효과를 버린 채 CTA 순서 재배치만으로 L2 residency를 더 끌어올릴 수 있음을 보여 준다. 즉 남은 gap은 epilogue가 아니라 load-side reuse, issue density, cache residency, 그리고 backend codegen choice 쪽이다.
- vec4 family의 남은 gap도 이제 단순 vec4 유무가 아니라 `LDGSTS` direct copy와 더 큰 `256x128` tile이 만드는 마지막 arithmetic density 차이다.

즉 지금 상태는 fixed-shape warptiling이 두 CUTLASS reference를 분명히 앞지르면서도, `torch`의 마지막 몇 퍼센트를 아직 남겨 둔 단계다. 다음 단계의 핵심은 `v7_vec4`의 residency/issue 정돈도와 `v7~v11`의 정돈된 epilogue를 동시에 유지한 채, `torch`가 쓰는 직접적인 global->shared 복사 경로와 더 큰 tile density 쪽으로 접근하는 것이다. `v10`과 `v11`이 함께 보여 준 것은, 그 마지막 gap이 소스 구조뿐 아니라 backend codegen choice와 CTA scheduling choice에도 상당히 민감하다는 점이다.

## 9. 최종 결론

이번 artifacts 기준으로 warptiling의 큰 흐름은 그대로다. 가장 큰 성능 도약은 여전히 `v2_vec4`, `v3_vec4`에서 끝났고, 이후 `v4~v8`의 실험은 주로 정체 구간 안에서 정돈도와 트레이드오프를 재배치하는 과정이었다. 다만 scalar family 쪽에서는 `v7`의 epilogue reorder, `v8`의 choreography 정리, `v9`의 load timing 재배치가 차례로 누적되며 `22.27 TFLOPS`까지 올라갔고, 그 뒤에는 두 갈래의 미세 최적화가 나타났다. `v10`은 이름 기반 codegen 변화로 `22.74 TFLOPS`까지 오른 버전이고, `v11`은 threadblock swizzling으로 inter-CTA L2 residency를 끌어올려 `22.94 TFLOPS`까지 오른 버전이다. 즉 output path를 CUTLASS 급으로 정리한 뒤에는, 다음 이득이 mainloop timing과 dependency shaping에서만이 아니라 backend가 어떤 보조 명령열을 만들고 grid가 tile을 어떤 순서로 훑느냐에서도 나온다는 그림이 더 선명해졌다.

reference와의 차이도 이제 상당히 구체적이다. vec4 최고점인 `v7_vec4 23.76 TFLOPS`는 explicit CUTLASS `22.64`, universal CUTLASS `22.28`를 확실히 넘지만 `torch_matmul 23.87`에는 아직 못 미친다. scalar 최고점인 `v11 22.94 TFLOPS`도 이제 explicit / universal CUTLASS를 둘 다 넘는다. 다만 `v10`의 승리는 backend heuristic의 도움을 크게 받고, `v11`의 승리는 inter-CTA cache residency 개선의 도움을 크게 받는다. 따라서 남은 과제는 새 epilogue를 더 만드는 것이 아니라, `LDGSTS` 계열 직접 global-to-shared 복사 경로, 더 큰 tile density, reference 수준의 issue efficiency, 그리고 compiler choice나 CTA ordering에 덜 흔들리는 더 안정적인 source-level 구조를 얼마나 가져올 수 있느냐다. 현재 문서가 보여 주는 결론은 명확하다. fixed-shape warptiling은 이미 CUTLASS reference를 넘는 단계까지 왔고, 이제 진짜 남은 gap은 `torch`가 보여 주는 ingress quality와 arithmetic density다.
