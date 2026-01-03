# SGEMM 최적화 적용 가이드

이 문서는 CUDA FP32 SIMT SGEMM 커널을 최적화할 때 어떤 최적화를 어떤 순서로 적용하면 좋은지 정리한 가이드다. 목적은 특정 구현을 그대로 복제하는 것이 아니라, 현재 병목이 무엇인지 확인하면서 한 단계씩 성능을 끌어올리는 일반적인 접근 순서를 설명하는 데 있다.

핵심 원칙은 단순하다.

- 좋은 thread mapping만으로는 충분하지 않다.
- memory path, register reuse, warp hierarchy, pipeline, epilogue가 함께 맞아야 한다.
- 최적화는 "복잡한 코드를 더하는 일"이 아니라 "현재 병목을 하나씩 제거하는 일"이다.
- 어떤 최적화든 먼저 넣는다고 항상 이득이 나는 것은 아니다.

이 순서를 이해할 때 도움이 되는 하드웨어 관점의 mental model도 있다. NVIDIA GPU에서 SGEMM 성능은 결국 warp scheduler를 얼마나 계속 바쁘게 유지하느냐, SM 안의 한정된 register/shared-memory 예산을 얼마나 효율적으로 쓰느냐, 그리고 warp가 만드는 memory transaction 수를 얼마나 줄이느냐에 크게 좌우된다. 이런 관점에서 ILP는 "instruction마다 전용 pipeline이 따로 있다"라고 이해하기보다, 이전 결과를 기다리는 동안에도 같은 warp 안에 scheduler가 고를 수 있는 독립적인 instruction이 여러 개 준비되어 있는 상태로 이해하는 편이 더 정확하다. SGEMM은 accumulator 때문에 occupancy가 낮아지기 쉬워서 이런 특성이 더 중요해진다.

아래의 하드웨어 설명은 NVIDIA CUDA C++ Programming Guide, CUDA C++ Best Practices Guide, 아키텍처 튜닝 가이드, CUTLASS GEMM 문서를 바탕으로 정리했다.

## 한눈에 보는 추천 적용 순서

| 순서 | 최적화 포인트 | 기대 효과 | 먼저 확인할 것 |
| --- | --- | --- | --- |
| 0 | 측정 기준 고정 | 비교 가능한 실험 환경 확보 | 정확도, TFLOPS, stall reason |
| 1 | shared memory tiling | global reload 대폭 감소 | DRAM 병목이 큰가 |
| 2 | register tiling | thread당 재사용 증가 | register 수와 occupancy |
| 3 | 2D outer-product 구조 | FMA 밀도 증가 | shared load 대비 연산량 |
| 4 | `BLOCK_K` 조정 | loop / barrier overhead 감소 | shared footprint, occupancy |
| 5 | shared layout / padding | bank conflict 감소 | access pattern, wavefront excess |
| 6 | vectorized memory path | memory instruction 수 감소 | alignment, contiguous packet |
| 7 | warp tiling | scheduler 친화적 계산 구조 | CTA 안의 warp 역할 분해 |
| 8 | warp microkernel 스케줄링 | operand reuse와 의존성 구조 개선 | issue efficiency, scoreboard |
| 9 | register lookahead / prefetch | load latency hiding | register cliff |
| 10 | epilogue 설계 | output path 비용 최적화 | store coalescing, extra barrier |
| 11 | 2-stage software pipeline | overlap 강화 | memory path와 warp 구조 안정성 |
| 12 | `cp.async`, multistage, Tensor Core | 다음 세대 최적화 | 아키텍처, 복잡도, 목표 성능 |

## 0. 측정 기준 먼저 고정하기

SGEMM 최적화는 구현보다 측정이 먼저다. 기준이 흔들리면 어떤 변화가 실제 개선인지 판단할 수 없다.

- 먼저 할 일: 비교할 문제 크기 집합을 정하고 계속 같은 집합으로 측정한다. `1024`, `2048`, `4096` 같은 tile 친화 크기를 써도 좋고, 실제 워크로드 크기를 함께 섞어도 좋다.
- 같이 봐야 할 지표: TFLOPS, `ptxas` register count, occupancy, `long scoreboard` 같은 stall reason, shared memory bank conflict, global/shared load-store 패턴.
- 실전 규칙: 한 번에 한 가지 최적화만 바꾸고, 성능과 정확도를 함께 확인한다.
- 추가로 권장하는 비교: tile 경계가 없는 경우와 경계가 있는 경우를 따로 본다. 경계가 없는 경우에 빠른 커널이 경계 조건에서는 전혀 다른 병목을 보일 수 있다.

흔한 실패는 TFLOPS만 보고 다음 단계로 넘어가는 것이다. preload, padding, epilogue 같은 변화는 TFLOPS보다 먼저 register pressure나 stall reason에서 부작용이 드러날 수 있다.

## 1. Shared Memory Tiling 도입

가장 먼저 해야 할 최적화는 shared memory tiling이다.

- 바꾸는 것: CTA가 `A`, `B`의 tile을 global memory에서 shared memory로 협력 로드하고, `K` 축을 `BLOCK_K` 단위로 잘라 `load -> sync -> compute` 구조를 만든다.
- 하드웨어 관점: global memory는 정렬된 transaction 단위로 서비스되므로, warp의 주소가 흩어지면 transaction 수와 낭비 대역폭이 같이 늘어난다. shared memory는 이런 off-chip fetch를 한 번의 협력 로드와 여러 번의 on-chip reuse로 바꿔 준다.
- 기대 효과: 같은 `A` row와 `B` column을 여러 thread가 DRAM에서 반복해서 다시 읽는 일을 크게 줄일 수 있다.
- 언제 도입할까: baseline이 global memory load에 묶여 있고, `LG Throttle`이나 low issue rate가 크게 보일 때다.
- 확인할 것: global traffic은 줄어드는 대신 shared load/store와 barrier 비용이 새 병목으로 올라오는지 본다.
- 흔한 실패: shared memory를 도입했는데도 thread당 output이 너무 적으면, 병목이 DRAM에서 shared path로만 옮겨간다. 이 단계는 끝점이 아니라 기초 공사다.

> 참고 예시: [`02_smem_tiling.cuh`](./02_smem_tiling.cuh)

## 2. Register Tiling 도입

shared tile을 쓴다고 해도 thread가 output 하나만 계산하면 shared load 대비 연산량이 충분히 높지 않다. 그래서 다음 단계는 thread당 여러 accumulator를 register에 두는 것이다.

- 바꾸는 것: thread 하나가 `C` 원소 하나가 아니라 작은 strip 또는 작은 tile을 계산하게 만든다.
- 하드웨어 관점: register는 thread가 쓸 수 있는 가장 가까운 on-chip 저장소이므로, accumulator를 register에 더 오래 두면 load 하나당 유효 연산량이 늘어난다. 반면 그 accumulator들은 SM의 고정된 register file을 소비하므로 resident warp 수를 줄일 수 있다.
- 기대 효과: shared memory에서 읽은 `A` 또는 `B` 값을 더 많이 재사용할 수 있어 instruction당 유효 연산량이 늘어난다.
- 언제 도입할까: shared memory tiling 이후에도 shared load가 과도하고, thread 수준의 재사용이 부족할 때다.
- 확인할 것: register 수 증가, occupancy 변화, shared load 대비 FMA 증가.
- 흔한 실패: thread tile을 너무 크게 잡아 register pressure를 급격히 올리는 것이다. 다음 단계인 warp hierarchy까지 고려하면 초반에는 과한 thread tile을 피하는 편이 안전하다.

> 참고 예시: [`03_blocktiling_1d.cuh`](./03_blocktiling_1d.cuh)

## 3. 2D Register Tiling으로 Outer Product 구조 만들기

1D register tiling 다음에는 계산을 2D outer-product 구조로 정리하는 것이 좋다.

- 바꾸는 것: thread가 `TM x TN` 크기의 `C` sub-tile을 가진다고 보고, inner loop에서 `A` operand 하나와 `B` operand 하나를 읽은 뒤 여러 accumulator를 갱신하게 만든다.
- 하드웨어 관점: 이제 `A` 값 하나와 `B` 값 하나가 여러 개의 독립적인 FFMA 갱신으로 이어진다. 최근 NVIDIA 아키텍처에서 산술 의존성 latency는 몇 cycle 수준이므로, 독립적인 accumulator update가 많을수록 이전 FFMA 결과를 기다리는 동안 scheduler가 발행할 수 있는 ready work가 늘어난다.
- 기대 효과: 같은 load로 더 많은 FMA를 만들 수 있고, register reuse가 가장 단순하게 드러난다.
- 언제 도입할까: 1D strip 구조로는 shared load 대비 연산 밀도가 더 이상 잘 늘지 않을 때다.
- 확인할 것: shared operand 하나를 읽고 몇 개의 accumulator를 갱신하는지, load-use pattern이 더 규칙적으로 바뀌는지 본다.
- 흔한 실패: 계산 구조는 좋아졌는데 memory path가 여전히 scalar라서 전체 성능이 기대만큼 오르지 않는 경우다. 이 단계는 compute 구조를 정리하는 단계이지 memory path를 끝내는 단계가 아니다.

> 참고 예시: [`04_blocktiling_2d.cuh`](./04_blocktiling_2d.cuh)

## 4. `BLOCK_K` 조정

register reuse 구조가 잡힌 뒤에는 `BLOCK_K`를 조정할 가치가 생긴다.

- 바꾸는 것: CTA tile은 유지하면서 `BLOCK_K`를 `8`, `16`처럼 바꿔 `K` loop 횟수와 barrier 횟수를 줄여 본다.
- 하드웨어 관점: `BLOCK_K`가 커지면 loop control, iterator update, CTA-wide barrier 같은 고정 비용이 더 많은 연산에 분산된다. 대신 shared-memory stage 크기와 live register state가 함께 커지기 쉬워서, 지나치면 residency를 깎는다.
- 기대 효과: loop control, iterator update, barrier 같은 고정 오버헤드를 줄일 수 있다.
- 언제 도입할까: shared tiling과 register tiling이 이미 어느 정도 안정화된 뒤다.
- 확인할 것: shared memory footprint, register usage, occupancy, mainloop 반복 횟수 변화.
- 흔한 실패: `BLOCK_K`를 무조건 크게 잡는 것이다. reuse가 약한 상태에서는 이득이 작고, shared와 register 사용량만 늘어 occupancy를 해칠 수 있다.

`BLOCK_K`는 만능 스위치가 아니라, 앞 단계가 정리된 뒤에 조정해야 효과가 보이는 파라미터다.

## 5. Shared Memory Layout과 Padding 튜닝

SGEMM에서 shared memory는 단순 캐시가 아니라 재배열 버퍼다. 특히 `A`를 transpose staging하는 순간 layout 설계가 중요해진다.

- 바꾸는 것: `A`와 `B`를 shared memory에 올릴 때, compute access pattern에 맞는 layout을 설계하고 필요한 경우 stage stride에 padding을 넣는다.
- 하드웨어 관점: shared memory는 banked structure라서 warp의 여러 주소가 같은 bank로 몰리면 하드웨어가 그 요청을 여러 conflict-free 조각으로 나눠 직렬 처리한다. padding이 효과를 내는 이유는 stride의 modulo 패턴을 바꿔 bank 매핑을 바꾸기 때문이다.
- 기대 효과: 같은 수의 shared load/store를 해도 bank conflict를 줄이면 실제 처리량이 크게 달라질 수 있다.
- 언제 도입할까: shared wavefront excess, bank conflict, `MIO Throttle`가 두드러질 때다.
- 확인할 것: 어떤 operand가 transpose staging되는지, warp가 어느 stride로 shared memory를 읽고 쓰는지, padding이 실제로 conflict를 줄이는지 본다.
- 흔한 실패: access pattern을 보지 않고 `+1`, `+4` 같은 padding을 기계적으로 넣는 것이다. padding은 가벼운 요령이 아니라 iterator와 layout 전체를 바꾸는 일이다.

CUTLASS의 SIMT 경로도 이런 이유로 stage에 skew나 padding을 넣어 bank 집중을 완화한다. 중요한 것은 특정 숫자를 외우는 일이 아니라, 현재 access pattern을 보고 bank 충돌을 피하는 stride를 설계하는 것이다.

> 참고 예시: [`03_blocktiling_1d.cuh`](./03_blocktiling_1d.cuh), [`04_blocktiling_2d.cuh`](./04_blocktiling_2d.cuh)

## 6. Vectorized Memory Path 도입

memory path를 `float4` 같은 packet 단위로 재구성하는 것은 매우 자주 효과가 큰 단계다.

- 바꾸는 것: global load/store, shared store/load, epilogue store 가운데 contiguous `4-float` packet을 만들 수 있는 구간을 vectorized path로 바꾼다.
- 하드웨어 관점: warp가 정렬된 contiguous packet을 만들 수 있으면 하드웨어는 더 적은 memory instruction과 더 적은 transaction fragment로 같은 데이터를 처리할 수 있다. 그래서 vectorization은 thread map이 이미 contiguous address를 잘 만드는 구간에서 특히 잘 먹힌다.
- 기대 효과: 같은 데이터를 더 적은 memory instruction으로 옮길 수 있고, load/store 경로가 더 정리된다.
- 언제 도입할까: register reuse와 layout이 어느 정도 정리됐는데 memory instruction 수가 여전히 많고, contiguous packet을 만들 수 있을 때다.
- 확인할 것: alignment, thread mapping, edge predication, transpose scatter 구간에서 실제로 vectorization이 가능한지 본다.
- 흔한 실패: vectorized path를 모든 구간에 억지로 밀어 넣는 것이다. 목적지가 흩어지는 구간은 scalar path가 더 단순하고 빠를 수 있다.

주의할 점은 "빠른 커널은 반드시 per-thread `float4` load를 쓴다"가 아니라는 점이다. 정말 중요한 것은 전체 memory path가 contiguous packet을 잘 만들고, 그 이득이 register pressure와 predication 비용을 상쇄하느냐이다.

> 참고 예시: [`05_blocktiling_2d_vec4.cuh`](./05_blocktiling_2d_vec4.cuh), [`06_warptiling.cuh`](./06_warptiling.cuh)의 `sgemm_warptiling_v4`

## 7. Warp Tiling과 Warp-Aware Decomposition

CTA 수준 구조가 정리되면, 다음에는 CTA 내부 계산을 warp 기준으로 다시 짜는 것이 좋다.

- 바꾸는 것: CTA tile을 여러 warp tile로 분해하고, 각 warp가 담당하는 `A/B` fragment와 `C` sub-tile을 명시적으로 정한다.
- 하드웨어 관점: 스케줄링과 issue의 실제 단위는 CTA가 아니라 warp다. warp ownership이 분명한 구조는 scheduler에게 더 규칙적인 ready work를 주고, shared -> register -> FMA 흐름도 더 체계적으로 정리하게 해 준다.
- 기대 효과: scheduler가 실제로 발행하는 단위가 warp이므로, 계산 구조를 warp 기준으로 맞추면 shared -> register -> FMA 흐름을 더 규칙적으로 만들 수 있다.
- 언제 도입할까: blocktiling이 어느 정도 정리됐고, CTA 안의 thread 협업을 더 구조화하고 싶을 때다.
- 확인할 것: warp별 역할이 명확한지, warp tile shape가 shared access와 output store에 모두 무리가 없는지 본다.
- 흔한 실패: warp tiling만 넣으면 모든 것이 해결될 것이라고 기대하는 것이다. memory path나 epilogue가 약하면 warp decomposition만으로는 한계가 남는다.

> 참고 예시: [`06_warptiling.cuh`](./06_warptiling.cuh)의 `sgemm_warptiling_v0`, `sgemm_warptiling_v1`

## 8. Warp Microkernel 스케줄링

warp tiling 다음에는 warp 내부 FMA 순서를 정리해야 한다.

- 바꾸는 것: lane이 읽은 `A/B` operand를 작은 `2x2` 또는 `4x4` island 단위로 재사용하도록 microkernel 순서를 정한다.
- 하드웨어 관점: scheduler가 원하는 것은 모호한 "register locality"가 아니라 독립적으로 발행 가능한 ready instruction이다. accumulator update를 작은 outer-product island로 묶으면 dependency chain이 짧아지고, operand 하나를 버리기 전에 여러 FFMA에 연달아 쓰게 된다.
- 기대 효과: operand reuse가 좋아지고, dependency structure가 더 규칙적이어서 scheduler가 다루기 쉬운 계산 패턴이 된다.
- 언제 도입할까: warp tiling은 넣었지만 scoreboard dependency나 issue efficiency가 여전히 좋지 않을 때다.
- 확인할 것: operand를 읽고 얼마나 빨리 여러 accumulator 갱신에 쓰는지, accumulator update가 작은 블록 단위로 잘 묶이는지 본다.
- 흔한 실패: 이를 "register physical locality" 같은 모호한 개념으로 설명하는 것이다. 더 정확한 관점은 `operand reuse`, `use-distance`, `dependency structure`다.

좋은 microkernel은 operand를 읽자마자 여러 FMA에 사용하고, 다음 operand로 넘어가기 전에 현재 operand의 가치를 최대한 뽑아낸다.

## 9. Single-Stage에서도 Register Lookahead는 중요하다

이 단계는 자주 오해된다. `single-stage`는 shared-memory stage 수를 뜻할 뿐, register lookahead가 불가능하다는 뜻이 아니다.

- 바꾸는 것: 현재 tile을 계산하는 동안 다음 tile의 일부 operand를 thread-private register에 미리 읽어 두어 load-use distance를 늘린다.
- 하드웨어 관점: off-chip memory latency는 산술 latency보다 훨씬 길다. lookahead는 load를 시간상 더 앞당겨 dependency path에서 떼어내는 효과가 있지만, 동시에 fragment live range를 늘리고 occupancy를 좌우하는 같은 register budget을 더 많이 잡아먹는다.
- 기대 효과: total traffic은 같아도 global load가 직접적인 의존 경로에서 조금 더 앞당겨져 latency hiding이 쉬워진다.
- 언제 도입할까: warp tiling과 memory path가 어느 정도 정리됐고, shared stage를 아직 늘리고 싶지 않을 때다.
- 확인할 것: preload fragment 크기, register count, occupancy 변화, stall reason 변화.
- 흔한 실패: fragment가 큰데도 lookahead를 넣어 live range만 길게 만드는 것이다. scalar fragment 위주이거나 accumulator가 큰 커널에서는 손해가 될 수 있다.

보통 fragment가 작고 벡터화된 memory path가 이미 잡혀 있으면 lookahead가 잘 먹히고, 반대로 accumulator와 temporary가 이미 많은 커널에서는 register cliff로 바뀌기 쉽다.

> 참고 예시: [`06_warptiling.cuh`](./06_warptiling.cuh)의 `sgemm_warptiling_v3`, `sgemm_warptiling_v4`

## 10. Epilogue는 계산 구조와 따로 설계한다

`C`를 어떻게 저장할지는 mainloop만큼 중요하다. 여기서는 `compute ownership`과 `final output ownership`을 구분해서 생각해야 한다.

- 바꾸는 것: accumulator를 바로 global store할지, shared scratch를 거쳐 재배열한 뒤 store할지 결정한다.
- 하드웨어 관점: FFMA에 좋은 thread map과 coalesced global store에 좋은 thread map은 서로 다를 수 있다. epilogue scratch buffer는 그 차이를 메우는 remapping 장치이고, 더 좋은 store pattern이 extra shared footprint, barrier, reorder instruction 비용을 이길 때만 가치가 있다.
- 기대 효과: output path를 compute path와 독립적으로 최적화할 수 있다.
- 언제 도입할까: mainloop는 충분히 빨라졌는데 store path, edge predication, output mapping이 병목일 때다.
- 확인할 것: store coalescing, extra shared footprint, extra barrier, reorder pass 비용.
- 흔한 실패: 고급 프레임워크의 epilogue 구조를 그대로 복사하는 것이다. shared scratch epilogue는 일반성과 mapping 분리에 강하지만, 특화 커널에서는 오히려 손해일 수 있다.

CUTLASS가 scratch epilogue를 쓰는 대표적인 이유도 compute에 좋은 thread map과 final store에 좋은 thread map을 분리하기 위해서다. 하지만 lane이 최종 `(row, col)` 좌표를 이미 알고 있고 contiguous segment를 바로 쓸 수 있다면 direct store가 더 낫다.

> 참고 예시: direct vector store 경로는 [`05_blocktiling_2d_vec4.cuh`](./05_blocktiling_2d_vec4.cuh)에서 볼 수 있다.

## 11. 2-Stage Software Pipeline

memory path와 warp hierarchy가 어느 정도 정리되면, 다음 큰 단계는 2-stage software pipeline이다.

- 바꾸는 것: shared tile을 두 벌 두고, 한 stage는 현재 계산에 쓰고 다른 stage는 다음 tile 준비에 쓰게 만든다. prologue, steady-state, epilogue를 명확히 나눈다.
- 하드웨어 관점: SGEMM mainloop는 accumulator와 fragment 때문에 원래 occupancy가 낮아지기 쉽다. double buffering은 resident warp를 더 늘리는 대신, 같은 threadblock 안에서 memory movement와 math를 겹치게 만들어 그 약점을 보완한다.
- 기대 효과: global -> shared, shared -> register, FFMA 사이의 overlap이 강화된다.
- 언제 도입할까: single-stage + lookahead가 이미 어느 정도 안정적으로 동작하고, 아직 memory latency가 주요 병목으로 남아 있을 때다.
- 확인할 것: shared memory 예산, register 증가, pipeline steady-state가 실제로 유지되는지.
- 흔한 실패: memory path와 warp decomposition이 정리되기 전에 pipeline부터 올리는 것이다. 이런 경우 코드만 복잡해지고 기대한 만큼 안 오른다.

CUTLASS의 SIMT mainloop도 이런 pipelined 구조를 적극적으로 사용한다. 중요한 것은 특정 프레임워크를 모사하는 것이 아니라, 현재 커널에서 overlap을 만들 수 있는 준비가 끝났는지 먼저 확인하는 것이다.

## 12. 다음 단계: Threadblock Swizzling, `cp.async`, Multistage, Tensor Core

여기부터는 SIMT SGEMM의 다음 단계다. 앞선 단계가 정리되지 않은 상태에서 바로 들어가면 구현 복잡도만 커질 가능성이 높다.

### `cp.async`

- 핵심: global -> register -> shared의 소프트웨어 경로 일부를 async copy로 대체해 global -> shared latency hiding을 더 강화한다.
- 하드웨어 관점: Ampere 계열 이상에서는 async copy가 하드웨어 가속을 받으며, 일반 copy가 중간에 거치던 register를 줄이고 필요하면 L1도 우회할 수 있다.
- 언제 볼까: 2-stage software pipeline이 이미 잘 동작하고, SM80 이상을 적극적으로 타겟팅할 때.

### Multistage

- 핵심: double buffer보다 더 긴 pipeline을 두어 memory latency를 더 깊게 숨긴다.
- 언제 볼까: `BLOCK_K`가 작고 memory latency가 여전히 지배적이며, shared memory budget과 register budget이 허용될 때.

### Threadblock Swizzling

- 핵심: CTA가 tile을 방문하는 순서를 바꿔 inter-CTA cache reuse를 더 잘 만들고, 특히 L2에 특정 operand tile이 더 오래 머물도록 한다.
- 왜 중요한가: single-CTA 내부의 `LDG -> STS -> LDS -> FFMA` 압축이 이미 어느 정도 정리된 뒤에는, 같은 kernel 구조로도 CTA ordering만 바꿔 L2 hit rate와 DRAM traffic을 크게 바꿀 수 있다.
- 언제 볼까: epilogue와 2-stage pipeline이 이미 안정적이고, load-side L2 residency가 여전히 reference보다 낮을 때다.
- 무엇을 기대할까: threadblock swizzling은 instruction class를 바꾸는 최적화가 아니라 launch order를 바꾸는 최적화이므로, seam-local dependency보다 cache residency와 DRAM bandwidth 쪽 지표 변화로 읽는 편이 맞다.
- 흔한 오해: swizzling이 mainloop 자체를 더 CUTLASS-like하게 바꾸는 것은 아니다. single-CTA seam이 아니라 inter-CTA reuse를 건드린다.

### Tensor Core

- 핵심: FFMA 기반 SIMT kernel을 미세 조정하는 것이 아니라, 연산 경로 자체를 MMA 기반으로 전환한다.
- 왜 별도 단계인가: tile shape, fragment layout, alignment 요구사항, datatype 선택이 전부 달라진다. 즉 "SIMT 최적화의 마지막 단계"라기보다 "다른 클래스의 커널"에 가깝다.

## 실제 적용 순서 요약

처음부터 모든 최적화를 한 번에 넣지 말고, 아래 순서로 올라가는 것이 안전하다.

1. 측정 기준을 고정한다.
2. shared memory tiling으로 global reload를 줄인다.
3. register tiling으로 thread당 재사용을 늘린다.
4. 2D outer-product 구조로 계산 밀도를 높인다.
5. `BLOCK_K`를 조정해 loop / barrier overhead를 줄인다.
6. shared layout과 padding을 정리해 bank conflict를 줄인다.
7. vectorized memory path로 load/store instruction 수를 줄인다.
8. warp tiling으로 CTA 내부 역할을 warp 기준으로 재배열한다.
9. warp microkernel로 operand reuse와 dependency 구조를 다듬는다.
10. register lookahead로 load latency를 더 숨긴다.
11. epilogue를 별도로 최적화한다.
12. 2-stage software pipeline으로 overlap을 강화한다.
13. 필요하면 threadblock swizzling으로 inter-CTA cache reuse를 조정한다.
14. 그다음 필요할 때 `cp.async`, multistage, Tensor Core로 넘어간다.

이 순서가 좋은 이유는 앞 단계가 뒤 단계의 전제이기 때문이다. shared tiling 없이 register tiling만 키우면 DRAM 병목이 너무 크고, memory path가 약한 상태에서 epilogue나 pipeline만 고급화해도 전체 성능은 잘 오르지 않는다.

## 마지막 체크리스트

커널 하나를 바꿀 때마다 아래를 같이 확인하는 것이 좋다.

- 결과가 정확한가
- TFLOPS가 실제로 올랐는가
- register 수가 얼마나 변했는가
- occupancy가 너무 많이 줄지 않았는가
- `long scoreboard`, `MIO Throttle`, `not selected` 같은 stall reason이 어떻게 바뀌었는가
- shared bank conflict와 wavefront excess가 줄었는가
- global/shared load-store path가 의도한 폭과 alignment를 실제로 만들고 있는가
- prefetch 뒤에 live range가 과도하게 길어지지 않았는가
- epilogue가 mainloop보다 더 큰 비용이 되지 않았는가
- 경계가 없는 경우와 경계가 있는 경우 모두에서 성능과 정합성이 유지되는가

SGEMM 최적화의 핵심은 "정답 커널을 한 번에 쓰는 것"이 아니라, 현재 병목이 global memory인지, shared memory인지, register pressure인지, scheduler인지, epilogue인지 계속 확인하면서 한 단계씩 제거하는 데 있다.
