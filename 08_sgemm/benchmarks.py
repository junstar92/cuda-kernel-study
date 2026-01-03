import argparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import median

import numpy as np
import pandas as pd
import sgemm
import torch
from tqdm import tqdm

DEVICE = torch.device("cuda", index=0)
DEFAULT_SIZES = (256, 512, 1024, 2048, 4096)
DEFAULT_OUTPUT_PATH = Path("artifacts/benchmarks/results.csv")


@dataclass(frozen=True)
class MeasurementPlanItem:
    round_index: int
    kernel_name: str
    size: int


def build_measurement_plan(
    kernel_names,
    sizes,
    rounds,
    shuffle_kernels_per_round,
    seed,
):
    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    if rounds == 1 and not shuffle_kernels_per_round:
        return [
            MeasurementPlanItem(
                round_index=0,
                kernel_name=kernel_name,
                size=size,
            )
            for kernel_name in kernel_names
            for size in sizes
        ]

    rng = Random(seed)
    plan = []
    for size in sizes:
        for round_index in range(rounds):
            names = list(kernel_names)
            if shuffle_kernels_per_round:
                rng.shuffle(names)

            for kernel_name in names:
                plan.append(
                    MeasurementPlanItem(
                        round_index=round_index,
                        kernel_name=kernel_name,
                        size=size,
                    )
                )

    return plan


def summarize_round_samples(samples):
    return {key: float(median(values)) for key, values in samples.items()}


@dataclass(frozen=True)
class KernelSpec:
    name: str
    group: str
    kind: str
    function: Callable


@dataclass(frozen=True)
class BenchmarkResult:
    kernel_name: str
    group: str
    kind: str
    matrix_size: tuple[int, int, int]  # (M, N, K)
    elapsed_time_ms: float
    tflops: float

    def __str__(self) -> str:
        M, N, K = self.matrix_size
        assert M == N == K
        return f"{self.kernel_name:50} | {M:10} | {self.elapsed_time_ms:8.3f} ms | {self.tflops:8.3f} TFLOPS"


def make_summary_table(
    df: pd.DataFrame,
    *,
    value_column: str,
    kernel_names: list[str],
) -> pd.DataFrame:
    ordered = df.copy()
    ordered["kernel_name"] = pd.Categorical(
        ordered["kernel_name"],
        categories=kernel_names,
        ordered=True,
    )
    ordered = ordered.sort_values(["kernel_name", "M"])
    return ordered.pivot_table(
        index="kernel_name",
        columns="M",
        values=value_column,
        sort=False,
        observed=False,
    )


def resolve_kernel_specs(
    groups: list[str] | None,
    kernels: list[str] | None,
    *,
    include_reference: bool,
) -> list[KernelSpec]:
    available_specs = {spec["name"]: spec for spec in sgemm.KERNEL_SPECS}

    selected = [
        (
            KernelSpec(
                name=sgemm.TORCH_MATMUL_NAME,
                group="torch",
                kind="reference",
                function=lambda a, b, out: torch.matmul(a, b, out=out),
            )
            if name == sgemm.TORCH_MATMUL_NAME
            else KernelSpec(
                name=name,
                group=available_specs[name]["group"],
                kind=available_specs[name]["kind"],
                function=getattr(sgemm, name),
            )
        )
        for name in sgemm.select_kernel_names(
            groups=groups,
            kernels=kernels,
            include_reference=include_reference,
        )
    ]

    if not groups or (groups and "autokernel" not in groups):
        selected = [spec for spec in selected if spec.group != "autokernel"]

    return selected


def run_mm(
    kernel: Callable,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> float:
    total_iterations = warmup_iterations + benchmark_iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iterations)]

    for i in range(total_iterations):
        torch.cuda._sleep(9000000)
        start_events[i].record()
        kernel(a, b, out=c)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(total_iterations)]
    return float(np.mean(times[warmup_iterations:]))


def calculate_tflops(ms: float, m_size: int, n_size: int, k_size: int) -> float:
    return m_size * n_size * 2 * (k_size - 1) / ms / 1e9


def benchmark_kernels(
    kernel_specs: list[KernelSpec],
    sizes: list[int],
    *,
    warmup_iterations: int,
    benchmark_iterations: int,
    rounds: int,
    shuffle_kernels_per_round: bool,
    seed: int,
) -> list[BenchmarkResult]:
    kernel_specs_by_name = {spec.name: spec for spec in kernel_specs}
    plan = build_measurement_plan(
        kernel_names=[spec.name for spec in kernel_specs],
        sizes=sizes,
        rounds=rounds,
        shuffle_kernels_per_round=shuffle_kernels_per_round,
        seed=seed,
    )
    round_samples: dict[tuple[str, int], list[float]] = defaultdict(list)

    total_cases = len(plan)
    with tqdm(total=total_cases, unit="case") as progress:
        for item in plan:
            kernel_spec = kernel_specs_by_name[item.kernel_name]
            m_size = n_size = k_size = item.size
            progress.set_postfix_str(f"round {item.round_index + 1}/{rounds} @ {item.size}")

            a = torch.randn(m_size, k_size, dtype=torch.float32, device=DEVICE) / 1e2
            b = torch.randn(k_size, n_size, dtype=torch.float32, device=DEVICE) / 1e2
            c = torch.empty(m_size, n_size, dtype=torch.float32, device=DEVICE)

            elapsed_time_ms = run_mm(
                kernel_spec.function,
                a,
                b,
                c,
                warmup_iterations=warmup_iterations,
                benchmark_iterations=benchmark_iterations,
            )
            round_samples[(kernel_spec.name, item.size)].append(elapsed_time_ms)
            progress.update(1)

    elapsed_summary = summarize_round_samples(round_samples)
    results: list[BenchmarkResult] = []
    for kernel_spec in kernel_specs:
        for size in sizes:
            elapsed_time_ms = elapsed_summary[(kernel_spec.name, size)]
            results.append(
                BenchmarkResult(
                    kernel_name=kernel_spec.name,
                    group=kernel_spec.group,
                    kind=kernel_spec.kind,
                    matrix_size=(size, size, size),
                    elapsed_time_ms=elapsed_time_ms,
                    tflops=calculate_tflops(elapsed_time_ms, size, size, size),
                )
            )

    return results


def create_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "kernel_name": result.kernel_name,
                "group": result.group,
                "kind": result.kind,
                "M": result.matrix_size[0],
                "N": result.matrix_size[1],
                "K": result.matrix_size[2],
                "elapsed_time_ms": result.elapsed_time_ms,
                "tflops": result.tflops,
            }
            for result in results
        ]
    )
    return df


def print_summary(df: pd.DataFrame) -> None:
    kernel_names = list(dict.fromkeys(df["kernel_name"].tolist()))
    elapsed_summary = make_summary_table(
        df,
        value_column="elapsed_time_ms",
        kernel_names=kernel_names,
    )
    tflops_summary = make_summary_table(
        df,
        value_column="tflops",
        kernel_names=kernel_names,
    )

    print("\nElapsed Time (ms)")
    print(elapsed_summary)
    print("\nTFLOPS")
    print(tflops_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SGEMM kernels.")
    parser.add_argument("--groups", nargs="*", default=None, choices=sorted(sgemm.KERNEL_GROUPS))
    parser.add_argument("--kernels", nargs="*", default=None, choices=sorted(sgemm.ALL_KERNELS))
    parser.add_argument("--sizes", nargs="*", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--output-path", "-o", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--skip-reference", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run SGEMM benchmarks.")

    torch.set_default_device(DEVICE)

    kernel_specs = resolve_kernel_specs(
        args.groups,
        args.kernels,
        include_reference=not args.skip_reference,
    )
    if args.reverse:
        kernel_specs.reverse()
    results = benchmark_kernels(
        kernel_specs,
        args.sizes,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iters,
        rounds=args.rounds,
        shuffle_kernels_per_round=args.shuffle,
        seed=args.seed,
    )
    results_df = create_dataframe(results)
    print_summary(results_df)

    if args.save_output:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output_path, index=False)
        print(f"\nWrote benchmark results to {args.output_path}")


if __name__ == "__main__":
    main()
