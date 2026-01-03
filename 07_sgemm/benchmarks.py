from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import torch
from test import KERNELS
from tqdm import tqdm

DEVICE = torch.device("cuda", index=0)


@dataclass
class BenchmarkResult:
    kernel_name: str
    matrix_size: tuple[int, int, int]  # (M, N, K)
    elapsed_time_ms: float
    tflops: float

    def __str__(self) -> str:
        M, N, K = self.matrix_size
        assert M == N == K
        return f"{self.kernel_name:50} | {M:10} | {self.elapsed_time_ms:8.3f} ms | {self.tflops:8.3f} TFLOPS"


@dataclass
class BenchmarkConfig:
    matrix_sizes: list[int]
    warmup_iterations: int = 50
    benchmark_iterations: int = 100


@dataclass
class KernelInfo:
    name: str
    function: Callable


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, skip_kernels: list[str] | None = None):
        skip_kernels = skip_kernels or ["naive_col"]
        self.config = config
        self.kernels = []

        for name, func in KERNELS:
            if name in skip_kernels:
                continue
            self.kernels.append(KernelInfo(name, func))

    def run_all_benchmarks(self) -> list[BenchmarkResult]:
        results = []
        errors = []
        total_cases = len(self.kernels) * len(self.config.matrix_sizes)

        with tqdm(total=total_cases, unit="case") as progress:
            for kernel_info, matrix_size in product(self.kernels, self.config.matrix_sizes):
                M, N, K = matrix_size, matrix_size, matrix_size
                progress.set_postfix_str(f"{kernel_info.name} @ ({M}x{N}x{K})")

                a = torch.randn(M, K, dtype=torch.float32, device=DEVICE) / 1e2
                b = torch.randn(K, N, dtype=torch.float32, device=DEVICE) / 1e2
                c = torch.empty(M, N, dtype=torch.float32, device=DEVICE)

                try:
                    elapsed_time_ms = run_mm(
                        kernel_info.function,
                        a,
                        b,
                        c,
                        warmup_iterations=self.config.warmup_iterations,
                        benchmark_iterations=self.config.benchmark_iterations,
                    )

                    tflops = self.calculate_tflops(elapsed_time_ms, M, N, K)

                    result = BenchmarkResult(
                        kernel_name=kernel_info.name,
                        matrix_size=(M, N, K),
                        elapsed_time_ms=elapsed_time_ms,
                        tflops=tflops,
                    )

                    results.append(result)
                except Exception as e:
                    errors.append((kernel_info.name, (M, N, K), str(e)))
                finally:
                    progress.update(1)

        if errors:
            print("\nBenchmark failures:")
            for kernel_name, matrix_size, error in errors:
                print(f"{kernel_name} @ {matrix_size}: {error}")

        return results

    @staticmethod
    def calculate_tflops(ms: float, M: int, N: int, K: int) -> float:
        return M * N * 2 * (K - 1) / ms / 1e9


class ResultFormatter:
    @staticmethod
    def create_dataframes(results: list[BenchmarkResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
        kernel_results = {}
        matrix_sizes = set()

        for result in results:
            size = result.matrix_size[0]
            matrix_sizes.add(size)

            if result.kernel_name not in kernel_results:
                kernel_results[result.kernel_name] = {}
            kernel_results[result.kernel_name][size] = (
                result.elapsed_time_ms,
                result.tflops,
            )

        matrix_sizes = sorted(matrix_sizes)

        msec = {}
        tflops = {}
        for size in matrix_sizes:
            msec[f"{size}"] = []
            tflops[f"{size}"] = []

        kernel_names = kernel_results.keys()

        for kernel_name in kernel_names:
            for size in matrix_sizes:
                if size in kernel_results[kernel_name]:
                    msec[f"{size}"].append(kernel_results[kernel_name][size][0])
                    tflops[f"{size}"].append(kernel_results[kernel_name][size][1])
                else:
                    msec[f"{size}"].append(None)
                    tflops[f"{size}"].append(None)

        msec_df = pd.DataFrame(msec, index=kernel_names)
        tflops_df = pd.DataFrame(tflops, index=kernel_names)
        return msec_df, tflops_df


def run_mm(
    kernel,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    warmup_iterations: int = 50,
    benchmark_iterations: int = 100,
) -> float:
    total_iterations = warmup_iterations + benchmark_iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iterations)]

    for i in range(total_iterations):
        start_events[i].record()
        kernel(a, b, out=c)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) for i in range(total_iterations)]
    return np.mean(times[warmup_iterations:])


def main():
    torch.set_default_device(DEVICE)

    config = BenchmarkConfig(matrix_sizes=[128, 256, 512, 1024, 2048, 4096])

    runner = BenchmarkRunner(config)
    results = runner.run_all_benchmarks()

    formatter = ResultFormatter()
    msec_df, tflops_df = formatter.create_dataframes(results)

    print(f"\n\nDevice: {DEVICE}")
    print(f"Warmup iterations: {config.warmup_iterations}")
    print(f"Benchmark iterations: {config.benchmark_iterations}")
    print(f"Matrix sizes: {config.matrix_sizes}\n")

    print(f"\n{'Elapsed Time (M=N=K):'}")
    print(msec_df)

    print(f"\n{'TFLOPS (M=N=K):'}")
    print(tflops_df)

    print(f"\n{'=' * 100}")
    print(f"{'BENCHMARK COMPLETED':^100}")
    print(f"{'=' * 100}")

    return tflops_df


if __name__ == "__main__":
    df = main()
    # df.to_csv("sgemm_performance.csv")
