from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sgemm
import torch
from kernels import KERNELS

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
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        # Start with torch baseline
        self.kernels = [KernelInfo("torch", torch.matmul)]
        # Auto-load all kernels from configuration
        for kernel in KERNELS:
            func = getattr(sgemm, kernel["python_name"])
            self.kernels.append(KernelInfo(kernel["name"], func))

    def run_all_benchmarks(self) -> list[BenchmarkResult]:
        results = []

        for kernel_info in self.kernels:
            print(f"\n{kernel_info.name}")

            for matrix_size in self.config.matrix_sizes:
                M, N, K = matrix_size, matrix_size, matrix_size
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
                    print(f"    {elapsed_time_ms:8.3f} ms, {tflops:8.3f} TFLOPS @ {M}x{N}x{K}")

                except Exception as e:
                    print(f"    ERROR: {str(e)} @ {M}x{N}x{K}")

        return results

    @staticmethod
    def calculate_tflops(ms: float, M: int, N: int, K: int) -> float:
        return M * N * 2 * (K - 1) / ms / 1e9


class ResultFormatter:
    @staticmethod
    def create_tflops_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
        kernel_results = {}
        matrix_sizes = set()

        for result in results:
            size = result.matrix_size[0]
            matrix_sizes.add(size)

            if result.kernel_name not in kernel_results:
                kernel_results[result.kernel_name] = {}
            kernel_results[result.kernel_name][size] = result.tflops

        matrix_sizes = sorted(matrix_sizes)

        data = {}
        for size in matrix_sizes:
            data[f"{size}"] = []

        kernel_names = kernel_results.keys()

        for kernel_name in kernel_names:
            for size in matrix_sizes:
                if size in kernel_results[kernel_name]:
                    data[f"{size}"].append(kernel_results[kernel_name][size])
                else:
                    data[f"{size}"].append(None)

        df = pd.DataFrame(data, index=kernel_names)
        return df


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

    config = BenchmarkConfig(matrix_sizes=[256, 512, 1024, 2048, 4096])

    runner = BenchmarkRunner(config)
    results = runner.run_all_benchmarks()

    formatter = ResultFormatter()
    tflops_df = formatter.create_tflops_dataframe(results)

    print(f"\n\nDevice: {DEVICE}")
    print(f"Warmup iterations: {config.warmup_iterations}")
    print(f"Benchmark iterations: {config.benchmark_iterations}")
    print(f"Matrix sizes: {config.matrix_sizes}\n")

    print(f"{'TFLOPS Performance (M=N=K):'}")
    print(tflops_df)

    print(f"\n{'=' * 100}")
    print(f"{'BENCHMARK COMPLETED':^100}")
    print(f"{'=' * 100}")

    return tflops_df


if __name__ == "__main__":
    df = main()
    # df.to_csv("sgemm_performance.csv")
