import argparse

import torch
import triton
from reduction import (
    reduction_atomic,
    reduction_atomic_shm,
    reduction_cg,
    reduction_cg_unroll_2,
    reduction_cub,
    reduction_shm,
    reduction_shuffle,
    reduction_unroll,
)

DEVICE = torch.device("cuda", index=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(16, 26, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=[
            "torch",
            "atomic_global",
            "atomic_shm",
            "shm",
            "shuffle",
            "unroll",
            "cg",
            "cg_unroll_2",
            "cub",
        ],
        line_names=[
            "Torch",
            "Atomic (global)",
            "Atomic (shared)",
            "Shared",
            "Shuffle",
            "Unroll",
            "CG",
            "CG (unroll 2)",
            "CUB",
        ],
        styles=[
            ("red", "-"),
            ("blue", "--"),
            ("green", "--"),
            ("purple", "--"),
            ("orange", "--"),
            ("brown", "--"),
            ("pink", "--"),
            ("gray", "--"),
            ("olive", "--"),
            ("cyan", "--"),
        ],
        ylabel="Time (ms)",
        plot_name="Reduction Performance (ms)",
        args={},
    )
)
def benchmark(N, provider, dtype):
    if dtype.is_floating_point:
        x = torch.randn(N, dtype=dtype)
    else:
        x = torch.randint(-100, 100, (N,), dtype=dtype)
    out = torch.tensor(0, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x.sum(), quantiles=quantiles)
    if provider == "atomic_global":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: reduction_atomic(out, x), quantiles=quantiles)
    if provider == "atomic_shm":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: reduction_atomic_shm(out, x), quantiles=quantiles)
    if provider == "shm":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: reduction_shm(out, x), quantiles=quantiles)
    if provider == "shuffle":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: reduction_shuffle(out, x), quantiles=quantiles)
    if provider == "unroll":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: reduction_unroll(out, x), quantiles=quantiles)
    if provider == "cg":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: reduction_cg(out, x), quantiles=quantiles)
    if provider == "cg_unroll_2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: reduction_cg_unroll_2(out, x), quantiles=quantiles)
    if provider == "cub":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: reduction_cub(out, x), quantiles=quantiles)

    return ms, min_ms, max_ms


class DtypeAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        assert isinstance(values, str), "values must be string"
        if values == "float32":
            dtype = torch.float32
        elif values == "float16":
            dtype = torch.float16
        elif values == "bfloat16":
            dtype = torch.bfloat16
        elif values == "int32":
            dtype = torch.int32
        elif values == "int64":
            dtype = torch.int64
        else:
            assert False, f"unsupported type: {values}"

        setattr(namespace, self.dest, dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        default=torch.float32,
        choices=["float32", "double", "int32"],
        action=DtypeAction,
    )
    parser.add_argument("--save", action="store_true", help="save results")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_default_device(DEVICE)

    benchmark.run(print_data=True, save_path="results" if args.save else None, dtype=args.dtype)
