import argparse

import torch
import triton
from ldg import load_with_ldg, load_without_ldg

DEVICE = torch.device("cuda", index=0)


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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(24, 32, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=[
            "non-ldg",
            "ldg",
        ],
        line_names=[
            "Copy without __ldg",
            "Copy with __ldg",
        ],
        styles=[
            ("red", "-"),
            ("blue", "--"),
        ],
        ylabel="Time (ms)",
        plot_name="__ldg Performance (ms)",
        args={},
    )
)
def benchmark(N, provider, dtype):
    if dtype.is_floating_point:
        x = torch.randn(N, dtype=dtype)
    else:
        x = torch.randint(-10, 10, (N,), dtype=dtype)
    out = torch.empty_like(x)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "non-ldg":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: load_without_ldg(out, x), quantiles=quantiles)
    if provider == "ldg":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: load_with_ldg(out, x), quantiles=quantiles)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        default=torch.float16,
        choices=["float32", "float16", "bfloat16", "int32", "int64"],
        action=DtypeAction,
    )
    parser.add_argument("--save", action="store_true", help="save results")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_default_device(DEVICE)

    benchmark.run(print_data=True, save_path="results" if args.save else None, dtype=args.dtype)
