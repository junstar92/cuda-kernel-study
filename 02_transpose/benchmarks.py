import argparse
import random

import numpy as np
import torch
import triton
from transpose_no_boundary_check import (
    copy,
    transpose_col,
    transpose_col_unroll_4,
    transpose_col_unroll_n,
    transpose_row,
    transpose_shm,
)

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
        x_vals=[2**i for i in range(9, 16, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=[
            "copy",
            "transpose_row",
            "transpose_col",
            "transpose_col_unroll_4",
            "transpose_col_unroll_n",
            "transpose_shm_v0",
            "transpose_shm_v1",
            "transpose_shm_v0_2",
            "transpose_shm_v1_2",
            "transpose_shm_v1_3",
            "transpose_shm_v1_unroll_2",
        ],
        line_names=[
            "Copy",
            "Row",
            "Col",
            "Unroll 4",
            "Unroll n",
            "Smem V0 (16x16)",
            "Smem V1 (16x16)",
            "Smem V0 (32x16)",
            "Smem V1 (32x16)",
            "Smem V1 (32x16, Pad)",
            "Smem V1 (32x16, Unroll 2)",
        ],
        styles=[
            ("red", "-"),
            ("blue", ":"),
            ("green", "--"),
            ("purple", "--"),
            ("orange", "--"),
            ("brown", "--"),
            ("pink", "--"),
            ("gray", "--"),
            ("olive", "--"),
            ("cyan", "--"),
            ("blue", "-"),
        ],
        ylabel="Bandwidth (GB/s)",
        plot_name="Transpose Bandwidth (GB/s)",
        args={},
    )
)
def benchmark(N, provider, dtype):
    if dtype.is_floating_point:
        x = torch.randn((N, N), dtype=dtype)
    else:
        x = torch.randint(-10, 10, (N, N), dtype=dtype)
    out = torch.empty_like(x)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "copy":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: copy(out, x), quantiles=quantiles)
    if provider == "transpose_row":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_row(out, x), quantiles=quantiles)
    if provider == "transpose_col":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_col(out, x), quantiles=quantiles)
    if provider == "transpose_col_unroll_4":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_col_unroll_4(out, x), quantiles=quantiles)
    if provider == "transpose_col_unroll_n":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_col_unroll_n(out, x), quantiles=quantiles)
    if provider == "transpose_shm_v0":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_shm(out, x, 0), quantiles=quantiles)
    if provider == "transpose_shm_v1":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_shm(out, x, 1), quantiles=quantiles)
    if provider == "transpose_shm_v0_2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_shm(out, x, 2), quantiles=quantiles)
    if provider == "transpose_shm_v1_2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_shm(out, x, 3), quantiles=quantiles)
    if provider == "transpose_shm_v1_3":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_shm(out, x, 4), quantiles=quantiles)
    if provider == "transpose_shm_v1_unroll_2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_shm(out, x, 5), quantiles=quantiles)

    def get_bandwidth_gbs(ms: float) -> float:
        return (2 * N * N * dtype.itemsize) / (ms * 1e6)

    return get_bandwidth_gbs(ms), get_bandwidth_gbs(max_ms), get_bandwidth_gbs(min_ms)


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

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.set_default_device(DEVICE)

    benchmark.run(print_data=True, save_path="results" if args.save else None, dtype=args.dtype)
