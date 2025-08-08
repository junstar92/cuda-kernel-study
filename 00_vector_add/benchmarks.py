import argparse
import random

import numpy as np
import torch
import triton
import triton.language as tl
from vector_add import add, add_element2, add_element2_interleaved

TRITON_BLOCK_SIZE = 512
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


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    out = a + b
    tl.store(out_ptr + offsets, out, mask=mask)


def add_triton(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
    assert a.device == b.device == out.device
    n_elements = out.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=TRITON_BLOCK_SIZE)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(12, 32, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton", "cuda", "cuda_elem2", "cuda_elem2_interleaved"],
        line_names=["PyTorch", "Triton", "add", "add 2 elems", "add 2 elems (interleaved)"],
        styles=[
            ("red", "-"),
            ("blue", "--"),
            ("green", (0, (1, 2))),
            ("yellow", (0, (1, 4))),
            ("purple", (0, (1, 6))),
        ],
        ylabel="Time (ms)",
        plot_name="Add Performance (ms)",
        args={},
    )
)
def benchmark(N, provider, dtype):
    if dtype.is_floating_point:
        a = torch.randn(N, dtype=dtype)
        b = torch.randn(N, dtype=dtype)
    else:
        a = torch.randint(-1000, 1000, (N,), dtype=dtype)
        b = torch.randint(-1000, 1000, (N,), dtype=dtype)
    out = torch.empty_like(a)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: a + b, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_triton(a, b, out), quantiles=quantiles)
    if provider == "cuda":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(a, b, out), quantiles=quantiles)
    if provider == "cuda_elem2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_element2(a, b, out), quantiles=quantiles)
    if provider == "cuda_elem2_interleaved":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_element2_interleaved(a, b, out), quantiles=quantiles)

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

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.set_default_device(DEVICE)

    benchmark.run(print_data=True, save_path="results" if args.save else None, dtype=args.dtype)
