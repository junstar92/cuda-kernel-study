import argparse

import sgemm
import torch

DEVICE = "cuda"
BASE_SHAPE = (4096, 4096, 4096)
ATOL = 1e-4
RTOL = 1e-4


def resolve_kernel_names(
    groups: list[str] | None = None,
    kernels: list[str] | None = None,
    *,
    include_reference: bool = True,
) -> list[str]:
    return sgemm.select_kernel_names(
        groups=groups,
        kernels=kernels,
        include_reference=include_reference,
    )


def test_kernel(kernel_name: str, shape: tuple[int, int, int]):
    kernel = getattr(sgemm, kernel_name)
    M, N, K = shape
    input = torch.randn(M, K, dtype=torch.float32, device=DEVICE) / 1e2
    other = torch.randn(K, N, dtype=torch.float32, device=DEVICE) / 1e2
    out = torch.empty(M, N, dtype=torch.float32, device=DEVICE)
    ref_out = torch.matmul(input, other)

    out.zero_()
    range_id = torch.cuda.nvtx.range_start("sgemm_profile")
    try:
        kernel(input, other, out=out)
    finally:
        torch.cuda.nvtx.range_end(range_id)

    assert torch.allclose(ref_out, out, atol=ATOL, rtol=RTOL), f"{kernel_name} failed for shape M={M}, N={N}, K={K}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SGEMM corarectness checks.")
    parser.add_argument("--groups", nargs="*", default=None, choices=sorted(sgemm.KERNEL_GROUPS))
    parser.add_argument("--kernels", nargs="*", default=None, choices=sorted(sgemm.ALL_KERNELS))
    parser.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"), default=BASE_SHAPE)
    parser.add_argument("--skip-ref", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    kernel_names = resolve_kernel_names(
        args.groups,
        args.kernels,
        include_reference=not args.skip_ref,
    )

    for kernel_name in kernel_names:
        print(f"test {kernel_name} @ {tuple(args.shape)}")
        test_kernel(kernel_name, tuple(args.shape))


if __name__ == "__main__":
    main()
