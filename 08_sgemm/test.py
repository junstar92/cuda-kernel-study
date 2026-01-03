import torch
from sgemm import (
    sgemm_blocktiling_1d_64x64x8_8_no_pad,
    sgemm_blocktiling_1d_64x64x8_8_pad4,
    sgemm_blocktiling_1d_64x64x8_16_pad4,
    sgemm_blocktiling_1d_128x128x8_64_pad4,
    sgemm_blocktiling_1d_128x128x16_64_pad4,
    sgemm_blocktiling_2d_128x128x8_8x8_pad4,
    sgemm_blocktiling_2d_128x128x16_8x8_pad4,
    sgemm_blocktiling_2d_vec4_128x128x8_8x8_pad4,
    sgemm_blocktiling_2d_vec4_128x128x16_8x8_pad4,
    sgemm_cutlass_simt_128x128x8_32x64x8_2stage,
    sgemm_cutlass_simt_128x128x8_64x32x8_2stage,
    sgemm_naive_col,
    sgemm_naive_row,
    sgemm_smem_tiling,
    sgemm_warptiling_128x128x8_32x64x8,
    sgemm_warptiling_128x128x8_64x32x8,
    sgemm_warptiling_vec4_128x128x8_32x64x8,
    sgemm_warptiling_vec4_128x128x8_64x32x8,
)

DEVICE = "cuda"
BASE_SHAPE = (4096, 4096, 4096)
ATOL = 1e-4
RTOL = 1e-4


def test_kernel(kernel, shape):
    M, N, K = shape
    input = torch.randn(M, K, dtype=torch.float32, device=DEVICE) / 1e2
    other = torch.randn(K, N, dtype=torch.float32, device=DEVICE) / 1e2
    out = torch.empty(M, N, dtype=torch.float32, device=DEVICE)
    ref_out = torch.matmul(input, other)

    out.zero_()
    range_id = torch.cuda.nvtx.range_start("sgemm_profile")
    kernel(input, other, out=out)
    torch.cuda.nvtx.range_end(range_id)
    assert torch.allclose(ref_out, out, atol=ATOL, rtol=RTOL), f"kernel failed for shape M={M}, N={N}, K={K}"


KERNELS = [
    ("torch.matmul", torch.matmul),
    ("cutlass_simt_128x128x8_64x32x8_2stage", sgemm_cutlass_simt_128x128x8_64x32x8_2stage),
    ("cutlass_simt_128x128x8_32x64x8_2stage", sgemm_cutlass_simt_128x128x8_32x64x8_2stage),
    ("naive_row", sgemm_naive_row),
    ("naive_col", sgemm_naive_col),
    ("smem_tiling", sgemm_smem_tiling),
    ("blocktiling_1d_64x64x8_8_no_pad", sgemm_blocktiling_1d_64x64x8_8_no_pad),
    ("blocktiling_1d_64x64x8_8_pad4", sgemm_blocktiling_1d_64x64x8_8_pad4),
    ("blocktiling_1d_64x64x8_16_pad4", sgemm_blocktiling_1d_64x64x8_16_pad4),
    ("blocktiling_1d_128x128x8_64_pad4", sgemm_blocktiling_1d_128x128x8_64_pad4),
    ("blocktiling_1d_128x128x16_64_pad4", sgemm_blocktiling_1d_128x128x16_64_pad4),
    ("blocktiling_2d_128x128x8_8x8_pad4", sgemm_blocktiling_2d_128x128x8_8x8_pad4),
    ("blocktiling_2d_128x128x16_8x8_pad4", sgemm_blocktiling_2d_128x128x16_8x8_pad4),
    ("blocktiling_2d_vec4_128x128x8_8x8_pad4", sgemm_blocktiling_2d_vec4_128x128x8_8x8_pad4),
    ("blocktiling_2d_vec4_128x128x16_8x8_pad4", sgemm_blocktiling_2d_vec4_128x128x16_8x8_pad4),
    ("warptiling_128x128x8_32x64x8", sgemm_warptiling_128x128x8_32x64x8),
    ("warptiling_128x128x8_64x32x8", sgemm_warptiling_128x128x8_64x32x8),
    ("warptiling_vec4_128x128x8_32x64x8", sgemm_warptiling_vec4_128x128x8_32x64x8),
    ("warptiling_vec4_128x128x8_64x32x8", sgemm_warptiling_vec4_128x128x8_64x32x8),
]


def main():
    torch.manual_seed(42)

    for name, kernel in KERNELS:
        print(f"test {name} @ {BASE_SHAPE}")
        test_kernel(kernel, BASE_SHAPE)


if __name__ == "__main__":
    main()
