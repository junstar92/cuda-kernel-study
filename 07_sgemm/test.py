import torch
from sgemm import (
    sgemm_64x64x8_1d,
    sgemm_128x128x8_8x8x1,
    sgemm_naive,
    sgemm_shmem,
    sgemm_vec_128x128x8_8x8x1,
    sgemm_vec_128x128x8_8x8x1_pad,
    sgemm_vec_128x128x16_8x8x1_pad,
    sgemm_warptiling_128x128x16_64x32x1_8x8x1,
)

torch.manual_seed(42)

M = N = K = 1024
DEVICE = "cuda"

input = torch.randn(M, K, dtype=torch.float32, device=DEVICE) / 1e2
other = torch.randn(K, N, dtype=torch.float32, device=DEVICE) / 1e2

ref_out = torch.matmul(input, other)
out = torch.empty_like(ref_out)

# naive
sgemm_naive(input, other, out)
assert torch.allclose(ref_out, out)

# shmem
out.copy_(torch.zeros_like(out))
sgemm_shmem(input, other, out)
assert torch.allclose(ref_out, out)

# shmem 1d blocktile (64x64x8_8x1x1)
out.copy_(torch.zeros_like(out))
sgemm_64x64x8_1d(input, other, out)
assert torch.allclose(ref_out, out)

# shmem 2d blocktile (128x128x8_8x8x1)
out.copy_(torch.zeros_like(out))
sgemm_128x128x8_8x8x1(input, other, out)
assert torch.allclose(ref_out, out)

# shmem vectorized 2d blocktile (128x128x8_8x8x1)
out.copy_(torch.zeros_like(out))
sgemm_vec_128x128x8_8x8x1(input, other, out)
assert torch.allclose(ref_out, out)

# shmem vectorized 2d blocktile (128x128x8_8x8x1_pad)
out.copy_(torch.zeros_like(out))
sgemm_vec_128x128x8_8x8x1_pad(input, other, out)
assert torch.allclose(ref_out, out)


# shmem vectorized 2d blocktile (128x128x16_8x8x1_pad)
out.copy_(torch.zeros_like(out))
sgemm_vec_128x128x16_8x8x1_pad(input, other, out)
assert torch.allclose(ref_out, out)

# shmem warp tiling 2d blocktile (128x128x8_64x32x1_8x8x1)
out.copy_(torch.zeros_like(out))
sgemm_warptiling_128x128x16_64x32x1_8x8x1(input, other, out)
assert torch.allclose(ref_out, out)
