import torch
from sgemm_2stage import (
    sgemm_shmem,
    sgemm_warptiling_128x128x8_32x64x8,
    sgemm_warptiling_128x128x8_32x64x8_padding,
    sgemm_warptiling_128x128x8_32x64x8_padding_2stage,
)

torch.manual_seed(42)

M = N = K = 1024
DEVICE = "cuda"

input = torch.randn(M, K, dtype=torch.float32, device=DEVICE) / 1e2
other = torch.randn(K, N, dtype=torch.float32, device=DEVICE) / 1e2

ref_out = torch.matmul(input, other)
out = torch.empty_like(ref_out)

# shmem
out.copy_(torch.zeros_like(out))
sgemm_shmem(input, other, out)
assert torch.allclose(ref_out, out)

# shmem warp tiling 2d blocktile (128x128x8_32x64x8)
out.copy_(torch.zeros_like(out))
sgemm_warptiling_128x128x8_32x64x8(input, other, out)
assert torch.allclose(ref_out, out)

# shmem warp tiling 2d blocktile with padding (128x128x8_32x64x8)
out.copy_(torch.zeros_like(out))
sgemm_warptiling_128x128x8_32x64x8_padding(input, other, out)
assert torch.allclose(ref_out, out)

# shmem warp tiling 2d blocktile with padding and 2-stage pipelining (128x128x8_32x64x8)
out.copy_(torch.zeros_like(out))
sgemm_warptiling_128x128x8_32x64x8_padding_2stage(input, other, out)
assert torch.allclose(ref_out, out)
