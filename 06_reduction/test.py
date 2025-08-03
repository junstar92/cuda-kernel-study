import torch
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

N = 2**12 + 1
DEVICE = "cuda"
DTYPE = torch.int

x = torch.randint(-100, 100, (N,), dtype=DTYPE, device=DEVICE)
ref_out = x.sum(dtype=DTYPE)
print(ref_out)

out = torch.tensor(0, dtype=DTYPE, device=DEVICE)
reduction_atomic(out, x)
print("reduction_atomic:", out.item())
assert torch.allclose(out, ref_out)

out = torch.tensor(0, dtype=DTYPE, device=DEVICE)
reduction_atomic_shm(out, x)
print("reduction_atomic_shm:", out.item())
assert torch.allclose(out, ref_out)

out = torch.tensor(0, dtype=DTYPE, device=DEVICE)
reduction_shm(out, x)
print("reduction_shm:", out.item())
assert torch.allclose(out, ref_out)

out = torch.tensor(0, dtype=DTYPE, device=DEVICE)
reduction_shuffle(out, x)
print("reduction_shuffle:", out.item())
# assert torch.allclose(out, ref_out)

out = torch.tensor(0, dtype=DTYPE, device=DEVICE)
reduction_unroll(out, x)
print("reduction_unroll:", out.item())
assert torch.allclose(out, ref_out)

out = torch.tensor(0, dtype=DTYPE, device=DEVICE)
reduction_cg(out, x)
print("reduction_cg:", out.item())
assert torch.allclose(out, ref_out)

out = torch.tensor(0, dtype=DTYPE, device=DEVICE)
reduction_cg_unroll_2(out, x)
print("reduction_cg_unroll_2:", out.item())
assert torch.allclose(out, ref_out)


out = torch.tensor(0, dtype=DTYPE, device=DEVICE)
reduction_cub(out, x)
print("reduction_cub:", out.item())
assert torch.allclose(out, ref_out)
