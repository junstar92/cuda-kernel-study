import torch
from transpose_no_boundary_check import (
    copy,
    transpose_col,
    transpose_col_unroll_4,
    transpose_col_unroll_n,
    transpose_row,
    transpose_shm,
)

if __name__ == "__main__":
    M, N = 1024, 2048
    x = torch.randn((M, N), dtype=torch.float16, device="cuda")

    out = torch.empty_like(x)
    copy(out, x)
    assert torch.allclose(x, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_row(out, x)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_col(out, x)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_col_unroll_4(out, x)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_col_unroll_n(out, x)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 0)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 1)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 2)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 3)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 4)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 5)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)
