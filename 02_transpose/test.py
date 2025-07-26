import torch
from transpose import (
    copy,
    transpose_col,
    transpose_col_unroll_4,
    transpose_col_unroll_n,
    transpose_row,
    transpose_shm,
)

if __name__ == "__main__":
    M = N = 2**12 + 1
    x = torch.randn((M, N), dtype=torch.float16, device="cuda")
    print("input tensor:\n", x)

    out = torch.empty_like(x)
    copy(out, x)
    assert torch.allclose(x, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_row(out, x)
    print("transpose row\n", out)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_col(out, x)
    print("transpose col\n", out)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_col_unroll_4(out, x)
    print("transpose col unroll 4\n", out)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_col_unroll_n(out, x)
    print("transpose col unroll n\n", out)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 0)
    print("transpose smem\n", out)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 1)
    print("transpose smem v1\n", out)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 2)
    print("transpose smem v2\n", out)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 3)
    print("transpose smem v3\n", out)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 4)
    print("transpose smem v4\n", out)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
    transpose_shm(out, x, 5)
    print("transpose smem v5\n", out)
    assert torch.allclose(x.T, out, atol=0.0, rtol=0.0)
