import torch
import triton
import triton.language as tl

from vectorized_copy import copy, vectorized_copy


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    out = x
    tl.store(out_ptr + offsets, out, mask=mask)


def copy_triton(x: torch.Tensor, out: torch.Tensor):
    assert x.device == out.device
    n_elements = out.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    copy_kernel[grid](x, out, n_elements, BLOCK_SIZE=512)


if __name__ == "__main__":
    numel = 2**16 + 123
    x = torch.randn(numel, dtype=torch.float16, device="cuda")

    out = torch.zeros_like(x)
    copy(x, out)

    print("input:\n", x)
    print("output:\n", out)
    assert torch.allclose(x, out, atol=0, rtol=0)

    out = torch.zeros_like(x)
    vectorized_copy(x, out)

    print("output (vectorized):\n", out)
    assert torch.allclose(x, out, atol=0, rtol=0)

    out = torch.zeros_like(x)
    torch.cuda.profiler.start()
    copy_triton(x, out)
    torch.cuda.profiler.stop()

    print("output (triton):\n", out)
    assert torch.allclose(x, out, atol=0, rtol=0)
