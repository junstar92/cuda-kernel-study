import torch
import triton
import triton.language as tl

from vector_add import add, add_element2, add_element2_interleaved


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
    add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=512)


if __name__ == "__main__":
    numel = 2**16 + 123
    x = torch.randn(numel, dtype=torch.float16, device="cuda")
    y = torch.randn(numel, dtype=torch.float16, device="cuda")
    output_torch = x + y

    out = torch.zeros_like(x)
    add(x, y, out)

    print("torch output:\n", output_torch)
    print("cuda output (1 elem):\n", out)
    assert torch.allclose(output_torch, out)

    out = torch.zeros_like(x)
    add_element2(x, y, out)

    print("cuda output (2 elems):\n", out)
    assert torch.allclose(output_torch, out)

    out = torch.zeros_like(x)
    add_element2_interleaved(x, y, out)

    print("cuda output (2 elems interleaved):\n", out)
    assert torch.allclose(output_torch, out)

    out = torch.zeros_like(x)
    torch.cuda.profiler.start()
    add_triton(x, y, out)
    torch.cuda.profiler.stop()
    print("triton output:\n", out)
    assert torch.allclose(output_torch, out)
