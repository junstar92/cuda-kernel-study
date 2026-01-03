import sgemm
import torch
from kernels import KERNELS


def test_kernel(kernel, input, other, out, ref_out):
    out.copy_(torch.zeros_like(out))
    kernel(input, other, out)
    assert torch.allclose(ref_out, out)


torch.manual_seed(42)

M = N = K = 4096
DEVICE = "cuda"

input = torch.randn(M, K, dtype=torch.float32, device=DEVICE) / 1e2
other = torch.randn(K, N, dtype=torch.float32, device=DEVICE) / 1e2

ref_out = torch.matmul(input, other)
out = torch.empty_like(ref_out)

# Test all kernels from configuration
print(f"Testing {len(KERNELS)} kernels...")
for kernel_config in KERNELS:
    kernel_func = getattr(sgemm, kernel_config["python_name"])
    print(f"  Testing {kernel_config['name']}...", end=" ")
    test_kernel(kernel_func, input, other, out, ref_out)
    print("✓ PASSED")

print(f"\nAll {len(KERNELS)} kernels passed!")
