import subprocess

from kernels import KERNELS
from torch.utils import cpp_extension

# Generate bindings before building
subprocess.run(["python", "generate_bindings.py"], check=True, cwd=".")

# Collect unique source files from kernel configurations
sources = list(set([k["file"] for k in KERNELS])) + ["sgemm_bindings.cu"]

module = cpp_extension.load(
    name="module",
    sources=sources,
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# Auto-export all kernel functions
for kernel in KERNELS:
    globals()[kernel["python_name"]] = getattr(module, kernel["python_name"])
