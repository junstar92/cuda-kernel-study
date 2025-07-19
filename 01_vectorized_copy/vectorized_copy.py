from torch.utils import cpp_extension

module = cpp_extension.load(
    name="module",
    sources=["vectorized_copy.cu"],
    extra_cuda_cflags=["-lineinfo"],
    verbose=False,
)
copy = module.copy
vectorized_copy = module.vectorized_copy
