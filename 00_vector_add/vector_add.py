from torch.utils import cpp_extension

add_module = cpp_extension.load(
    name="add_module",
    sources=["vector_add.cu"],
    extra_cuda_cflags=["-lineinfo"],
    verbose=False,
)
add = add_module.add
add_element2 = add_module.add_element2
add_element2_interleaved = add_module.add_element2_interleaved
