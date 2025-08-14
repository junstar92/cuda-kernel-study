from torch.utils import cpp_extension

module = cpp_extension.load(
    name="module",
    sources=["transpose_no_boundary_check.cu"],
    extra_cuda_cflags=["-lineinfo"],
    verbose=False,
)
copy = module.copy
transpose_row = module.transpose_row
transpose_col = module.transpose_col
transpose_col_unroll_4 = module.transpose_col_unroll_4
transpose_col_unroll_n = module.transpose_col_unroll_n
transpose_shm = module.transpose_shm
