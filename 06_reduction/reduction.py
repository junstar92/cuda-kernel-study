from torch.utils import cpp_extension

module = cpp_extension.load(
    name="module",
    sources=["reduction.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

reduction_atomic = module.reduction_atomic
reduction_atomic_shm = module.reduction_atomic_shm
reduction_shm = module.reduction_shm
reduction_shuffle = module.reduction_shuffle
reduction_unroll = module.reduction_unroll
reduction_cg = module.reduction_cg
reduction_cg_unroll_2 = module.reduction_cg_unroll_2
reduction_cub = module.reduction_cub
