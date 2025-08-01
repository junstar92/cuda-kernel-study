from torch.utils import cpp_extension

module = cpp_extension.load(
    name="module",
    sources=["shuffle.cu"],
    extra_cuda_cflags=["-lineinfo"],
    verbose=False,
)

shuffle_sync = module.shuffle_sync
shuffle_up_sync = module.shuffle_up_sync
shuffle_down_sync = module.shuffle_down_sync
shuffle_xor_sync = module.shuffle_xor_sync
