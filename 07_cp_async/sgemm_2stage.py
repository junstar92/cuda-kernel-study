from torch.utils import cpp_extension

module = cpp_extension.load(
    name="module",
    sources=["sgemm_pipeline.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

sgemm_shmem = module.sgemm_shmem
sgemm_warptiling_128x128x8_32x64x8 = module.sgemm_warptiling_128x128x8_32x64x8
sgemm_warptiling_128x128x8_32x64x8_padding = module.sgemm_warptiling_128x128x8_32x64x8_padding
sgemm_warptiling_128x128x8_32x64x8_padding_2stage = module.sgemm_warptiling_128x128x8_32x64x8_padding_2stage
