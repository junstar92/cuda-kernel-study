from torch.utils import cpp_extension

module = cpp_extension.load(
    name="module",
    sources=["sgemm.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

sgemm_naive = module.sgemm_naive
sgemm_shmem = module.sgemm_shmem
sgemm_64x64x8_1d = module.sgemm_64x64x8_1d
sgemm_64x64x16_1d = module.sgemm_64x64x16_1d
sgemm_128x128x8_8x8x1 = module.sgemm_128x128x8_8x8x1
sgemm_vec_128x128x8_8x8x1 = module.sgemm_vec_128x128x8_8x8x1
sgemm_vec_128x128x8_8x8x1_pad = module.sgemm_vec_128x128x8_8x8x1_pad
sgemm_vec_128x128x16_8x8x1_pad = module.sgemm_vec_128x128x16_8x8x1_pad
sgemm_warptiling_128x128x16_64x32x1_8x8x1 = module.sgemm_warptiling_128x128x16_64x32x1_8x8x1
