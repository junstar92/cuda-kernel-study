from pathlib import Path

from torch.utils import cpp_extension

ROOT_DIR = Path(__file__).resolve().parent
CUTLASS_INCLUDE_DIR = ROOT_DIR / "cutlass" / "include"

module = cpp_extension.load(
    name="module",
    sources=[
        "bindings.cu",
    ],
    extra_include_paths=[str(CUTLASS_INCLUDE_DIR)],
    extra_cuda_cflags=[
        "-O3",
        f"-I{CUTLASS_INCLUDE_DIR}",
        "--expt-relaxed-constexpr",
    ],
    # verbose=True,
)

sgemm_cutlass_simt_128x128x8_32x64x8_2stage = module.sgemm_cutlass_simt_128x128x8_32x64x8_2stage
sgemm_cutlass_simt_128x128x8_64x32x8_2stage = module.sgemm_cutlass_simt_128x128x8_64x32x8_2stage
sgemm_cutlass_universal_simt_128x256_8x4 = module.sgemm_cutlass_universal_simt_128x256_8x4
sgemm_naive_row = module.sgemm_naive_row
sgemm_naive_col = module.sgemm_naive_col
sgemm_smem_tiling = module.sgemm_smem_tiling
sgemm_blocktiling_1d_64x64x8_8_no_pad = module.sgemm_blocktiling_1d_64x64x8_8_no_pad
sgemm_blocktiling_1d_64x64x8_8_pad4 = module.sgemm_blocktiling_1d_64x64x8_8_pad4
sgemm_blocktiling_1d_64x64x8_16_pad4 = module.sgemm_blocktiling_1d_64x64x8_16_pad4
sgemm_blocktiling_1d_128x128x8_64_pad4 = module.sgemm_blocktiling_1d_128x128x8_64_pad4
sgemm_blocktiling_1d_128x128x16_64_pad4 = module.sgemm_blocktiling_1d_128x128x16_64_pad4
sgemm_blocktiling_2d_128x128x8_8x8_pad4 = module.sgemm_blocktiling_2d_128x128x8_8x8_pad4
sgemm_blocktiling_2d_128x128x16_8x8_pad4 = module.sgemm_blocktiling_2d_128x128x16_8x8_pad4
sgemm_blocktiling_2d_vec4_128x128x8_8x8_pad4 = module.sgemm_blocktiling_2d_vec4_128x128x8_8x8_pad4
sgemm_blocktiling_2d_vec4_128x128x16_8x8_pad4 = module.sgemm_blocktiling_2d_vec4_128x128x16_8x8_pad4
sgemm_warptiling_v0_128x128x8_32x64x8 = module.sgemm_warptiling_v0_128x128x8_32x64x8
sgemm_warptiling_v0_128x128x8_64x32x8 = module.sgemm_warptiling_v0_128x128x8_64x32x8
sgemm_warptiling_v1_128x128x8_32x64x8 = module.sgemm_warptiling_v1_128x128x8_32x64x8
sgemm_warptiling_v1_128x128x8_64x32x8 = module.sgemm_warptiling_v1_128x128x8_64x32x8
sgemm_warptiling_v2_128x128x8_32x64x8 = module.sgemm_warptiling_v2_128x128x8_32x64x8
sgemm_warptiling_v2_128x128x8_64x32x8 = module.sgemm_warptiling_v2_128x128x8_64x32x8
sgemm_warptiling_v2_128x128x16_32x64x16 = module.sgemm_warptiling_v2_128x128x16_32x64x16
sgemm_warptiling_v3_128x128x8_32x64x8 = module.sgemm_warptiling_v3_128x128x8_32x64x8
sgemm_warptiling_v3_128x128x8_64x32x8 = module.sgemm_warptiling_v3_128x128x8_64x32x8
sgemm_warptiling_v3_128x128x16_32x64x16 = module.sgemm_warptiling_v3_128x128x16_32x64x16
sgemm_warptiling_v4_128x128x8_32x64x8 = module.sgemm_warptiling_v4_128x128x8_32x64x8
sgemm_warptiling_v4_128x128x8_64x32x8 = module.sgemm_warptiling_v4_128x128x8_64x32x8
sgemm_warptiling_v4_128x128x16_32x64x16 = module.sgemm_warptiling_v4_128x128x16_32x64x16
sgemm_warptiling_v5_128x128x8_32x64x8 = module.sgemm_warptiling_v5_128x128x8_32x64x8
sgemm_warptiling_v5_128x128x8_64x32x8 = module.sgemm_warptiling_v5_128x128x8_64x32x8
sgemm_warptiling_v5_128x128x16_32x64x16 = module.sgemm_warptiling_v5_128x128x16_32x64x16
