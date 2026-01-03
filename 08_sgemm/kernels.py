"""SGEMM kernel configuration.

This file defines all SGEMM kernels with their metadata.
Add new kernels here to automatically integrate them into build, test, and benchmark.
"""

KERNELS = [
    {
        "name": "naive",
        "python_name": "sgemm_naive",
        "file": "01_naive.cu",
        "launch_func": "launch_sgemm_naive",
        "template_params": [],
    },
    {
        "name": "smem_naive",
        "python_name": "sgemm_smem_naive",
        "file": "02_smem_naive.cu",
        "launch_func": "launch_sgemm_smem_naive",
        "template_params": [32],
    },
    {
        "name": "1d_blocktiling_64x64x8_8",
        "python_name": "sgemm_1d_blocktiling_64x64x8_8",
        "file": "03_1d_blocktiling.cu",
        "launch_func": "launch_sgemm_1d_blocktiling",
        "template_params": [64, 64, 8, 8],
    },
    {
        "name": "1d_blocktiling_128x128x8_64",
        "python_name": "sgemm_1d_blocktiling_128x128x8_64",
        "file": "03_1d_blocktiling.cu",
        "launch_func": "launch_sgemm_1d_blocktiling",
        "template_params": [128, 128, 8, 64],
    },
    {
        "name": "2d_blocktiling_128x128x8_8x8",
        "python_name": "sgemm_2d_blocktiling_128x128x8_8x8",
        "file": "04_2d_blocktiling.cu",
        "launch_func": "launch_sgemm_2d_blocktiling",
        "template_params": [128, 128, 8, 8, 8],
    },
    {
        "name": "2d_blocktiling_vec_128x128x8_8x8",
        "python_name": "sgemm_2d_blocktiling_vec_128x128x8_8x8",
        "file": "05_2d_blocktiling_vec.cu",
        "launch_func": "launch_sgemm_2d_blocktiling_vec",
        "template_params": [128, 128, 8, 8, 8],
    },
    {
        "name": "warptiling_128x128x8_64x32x8",
        "python_name": "sgemm_warptiling_128x128x8_64x32x8",
        "file": "06_warptiling.cu",
        "launch_func": "launch_sgemm_warptiling",
        "template_params": [128, 128, 8, 64, 32, 8],
    },
]
