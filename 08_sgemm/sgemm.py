from pathlib import Path

import torch
from torch.utils import cpp_extension

ROOT_DIR = Path(__file__).resolve().parent
EXTENSION_NAME = "sgemm_cuda"
BUILD_DIR = ROOT_DIR / ".cache" / EXTENSION_NAME
TORCH_MATMUL_NAME = "torch_matmul"
CUTLASS_INCLUDE_DIR = ROOT_DIR / "cutlass" / "include"


def _build_kernel_api(module):
    functions = {}
    kernel_specs = []
    all_kernel_names: list[str] = []
    study_kernel_names: list[str] = []
    reference_kernel_names: list[str] = []
    grouped_kernel_names: dict[str, list[str]] = {}

    for spec in module.get_kernel_list():
        kernel_specs.append(spec)
        name = spec["name"]
        group = spec["group"]
        kind = spec["kind"]

        functions[name] = getattr(module, name) if name != TORCH_MATMUL_NAME else torch.matmul
        all_kernel_names.append(name)
        grouped_kernel_names.setdefault(group, []).append(name)

        if kind == "study":
            study_kernel_names.append(name)
        elif kind == "reference":
            reference_kernel_names.append(name)
        else:
            raise ValueError(f"Unknown kernel kind: {kind}")

    return (
        tuple(kernel_specs),
        functions,
        tuple(all_kernel_names),
        tuple(study_kernel_names),
        tuple(reference_kernel_names),
        {group: tuple(names) for group, names in grouped_kernel_names.items()},
    )


def _load_extension():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return cpp_extension.load(
        name=EXTENSION_NAME,
        sources=[
            str(ROOT_DIR / "bindings.cu"),
        ],
        build_directory=str(BUILD_DIR),
        extra_include_paths=[str(CUTLASS_INCLUDE_DIR), str(ROOT_DIR)],
        extra_cuda_cflags=[
            "-O3",
            f"-I{CUTLASS_INCLUDE_DIR}",
            f"-I{ROOT_DIR}",
            "--expt-relaxed-constexpr",
        ],
        # verbose=True,
    )


module = _load_extension()
(
    KERNEL_SPECS,
    _functions,
    ALL_KERNELS,
    STUDY_KERNELS,
    REFERENCE_KERNELS,
    KERNEL_GROUPS,
) = _build_kernel_api(module)

globals().update(_functions)

list_kernels = module.get_kernel_list
list_kernel_groups = module.get_group_list


def select_kernel_names(
    groups: list[str] | None = None,
    kernels: list[str] | None = None,
    *,
    include_reference: bool = True,
) -> list[str]:
    if kernels:
        selected = list(kernels)
    elif not groups:
        selected = list(ALL_KERNELS)
    else:
        selected_groups = set(groups)
        selected = [spec["name"] for spec in KERNEL_SPECS if spec["group"] in selected_groups]

    if include_reference:
        ref_kernels = []
        if TORCH_MATMUL_NAME not in selected:
            ref_kernels.append(TORCH_MATMUL_NAME)

        for ref_kernel_name in REFERENCE_KERNELS:
            if ref_kernel_name not in selected:
                ref_kernels.append(ref_kernel_name)

        selected = [*ref_kernels, *selected]

    return selected


__all__ = [
    "TORCH_MATMUL_NAME",
    "KERNEL_SPECS",
    "ALL_KERNELS",
    "STUDY_KERNELS",
    "REFERENCE_KERNELS",
    "KERNEL_GROUPS",
    "list_kernels",
    "list_kernel_groups",
    "select_kernel_names",
    *ALL_KERNELS,
]
