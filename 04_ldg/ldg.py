from torch.utils import cpp_extension

module = cpp_extension.load(
    name="module",
    sources=["ldg.cu"],
    verbose=False,
)

load_with_ldg = module.load_with_ldg
load_without_ldg = module.load_without_ldg
