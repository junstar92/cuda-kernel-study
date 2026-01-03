#include <torch/extension.h>

#include "00_cutlass.cuh"
#include "01_naive.cuh"
#include "02_smem_tiling.cuh"
#include "03_blocktiling_1d.cuh"
#include "04_blocktiling_2d.cuh"
#include "05_blocktiling_2d_vec4.cuh"
#include "06_warptiling_v0.cuh"
#include "06_warptiling_v1.cuh"
#include "06_warptiling_v10.cuh"
#include "06_warptiling_v11.cuh"
#include "06_warptiling_v2.cuh"
#include "06_warptiling_v2_vec4.cuh"
#include "06_warptiling_v3.cuh"
#include "06_warptiling_v3_vec4.cuh"
#include "06_warptiling_v4.cuh"
#include "06_warptiling_v4_vec4.cuh"
#include "06_warptiling_v5.cuh"
#include "06_warptiling_v5_vec4.cuh"
#include "06_warptiling_v6.cuh"
#include "06_warptiling_v6_vec4.cuh"
#include "06_warptiling_v7.cuh"
#include "06_warptiling_v7_vec4.cuh"
#include "06_warptiling_v8.cuh"
#include "06_warptiling_v8_vec4.cuh"
#include "06_warptiling_v9.cuh"

namespace py = pybind11;

using SgemmLauncher = void (*)(float*, float const*, float const*, int, int,
                               int);

#define SGEMM_FOR_EACH_KERNEL(SGEMM_KERNEL)                                    \
  SGEMM_KERNEL(reference, cutlass, cutlass_simt_128x128x8_32x64x8_2stage,      \
               (sgemm::cutlass::launch<128, 128, 8, 32, 64, 8, 2>), 128, 128,  \
               8, 32, 64, 8, 2)                                                \
  SGEMM_KERNEL(                                                                \
      reference, cutlass, cutlass_universal_simt_256x128_8x4,                  \
      (sgemm::cutlass::launch_universal<256, 128, 8, 64, 64, 8, 4, 8>), 256,   \
      128, 8, 64, 64, 8, 4)                                                    \
  SGEMM_KERNEL(study, naive, naive_row, (sgemm::naive::row::launch), 0, 0, 0,  \
               0, 0, 0, 1)                                                     \
  SGEMM_KERNEL(study, naive, naive_col, (sgemm::naive::col::launch), 0, 0, 0,  \
               0, 0, 0, 1)                                                     \
  SGEMM_KERNEL(study, smem_tiling, smem_tiling, (sgemm::smem_tiling::launch),  \
               32, 32, 32, 0, 0, 0, 1)                                         \
  SGEMM_KERNEL(study, blocktiling_1d, blocktiling_1d_v0_64x64x8_8,             \
               (sgemm::blocktiling_1d::v0::launch<64, 64, 8, 8, 4>), 64, 64,   \
               8, 0, 0, 0, 1)                                                  \
  SGEMM_KERNEL(study, blocktiling_1d, blocktiling_1d_v1_64x64x8_8,             \
               (sgemm::blocktiling_1d::v1::launch<64, 64, 8, 8, 4>), 64, 64,   \
               8, 0, 0, 0, 1)                                                  \
  SGEMM_KERNEL(study, blocktiling_1d, blocktiling_1d_v2_64x64x8_8,             \
               (sgemm::blocktiling_1d::v2::launch<64, 64, 8, 8, 4>), 64, 64,   \
               8, 0, 0, 0, 1)                                                  \
  SGEMM_KERNEL(study, blocktiling_1d, blocktiling_1d_v3_64x64x8_8,             \
               (sgemm::blocktiling_1d::v3::launch<64, 64, 8, 8, 4>), 64, 64,   \
               8, 0, 0, 0, 1)                                                  \
  SGEMM_KERNEL(study, blocktiling_1d, blocktiling_1d_v2_64x64x8_16,            \
               (sgemm::blocktiling_1d::v2::launch<64, 64, 8, 16, 4>), 64, 64,  \
               8, 0, 0, 0, 1)                                                  \
  SGEMM_KERNEL(study, blocktiling_1d, blocktiling_1d_v2_128x128x8_64,          \
               (sgemm::blocktiling_1d::v2::launch<128, 128, 8, 64, 4>), 128,   \
               128, 8, 0, 0, 0, 1)                                             \
  SGEMM_KERNEL(study, blocktiling_1d, blocktiling_1d_v2_128x128x16_64,         \
               (sgemm::blocktiling_1d::v2::launch<128, 128, 16, 64, 4>), 128,  \
               128, 16, 0, 0, 0, 1)                                            \
  SGEMM_KERNEL(study, blocktiling_2d, blocktiling_2d_128x128x8_8x8,            \
               (sgemm::blocktiling_2d::launch<128, 128, 8, 8, 8, 4>), 128,     \
               128, 8, 0, 0, 0, 1)                                             \
  SGEMM_KERNEL(study, blocktiling_2d, blocktiling_2d_128x128x16_8x8,           \
               (sgemm::blocktiling_2d::launch<128, 128, 16, 8, 8, 4>), 128,    \
               128, 16, 0, 0, 0, 1)                                            \
  SGEMM_KERNEL(study, blocktiling_2d_vec4,                                     \
               blocktiling_2d_vec4_v0_128x128x8_8x8,                           \
               (sgemm::blocktiling_2d_vec4::v0::launch<128, 128, 8, 8, 8, 4>), \
               128, 128, 8, 0, 0, 0, 1)                                        \
  SGEMM_KERNEL(                                                                \
      study, blocktiling_2d_vec4, blocktiling_2d_vec4_v0_128x128x16_8x8,       \
      (sgemm::blocktiling_2d_vec4::v0::launch<128, 128, 16, 8, 8, 4>), 128,    \
      128, 16, 0, 0, 0, 1)                                                     \
  SGEMM_KERNEL(study, blocktiling_2d_vec4,                                     \
               blocktiling_2d_vec4_v1_128x128x8_8x8,                           \
               (sgemm::blocktiling_2d_vec4::v1::launch<128, 128, 8, 8, 8, 4>), \
               128, 128, 8, 0, 0, 0, 1)                                        \
  SGEMM_KERNEL(                                                                \
      study, blocktiling_2d_vec4, blocktiling_2d_vec4_v1_128x128x16_8x8,       \
      (sgemm::blocktiling_2d_vec4::v1::launch<128, 128, 16, 8, 8, 4>), 128,    \
      128, 8, 0, 0, 0, 1)                                                      \
  SGEMM_KERNEL(study, warptiling, warptiling_v0_128x128x8_32x64x8,             \
               (sgemm::warptiling::v0::launch<128, 128, 8, 32, 64, 8>), 128,   \
               128, 8, 32, 64, 8, 1)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v0_128x128x16_32x64x16,           \
               (sgemm::warptiling::v0::launch<128, 128, 16, 32, 64, 16>), 128, \
               128, 16, 32, 64, 16, 1)                                         \
  SGEMM_KERNEL(study, warptiling, warptiling_v1_128x128x8_32x64x8,             \
               (sgemm::warptiling::v1::launch<128, 128, 8, 32, 64, 8>), 128,   \
               128, 8, 32, 64, 8, 1)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v2_128x128x8_32x64x8,             \
               (sgemm::warptiling::v2::launch<128, 128, 8, 32, 64, 8>), 128,   \
               128, 8, 32, 64, 8, 2)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v2_vec4_128x128x8_32x64x8,        \
               (sgemm::warptiling::v2::vec4::launch<128, 128, 8, 32, 64, 8>),  \
               128, 128, 8, 32, 64, 8, 2)                                      \
  SGEMM_KERNEL(study, warptiling, warptiling_v3_128x128x8_32x64x8,             \
               (sgemm::warptiling::v3::launch<128, 128, 8, 32, 64, 8>), 128,   \
               128, 8, 32, 64, 8, 2)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v3_vec4_128x128x8_32x64x8,        \
               (sgemm::warptiling::v3::vec4::launch<128, 128, 8, 32, 64, 8>),  \
               128, 128, 8, 32, 64, 8, 2)                                      \
  SGEMM_KERNEL(study, warptiling, warptiling_v4_128x128x8_32x64x8,             \
               (sgemm::warptiling::v4::launch<128, 128, 8, 32, 64, 8>), 128,   \
               128, 8, 32, 64, 8, 2)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v4_vec4_128x128x8_32x64x8,        \
               (sgemm::warptiling::v4::vec4::launch<128, 128, 8, 32, 64, 8>),  \
               128, 128, 8, 32, 64, 8, 2)                                      \
  SGEMM_KERNEL(study, warptiling, warptiling_v5_128x128x8_32x64x8,             \
               (sgemm::warptiling::v5::launch<128, 128, 8, 32, 64, 8>), 128,   \
               128, 8, 32, 64, 8, 2)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v5_vec4_128x128x8_32x64x8,        \
               (sgemm::warptiling::v5::vec4::launch<128, 128, 8, 32, 64, 8>),  \
               128, 128, 8, 32, 64, 8, 2)                                      \
  SGEMM_KERNEL(study, warptiling, warptiling_v6_128x128x8_32x64x8,             \
               (sgemm::warptiling::v6::launch<128, 128, 8, 32, 64, 8>), 128,   \
               128, 8, 32, 64, 8, 2)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v6_vec4_128x128x8_32x64x8,        \
               (sgemm::warptiling::v6::vec4::launch<128, 128, 8, 32, 64, 8>),  \
               128, 128, 8, 32, 64, 8, 2)                                      \
  SGEMM_KERNEL(study, warptiling, warptiling_v7_128x128x8_32x64x8,             \
               (sgemm::warptiling::v7::launch<128, 128, 8, 32, 64, 8>), 128,   \
               128, 8, 32, 64, 8, 2)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v7_vec4_128x128x8_32x64x8,        \
               (sgemm::warptiling::v7::vec4::launch<128, 128, 8, 32, 64, 8>),  \
               128, 128, 8, 32, 64, 8, 2)                                      \
  SGEMM_KERNEL(study, warptiling, warptiling_v8_128x128x8_32x64x8,             \
               (sgemm::warptiling::v8::launch<128, 128, 8, 32, 64, 8>), 128,   \
               128, 8, 32, 64, 8, 2)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v8_vec4_128x128x8_32x64x8,        \
               (sgemm::warptiling::v8::vec4::launch<128, 128, 8, 32, 64, 8>),  \
               128, 128, 8, 32, 64, 8, 2)                                      \
  SGEMM_KERNEL(study, warptiling, warptiling_v9_128x128x8_32x64x8,             \
               (sgemm::warptiling::v9::launch<128, 128, 8, 32, 64, 8>), 128,   \
               128, 8, 32, 64, 8, 2)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v10_128x128x8_32x64x8,            \
               (sgemm::warptiling::v10::launch<128, 128, 8, 32, 64, 8>), 128,  \
               128, 8, 32, 64, 8, 2)                                           \
  SGEMM_KERNEL(study, warptiling, warptiling_v11_128x128x8_32x64x8,            \
               (sgemm::warptiling::v11::launch<128, 128, 8, 32, 64, 8>), 128,  \
               128, 8, 32, 64, 8, 2)

struct KernelMetadata {
  char const* name;
  char const* group;
  char const* kind;
  int threadblock_m{};
  int threadblock_n{};
  int threadblock_k{};
  int warp_m{};
  int warp_n{};
  int warp_k{};
  int stages{};
};

#define SGEMM_METADATA(kind, group, name, launch_expr, tb_m, tb_n, tb_k, \
                       warp_m, warp_n, warp_k, stages)                   \
  KernelMetadata{#name, #group, #kind,  tb_m,   tb_n,                    \
                 tb_k,  warp_m, warp_n, warp_k, stages},

constexpr auto kKernelConfig =
    std::array{SGEMM_FOR_EACH_KERNEL(SGEMM_METADATA)};

#undef SGEMM_METADATA

inline void check_cuda_matrix(torch::Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a 2D tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name,
              " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_sgemm_args(torch::Tensor const& input,
                             torch::Tensor const& other,
                             torch::Tensor const& out) {
  check_cuda_matrix(input, "input");
  check_cuda_matrix(other, "other");
  check_cuda_matrix(out, "out");

  TORCH_CHECK(input.device() == other.device(),
              "input and other must be on the same CUDA device");
  TORCH_CHECK(input.device() == out.device(),
              "input and out must be on the same CUDA device");
  TORCH_CHECK(input.size(1) == other.size(0),
              "matmul shape mismatch: input is (", input.size(0), ", ",
              input.size(1), "), other is (", other.size(0), ", ",
              other.size(1), ")");
  TORCH_CHECK(out.size(0) == input.size(0) && out.size(1) == other.size(1),
              "out must have shape (", input.size(0), ", ", other.size(1), ")");
}

template <SgemmLauncher Launch>
void dispatch_sgemm(torch::Tensor const& input, torch::Tensor const& other,
                    torch::Tensor const& out) {
  check_sgemm_args(input, other, out);

  int const M = static_cast<int>(input.size(0));
  int const K = static_cast<int>(input.size(1));
  int const N = static_cast<int>(other.size(1));

  Launch(out.data_ptr<float>(), input.data_ptr<float>(),
         other.data_ptr<float>(), M, N, K);
}

template <SgemmLauncher Launch>
void bind_sgemm(py::module_& m, char const* name) {
  m.def(name, &dispatch_sgemm<Launch>, py::arg("input"), py::arg("other"),
        py::arg("out"));
}

py::list get_kernel_list() {
  py::list kernels;

  {
    py::dict entry;
    entry["name"] = "torch_matmul";
    entry["group"] = "torch";
    entry["kind"] = "reference";
    entry["threadblock_m"] = 0;
    entry["threadblock_n"] = 0;
    entry["threadblock_k"] = 0;
    entry["warp_m"] = 0;
    entry["warp_n"] = 0;
    entry["warp_k"] = 0;
    entry["stages"] = 0;
    kernels.append(entry);
  }

  for (auto const& kernel_config : kKernelConfig) {
    py::dict entry;
    entry["name"] = kernel_config.name;
    entry["group"] = kernel_config.group;
    entry["kind"] = kernel_config.kind;
    entry["threadblock_m"] = kernel_config.threadblock_m;
    entry["threadblock_n"] = kernel_config.threadblock_n;
    entry["threadblock_k"] = kernel_config.threadblock_k;
    entry["warp_m"] = kernel_config.warp_m;
    entry["warp_n"] = kernel_config.warp_n;
    entry["warp_k"] = kernel_config.warp_k;
    entry["stages"] = kernel_config.stages;
    kernels.append(entry);
  }

  return kernels;
}

py::dict get_group_list() {
  py::dict groups;

  for (auto const& kernel_config : kKernelConfig) {
    py::str key(kernel_config.group);
    if (!groups.attr("__contains__")(key).cast<bool>()) {
      groups[key] = py::list();
    }
    groups[key].cast<py::list>().append(kernel_config.name);
  }

  return groups;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define SGEMM_BIND(kind, group, name, launch_expr, tb_m, tb_n, tb_k, warp_m, \
                   warp_n, warp_k, stages)                                   \
  bind_sgemm<&launch_expr>(m, #name);

  SGEMM_FOR_EACH_KERNEL(SGEMM_BIND)
#undef SGEMM_BIND

  m.def("get_kernel_list", &get_kernel_list);
  m.def("get_group_list", &get_group_list);
}
