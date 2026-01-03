#include <torch/extension.h>

#include "00_cutlass.cuh"
#include "01_naive.cuh"
#include "02_smem_tiling.cuh"
#include "03_blocktiling_1d.cuh"
#include "04_blocktiling_2d.cuh"
#include "05_blocktiling_2d_vec4.cuh"
#include "06_warptiling.cuh"

namespace py = pybind11;

using SgemmLauncher = void (*)(float*, float const*, float const*, int, int,
                               int);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_sgemm<&sgemm::cutlass::launch<128, 128, 8, 32, 64, 8, 2>>(
      m, "sgemm_cutlass_simt_128x128x8_32x64x8_2stage");
  bind_sgemm<&sgemm::cutlass::launch<128, 128, 8, 64, 32, 8, 2>>(
      m, "sgemm_cutlass_simt_128x128x8_64x32x8_2stage");
  bind_sgemm<&sgemm::cutlass::launch_universal<128, 256, 8, 64, 64, 8, 4, 8>>(
      m, "sgemm_cutlass_universal_simt_128x256_8x4");
  bind_sgemm<&sgemm::naive::row::launch>(m, "sgemm_naive_row");
  bind_sgemm<&sgemm::naive::col::launch>(m, "sgemm_naive_col");
  bind_sgemm<&sgemm::smem_tiling::launch>(m, "sgemm_smem_tiling");
  bind_sgemm<&sgemm::blocktiling_1d::launch<64, 64, 8, 8, 0>>(
      m, "sgemm_blocktiling_1d_64x64x8_8_no_pad");
  bind_sgemm<&sgemm::blocktiling_1d::launch<64, 64, 8, 8, 4>>(
      m, "sgemm_blocktiling_1d_64x64x8_8_pad4");
  bind_sgemm<&sgemm::blocktiling_1d::launch<64, 64, 8, 16, 4>>(
      m, "sgemm_blocktiling_1d_64x64x8_16_pad4");
  bind_sgemm<&sgemm::blocktiling_1d::launch<128, 128, 8, 64, 4>>(
      m, "sgemm_blocktiling_1d_128x128x8_64_pad4");
  bind_sgemm<&sgemm::blocktiling_1d::launch<128, 128, 16, 64, 4>>(
      m, "sgemm_blocktiling_1d_128x128x16_64_pad4");
  bind_sgemm<&sgemm::blocktiling_2d::launch<128, 128, 8, 8, 8, 4>>(
      m, "sgemm_blocktiling_2d_128x128x8_8x8_pad4");
  bind_sgemm<&sgemm::blocktiling_2d::launch<128, 128, 16, 8, 8, 4>>(
      m, "sgemm_blocktiling_2d_128x128x16_8x8_pad4");
  bind_sgemm<&sgemm::blocktiling_2d_vec4::launch<128, 128, 8, 8, 8, 4>>(
      m, "sgemm_blocktiling_2d_vec4_128x128x8_8x8_pad4");
  bind_sgemm<&sgemm::blocktiling_2d_vec4::launch<128, 128, 16, 8, 8, 4>>(
      m, "sgemm_blocktiling_2d_vec4_128x128x16_8x8_pad4");
  bind_sgemm<&sgemm::warptiling::v0::launch<128, 128, 8, 32, 64, 8>>(
      m, "sgemm_warptiling_v0_128x128x8_32x64x8");
  bind_sgemm<&sgemm::warptiling::v0::launch<128, 128, 8, 64, 32, 8>>(
      m, "sgemm_warptiling_v0_128x128x8_64x32x8");
  bind_sgemm<&sgemm::warptiling::v1::launch<128, 128, 8, 32, 64, 8>>(
      m, "sgemm_warptiling_v1_128x128x8_32x64x8");
  bind_sgemm<&sgemm::warptiling::v1::launch<128, 128, 8, 64, 32, 8>>(
      m, "sgemm_warptiling_v1_128x128x8_64x32x8");
  bind_sgemm<&sgemm::warptiling::v2::launch<128, 128, 8, 32, 64, 8>>(
      m, "sgemm_warptiling_v2_128x128x8_32x64x8");
  bind_sgemm<&sgemm::warptiling::v2::launch<128, 128, 8, 64, 32, 8>>(
      m, "sgemm_warptiling_v2_128x128x8_64x32x8");
  bind_sgemm<&sgemm::warptiling::v2::launch<128, 128, 16, 32, 64, 16>>(
      m, "sgemm_warptiling_v2_128x128x16_32x64x16");
  bind_sgemm<&sgemm::warptiling::v3::launch<128, 128, 8, 32, 64, 8>>(
      m, "sgemm_warptiling_v3_128x128x8_32x64x8");
  bind_sgemm<&sgemm::warptiling::v3::launch<128, 128, 8, 64, 32, 8>>(
      m, "sgemm_warptiling_v3_128x128x8_64x32x8");
  bind_sgemm<&sgemm::warptiling::v3::launch<128, 128, 16, 32, 64, 16>>(
      m, "sgemm_warptiling_v3_128x128x16_32x64x16");
  bind_sgemm<&sgemm::warptiling::v4::launch<128, 128, 8, 32, 64, 8>>(
      m, "sgemm_warptiling_v4_128x128x8_32x64x8");
  bind_sgemm<&sgemm::warptiling::v4::launch<128, 128, 8, 64, 32, 8>>(
      m, "sgemm_warptiling_v4_128x128x8_64x32x8");
  bind_sgemm<&sgemm::warptiling::v4::launch<128, 128, 16, 32, 64, 16>>(
      m, "sgemm_warptiling_v4_128x128x16_32x64x16");
  bind_sgemm<&sgemm::warptiling::v5::launch<128, 128, 8, 32, 64, 8>>(
      m, "sgemm_warptiling_v5_128x128x8_32x64x8");
  bind_sgemm<&sgemm::warptiling::v5::launch<128, 128, 8, 64, 32, 8>>(
      m, "sgemm_warptiling_v5_128x128x8_64x32x8");
  bind_sgemm<&sgemm::warptiling::v5::launch<128, 128, 16, 32, 64, 16>>(
      m, "sgemm_warptiling_v5_128x128x16_32x64x16");
}
