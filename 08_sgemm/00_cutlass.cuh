#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

namespace sgemm {
namespace cutlass {

inline void check_status(::cutlass::Status status, char const* context) {
  if (status != ::cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cutlassGetStatusString(status));
  }
}

template <typename Gemm>
inline void run_gemm(typename Gemm::Arguments const& arguments) {
  Gemm gemm_op;

  size_t const workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;

  if (workspace_size > 0) {
    cudaError_t const error = cudaMalloc(&workspace, workspace_size);
    if (error != cudaSuccess) {
      throw std::runtime_error("cutlass cudaMalloc workspace failed");
    }
  }

  try {
    check_status(gemm_op.can_implement(arguments), "cutlass can_implement");
    check_status(gemm_op.initialize(arguments, workspace),
                 "cutlass initialize");
    check_status(gemm_op(), "cutlass run");
  } catch (...) {
    if (workspace) {
      cudaFree(workspace);
    }
    throw;
  }

  if (workspace) {
    cudaFree(workspace);
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K, int STAGES>
struct KernelTraits {
  using Element = float;
  using LayoutA = ::cutlass::layout::RowMajor;
  using LayoutB = ::cutlass::layout::RowMajor;
  using LayoutC = ::cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using OperatorClass = ::cutlass::arch::OpClassSimt;
  using ArchTag = ::cutlass::arch::Sm80;

  using ThreadblockShape =
      ::cutlass::gemm::GemmShape<BLOCK_M, BLOCK_N, BLOCK_K>;
  using WarpShape = ::cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>;
  using InstructionShape = ::cutlass::gemm::GemmShape<1, 1, 1>;
  using ThreadblockSwizzle =
      ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp =
      ::cutlass::epilogue::thread::LinearCombination<Element,
                                                     1,  // align1
                                                     ElementAccumulator,
                                                     ElementAccumulator>;

  static constexpr int kStages = STAGES;
  static constexpr int kAlignmentA = 1;
  static constexpr int kAlignmentB = 1;

  using Gemm = ::cutlass::gemm::device::Gemm<
      Element, LayoutA, Element, LayoutB, Element, LayoutC, ElementAccumulator,
      OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
      EpilogueOp, ThreadblockSwizzle, kStages, kAlignmentA, kAlignmentB>;
};

template <typename Traits>
inline void launch_with_source(float* d_ptr, float const* c_ptr,
                               float const* a_ptr, float const* b_ptr,
                               int const M, int const N, int const K,
                               float const alpha = 1.0f,
                               float const beta = 0.0f) {
  typename Traits::Gemm::Arguments arguments{
      {M, N, K},  {a_ptr, K},    {b_ptr, N}, {c_ptr, N},
      {d_ptr, N}, {alpha, beta}, 1};

  run_gemm<typename Traits::Gemm>(arguments);
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K, int STAGES>
inline void launch(float* c_ptr, float const* a_ptr, float const* b_ptr,
                   int const M, int const N, int const K) {
  using Traits =
      KernelTraits<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGES>;
  launch_with_source<Traits>(c_ptr, c_ptr, a_ptr, b_ptr, M, N, K, 1.0f, 0.0f);
}

}  // namespace cutlass
}  // namespace sgemm
