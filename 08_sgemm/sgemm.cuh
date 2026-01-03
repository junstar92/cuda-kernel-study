#pragma once

void launch_sgemm_naive(float* c_ptr, float const* a_ptr, float const* b_ptr,
                        int const M, int const N, int const K);

template <int BLOCK_SZE>
void launch_sgemm_smem_naive(float* c_ptr, float const* a_ptr,
                             float const* b_ptr, int const M, int const N,
                             int const K);

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M,
          int A_SMEM_PADDING = 4>
void launch_sgemm_1d_blocktiling(float* c_ptr, float const* a_ptr,
                                 float const* b_ptr, int const M, int const N,
                                 int const K);

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
          int A_SMEM_PADDING = 4>
void launch_sgemm_2d_blocktiling(float* c_ptr, float const* a_ptr,
                                 float const* b_ptr, int const M, int const N,
                                 int const K);

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
          int A_SMEM_PADDING = 4>
void launch_sgemm_2d_blocktiling_vec(float* c_ptr, float const* a_ptr,
                                     float const* b_ptr, int const M,
                                     int const N, int const K);

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N,
          int WARP_K, int A_SMEM_PADDING = 4>
void launch_sgemm_warptiling(float* c_ptr, float const* a_ptr,
                             float const* b_ptr, int const M, int const N,
                             int const K);
