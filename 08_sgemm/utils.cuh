#pragma once

namespace sgemm {
namespace utils {

template <typename T>
struct sizeof_bits;

template <>
struct sizeof_bits<float> {
  static constexpr int value = 32;
};

template <typename T>
__host__ __device__ constexpr T max_value(T lhs, T rhs) {
  return lhs > rhs ? lhs : rhs;
}

__host__ __device__ constexpr int ceil_div(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

__device__ __forceinline__ float4 load_float4(float const* ptr) {
  return *reinterpret_cast<float4 const*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 value) {
  *reinterpret_cast<float4*>(ptr) = value;
}

__device__ __forceinline__ void unpack_float4(float* ptr, float4 value) {
  ptr[0] = value.x;
  ptr[1] = value.y;
  ptr[2] = value.z;
  ptr[3] = value.w;
}

__device__ __forceinline__ float4 make_float4_from_array(float const* values) {
  return make_float4(values[0], values[1], values[2], values[3]);
}

// Load an A tile from global memory to shared memory with transpose
template <int NUM_THREADS, int TILE_M, int TILE_K, int TILE_STRIDE>
__forceinline__ __device__ void load_a_threadblock_tile(float* tb_tile_a,
                                                        float const* a_ptr,
                                                        int lda) {
  constexpr int num_tile_row_iters = (TILE_M * TILE_K) / NUM_THREADS;
  constexpr int row_offsets_per_iter = NUM_THREADS / TILE_K;

  int const tile_row = threadIdx.x / TILE_K;
  int const tile_col = threadIdx.x % TILE_K;

#pragma unroll
  for (int row_iter = 0; row_iter < num_tile_row_iters; ++row_iter) {
    int const row = row_offsets_per_iter * row_iter + tile_row;
    int const col = tile_col;

    float const* src = a_ptr + row * lda + col;
    float* dst = tb_tile_a + col * TILE_STRIDE + row;

    *dst = *src;
  }
}

// Load a B tile from global memory to shared memory without transpose
template <int NUM_THREADS, int TILE_K, int TILE_N, int TILE_STRIDE>
__forceinline__ __device__ void load_b_threadblock_tile(float* tb_tile_b,
                                                        float const* b_ptr,
                                                        int ldb) {
  constexpr int num_tile_row_iters = (TILE_K * TILE_N) / NUM_THREADS;
  constexpr int row_offsets_per_iter = NUM_THREADS / TILE_N;

  int const tile_row = threadIdx.x / TILE_N;
  int const tile_col = threadIdx.x % TILE_N;

#pragma unroll
  for (int row_iter = 0; row_iter < num_tile_row_iters; ++row_iter) {
    int const row = row_offsets_per_iter * row_iter + tile_row;
    int const col = tile_col;

    float const* src = b_ptr + row * ldb + col;
    float* dst = tb_tile_b + row * TILE_STRIDE + col;

    *dst = *src;
  }
}

// Load an A tile from global memory to shared memory with transpose
// using float4 vectorized memory access
template <int NUM_THREADS, int TILE_M, int TILE_K, int TILE_STRIDE>
__forceinline__ __device__ void load_a_threadblock_tile_vec4(float* tb_tile_a,
                                                             float const* a_ptr,
                                                             int lda) {
  constexpr int vec_width = 4;
  static_assert(TILE_K % vec_width == 0);

  constexpr int num_tile_row_iters =
      (TILE_M * TILE_K) / (NUM_THREADS * vec_width);
  constexpr int threads_per_tile_row = TILE_K / vec_width;
  constexpr int row_offsets_per_iter = NUM_THREADS / threads_per_tile_row;

  int const tile_row = threadIdx.x / threads_per_tile_row;
  int const tile_col_vec = threadIdx.x % threads_per_tile_row;

#pragma unroll
  for (int row_iter = 0; row_iter < num_tile_row_iters; ++row_iter) {
    int const row = row_offsets_per_iter * row_iter + tile_row;
    int const col = vec_width * tile_col_vec;

    float4 const value = utils::load_float4(&a_ptr[row * lda + col]);
    tb_tile_a[col * TILE_STRIDE + row] = value.x;
    tb_tile_a[(col + 1) * TILE_STRIDE + row] = value.y;
    tb_tile_a[(col + 2) * TILE_STRIDE + row] = value.z;
    tb_tile_a[(col + 3) * TILE_STRIDE + row] = value.w;
  }
}

// Load a B tile from global memory to shared memory without transpose
// using float4 vectorized memory access
template <int NUM_THREADS, int TILE_K, int TILE_N, int TILE_STRIDE>
__forceinline__ __device__ void load_b_threadblock_tile_vec4(
    float* smem_b, float const* gmem_b, int gmem_b_ld) {
  constexpr int vec_width = 4;
  static_assert(TILE_N % vec_width == 0);

  constexpr int num_tile_row_iters =
      (TILE_K * TILE_N) / (NUM_THREADS * vec_width);
  constexpr int threads_per_tile_row = TILE_N / vec_width;
  constexpr int row_offsets_per_iter = NUM_THREADS / threads_per_tile_row;

  int const tile_row = threadIdx.x / threads_per_tile_row;
  int const tile_col_vec = threadIdx.x % threads_per_tile_row;

#pragma unroll
  for (int row_iter = 0; row_iter < num_tile_row_iters; ++row_iter) {
    int const row = row_offsets_per_iter * row_iter + tile_row;
    int const col = vec_width * tile_col_vec;

    float4 const value = utils::load_float4(&gmem_b[row * gmem_b_ld + col]);
    utils::store_float4(&smem_b[row * TILE_STRIDE + col], value);
  }
}

__host__ __device__ constexpr int simt_transpose_padding(int threads,
                                                         int crosswise,
                                                         int size_in_bits) {
  return (size_in_bits >= 32 ? threads / crosswise / (size_in_bits / 32)
                             : threads / crosswise * (32 / size_in_bits));
}

}  // namespace utils
}  // namespace sgemm