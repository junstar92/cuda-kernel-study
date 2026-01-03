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
  constexpr int num_iters = (TILE_M * TILE_K) / NUM_THREADS;
  constexpr int row_advance_per_iter = NUM_THREADS / TILE_K;
  constexpr int col_advance_per_iter = NUM_THREADS % TILE_K;

  int const src_advance_offset =
      row_advance_per_iter * lda + col_advance_per_iter;
  int const dst_advance_offset =
      col_advance_per_iter * TILE_STRIDE + row_advance_per_iter;
  int const base_row = threadIdx.x / TILE_K;
  int const base_col = threadIdx.x % TILE_K;
  int src_offset = base_row * lda + base_col;
  int dst_offset = base_col * TILE_STRIDE + base_row;

#pragma unroll
  for (int iter = 0; iter < num_iters; ++iter) {
    tb_tile_a[dst_offset] = a_ptr[src_offset];
    src_offset += src_advance_offset;
    dst_offset += dst_advance_offset;
  }
}

// Load a B tile from global memory to shared memory without transpose
template <int NUM_THREADS, int TILE_K, int TILE_N, int TILE_STRIDE>
__forceinline__ __device__ void load_b_threadblock_tile(float* tb_tile_b,
                                                        float const* b_ptr,
                                                        int ldb) {
  constexpr int num_iters = (TILE_K * TILE_N) / NUM_THREADS;
  constexpr int row_advance_per_iter = NUM_THREADS / TILE_N;
  constexpr int col_advance_per_iter = NUM_THREADS % TILE_N;

  int const src_advance_offset =
      row_advance_per_iter * ldb + col_advance_per_iter;
  int const dst_advance_offset =
      row_advance_per_iter * TILE_STRIDE + col_advance_per_iter;
  int const base_row = threadIdx.x / TILE_N;
  int const base_col = threadIdx.x % TILE_N;
  int src_offset = base_row * ldb + base_col;
  int dst_offset = base_row * TILE_STRIDE + base_col;

#pragma unroll
  for (int iter = 0; iter < num_iters; ++iter) {
    tb_tile_b[dst_offset] = b_ptr[src_offset];
    src_offset += src_advance_offset;
    dst_offset += dst_advance_offset;
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

  constexpr int vec4s_per_row = TILE_K / vec_width;
  constexpr int num_iters = (TILE_M * vec4s_per_row) / NUM_THREADS;
  constexpr int row_advance_per_iter = NUM_THREADS / vec4s_per_row;
  constexpr int packet_advance_per_iter = NUM_THREADS % vec4s_per_row;

  int const src_advance_offset =
      row_advance_per_iter * lda + packet_advance_per_iter * vec_width;
  int const dst_advance_offset =
      packet_advance_per_iter * vec_width * TILE_STRIDE + row_advance_per_iter;

  int const base_row = threadIdx.x / vec4s_per_row;
  int const base_col = (threadIdx.x % vec4s_per_row) * vec_width;

  int src_offset = base_row * lda + base_col;
  int dst_offset = base_col * TILE_STRIDE + base_row;

#pragma unroll
  for (int iter = 0; iter < num_iters; ++iter) {
    float4 value = utils::load_float4(a_ptr + src_offset);

    tb_tile_a[dst_offset + 0 * TILE_STRIDE] = value.x;
    tb_tile_a[dst_offset + 1 * TILE_STRIDE] = value.y;
    tb_tile_a[dst_offset + 2 * TILE_STRIDE] = value.z;
    tb_tile_a[dst_offset + 3 * TILE_STRIDE] = value.w;

    src_offset += src_advance_offset;
    dst_offset += dst_advance_offset;
  }
}

// Load a B tile from global memory to shared memory without transpose
// using float4 vectorized memory access
template <int NUM_THREADS, int TILE_K, int TILE_N, int TILE_STRIDE>
__forceinline__ __device__ void load_b_threadblock_tile_vec4(float* tb_tile_b,
                                                             float const* b_ptr,
                                                             int ldb) {
  constexpr int vec_width = 4;
  static_assert(TILE_N % vec_width == 0);

  constexpr int vec4s_per_row = TILE_N / vec_width;
  constexpr int num_iters = (TILE_K * vec4s_per_row) / NUM_THREADS;
  constexpr int row_advance_per_iter = NUM_THREADS / vec4s_per_row;
  constexpr int packet_advance_per_iter = NUM_THREADS % vec4s_per_row;

  int const src_advance_offset =
      row_advance_per_iter * ldb + packet_advance_per_iter * vec_width;
  int const dst_advance_offset =
      row_advance_per_iter * TILE_STRIDE + packet_advance_per_iter * vec_width;

  int const base_row = threadIdx.x / vec4s_per_row;
  int const base_col = (threadIdx.x % vec4s_per_row) * vec_width;

  int src_offset = base_row * ldb + base_col;
  int dst_offset = base_row * TILE_STRIDE + base_col;

#pragma unroll
  for (int iter = 0; iter < num_iters; ++iter) {
    float4 value = utils::load_float4(b_ptr + src_offset);
    utils::store_float4(tb_tile_b + dst_offset, value);

    src_offset += src_advance_offset;
    dst_offset += dst_advance_offset;
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