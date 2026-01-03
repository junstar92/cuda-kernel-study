#pragma once

#define __HOST_DEVICE__ __forceinline__ __host__ __device__
#define __DEVICE__ __forceinline__ __device__

namespace sgemm {
namespace utils {

template <typename T>
struct sizeof_bits;

template <>
struct sizeof_bits<float> {
  static constexpr int value = 32;
};

template <typename T>
__HOST_DEVICE__ constexpr T max_value(T lhs, T rhs) {
  return lhs > rhs ? lhs : rhs;
}

__HOST_DEVICE__ constexpr int ceil_div(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <int GroupRows>
struct GroupedThreadblockSwizzle {
  static_assert(GroupRows > 0, "Threadblock swizzle group size must be > 0.");

  __HOST_DEVICE__ static constexpr int effective_group_rows(int tile_rows) {
    return tile_rows < GroupRows ? tile_rows : GroupRows;
  }

  __DEVICE__ static void get_tile_offset(int& tile_m_idx, int& tile_n_idx) {
    int const grid_tiles_n = int(gridDim.x);
    int const grid_tiles_m = int(gridDim.y);
    int const group_rows = effective_group_rows(grid_tiles_m);

    int const linear_idx = int(blockIdx.x) + int(blockIdx.y) * grid_tiles_n;
    int const full_group_count = grid_tiles_m / group_rows;
    int const tail_rows = grid_tiles_m - full_group_count * group_rows;
    int const full_group_tiles = full_group_count * group_rows * grid_tiles_n;

    if (linear_idx < full_group_tiles) {
      int const tiles_per_group = group_rows * grid_tiles_n;
      int const group_idx = linear_idx / tiles_per_group;
      int const residual = linear_idx - group_idx * tiles_per_group;
      tile_n_idx = residual / group_rows;
      tile_m_idx = group_idx * group_rows + (residual - tile_n_idx * group_rows);
      return;
    }

    if (tail_rows > 0) {
      int const residual = linear_idx - full_group_tiles;
      tile_n_idx = residual / tail_rows;
      tile_m_idx = full_group_count * group_rows +
                   (residual - tile_n_idx * tail_rows);
      return;
    }

    tile_m_idx = int(blockIdx.y);
    tile_n_idx = int(blockIdx.x);
  }
};

__DEVICE__ float4 load_float4(float const* ptr) {
  return *reinterpret_cast<float4 const*>(ptr);
}

__DEVICE__ void store_float4(float* ptr, float4 value) {
  *reinterpret_cast<float4*>(ptr) = value;
}

__DEVICE__ void unpack_float4(float* ptr, float4 value) {
  ptr[0] = value.x;
  ptr[1] = value.y;
  ptr[2] = value.z;
  ptr[3] = value.w;
}

__DEVICE__ float4 make_float4_from_array(float const* values) {
  return make_float4(values[0], values[1], values[2], values[3]);
}

template <typename AccessType, int Bytes = sizeof(AccessType)>
struct global_load;

template <typename AccessType>
struct global_load<AccessType, 4> {
  __DEVICE__ global_load(AccessType& dst, void const* ptr, bool pred = true) {
    unsigned& data = reinterpret_cast<unsigned&>(dst);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  mov.b32 %0, %3;\n"
        "  @p ld.global.L2::128B.u32 %0, [%1];\n"
        "}\n"
        : "=r"(data)
        : "l"(ptr), "r"((int)pred), "r"(data));
  }
};

template <typename AccessType>
struct global_load<AccessType, 8> {
  __DEVICE__ global_load(AccessType& dst, void const* ptr, bool pred = true) {
    uint2& data = reinterpret_cast<uint2&>(dst);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %3, 0;\n"
        "  mov.b32 %0, %4;\n"
        "  mov.b32 %1, %5;\n"
        "  @p ld.global.L2::128B.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y)
        : "l"(ptr), "r"((int)pred), "r"(data.x), "r"(data.y));
  }
};

template <typename AccessType>
struct global_load<AccessType, 16> {
  __DEVICE__ global_load(AccessType& dst, void const* ptr, bool pred = true) {
    uint4& data = reinterpret_cast<uint4&>(dst);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
        "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred), "r"(data.x), "r"(data.y), "r"(data.z),
          "r"(data.w));
  }
};

template <int Size>
struct FloatArray {
  float data[Size];

  __HOST_DEVICE__ float& operator[](int idx) { return data[idx]; }
  __HOST_DEVICE__ float const& operator[](int idx) const { return data[idx]; }

  FloatArray() = default;
  FloatArray(FloatArray const&) = default;
  FloatArray& operator=(FloatArray const&) = default;
  FloatArray(FloatArray&&) noexcept = default;
  FloatArray& operator=(FloatArray&&) noexcept = default;
  ~FloatArray() = default;

  static constexpr int size() { return Size; }
  __HOST_DEVICE__ FloatArray(float x) {
    for (int i = 0; i < Size; i++) {
      data[i] = x;
    }
  }
};

template <int Elements, int Alignment = Elements * int(sizeof(float))>
struct alignas(Alignment) AlignedFloatArray {
  float values[Elements];

  __DEVICE__ float& operator[](int idx) { return values[idx]; }
  __DEVICE__ float const& operator[](int idx) const { return values[idx]; }
};

template <int Contiguous, int Strided, int Threads, int ElementPerAccess = 1>
struct PitchLinearStripminedThreadMap {
  static constexpr int kContiguous = Contiguous;
  static constexpr int kStrided = Strided;
  static constexpr int kThreads = Threads;
  static constexpr int kElementsPerAccess = ElementPerAccess;
  static constexpr int kShapeVecContiguous = kContiguous / kElementsPerAccess;

  static_assert(kContiguous % kElementsPerAccess == 0,
                "Thread map contiguous must be divisible by access size.");
  static_assert(
      (kThreads < kShapeVecContiguous && !(kShapeVecContiguous % kThreads)) ||
          (!(kThreads % kShapeVecContiguous)),
      "Shape must be divisible by number of iterations of each thread.");

  static constexpr int kIterationsContiguous =
      (kThreads >= kShapeVecContiguous) ? 1 : (kShapeVecContiguous / kThreads);
  static constexpr int kIterationsStrided =
      (kThreads >= kShapeVecContiguous)
          ? ((kStrided + (kThreads / kShapeVecContiguous) - 1) /
             (kThreads / kShapeVecContiguous))
          : kStrided;
  static constexpr int kIterationsCount =
      kIterationsContiguous * kIterationsStrided;
  static constexpr int kDeltaContiguous =
      (kThreads >= kShapeVecContiguous) ? 1 : (kThreads * kElementsPerAccess);
  static constexpr int kDeltaStrided =
      (kThreads >= kShapeVecContiguous) ? (kThreads / kShapeVecContiguous) : 1;

  __DEVICE__ static void initial_offset(int thread_id, int& contiguous,
                                        int& strided) {
    contiguous = (thread_id % kShapeVecContiguous) * kElementsPerAccess;
    strided = thread_id / kShapeVecContiguous;
  }
};

template <typename ThreadMap>
struct TransposePitchLinearThreadMapSimt {
  static constexpr int kContiguous = ThreadMap::kStrided;
  static constexpr int kStrided = ThreadMap::kContiguous;
  static constexpr int kThreads = ThreadMap::kThreads;
  static constexpr int kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static constexpr int kIterationsContiguous = ThreadMap::kIterationsStrided;
  static constexpr int kIterationsStrided = ThreadMap::kIterationsContiguous;
  static constexpr int kIterationsCount = ThreadMap::kIterationsCount;
  static constexpr int kDeltaContiguous = ThreadMap::kDeltaStrided;
  static constexpr int kDeltaStrided = ThreadMap::kDeltaContiguous;

  static_assert(kElementsPerAccess == 1,
                "SIMT transpose only supports scalar access.");
  static_assert(kIterationsStrided == 1);

  __DEVICE__ static void initial_offset(int thread_id, int& contiguous,
                                        int& strided) {
    int base_contiguous = 0;
    int base_strided = 0;
    ThreadMap::initial_offset(thread_id, base_contiguous, base_strided);
    contiguous = base_strided;
    strided = base_contiguous;
  }
};

template <int Contiguous, int Strided, typename ThreadMap, int AdvanceRank>
struct ThreadblockTileIteratorV0 {
  using Fragment =
      FloatArray<ThreadMap::kIterationsCount * ThreadMap::kElementsPerAccess>;
  static constexpr int kContiguous = Contiguous;
  static constexpr int kStrided = Strided;

  float const* pointer;
  int stride;
  int increment_strided;
  int increment_advance;

  __HOST_DEVICE__ ThreadblockTileIteratorV0(float const* ptr, int ld,
                                            int thread_idx) {
    int contiguous = 0;
    int strided = 0;
    ThreadMap::initial_offset(thread_idx, contiguous, strided);
    pointer = ptr + strided * ld + contiguous;
    stride = ld;
    increment_strided = ld * ThreadMap::kDeltaStrided;
    increment_advance = (AdvanceRank == 0 ? kContiguous : kStrided * ld);
  }

  __HOST_DEVICE__ void load(Fragment& frag) {
    float* frag_ptr = reinterpret_cast<float*>(&frag);
    float const* ptr = pointer;

#pragma unroll
    for (int s = 0; s < ThreadMap::kIterationsStrided; s++) {
      float const* access_ptr = ptr;
#pragma unroll
      for (int c = 0; c < ThreadMap::kIterationsContiguous; c++) {
        int idx = c + s * ThreadMap::kIterationsContiguous;
        frag_ptr[idx] = access_ptr[c * ThreadMap::kDeltaContiguous /
                                   ThreadMap::kElementsPerAccess];
      }

      if (s + 1 < ThreadMap::kIterationsStrided) {
        ptr += increment_strided;
      }
    }
  }

  __HOST_DEVICE__ ThreadblockTileIteratorV0& operator++() {
    pointer += increment_advance;
    return *this;
  }

  __HOST_DEVICE__ ThreadblockTileIteratorV0& operator--() {
    pointer -= increment_advance;
    return *this;
  }

  __HOST_DEVICE__ void add_tile_offset(int tile_m, int tile_n) {
    int offset = tile_n * kContiguous + tile_m * kStrided * stride;
    pointer += offset;
  }
};

template <int Contiguous, int Strided, typename ThreadMap, int AdvanceRank>
struct ThreadblockTileIterator {
  using AccessType = AlignedFloatArray<ThreadMap::kElementsPerAccess>;
  using Fragment = AlignedFloatArray<ThreadMap::kIterationsCount *
                                         ThreadMap::kElementsPerAccess,
                                     alignof(AccessType)>;
  static constexpr int kContiguous = Contiguous;
  static constexpr int kStrided = Strided;

  char const* pointer;
  int stride;
  int increment_strided;
  int increment_advance;

  __HOST_DEVICE__ ThreadblockTileIterator(float const* ptr, int ld,
                                          int thread_idx) {
    int contiguous = 0;
    int strided = 0;
    ThreadMap::initial_offset(thread_idx, contiguous, strided);
    pointer = reinterpret_cast<char const*>(ptr + strided * ld + contiguous);
    stride = ld;
    increment_strided = sizeof(float) * ld * ThreadMap::kDeltaStrided;
    increment_advance =
        sizeof(float) * (AdvanceRank == 0 ? kContiguous : kStrided * ld);
  }

  __HOST_DEVICE__ void load(Fragment& frag) {
    AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
    char const* byte_pointer = pointer;

#pragma unroll
    for (int s = 0; s < ThreadMap::kIterationsStrided; s++) {
      AccessType const* access_ptr =
          reinterpret_cast<AccessType const*>(byte_pointer);
#pragma unroll
      for (int c = 0; c < ThreadMap::kIterationsContiguous; c++) {
        int idx = c + s * ThreadMap::kIterationsContiguous;
        frag_ptr[idx] = access_ptr[c * ThreadMap::kDeltaContiguous /
                                   ThreadMap::kElementsPerAccess];
      }

      if (s + 1 < ThreadMap::kIterationsStrided) {
        byte_pointer += increment_strided;
      }
    }
  }

  __HOST_DEVICE__ void load(Fragment& frag, bool pred) {
    AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
    char const* byte_pointer = pointer;

#pragma unroll
    for (int s = 0; s < ThreadMap::kIterationsStrided; s++) {
      AccessType const* access_ptr =
          reinterpret_cast<AccessType const*>(byte_pointer);
#pragma unroll
      for (int c = 0; c < ThreadMap::kIterationsContiguous; c++) {
        int idx = c + s * ThreadMap::kIterationsContiguous;
        int access_idx =
            c * ThreadMap::kDeltaContiguous / ThreadMap::kElementsPerAccess;
        global_load<AccessType>(frag_ptr[idx], access_ptr + access_idx, pred);
      }

      if (s + 1 < ThreadMap::kIterationsStrided) {
        byte_pointer += increment_strided;
      }
    }
  }

  __HOST_DEVICE__ void load_ptx(Fragment& frag) {
    AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
    char const* byte_pointer = pointer;

#pragma unroll
    for (int s = 0; s < ThreadMap::kIterationsStrided; s++) {
      AccessType const* access_ptr =
          reinterpret_cast<AccessType const*>(byte_pointer);
#pragma unroll
      for (int c = 0; c < ThreadMap::kIterationsContiguous; c++) {
        int idx = c + s * ThreadMap::kIterationsContiguous;
        int access_idx =
            c * ThreadMap::kDeltaContiguous / ThreadMap::kElementsPerAccess;
        global_load<AccessType>(frag_ptr[idx], access_ptr + access_idx, true);
      }

      if (s + 1 < ThreadMap::kIterationsStrided) {
        byte_pointer += increment_strided;
      }
    }
  }

  __HOST_DEVICE__ ThreadblockTileIterator& operator++() {
    pointer += increment_advance;
    return *this;
  }

  __HOST_DEVICE__ ThreadblockTileIterator& operator--() {
    pointer -= increment_advance;
    return *this;
  }

  __HOST_DEVICE__ void add_tile_offset(int tile_m, int tile_n) {
    int offset = tile_n * kContiguous + tile_m * kStrided * stride;
    pointer += sizeof(float) * offset;
  }
};

template <int Contiguous, int Strided, typename ThreadMap, int AdvanceRank>
struct SmemTileIteratorV0 {
  using Fragment =
      FloatArray<ThreadMap::kIterationsCount * ThreadMap::kElementsPerAccess>;
  static constexpr int kContiguous = Contiguous;
  static constexpr int kStrided = Strided;

  float* pointer;
  int stride;
  int increment_strided;
  int increment_advance;

  __HOST_DEVICE__ SmemTileIteratorV0(float* ptr, int smem_stride,
                                     int thread_idx) {
    int contiguous = 0;
    int strided = 0;
    ThreadMap::initial_offset(thread_idx, contiguous, strided);
    pointer = ptr + strided * smem_stride + contiguous;
    stride = smem_stride;
    increment_strided = smem_stride * ThreadMap::kDeltaStrided;
    increment_advance =
        (AdvanceRank == 0 ? kContiguous : kStrided * smem_stride);
  }

  __HOST_DEVICE__ void store(Fragment const& frag) {
    float const* frag_ptr = reinterpret_cast<float const*>(&frag);
    float* ptr = pointer;

#pragma unroll
    for (int s = 0; s < ThreadMap::kIterationsStrided; s++) {
      float* access_ptr = reinterpret_cast<float*>(ptr);
#pragma unroll
      for (int c = 0; c < ThreadMap::kIterationsContiguous; c++) {
        int idx = c + s * ThreadMap::kIterationsContiguous;
        access_ptr[c * ThreadMap::kDeltaContiguous /
                   ThreadMap::kElementsPerAccess] = frag_ptr[idx];
      }

      if (s + 1 < ThreadMap::kIterationsStrided) {
        ptr += increment_strided;
      }
    }
  }

  __HOST_DEVICE__ SmemTileIteratorV0& operator++() {
    pointer += increment_advance;
    return *this;
  }

  __HOST_DEVICE__ SmemTileIteratorV0& operator--() {
    pointer -= increment_advance;
    return *this;
  }

  __HOST_DEVICE__ void add_tile_offset(int tile_m, int tile_n) {
    int offset = tile_n * kContiguous + tile_m * kStrided * stride;
    pointer += offset;
  }
};

template <int Contiguous, int Strided, typename ThreadMap, int AdvanceRank>
struct SmemTileIterator {
  using AccessType = AlignedFloatArray<ThreadMap::kElementsPerAccess>;
  using Fragment = AlignedFloatArray<ThreadMap::kIterationsCount *
                                         ThreadMap::kElementsPerAccess,
                                     alignof(AccessType)>;
  static constexpr int kContiguous = Contiguous;
  static constexpr int kStrided = Strided;

  char* pointer;
  int stride;
  int increment_strided;
  int increment_advance;

  __HOST_DEVICE__ SmemTileIterator(float* ptr, int smem_stride,
                                   int thread_idx) {
    int contiguous = 0;
    int strided = 0;
    ThreadMap::initial_offset(thread_idx, contiguous, strided);
    pointer = reinterpret_cast<char*>(ptr + strided * smem_stride + contiguous);
    stride = smem_stride;
    increment_strided = sizeof(float) * smem_stride * ThreadMap::kDeltaStrided;
    increment_advance =
        sizeof(float) *
        (AdvanceRank == 0 ? kContiguous : kStrided * smem_stride);
  }

  __HOST_DEVICE__ void store(Fragment const& frag) {
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
    char* byte_pointer = pointer;

#pragma unroll
    for (int s = 0; s < ThreadMap::kIterationsStrided; s++) {
      AccessType* access_ptr = reinterpret_cast<AccessType*>(byte_pointer);
#pragma unroll
      for (int c = 0; c < ThreadMap::kIterationsContiguous; c++) {
        int idx = c + s * ThreadMap::kIterationsContiguous;
        access_ptr[c * ThreadMap::kDeltaContiguous /
                   ThreadMap::kElementsPerAccess] = frag_ptr[idx];
      }

      if (s + 1 < ThreadMap::kIterationsStrided) {
        byte_pointer += increment_strided;
      }
    }
  }

  __HOST_DEVICE__ SmemTileIterator& operator++() {
    pointer += increment_advance;
    return *this;
  }

  __HOST_DEVICE__ SmemTileIterator& operator--() {
    pointer -= increment_advance;
    return *this;
  }

  __HOST_DEVICE__ SmemTileIterator& add_stage_offset(int stages) {
    pointer += stages * increment_advance;
    return *this;
  }
};

template <typename ThreadMap>
struct TransposePitchLinearThreadMapSimtVec4Store {
  static constexpr int kContiguous = ThreadMap::kStrided;
  static constexpr int kStrided = ThreadMap::kContiguous;
  static constexpr int kThreads = ThreadMap::kThreads;
  static constexpr int kElementsPerAccess = ThreadMap::kElementsPerAccess;

  // Fragment traversal order is preserved from the global vec4 thread map.
  static constexpr int kIterationsContiguous = ThreadMap::kIterationsContiguous;
  static constexpr int kIterationsStrided = ThreadMap::kIterationsStrided;
  static constexpr int kIterationsCount = ThreadMap::kIterationsCount;
  static constexpr int kDeltaContiguous = ThreadMap::kDeltaContiguous;
  static constexpr int kDeltaStrided = ThreadMap::kDeltaStrided;

  static_assert(kElementsPerAccess == 4,
                "Vec4 transpose store requires 4 elements per access.");

  __DEVICE__ static void initial_offset(int thread_id, int& contiguous,
                                        int& strided) {
    int base_contiguous = 0;
    int base_strided = 0;
    ThreadMap::initial_offset(thread_id, base_contiguous, base_strided);

    // Shared memory is indexed as smem_a[k * stride + m].
    contiguous = base_strided;  // m
    strided = base_contiguous;  // k packet base
  }
};

template <int Contiguous, int Strided, typename ThreadMap, int AdvanceRank>
struct SmemTransposeVec4TileIterator {
  using AccessType = AlignedFloatArray<ThreadMap::kElementsPerAccess>;
  using Fragment = AlignedFloatArray<ThreadMap::kIterationsCount *
                                         ThreadMap::kElementsPerAccess,
                                     alignof(AccessType)>;

  static constexpr int kContiguous = Contiguous;
  static constexpr int kStrided = Strided;

  char* pointer;
  int stride;
  int increment_contiguous;
  int increment_strided;
  int increment_advance;

  static_assert(ThreadMap::kElementsPerAccess == 4,
                "This iterator is specialized for vec4 transpose stores.");

  __HOST_DEVICE__ SmemTransposeVec4TileIterator(float* ptr, int smem_stride,
                                                int thread_idx) {
    int contiguous = 0;
    int strided = 0;
    ThreadMap::initial_offset(thread_idx, contiguous, strided);

    pointer = reinterpret_cast<char*>(ptr + strided * smem_stride + contiguous);
    stride = smem_stride;

    // c-loop advances along K packets, which becomes the strided dimension
    // in shared memory after transpose.
    increment_contiguous =
        sizeof(float) * ThreadMap::kDeltaContiguous * smem_stride;

    // s-loop advances along M rows, which becomes the contiguous dimension.
    increment_strided = sizeof(float) * ThreadMap::kDeltaStrided;

    increment_advance =
        sizeof(float) *
        (AdvanceRank == 0 ? kContiguous : kStrided * smem_stride);
  }

  __HOST_DEVICE__ void store(Fragment const& frag) {
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
    char* strided_byte_pointer = pointer;

#pragma unroll
    for (int s = 0; s < ThreadMap::kIterationsStrided; ++s) {
      char* contiguous_byte_pointer = strided_byte_pointer;

#pragma unroll
      for (int c = 0; c < ThreadMap::kIterationsContiguous; ++c) {
        int idx = c + s * ThreadMap::kIterationsContiguous;
        float const* access = reinterpret_cast<float const*>(&frag_ptr[idx]);
        float* scalar_ptr = reinterpret_cast<float*>(contiguous_byte_pointer);

        scalar_ptr[0 * stride] = access[0];
        scalar_ptr[1 * stride] = access[1];
        scalar_ptr[2 * stride] = access[2];
        scalar_ptr[3 * stride] = access[3];

        if (c + 1 < ThreadMap::kIterationsContiguous) {
          contiguous_byte_pointer += increment_contiguous;
        }
      }

      if (s + 1 < ThreadMap::kIterationsStrided) {
        strided_byte_pointer += increment_strided;
      }
    }
  }

  __HOST_DEVICE__ SmemTransposeVec4TileIterator& operator++() {
    pointer += increment_advance;
    return *this;
  }

  __HOST_DEVICE__ SmemTransposeVec4TileIterator& operator--() {
    pointer -= increment_advance;
    return *this;
  }

  __HOST_DEVICE__ SmemTransposeVec4TileIterator& add_stage_offset(int stages) {
    pointer += stages * increment_advance;
    return *this;
  }
};

template <int WarpM, int WarpThreadsM, int ThreadTileM, int LaneMmaM,
          int SmemStrideA, int SmemStageElemsA>
struct WarpTileIteratorA {
  using AccessType = AlignedFloatArray<LaneMmaM>;
  using Fragment = AlignedFloatArray<ThreadTileM, alignof(AccessType)>;

  static constexpr int kContiguous = LaneMmaM;
  static constexpr int kStrided = ThreadTileM / kContiguous;
  static constexpr int kElementsPerAccess = kContiguous;

  static constexpr int kIterationsRow = ThreadTileM / LaneMmaM;
  static constexpr int kIterationsColumn = 1;
  static constexpr int kIterationsCount = kIterationsRow * kIterationsColumn;

  static constexpr int kDeltaRow = WarpThreadsM;
  static constexpr int kDeltaColumn = 0;

  AccessType const* pointer;
  int increment_advance;

  static_assert(ThreadTileM % LaneMmaM == 0,
                "ThreadTileM must be divisible by LaneMmaM.");
  static_assert(SmemStrideA % LaneMmaM == 0,
                "SmemStrideA must preserve AccessType alignment.");

  __HOST_DEVICE__ WarpTileIteratorA(float const* shared_storage, int stage,
                                    int warp_tile_m_idx, int lane_m_idx) {
    float const* ref = shared_storage + stage * SmemStageElemsA +
                       warp_tile_m_idx * WarpM + lane_m_idx * LaneMmaM;

    pointer = reinterpret_cast<AccessType const*>(ref);
    increment_advance = SmemStrideA / LaneMmaM;
  }

  __HOST_DEVICE__ void load(Fragment& frag) const {
    AccessType* dst_ptr = reinterpret_cast<AccessType*>(&frag);

#pragma unroll
    for (int k = 0; k < kIterationsColumn; ++k) {
#pragma unroll
      for (int m = 0; m < kIterationsRow; ++m) {
        dst_ptr[m + k * kIterationsRow] =
            pointer[m * kDeltaRow + k * kDeltaColumn];
      }
    }
  }

  __HOST_DEVICE__ WarpTileIteratorA& operator++() {
    pointer += increment_advance;
    return *this;
  }

  __HOST_DEVICE__ WarpTileIteratorA& operator--() {
    pointer -= increment_advance;
    return *this;
  }

  __HOST_DEVICE__ WarpTileIteratorA& add_tile_offset(int tile_k) {
    pointer += tile_k * increment_advance;
    return *this;
  }
};

template <int WarpN, int WarpThreadsN, int ThreadTileN, int LaneMmaN,
          int SmemStrideB, int SmemStageElemsB>
struct WarpTileIteratorB {
  using AccessType = AlignedFloatArray<LaneMmaN>;
  using Fragment = AlignedFloatArray<ThreadTileN, alignof(AccessType)>;

  static constexpr int kContiguous = LaneMmaN;
  static constexpr int kStrided = ThreadTileN / kContiguous;
  static constexpr int kElementsPerAccess = kContiguous;

  static constexpr int kIterationsRow = 1;
  static constexpr int kIterationsColumn = kStrided;
  static constexpr int kIterationsCount = kIterationsRow * kIterationsColumn;

  static constexpr int kDeltaRow = 0;
  static constexpr int kDeltaColumn = WarpThreadsN;

  AccessType const* pointer;
  int increment_advance;

  static_assert(ThreadTileN % LaneMmaN == 0,
                "ThreadTileN must be divisible by LaneMmaN.");
  static_assert(SmemStrideB % LaneMmaN == 0,
                "SmemStrideB must preserve AccessType alignment.");

  __HOST_DEVICE__ WarpTileIteratorB(float const* shared_storage, int stage,
                                    int warp_tile_n_idx, int lane_n_idx) {
    float const* ref = shared_storage + stage * SmemStageElemsB +
                       warp_tile_n_idx * WarpN + lane_n_idx * LaneMmaN;

    pointer = reinterpret_cast<AccessType const*>(ref);
    increment_advance = SmemStrideB / LaneMmaN;
  }

  __HOST_DEVICE__ void load(Fragment& frag) {
    AccessType* dst_ptr = reinterpret_cast<AccessType*>(&frag);

#pragma unroll
    for (int k = 0; k < kIterationsRow; ++k) {
#pragma unroll
      for (int n = 0; n < kIterationsColumn; ++n) {
        dst_ptr[n + k * kIterationsColumn] =
            pointer[k * kDeltaRow + n * kDeltaColumn];
      }
    }
  }

  __HOST_DEVICE__ WarpTileIteratorB& operator++() {
    pointer += increment_advance;
    return *this;
  }

  __HOST_DEVICE__ WarpTileIteratorB& operator--() {
    pointer -= increment_advance;
    return *this;
  }

  __HOST_DEVICE__ WarpTileIteratorB& add_tile_offset(int tile_k) {
    pointer += tile_k * increment_advance;
    return *this;
  }
};

template <typename Traits>
struct AccumulatorFragmentIterator {
  using AccessType = AlignedFloatArray<Traits::kLaneMmaN>;
  using Fragment = AlignedFloatArray<Traits::kThreadTileN, alignof(AccessType)>;

  static constexpr int kIterations = Traits::kThreadTileM;
  static constexpr int kAccessesPerIteration =
      Traits::kThreadTileN / Traits::kLaneMmaN;

  AccessType const* pointer;
  int increment_advance;

  __HOST_DEVICE__ AccumulatorFragmentIterator(float const* accum)
      : pointer(reinterpret_cast<AccessType const*>(accum)),
        increment_advance(kAccessesPerIteration) {}

  __HOST_DEVICE__ void load(Fragment& frag) const {
    AccessType* dst_ptr = reinterpret_cast<AccessType*>(&frag);

    for (int idx = 0; idx < kAccessesPerIteration; ++idx) {
      dst_ptr[idx] = pointer[idx];
    }
  }

  __HOST_DEVICE__ AccumulatorFragmentIterator& operator++() {
    pointer += increment_advance;
    return *this;
  }

  __HOST_DEVICE__ AccumulatorFragmentIterator& operator--() {
    pointer -= increment_advance;
    return *this;
  }
};

template <typename Traits>
struct WarpEpilogueStoreIterator {
  using AccessType = AlignedFloatArray<1>;
  using Fragment = typename AccumulatorFragmentIterator<Traits>::Fragment;

  static constexpr int kAccessesPerIteration =
      Traits::kThreadTileN / Traits::kLaneMmaN;

  float* pointer;
  int column_group_advance;

  __HOST_DEVICE__ WarpEpilogueStoreIterator(float* shared_storage,
                                            int warp_tile_m_idx,
                                            int warp_tile_n_idx, int lane_m_idx,
                                            int lane_n_idx, int stride) {
    int const row = warp_tile_m_idx * Traits::kWarpThreadsM + lane_m_idx;
    int const col =
        warp_tile_n_idx * Traits::kWarpN + lane_n_idx * Traits::kLaneMmaN;

    pointer = shared_storage + row * stride + col;
    column_group_advance = Traits::kWarpThreadsN * Traits::kLaneMmaN;
  }

  __HOST_DEVICE__ void store(Fragment const& frag) {
    float const* frag_ptr = reinterpret_cast<float const*>(&frag);

#pragma unroll
    for (int column_group = 0; column_group < kAccessesPerIteration;
         ++column_group) {
#pragma unroll
      for (int n = 0; n < Traits::kLaneMmaN; ++n) {
        pointer[column_group * column_group_advance + n] =
            frag_ptr[column_group * Traits::kLaneMmaN + n];
      }
    }
  }
};

template <typename Traits>
struct EpilogueSharedLoadIterator {
  using AccessType = AlignedFloatArray<1>;
  using Fragment = AlignedFloatArray<Traits::kThreadTileN, alignof(AccessType)>;

  static constexpr int kColumnIterations =
      Traits::kEpilogueColumns / Traits::kWarpSize;
  static constexpr int kRowIterations =
      Traits::kThreadTileN / kColumnIterations;

  static_assert(Traits::kEpilogueColumns % Traits::kWarpSize == 0,
                "Epilogue columns must be divisible by warp size.");
  static_assert(Traits::kThreadTileN % kColumnIterations == 0,
                "Thread tile N must match compact epilogue thread map.");

  float const* pointer;
  int row_base;
  int col_base;
  int stride;

  __HOST_DEVICE__ EpilogueSharedLoadIterator(float const* shared_pointer,
                                             int tid, int stride) {
    int const warp_idx = tid / Traits::kWarpSize;
    int const lane_idx = tid % Traits::kWarpSize;

    pointer = shared_pointer;
    row_base = (warp_idx / Traits::kWarpCountN) * Traits::kWarpThreadsM +
               (warp_idx % Traits::kWarpCountN);
    col_base = lane_idx;
    this->stride = stride;
  }

  __HOST_DEVICE__ void load(Fragment& frag) {
    AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

#pragma unroll
    for (int row_iter = 0; row_iter < kRowIterations; ++row_iter) {
#pragma unroll
      for (int col_iter = 0; col_iter < kColumnIterations; ++col_iter) {
        int idx = col_iter + row_iter * kColumnIterations;
        int row = row_base + row_iter * Traits::kWarpCountN;
        int col = col_base + col_iter * Traits::kWarpSize;

        frag_ptr[idx][0] = pointer[row * stride + col];
      }
    }
  }
};

template <typename Traits>
struct EpilogueOutputStoreIterator {
  using AccessType = AlignedFloatArray<1>;
  using Fragment = AlignedFloatArray<Traits::kThreadTileN, alignof(AccessType)>;

  static constexpr int kColumnIterations =
      Traits::kEpilogueColumns / Traits::kWarpSize;
  static constexpr int kRowIterations =
      Traits::kThreadTileN / kColumnIterations;

  static_assert(Traits::kEpilogueColumns % Traits::kWarpSize == 0,
                "Epilogue columns must be divisible by warp size.");
  static_assert(Traits::kThreadTileN % kColumnIterations == 0,
                "Thread tile N must match compact epilogue thread map.");

  float* pointer;
  int ld;
  int row_base;
  int col_base;
  int iteration;

  __HOST_DEVICE__ EpilogueOutputStoreIterator(float* ptr, int tid, int ld) {
    int const warp_idx = tid / Traits::kWarpSize;
    int const lane_idx = tid % Traits::kWarpSize;

    pointer = ptr;
    this->ld = ld;
    row_base = (warp_idx / Traits::kWarpCountN) * Traits::kWarpThreadsM +
               (warp_idx % Traits::kWarpCountN);
    col_base = lane_idx;
    iteration = 0;
  }

  __HOST_DEVICE__ void store(Fragment const& frag) {
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
    int const row_group = iteration / Traits::kLaneMmaM;
    int const row_minor = iteration % Traits::kLaneMmaM;

#pragma unroll
    for (int row_iter = 0; row_iter < kRowIterations; ++row_iter) {
#pragma unroll
      for (int col_iter = 0; col_iter < kColumnIterations; ++col_iter) {
        int idx = col_iter + row_iter * kColumnIterations;
        int compact_row = row_base + row_iter * Traits::kWarpCountN;
        int compact_col = col_base + col_iter * Traits::kWarpSize;
        int warp_m = compact_row / Traits::kWarpThreadsM;
        int lane_row = compact_row % Traits::kWarpThreadsM;
        int row = warp_m * Traits::kWarpM +
                  row_group * Traits::kWarpThreadsM * Traits::kLaneMmaM +
                  lane_row * Traits::kLaneMmaM + row_minor;

        pointer[row * ld + compact_col] = frag_ptr[idx][0];
      }
    }
  }

  __HOST_DEVICE__ EpilogueOutputStoreIterator& operator++() {
    ++iteration;
    return *this;
  }

  __HOST_DEVICE__ EpilogueOutputStoreIterator& operator--() {
    --iteration;
    return *this;
  }
};

template <typename Traits>
struct EpilogueSharedLoadIteratorVec4 {
  using AccessType = AlignedFloatArray<4>;
  using Fragment = AlignedFloatArray<Traits::kThreadTileN, alignof(AccessType)>;

  static constexpr int kElementsPerAccess = 4;
  static constexpr int kColumnIterations =
      Traits::kEpilogueColumns / (Traits::kWarpSize * kElementsPerAccess);
  static constexpr int kRowIterations =
      Traits::kThreadTileN / (kColumnIterations * kElementsPerAccess);

  static_assert(Traits::kEpilogueColumns %
                        (Traits::kWarpSize * kElementsPerAccess) ==
                    0,
                "Epilogue columns must support vec4 output stores.");
  static_assert(Traits::kThreadTileN %
                        (kColumnIterations * kElementsPerAccess) ==
                    0,
                "Thread tile N must match vec4 compact epilogue thread map.");

  float const* pointer;
  int row_base;
  int col_base;
  int stride;

  __HOST_DEVICE__ EpilogueSharedLoadIteratorVec4(float const* shared_pointer,
                                                 int tid, int stride) {
    int const warp_idx = tid / Traits::kWarpSize;
    int const lane_idx = tid % Traits::kWarpSize;

    pointer = shared_pointer;
    row_base = (warp_idx / Traits::kWarpCountN) * Traits::kWarpThreadsM +
               (warp_idx % Traits::kWarpCountN);
    col_base = lane_idx * kElementsPerAccess;
    this->stride = stride;
  }

  __HOST_DEVICE__ void load(Fragment& frag) {
    AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

#pragma unroll
    for (int row_iter = 0; row_iter < kRowIterations; ++row_iter) {
      int row = row_base + row_iter * Traits::kWarpCountN;
      float const* src = pointer + row * stride + col_base;

#pragma unroll
      for (int e = 0; e < kElementsPerAccess; ++e) {
        frag_ptr[row_iter][e] = src[e];
      }
    }
  }
};

template <typename Traits>
struct EpilogueOutputStoreIteratorVec4 {
  using AccessType = AlignedFloatArray<4>;
  using Fragment = AlignedFloatArray<Traits::kThreadTileN, alignof(AccessType)>;

  static constexpr int kElementsPerAccess = 4;
  static constexpr int kColumnIterations =
      Traits::kEpilogueColumns / (Traits::kWarpSize * kElementsPerAccess);
  static constexpr int kRowIterations =
      Traits::kThreadTileN / (kColumnIterations * kElementsPerAccess);

  static_assert(Traits::kEpilogueColumns %
                        (Traits::kWarpSize * kElementsPerAccess) ==
                    0,
                "Epilogue columns must support vec4 output stores.");
  static_assert(Traits::kThreadTileN %
                        (kColumnIterations * kElementsPerAccess) ==
                    0,
                "Thread tile N must match vec4 compact epilogue thread map.");

  float* pointer;
  int ld;
  int row_base;
  int col_base;
  int iteration;

  __HOST_DEVICE__ EpilogueOutputStoreIteratorVec4(float* ptr, int tid, int ld) {
    int const warp_idx = tid / Traits::kWarpSize;
    int const lane_idx = tid % Traits::kWarpSize;

    pointer = ptr;
    this->ld = ld;
    row_base = (warp_idx / Traits::kWarpCountN) * Traits::kWarpThreadsM +
               (warp_idx % Traits::kWarpCountN);
    col_base = lane_idx * kElementsPerAccess;
    iteration = 0;
  }

  __DEVICE__ void store(Fragment const& frag) {
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
    int const row_group = iteration / Traits::kLaneMmaM;
    int const row_minor = iteration % Traits::kLaneMmaM;

#pragma unroll
    for (int row_iter = 0; row_iter < kRowIterations; ++row_iter) {
      int compact_row = row_base + row_iter * Traits::kWarpCountN;
      int warp_m = compact_row / Traits::kWarpThreadsM;
      int lane_row = compact_row % Traits::kWarpThreadsM;
      int row = warp_m * Traits::kWarpM +
                row_group * Traits::kWarpThreadsM * Traits::kLaneMmaM +
                lane_row * Traits::kLaneMmaM + row_minor;

      store_float4(pointer + row * ld + col_base,
                   *reinterpret_cast<float4 const*>(&frag_ptr[row_iter]));
    }
  }

  __HOST_DEVICE__ EpilogueOutputStoreIteratorVec4& operator++() {
    ++iteration;
    return *this;
  }

  __HOST_DEVICE__ EpilogueOutputStoreIteratorVec4& operator--() {
    --iteration;
    return *this;
  }
};

}  // namespace utils
}  // namespace sgemm
