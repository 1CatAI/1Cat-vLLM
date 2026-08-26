#ifndef FUSED_MMA_H
#define FUSED_MMA_H

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 700)
#error "Volta WMMA: This header is for sm_70 ONLY! Compile with -arch=sm_70"
#endif

#include <cuda_fp16.h>

namespace volta {

struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};

enum layout_t {
    mem_row_major,
    mem_col_major
};

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment;

template <> struct fragment<matrix_a, 16, 16, 16, half, row_major> { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_a, 16, 16, 16, half, col_major> { uint32_t x[8]; static constexpr int num_elements = 16; };

template <> struct fragment<matrix_a, 32, 8, 16, half, row_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_a, 32, 8, 16, half, col_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };

template <> struct fragment<matrix_a, 8, 32, 16, half, row_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_a, 8, 32, 16, half, col_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };

template <> struct fragment<matrix_b, 16, 16, 16, half, row_major> { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_b, 16, 16, 16, half, col_major> { uint32_t x[8]; static constexpr int num_elements = 16; };

template <> struct fragment<matrix_b, 32, 8, 16, half, row_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_b, 32, 8, 16, half, col_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };

template <> struct fragment<matrix_b, 8, 32, 16, half, row_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_b, 8, 32, 16, half, col_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };

template <> struct fragment<accumulator, 16, 16, 16, float> { float x[8]; static constexpr int num_elements = 8; };
template <> struct fragment<accumulator, 32, 8, 16, float>  { float x[8]; static constexpr int num_elements = 8; };
template <> struct fragment<accumulator, 8, 32, 16, float>  { float x[8]; static constexpr int num_elements = 8; };

__device__ __forceinline__ unsigned get_lane_id() {
    unsigned lane_id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

template <int M, int N, int K>
__device__ __forceinline__ void fill_fragment(fragment<accumulator, M, N, K, float>& frag, float value) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) frag.x[i] = value;
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 16, 16, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.row.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 16, 16, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.col.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.row.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.col.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, 16, 16, 16, float>& frag,
    const float* smem_ptr, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.load.c.sync.aligned.row.m16n16k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.load.c.sync.aligned.col.m16n16k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    }
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 32, 8, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.row.m32n8k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 32, 8, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.col.m32n8k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 32, 8, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.row.m32n8k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 32, 8, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.col.m32n8k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, 32, 8, 16, float>& frag,
    const float* smem_ptr, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.load.c.sync.aligned.row.m32n8k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.load.c.sync.aligned.col.m32n8k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    }
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 8, 32, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.row.m8n32k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 8, 32, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.col.m8n32k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 8, 32, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.row.m8n32k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 8, 32, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.col.m8n32k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, 8, 32, 16, float>& frag,
    const float* smem_ptr, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.load.c.sync.aligned.row.m8n32k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.load.c.sync.aligned.col.m8n32k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    }
}

__device__ __forceinline__ void store_matrix_sync(
    float* smem_ptr,
    const fragment<accumulator, 16, 16, 16, float>& frag,
    unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.store.d.sync.aligned.row.m16n16k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.store.d.sync.aligned.col.m16n16k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    }
}

__device__ __forceinline__ void store_matrix_sync(
    float* smem_ptr,
    const fragment<accumulator, 32, 8, 16, float>& frag,
    unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.store.d.sync.aligned.row.m32n8k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.store.d.sync.aligned.col.m32n8k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    }
}

__device__ __forceinline__ void store_matrix_sync(
    float* smem_ptr,
    const fragment<accumulator, 8, 32, 16, float>& frag,
    unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.store.d.sync.aligned.row.m8n32k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.store.d.sync.aligned.col.m8n32k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    }
}

#define VOLTA_WMMA_MMA_F32(M, N, K, ALAY, BLAY) \
__device__ __forceinline__ void mma_sync( \
    fragment<accumulator, M, N, K, float>& d, \
    const fragment<matrix_a, M, N, K, half, ALAY##_major>& a, \
    const fragment<matrix_b, M, N, K, half, BLAY##_major>& b, \
    const fragment<accumulator, M, N, K, float>& c) { \
    asm volatile( \
        "wmma.mma.sync.aligned." #ALAY "." #BLAY ".m" #M "n" #N "k" #K ".f32.f32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, " \
        "{%8,%9,%10,%11,%12,%13,%14,%15}, " \
        "{%16,%17,%18,%19,%20,%21,%22,%23}, " \
        "{%24,%25,%26,%27,%28,%29,%30,%31};" \
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), \
          "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7]) \
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), \
          "r"(a.x[4]), "r"(a.x[5]), "r"(a.x[6]), "r"(a.x[7]), \
          "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]), \
          "r"(b.x[4]), "r"(b.x[5]), "r"(b.x[6]), "r"(b.x[7]), \
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), \
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]) \
    ); \
}

VOLTA_WMMA_MMA_F32(16, 16, 16, row, col)
VOLTA_WMMA_MMA_F32(16, 16, 16, row, row)
VOLTA_WMMA_MMA_F32(16, 16, 16, col, col)
VOLTA_WMMA_MMA_F32(16, 16, 16, col, row)

VOLTA_WMMA_MMA_F32(32, 8, 16, row, col)
VOLTA_WMMA_MMA_F32(32, 8, 16, row, row)
VOLTA_WMMA_MMA_F32(32, 8, 16, col, col)
VOLTA_WMMA_MMA_F32(32, 8, 16, col, row)

VOLTA_WMMA_MMA_F32(8, 32, 16, row, col)
VOLTA_WMMA_MMA_F32(8, 32, 16, row, row)
VOLTA_WMMA_MMA_F32(8, 32, 16, col, col)
VOLTA_WMMA_MMA_F32(8, 32, 16, col, row)

#undef VOLTA_WMMA_MMA_F32

}

#endif
