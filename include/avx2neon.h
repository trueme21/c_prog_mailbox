#ifndef AVX2NEON_H
#define AVX2NEON_H


#include <arm_neon.h>
#include "typedefs.h"

/* "__has_builtin" can be used to query support for built-in functions
 * provided by gcc/clang and other compilers that support it.
 */
#ifndef __has_builtin /* GCC prior to 10 or non-clang compilers */
/* Compatibility with gcc <= 9 */
#if __GNUC__ <= 9
#define __has_builtin(x) HAS##x
#define HAS__builtin_popcount 1
#define HAS__builtin_popcountll 1
#else
#define __has_builtin(x) 0
#endif
#endif

typedef struct {
    float32x4_t vect_f32[2];
} __m256;

#define vreinterpretq_m128_f32(x) (x)
#define vreinterpretq_f32_m128(x) (x)

FORCE_INLINE void _mm_prefetch(const void *p, int i)
{
    (void) i;
    __builtin_prefetch(p);
}

FORCE_INLINE __m256 _mm256_setzero_ps(void)
{
    __m256 ret;
    ret.vect_f32[0] = ret.vect_f32[1] = vdupq_n_f32(0.0f);
    return ret;
}

FORCE_INLINE __m256 _mm256_add_ps(__m256 a, __m256 b)
{
    __m256 res_m256;
    res_m256.vect_f32[0] = vaddq_f32(a.vect_f32[0], b.vect_f32[0]);
    res_m256.vect_f32[1] = vaddq_f32(a.vect_f32[1], b.vect_f32[1]);
    return res_m256;
}

FORCE_INLINE __m256 _mm256_sub_ps(__m256 a, __m256 b)
{
    __m256 res_m256;
    res_m256.vect_f32[0] = vsubq_f32(a.vect_f32[0], b.vect_f32[0]);
    res_m256.vect_f32[1] = vsubq_f32(a.vect_f32[1], b.vect_f32[1]);
    return res_m256;
}

FORCE_INLINE void _mm256_storeu_ps (float * mem_addr, __m256 a)
{
    vst1q_f32(mem_addr, vreinterpretq_f32_m128(a.vect_f32[0]));
    vst1q_f32(mem_addr + 4, vreinterpretq_f32_m128(a.vect_f32[1]));
}  // unaligned mem_addr?

// Quote from sse2neon:
// for neon, alignment doesn't matter, so _mm_load_ps and _mm_loadu_ps are
// equivalent for neon
FORCE_INLINE __m256 _mm256_loadu_ps (float const * mem_addr)
{
    __m256 res;
    res.vect_f32[0] = vreinterpretq_m128_f32(vld1q_f32((const float *)mem_addr));
    res.vect_f32[1] = vreinterpretq_m128_f32(vld1q_f32((const float *)mem_addr + 4));
    return res;
}

#endif
