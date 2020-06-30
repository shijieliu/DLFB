/*
 * @Author: liushijie
 * @Date: 2020-06-15 21:09:15
 * @Last Modified by:   liushijie
 * @Last Modified time: 2020-06-15 21:09:15
 */
#pragma once

#include <cstdio>
#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#include <cmath>

// AVX Support
namespace dl {
namespace SIMD {
inline void AvxVecAdd(const float *x, const float *y, float *res, int len) {
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_add_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            _mm256_storeu_ps(res, t);
            x += 8;
            y += 8;
            res += 8;
        }
    }
    // Don't forget the remaining values.
    for (; len > 0; len--) {
        *res = *x + *y;
        x++;
        y++;
        res++;
    }
}

inline void AvxVecSub(const float *x, const float *y, float *res, int len) {
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            _mm256_storeu_ps(res, t);
            x += 8;
            y += 8;
            res += 8;
        }
    }
    // Don't forget the remaining values.
    for (; len > 0; len--) {
        *res = *x - *y;
        x++;
        y++;
        res++;
    }
}

inline void AvxVecAdd(const float *x, const float delta_, float *res,
                      int len) {
    const __m256 delta = _mm256_broadcast_ss(&delta_);
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_add_ps(_mm256_loadu_ps(x), delta);
            _mm256_storeu_ps(res, t);
            x += 8;
            res += 8;
        }
    }
    // Don't forget the remaining values.
    for (; len > 0; len--) {
        *res = *x + delta_;
        x++;
        res++;
    }
}

inline void AvxVecMul(const float *x, const float *y, float *res, int len) {
    if (len > 7) {
        while (len > 7) {
            __m256 t = _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            _mm256_storeu_ps(res, t);
            x += 8;
            y += 8;
            res += 8;
            len -= 8;
        }
    }
    while (len > 0) {
        *res = (*x) * (*y);
        x++;
        y++;
        res++;
        len--;
    }
}

inline void AvxVecMul(const float *x, const float delta_, float *res, int len){
    const __m256 delta = _mm256_broadcast_ss(&delta_);
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_mul_ps(_mm256_loadu_ps(x), delta);
            _mm256_storeu_ps(res, t);
            x += 8;
            res += 8;
        }
    }
    // Don't forget the remaining values.
    for (; len > 0; len--) {
        *res = *x + delta_;
        x++;
        res++;
    }
}

inline void AvxVecDiv(const float *x, const float *y, float *res, int len) {
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_div_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            _mm256_storeu_ps(res, t);
            x += 8;
            y += 8;
            res += 8;
        }
    }
    for (; len > 0; len--) {
        *res = *x / *y;
        x++;
        y++;
        res++;
    }
}

inline float hsum256_ps_avx(__m256 v) {
    const __m128 x128 =
        _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline float AvxVecDotProduct(const float *x, const float *y, int len) {
    float result = 0;
    if (len > 7) {
        __m256 t = _mm256_setzero_ps();
        for (; len > 7; len -= 8) {
            t = _mm256_add_ps(
                t, _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y)));
            x += 8;
            y += 8;
        }
        result += hsum256_ps_avx(t);
    }
    for (; len > 0; len--) {
        result += (*x) * (*y);
        x++;
        y++;
    }
    return result;
}
}
}