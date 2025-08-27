#pragma once

#include <immintrin.h>
#include <cstdlib>

#include "util.hh"


namespace util {

// 32 byte alignment for use with 256-bit AVX types
constexpr std::size_t ALIGN = 32;
constexpr std::size_t CHUNK = 8; // changing this will break the AVX version
struct alignas(ALIGN) f4x8 { float x[CHUNK], y[CHUNK], z[CHUNK], m[CHUNK]; };

}


static inline __m256 
rsqrt_256(const __m256 x) {
    return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(x));
}

static inline __m256
rsqrt_nr_256(const __m256 x) {
    const __m256 three = _mm256_set1_ps(3.0f), half = _mm256_set1_ps(0.5f);
    __m256 res         = _mm256_rsqrt_ps(x);
    const __m256 muls  = _mm256_mul_ps(_mm256_mul_ps(res, res), x);
    res = _mm256_mul_ps(_mm256_mul_ps(half, res), _mm256_sub_ps(three, muls));
    return res;
}

static inline __m256
rsqrt_nr2_256(__m256 x) {
  __m256 three = _mm256_set1_ps(3.0f), half = _mm256_set1_ps(0.5f);
  __m256 res, muls;
  res = _mm256_rsqrt_ps(x); 
  muls = _mm256_mul_ps(_mm256_mul_ps(x, res), res); 
  res = _mm256_mul_ps(_mm256_mul_ps(half, res), _mm256_sub_ps(three, muls));
  muls = _mm256_mul_ps(_mm256_mul_ps(x, res), res); 
  res = _mm256_mul_ps(_mm256_mul_ps(half, res), _mm256_sub_ps(three, muls));
  return res;
}

static inline __m256
rsqrt_fast_256(const __m256 x) {
    return _mm256_rsqrt_ps(x);
}

static inline void grav_accel_kernel_avx(const int i, util::f4x8 const *pavx, util::f4x8 *ravx)
{   
    // Softening squared - TODO -- do something abut this
    constexpr float soft_sqrd = 0.00001f;

    // Structured 256-bit loads (8 unique floats per __m256)
    const __m256 p_xi = _mm256_load_ps(&pavx[i].x[0]);
    const __m256 p_yi = _mm256_load_ps(&pavx[i].y[0]);
    const __m256 p_zi = _mm256_load_ps(&pavx[i].z[0]);

    // Create 256-bit vectors for accumulating accelerations  
    __m256 result_x = _mm256_setzero_ps();
    __m256 result_y = _mm256_setzero_ps();
    __m256 result_z = _mm256_setzero_ps();

    // Loop through array of 256-bit vectors
    for (std::size_t j = 0; j < NBODIES/CHUNK; j++) {
        for (std::size_t k = 0; k < CHUNK; k++) {

            // Set all 8 floats in 256-bit vector to the same value
            const __m256 p_xj = _mm256_set1_ps(pavx[j].x[k]);
            const __m256 p_yj = _mm256_set1_ps(pavx[j].y[k]);
            const __m256 p_zj = _mm256_set1_ps(pavx[j].z[k]);
            const __m256 mass = _mm256_set1_ps(pavx[j].m[k]);

            // Distance between positions (p_x contains 8 unique floats and p_xx contains 8 copies of the same value)
            __m256 d_x = _mm256_sub_ps(p_xj, p_xi);
            __m256 d_y = _mm256_sub_ps(p_yj, p_yi);
            __m256 d_z = _mm256_sub_ps(p_zj, p_zi);

            // Square the distance and add the squared softening length
            __m256 d_sqrd = _mm256_add_ps( 
                            _mm256_add_ps( 
                            _mm256_mul_ps(d_x, d_x), 
                            _mm256_mul_ps(d_y, d_y)), 
                            _mm256_mul_ps(d_z, d_z));
            d_sqrd = _mm256_add_ps(d_sqrd, _mm256_set1_ps(soft_sqrd));

            // Calculate inv_d_cubed = 1 / d_sqrd^(3/2)
            __m256 inv_d = rsqrt_fast_256(d_sqrd); //rsqrt_nr_256(d_sqrd) or rsqrt_nr2_256(d_sqrd);
            __m256 inv_d_cubed = _mm256_mul_ps(inv_d,_mm256_mul_ps(inv_d, inv_d ));

            // impulse = mass_j * inv_d_cubed
            __m256 impulse = _mm256_mul_ps(mass, inv_d_cubed);

            // Get acceleration by giving the impulse a direction -- multiply it with d_x
            // accel = mass_j (m_1 * r_01) / (d^2 + e^2)^(3/2)
            // Accumulate accelerations into result vectors
            result_x = _mm256_add_ps(result_x, _mm256_mul_ps( d_x, impulse ) );
            result_y = _mm256_add_ps(result_y, _mm256_mul_ps( d_y, impulse ) );
            result_z = _mm256_add_ps(result_z, _mm256_mul_ps( d_z, impulse ) );
        }
    }
    _mm256_store_ps(&ravx[i].x[0], result_x);
    _mm256_store_ps(&ravx[i].y[0], result_y);
    _mm256_store_ps(&ravx[i].z[0], result_z);
}
