#include "cosine_simd.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static inline double dot_scalar(const double *x, const double *y, size_t d) {
    double sum = 0.0;
    for (size_t k = 0; k < d; k++) {
        sum += x[k] * y[k];
    }
    return sum;
}

static inline double dot_simd(const double *x, const double *y, size_t d) {
#if defined(__AVX2__)
    __m256d acc = _mm256_setzero_pd();
    size_t k = 0;
    for (; k + 4 <= d; k += 4) {
        __m256d vx = _mm256_loadu_pd(x + k);
        __m256d vy = _mm256_loadu_pd(y + k);
#if defined(__FMA__)
        acc = _mm256_fmadd_pd(vx, vy, acc);
#else
        acc = _mm256_add_pd(acc, _mm256_mul_pd(vx, vy));
#endif
    }

    double lanes[4];
    _mm256_storeu_pd(lanes, acc);
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; k < d; k++) {
        sum += x[k] * y[k];
    }
    return sum;
#else
    return dot_scalar(x, y, d);
#endif
}

double cosine_all_pairs_checksum_simd(const Dataset *ds) {
    const size_t N = ds->N;
    const size_t M = ds->M;
    const size_t D = ds->D;

    double *a_normed = (double *)malloc(M * D * sizeof(double));
    if (a_normed == NULL) {
        return 0.0;
    }

    for (size_t j = 0; j < M; j++) {
        const size_t a_base = j * D;
        double axis_norm_sq = 0.0;
        for (size_t k = 0; k < D; k++) {
            const double x = ds->A[a_base + k];
            axis_norm_sq += x * x;
        }

        const double axis_norm = sqrt(axis_norm_sq);
        const double inv_axis_norm = (axis_norm != 0.0) ? (1.0 / axis_norm) : 0.0;
        for (size_t k = 0; k < D; k++) {
            a_normed[a_base + k] = ds->A[a_base + k] * inv_axis_norm;
        }
    }

    double checksum = 0.0;
    for (size_t i = 0; i < N; i++) {
        const size_t e_base = i * D;
        const double *e_ptr = ds->E + e_base;

        double emb_norm_sq = 0.0;
        for (size_t k = 0; k < D; k++) {
            const double x = e_ptr[k];
            emb_norm_sq += x * x;
        }

        const double emb_norm = sqrt(emb_norm_sq);
        if (emb_norm == 0.0) {
            continue;
        }
        const double inv_emb_norm = 1.0 / emb_norm;

        for (size_t j = 0; j < M; j++) {
            const size_t a_base = j * D;
            const double dot = dot_simd(e_ptr, a_normed + a_base, D);
            checksum += dot * inv_emb_norm;
        }
    }

    free(a_normed);
    return checksum;
}
