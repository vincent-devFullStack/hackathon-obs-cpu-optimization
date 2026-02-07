#include "cosine_simd.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

[[maybe_unused]] double dot_scalar(const double *x, const double *y, std::size_t d) {
    double sum = 0.0;
    for (std::size_t k = 0; k < d; k++) {
        sum += x[k] * y[k];
    }
    return sum;
}

double dot_simd(const double *x, const double *y, std::size_t d) {
#if defined(__AVX2__)
    __m256d acc = _mm256_setzero_pd();
    std::size_t k = 0;
    for (; k + 4 <= d; k += 4) {
        __m256d vx = _mm256_loadu_pd(x + k);
        __m256d vy = _mm256_loadu_pd(y + k);
#if defined(__FMA__)
        acc = _mm256_fmadd_pd(vx, vy, acc);
#else
        acc = _mm256_add_pd(acc, _mm256_mul_pd(vx, vy));
#endif
    }

    alignas(32) double lanes[4] = {0.0, 0.0, 0.0, 0.0};
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

}  // namespace

double cosine_all_pairs_checksum_simd(const Dataset &ds) {
    std::vector<double> a_normed(ds.M * ds.D, 0.0);
    for (std::size_t j = 0; j < ds.M; j++) {
        const std::size_t a_base = j * ds.D;
        double axis_norm_sq = 0.0;
        for (std::size_t k = 0; k < ds.D; k++) {
            const double x = ds.A[a_base + k];
            axis_norm_sq += x * x;
        }

        const double axis_norm = std::sqrt(axis_norm_sq);
        const double inv_axis_norm = (axis_norm != 0.0) ? (1.0 / axis_norm) : 0.0;
        for (std::size_t k = 0; k < ds.D; k++) {
            a_normed[a_base + k] = ds.A[a_base + k] * inv_axis_norm;
        }
    }

    double checksum = 0.0;
    for (std::size_t i = 0; i < ds.N; i++) {
        const std::size_t e_base = i * ds.D;
        const double *e_ptr = ds.E.data() + e_base;

        double emb_norm_sq = 0.0;
        for (std::size_t k = 0; k < ds.D; k++) {
            const double x = e_ptr[k];
            emb_norm_sq += x * x;
        }

        const double emb_norm = std::sqrt(emb_norm_sq);
        if (emb_norm == 0.0) {
            continue;
        }
        const double inv_emb_norm = 1.0 / emb_norm;

        for (std::size_t j = 0; j < ds.M; j++) {
            const std::size_t a_base = j * ds.D;
            const double dot = dot_simd(e_ptr, a_normed.data() + a_base, ds.D);
            checksum += dot * inv_emb_norm;
        }
    }

    return checksum;
}
