#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 64
#endif

static inline void micro_kernel_4x4(int K,
                                   const double *A, int lda,
                                   const double *B, int ldb,
                                   double *C, int ldc) {
    // registers for C 4x4
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    for (int k = 0; k < K; ++k) {
        // load A column (4 elements)
        __m256d a_col = _mm256_loadu_pd(&A[k*lda]); // A is MR x K with column-major w.r.t k loop
        // load B row elements and broadcast
        double b0 = B[k*ldb + 0];
        double b1 = B[k*ldb + 1];
        double b2 = B[k*ldb + 2];
        double b3 = B[k*ldb + 3];

        __m256d bb0 = _mm256_set1_pd(b0);
        __m256d bb1 = _mm256_set1_pd(b1);
        __m256d bb2 = _mm256_set1_pd(b2);
        __m256d bb3 = _mm256_set1_pd(b3);

        c0 = _mm256_fmadd_pd(a_col, bb0, c0);
        c1 = _mm256_fmadd_pd(a_col, bb1, c1);
        c2 = _mm256_fmadd_pd(a_col, bb2, c2);
        c3 = _mm256_fmadd_pd(a_col, bb3, c3);
    }

    // store back to C (C is ld c rows)
    // c? contain 4 elements each column -> need to add to existing C
    double tmp[4];

    _mm256_storeu_pd(tmp, c0);
    for (int i = 0; i < 4; ++i) C[i*ldc + 0] += tmp[i];
    _mm256_storeu_pd(tmp, c1);
    for (int i = 0; i < 4; ++i) C[i*ldc + 1] += tmp[i];
    _mm256_storeu_pd(tmp, c2);
    for (int i = 0; i < 4; ++i) C[i*ldc + 2] += tmp[i];
    _mm256_storeu_pd(tmp, c3);
    for (int i = 0; i < 4; ++i) C[i*ldc + 3] += tmp[i];
}

void gemm_blocked(const double *A, const double *B, double *C, int M, int N, int K) {
    // Initialize C (caller may choose to pre-initialize)
    // We'll assume C already contains initial values (e.g., zeros) and accumulate into it.

    // Outer blocking over M and N
    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < M; i0 += BM) {
        int ib = (i0 + BM > M) ? (M - i0) : BM;
        for (int j0 = 0; j0 < N; j0 += BN) {
            int jb = (j0 + BN > N) ? (N - j0) : BN;

            // For each block pair, iterate over K in blocks
            for (int k0 = 0; k0 < K; k0 += BK) {
                int kb = (k0 + BK > K) ? (K - k0) : BK;

                // micro-tiling inside block: process 4x4 tiles
                for (int i = i0; i < i0 + ib; i += 4) {
                    int imax = (i + 4 <= i0 + ib) ? 4 : (i0 + ib - i);
                    for (int j = j0; j < j0 + jb; j += 4) {
                        int jmax = (j + 4 <= j0 + jb) ? 4 : (j0 + jb - j);

                        // Handle full 4x4 tiles with micro-kernel
                        if (imax == 4 && jmax == 4) {
                            // Prepare pointers for micro kernel
                            // A_block pointer: start at A[i, k0]
                            // We want A submatrix of shape (4 x kb) but micro kernel expects K on loop
                            // We'll create a small temporary packed buffer for A columns for efficiency
                            // Pack A: reshape to access A_col[k*4 + r]
                            double Acol[4 * BK];
                            const double *Ap = &A[i*K + k0];
                            // Pack A in column-major per k
                            for (int kk = 0; kk < kb; ++kk) {
                                for (int ii = 0; ii < 4; ++ii) {
                                    Acol[kk*4 + ii] = Ap[ii*K + kk];
                                }
                            }

                            // B pointer: &B[k0, j]
                            const double *Bp = &B[k0*N + j];

                            // C pointer: &C[i, j]
                            double *Cp = &C[i*N + j];

                            // For K in chunks of 1..kb, call micro kernel for first 4 elements at a time
                            // But micro_kernel_4x4 expects K and A packed as (K x 4) with lda=4 and B pointer with ldb=N
                            // We'll call micro kernel over the entire kb using our packed Acol and Bp.

                            // Create a small wrapper that iterates over kb in steps and uses micro kernel
                            // We'll run micro kernel manually with Acol as K x 4 (lda=4) and Bp as K x 4 (ldb=N)

                            // Because micro_kernel_4x4 expects B to have 4 columns contiguous (ldb >= 4)
                            // and uses B[k*ldb + col]

                            // We'll call micro_kernel_4x4 for the entire kb by looping with K=kb
                            micro_kernel_4x4(kb, Acol, 4, Bp, N, Cp, N);

                        } else {
                            // Handle edge tiles (small sizes) with scalar fallback
                            for (int ii = i; ii < i + imax; ++ii) {
                                for (int jj = j; jj < j + jmax; ++jj) {
                                    double sum = 0.0;
                                    for (int kk = k0; kk < k0 + kb; ++kk) {
                                        sum += A[ii*K + kk] * B[kk*N + jj];
                                    }
                                    C[ii*N + jj] += sum;
                                }
                            }
                        }

                    }
                }

            }
        }
    }
}

#ifdef TEST_MAIN
int main() {
    int M = 1024, K = 1024, N = 1024;
    double *A = aligned_alloc(64, M*K*sizeof(double));
    double *B = aligned_alloc(64, K*N*sizeof(double));
    double *C = aligned_alloc(64, M*N*sizeof(double));

    for (int i = 0; i < M*K; i++) A[i] = (i % 100) * 0.001;
    for (int i = 0; i < K*N; i++) B[i] = (i % 50) * 0.002;
    memset(C, 0, M*N*sizeof(double));

    double start = omp_get_wtime();
    gemm_blocked(A, B, C, M, N, K);
    double end = omp_get_wtime();
    printf("Done. Time = %.4f s\n", end - start);
    printf("C[0]=%.4f, C[last]=%.4f\n", C[0], C[M*N-1]);

    free(A); free(B); free(C);
    return 0;
}
#endif
