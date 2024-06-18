#include "AVX2Kernel.h"
#include <immintrin.h>

void SGemm6x16Avx2(int32_t kc, float* a, float* b, float* c, int32_t rs_a, int32_t rs_b, int32_t rs_c)
{
    // a: kc x MR
    // b: kc x NR

    // C: MR x NR (6 x 16)
    __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    __m256 a00, b00, b01;

    float* pc = c;
    c00 = _mm256_loadu_ps(pc);
    c01 = _mm256_loadu_ps(pc + 8);
    pc += rs_c;

    c10 = _mm256_loadu_ps(pc);
    c11 = _mm256_loadu_ps(pc + 8);
    pc += rs_c;

    c20 = _mm256_loadu_ps(pc);
    c21 = _mm256_loadu_ps(pc + 8);
    pc += rs_c;

    c30 = _mm256_loadu_ps(pc);
    c31 = _mm256_loadu_ps(pc + 8);
    pc += rs_c;

    c40 = _mm256_loadu_ps(pc);
    c41 = _mm256_loadu_ps(pc + 8);
    pc += rs_c;

    c50 = _mm256_loadu_ps(pc);
    c51 = _mm256_loadu_ps(pc + 8);
    pc += rs_c;

    float* pa = a;
    float* pb = b;

    for (int k = 0; k < kc; ++k) {
        b00 = _mm256_loadu_ps(pb);
        b01 = _mm256_loadu_ps(pb + 8);

        a00 = _mm256_broadcast_ss(pa);
        c00 = _mm256_fmadd_ps(a00, b00, c00);
        c01 = _mm256_fmadd_ps(a00, b01, c01);
        pa += 1;

        a00 = _mm256_broadcast_ss(pa);
        c10 = _mm256_fmadd_ps(a00, b00, c10);
        c11 = _mm256_fmadd_ps(a00, b01, c11);
        pa += 1;

        a00 = _mm256_broadcast_ss(pa);
        c20 = _mm256_fmadd_ps(a00, b00, c20);
        c21 = _mm256_fmadd_ps(a00, b01, c21);
        pa += 1;

        a00 = _mm256_broadcast_ss(pa);
        c30 = _mm256_fmadd_ps(a00, b00, c30);
        c31 = _mm256_fmadd_ps(a00, b01, c31);
        pa += 1;

        a00 = _mm256_broadcast_ss(pa);
        c40 = _mm256_fmadd_ps(a00, b00, c40);
        c41 = _mm256_fmadd_ps(a00, b01, c41);
        pa += 1;

        a00 = _mm256_broadcast_ss(pa);
        c50 = _mm256_fmadd_ps(a00, b00, c50);
        c51 = _mm256_fmadd_ps(a00, b01, c51);
        pa += 1;

        //pb += 16;
        pa += (rs_a - 6);
        pb += rs_b;
    }

    pc = c;
    _mm256_storeu_ps(pc, c00);
    _mm256_storeu_ps(pc + 8, c01);
    pc += rs_c;

    _mm256_storeu_ps(pc, c10);
    _mm256_storeu_ps(pc + 8, c11);
    pc += rs_c;

    _mm256_storeu_ps(pc, c20);
    _mm256_storeu_ps(pc + 8, c21);
    pc += rs_c;

    _mm256_storeu_ps(pc, c30);
    _mm256_storeu_ps(pc + 8, c31);
    pc += rs_c;

    _mm256_storeu_ps(pc, c40);
    _mm256_storeu_ps(pc + 8, c41);
    pc += rs_c;

    _mm256_storeu_ps(pc, c50);
    _mm256_storeu_ps(pc + 8, c51);
    pc += rs_c;
}

void SGemm6x16Avx2(int32_t kc, double* a, double* b, double* c, int32_t rs_a, int32_t rs_b, int32_t rs_c)
{
    // -> 6 x 16 (double)  6 x 4
    //__m256d c00, c01, c02, c03, c10, c11, c12, c13;
    //__m256d c20, c21, c22, c23, c30, c31, c32, c33;
    //__m256d c40, c41, c42, c43, c50, c51, c52, c53;

    __m256d _C[6 * 4]{};
    __m256d ta, tb[4]{};

    double* pc = c;

    for (int i = 0; i < 6; ++i)
    {
        _C[4 * i] = _mm256_loadu_pd(pc);
        _C[4 * i + 1] = _mm256_loadu_pd(pc + 4);
        _C[4 * i + 2] = _mm256_loadu_pd(pc + 8);
        _C[4 * i + 3] = _mm256_loadu_pd(pc + 12);
        pc += rs_c;
    }
    double* pa = a;
    double* pb = b;

    for (int k = 0; k < kc; ++k)
    {
        tb[0] = _mm256_loadu_pd(pb);
        tb[1] = _mm256_loadu_pd(pb + 4);
        tb[2] = _mm256_loadu_pd(pb + 8);
        tb[3] = _mm256_loadu_pd(pb + 12);

        ta = _mm256_broadcast_sd(pa);
        _C[4 * 0 + 0] = _mm256_fmadd_pd(ta, tb[0], _C[4 * 0 + 0]);
        _C[4 * 0 + 1] = _mm256_fmadd_pd(ta, tb[1], _C[4 * 0 + 1]);
        _C[4 * 0 + 2] = _mm256_fmadd_pd(ta, tb[2], _C[4 * 0 + 2]);
        _C[4 * 0 + 3] = _mm256_fmadd_pd(ta, tb[3], _C[4 * 0 + 3]);
        pa += 1;

        ta = _mm256_broadcast_sd(pa);
        _C[4 * 1 + 0] = _mm256_fmadd_pd(ta, tb[0], _C[4 * 1 + 0]);
        _C[4 * 1 + 1] = _mm256_fmadd_pd(ta, tb[1], _C[4 * 1 + 1]);
        _C[4 * 1 + 2] = _mm256_fmadd_pd(ta, tb[2], _C[4 * 1 + 2]);
        _C[4 * 1 + 3] = _mm256_fmadd_pd(ta, tb[3], _C[4 * 1 + 3]);
        pa += 1;

        ta = _mm256_broadcast_sd(pa);
        _C[4 * 2 + 0] = _mm256_fmadd_pd(ta, tb[0], _C[4 * 2 + 0]);
        _C[4 * 2 + 1] = _mm256_fmadd_pd(ta, tb[1], _C[4 * 2 + 1]);
        _C[4 * 2 + 2] = _mm256_fmadd_pd(ta, tb[2], _C[4 * 2 + 2]);
        _C[4 * 2 + 3] = _mm256_fmadd_pd(ta, tb[3], _C[4 * 2 + 3]);
        pa += 1;

        ta = _mm256_broadcast_sd(pa);
        _C[4 * 3 + 0] = _mm256_fmadd_pd(ta, tb[0], _C[4 * 3 + 0]);
        _C[4 * 3 + 1] = _mm256_fmadd_pd(ta, tb[1], _C[4 * 3 + 1]);
        _C[4 * 3 + 2] = _mm256_fmadd_pd(ta, tb[2], _C[4 * 3 + 2]);
        _C[4 * 3 + 3] = _mm256_fmadd_pd(ta, tb[3], _C[4 * 3 + 3]);
        pa += 1;

        ta = _mm256_broadcast_sd(pa);
        _C[4 * 4 + 0] = _mm256_fmadd_pd(ta, tb[0], _C[4 * 4 + 0]);
        _C[4 * 4 + 1] = _mm256_fmadd_pd(ta, tb[1], _C[4 * 4 + 1]);
        _C[4 * 4 + 2] = _mm256_fmadd_pd(ta, tb[2], _C[4 * 4 + 2]);
        _C[4 * 4 + 3] = _mm256_fmadd_pd(ta, tb[3], _C[4 * 4 + 3]);
        pa += 1;

        ta = _mm256_broadcast_sd(pa);
        _C[4 * 5 + 0] = _mm256_fmadd_pd(ta, tb[0], _C[4 * 5 + 0]);
        _C[4 * 5 + 1] = _mm256_fmadd_pd(ta, tb[1], _C[4 * 5 + 1]);
        _C[4 * 5 + 2] = _mm256_fmadd_pd(ta, tb[2], _C[4 * 5 + 2]);
        _C[4 * 5 + 3] = _mm256_fmadd_pd(ta, tb[3], _C[4 * 5 + 3]);
        pa += 1;

        pa += (rs_a - 6);
        pb += rs_b;
    }
    pc = c;

    for (int i = 0; i < 6; ++i)
    {
        _mm256_storeu_pd(pc, _C[i * 4]);
        _mm256_storeu_pd(pc + 4, _C[i * 4 + 1]);
        _mm256_storeu_pd(pc + 8, _C[i * 4 + 2]);
        _mm256_storeu_pd(pc + 12, _C[i * 4 + 3]);
        pc += rs_c;
    }
}

void SGemm6x16Avx2(int32_t kc, int32_t* a, int32_t* b, int32_t* c, int32_t rs_a, int32_t rs_b, int32_t rs_c)
{
    __m256i _C[6 * 2]{};
    __m256i ta, tb[2]{};
    int32_t* pc = c;

    for (int i = 0; i < 6; ++i)
    {
        _C[2 * i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pc));
        _C[2 * i + 1] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pc + 8));
        pc += rs_c;
    }

    int32_t* pa = a;
    int32_t* pb = b;

    for (int k = 0; k < kc; ++k)
    {
        tb[0] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb));
        tb[1] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + 8));
    
        ta = _mm256_set1_epi32(pa[0]);
        _C[2 * 0 + 0] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[0]), _C[2 * 0 + 0]);
        _C[2 * 0 + 1] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[1]), _C[2 * 0 + 1]);
        pa += 1;

        ta = _mm256_set1_epi32(pa[0]);
        _C[2 * 1 + 0] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[0]), _C[2 * 1 + 0]);
        _C[2 * 1 + 1] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[1]), _C[2 * 1 + 1]);
        pa += 1;

        ta = _mm256_set1_epi32(pa[0]);
        _C[2 * 2 + 0] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[0]), _C[2 * 2 + 0]);
        _C[2 * 2 + 1] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[1]), _C[2 * 2 + 1]);
        pa += 1;

        ta = _mm256_set1_epi32(pa[0]);
        _C[2 * 3 + 0] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[0]), _C[2 * 3 + 0]);
        _C[2 * 3 + 1] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[1]), _C[2 * 3 + 1]);
        pa += 1;

        ta = _mm256_set1_epi32(pa[0]);
        _C[2 * 4 + 0] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[0]), _C[2 * 4 + 0]);
        _C[2 * 4 + 1] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[1]), _C[2 * 4 + 1]);
        pa += 1;

        ta = _mm256_set1_epi32(pa[0]);
        _C[2 * 5 + 0] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[0]), _C[2 * 5 + 0]);
        _C[2 * 5 + 1] = _mm256_add_epi32(_mm256_mullo_epi32(ta, tb[1]), _C[2 * 5 + 1]);
        pa += 1;

        pa += (rs_a - 6);
        pb += rs_b;
    }
    pc = c;

    for (int i = 0; i < 6; ++i)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pc), _C[i * 2]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pc) + 1, _C[i * 2 + 1]);
        pc += rs_c;
    }
}
