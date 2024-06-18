#pragma once
#include "Gemm.h"

void Float32AVX2Gemm::apply(int32_t kc, float* a, float* b, float* c, int32_t rs_a, int32_t rs_b, int32_t rs_c)
{
	SGemm6x16Avx2(kc, a, b, c, rs_a, rs_b, rs_c);
}

void Float64AVX2Gemm::apply(int32_t kc, double* a, double* b, double* c, int32_t rs_a, int32_t rs_b, int32_t rs_c)
{
	SGemm6x16Avx2(kc, a, b, c, rs_a, rs_b, rs_c);
}

void Int32AVX2Gemm::apply(int32_t kc, int32_t* a, int32_t* b, int32_t* c, int32_t rs_a, int32_t rs_b, int32_t rs_c)
{
	SGemm6x16Avx2(kc, a, b, c, rs_a, rs_b, rs_c);
}