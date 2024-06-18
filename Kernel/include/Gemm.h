#include "GemmKernel.h"

static struct Float32AVX2Gemm
{
	static const int32_t MR = 6;
	static const int32_t NR = 16;

	static void apply(int32_t kc, float* a, float* b, float* c, int32_t rs_a, int32_t rs_b, int32_t rs_c);
};

static struct Float64AVX2Gemm
{
	static const int32_t MR = 6;
	static const int32_t NR = 16;

	static void apply(int32_t kc, double* a, double* b, double* c, int32_t rs_a, int32_t rs_b, int32_t rs_c);
};

static struct Int32AVX2Gemm
{
	static const int32_t MR = 6;
	static const int32_t NR = 16;

	static void apply(int32_t kc, int32_t* a, int32_t* b, int32_t* c, int32_t rs_a, int32_t rs_b, int32_t rs_c);
};

// please check c was filled with 0.
template<typename Dtype, class KernelCalcT>
void MatMul(Dtype* a, Dtype* b, Dtype* c, int32_t sa, int32_t sb, int32_t sc, int32_t m, int32_t n, int32_t k)
{
	GemmKernel<Dtype, KernelCalcT> GK(m, k, n);
	GEMMArgs<Dtype> args(m, n, k, sa, sb, sc, a, b, c);
	GK.apply(args);
}