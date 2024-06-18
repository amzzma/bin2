#pragma once
#include "AVX2Kernel.h"
#include <memory>
#include <algorithm>
#include <omp.h>
namespace 
{
	int num_procs = omp_get_num_procs();
}

template<typename Dtype>
struct GEMMArgs
{
	int32_t M;
	int32_t N;
	int32_t K;
	int32_t StrideA;
	int32_t StrideB;
	int32_t StrideC;
	Dtype* A;
	Dtype* B;
	Dtype* C;
};

template<typename Dtype>
struct Block 
{
	Dtype *data;
	int32_t Stride;
	int32_t NRow;
	int32_t NCol;

	Block Slice(int32_t RowStart, int32_t ColStart, int32_t NRow, int32_t NCol);
	void Transpose(Block& _Dst);
	void CopyTo(Block& _Dst, bool ATrans, bool BTrans, bool if_add=false);
};

template<typename Dtype, class KernelCalcT>
class GemmKernel
{
public:
	GemmKernel(int M, int K, int N);
	~GemmKernel()=default;

	void apply(GEMMArgs<Dtype>& args);
	void apply2(GEMMArgs<Dtype>& args);

	void SliceM();
	void SliceN(int32_t m, int32_t _m);
	void SliceK(int32_t m, int32_t _m, int32_t n, int32_t _n);

private:
	int M, K, N;
	std::unique_ptr<Dtype[]> _BufferTA;

	Block<Dtype> _InputA;
	Block<Dtype> _TA;
	Block<Dtype> _InputB;
	Block<Dtype> _OutputC;
};

template<typename Dtype>
inline Block<Dtype> Block<Dtype>::Slice(int32_t RowStart, int32_t ColStart, int32_t NRow, int32_t NCol)
{
	return Block
	(
		data + RowStart * Stride + ColStart,
		Stride,
		NRow,
		NCol
	);
}

//_Dst.data必须分配正好的空间大小
template<typename Dtype>
inline void Block<Dtype>::Transpose(Block& _Dst)
{
	for (int row = 0; row < NRow; ++row)
	{
		for (int col = 0; col < NCol; ++col)
		{
			_Dst.data[col * _Dst.Stride + row] = data[row * Stride + col];
		}
	}
	_Dst.Stride = NRow;
	_Dst.NCol = NRow;
	_Dst.NRow = NCol;

	return;
}

template<typename Dtype>
inline void Block<Dtype>::CopyTo(Block<Dtype>& _Dst, bool ATrans, bool BTrans, bool if_add)
{
	if (!if_add)
	{
		if (!ATrans && !BTrans)
		{
			for (int r = 0; r < NRow; ++r)
			{
				int _DstOffset = r * _Dst.Stride;
				int _SrcOffset = r * Stride;
				for (int c = 0; c < NCol; ++c)
					_Dst.data[_DstOffset + c] = data[_SrcOffset + c];
			}
		}
		else if (ATrans && !BTrans) //A是转置，B不是
		{
			for (int r = 0; r < NRow; ++r)
			{
				for (int c = 0; c < NCol; ++c)
					_Dst.data[r * _Dst.Stride + c] = data[c * Stride + r];
			}
		}
		else if (!ATrans && BTrans) //A不是转置，B是
		{
			for (int r = 0; r < NRow; ++r)
			{
				for (int c = 0; c < NCol; ++c)
					_Dst.data[c * _Dst.Stride + r] = data[r * Stride + c];
			}
		}
		else if (ATrans && BTrans)
		{
			for (int r = 0; r < NRow; ++r)
			{
				for (int c = 0; c < NCol; ++c)
					_Dst.data[c * _Dst.Stride + r] = data[c * Stride + r];
			}
		}
	}
	else if (if_add)
	{
		if (!ATrans && !BTrans)
		{
			for (int r = 0; r < NRow; ++r)
			{
				int _DstOffset = r * _Dst.Stride;
				int _SrcOffset = r * Stride;
				for (int c = 0; c < NCol; ++c) 
					_Dst.data[_DstOffset + c] += data[_SrcOffset + c];
			}
		}
		else if (ATrans && !BTrans)
		{
			for (int r = 0; r < NRow; ++r)
			{
				for (int c = 0; c < NCol; ++c)
					_Dst.data[r * _Dst.Stride + c] += data[c * Stride + r];
			}
		}
		else if (!ATrans && !BTrans)
		{
			for (int r = 0; r < NRow; ++r)
			{
				for (int c = 0; c < NCol; ++c)
					_Dst.data[r * _Dst.Stride + c] += data[r * Stride + c];
			}
		}
		else if (ATrans && BTrans)
		{
			for (int r = 0; r < NRow; ++r)
			{
				for (int c = 0; c < NCol; ++c)
					_Dst.data[c * _Dst.Stride + r] += data[c * Stride + r];
			}
		}
	}
}

template<typename Dtype, class KernelCalcT>
GemmKernel<Dtype, KernelCalcT>::GemmKernel(int M, int K, int N)
{
	this->M = M;
	this->N = N;
	this->K = K;
	_BufferTA = std::make_unique<Dtype[]>(M * K);
}

template<typename Dtype>
void SGemm_mxn(int32_t kc, Dtype* a, Dtype* b, Dtype* c, int32_t rs_a, int32_t rs_b, int32_t rs_c, int32_t m, int32_t n)
{
	// a: kc x m
	// b: kc x n
	// C: m x n

	Dtype* pc = c;
	for (int32_t i = 0; i < m; ++i)
	{
		for (int32_t j = 0; j < n; ++j)
		{
			for (int32_t k = 0; k < kc; ++k)
			{
				pc[i * rs_c + j] += a[k * rs_a + i] * b[k * rs_b + j];
			}
		}
	}
}

template<typename Dtype, class KernelCalcT>
void GemmKernel<Dtype, KernelCalcT>::apply(GEMMArgs<Dtype>& args)
{
	_InputA = Block<Dtype>{ args.A, args.StrideA, args.M, args.K };
	_TA = Block<Dtype>{ _BufferTA.get(), args.M, args.K, args.M };
	_InputA.Transpose(_TA);
	_InputB = Block<Dtype>{ args.B, args.StrideB, args.K, args.N };
	_OutputC = Block<Dtype>{ args.C, args.StrideC, args.M, args.N };
	const int32_t _sk = 40;
	int32_t m, n, k;
	//const int32_t Um = M - KernelCalcT::MR, Un = N - KernelCalcT::NR, Uk = K - _sk;

	#pragma omp parallel for num_threads(num_procs / 4 * 3) private(m, n, k) schedule(dynamic, 1) collapse(2)
	for (m = 0; m < M; m += KernelCalcT::MR) 
	{
		for (k = 0; k < K; k += _sk)
		{
			for (n = 0; n < N; n += KernelCalcT::NR) 
			{
				int32_t ai_s = m, ai_e = std::min(m + KernelCalcT::MR, M);
				int32_t bj_s = n, bj_e = std::min(n + KernelCalcT::NR, N);
				int32_t ks = k, ke = std::min(k + _sk, K);

				int32_t _m = ai_e - ai_s, _n = bj_e - bj_s, kd = ke - ks;

				//Block<Dtype> AB = _InputA.Slice(ai_s, ks, _m, kd);
				//Dtype TAd[KernelCalcT::MR * _sk]{};
				//Block<Dtype> TAB{ TAd, _m, kd, _m };
				//AB.CopyTo(TAB, false, true);

				Block<Dtype> TAB = _TA.Slice(ks, ai_s, kd, _m);

				Block<Dtype> BB = _InputB.Slice(ks, bj_s, kd, _n);
				Block<Dtype> CB = _OutputC.Slice(ai_s, bj_s, _m, _n);

				if (_m < KernelCalcT::MR || _n < KernelCalcT::NR) {
					Dtype Td[KernelCalcT::MR * KernelCalcT::NR]{};
					Block<Dtype> TB{ Td, _n, _m, _n };
					TB = TB.Slice(0, 0, _m, _n);

					KernelCalcT::apply(kd, TAB.data, BB.data, TB.data, TAB.Stride, BB.Stride, TB.Stride);
					TB.CopyTo(CB, false, false, true);
					continue;
				}
				KernelCalcT::apply(kd, TAB.data, BB.data, CB.data, TAB.Stride, BB.Stride, CB.Stride);
			}
		}
	}
}

template<typename Dtype, class KernelCalcT>
inline void GemmKernel<Dtype, KernelCalcT>::apply2(GEMMArgs<Dtype>& args)
{
	_InputA = Block<Dtype>{ args.A, args.StrideA, args.M, args.K };
	//_TA = Block<Dtype>{ _BufferTA.get(), args.M, args.K, args.M };
	//_InputA.Transpose(_TA);
	_InputB = Block<Dtype>{ args.B, args.StrideB, args.K, args.N };
	_OutputC = Block<Dtype>{ args.C, args.StrideC, args.M, args.N };

	SliceM();
}

template<typename Dtype, class KernelCalcT>
inline void GemmKernel<Dtype, KernelCalcT>::SliceM()
{
	#pragma omp parallel for num_threads(num_procs/4 * 3) schedule(dynamic)
	for (int32_t m = 0; m <= M - KernelCalcT::MR; m += KernelCalcT::MR)
	{
		SliceN(m, KernelCalcT::MR);
	}
	int32_t lm = M / KernelCalcT::MR * KernelCalcT::MR;
	SliceN(lm, M - lm);
}

template<typename Dtype, class KernelCalcT>
inline void GemmKernel<Dtype, KernelCalcT>::SliceN(int32_t m, int32_t _m)
{
	int32_t n;
	for (n = 0; n <= N - KernelCalcT::NR; n += KernelCalcT::NR)
	{
		SliceK(m, _m, n, KernelCalcT::NR);
	}
	SliceK(m, _m, n, N - n);
}

template<typename Dtype, class KernelCalcT>
inline void GemmKernel<Dtype, KernelCalcT>::SliceK(int32_t ai_s, int32_t _m, int32_t bj_s, int32_t _n)
{
	const int32_t _sk = 35;
	int32_t ks;
	for (ks = 0; ks <= K - _sk; ks += _sk)
	{
		int32_t kd = _sk;

		Block<Dtype> AB = _InputA.Slice(ai_s, ks, _m, kd);
		Dtype TAd[KernelCalcT::MR * _sk]{};
		Block<Dtype> TAB{ TAd, _m, kd, _m };
		AB.CopyTo(TAB, false, true);

		Block<Dtype> BB = _InputB.Slice(ks, bj_s, kd, _n);
		Block<Dtype> CB = _OutputC.Slice(ai_s, bj_s, _m, _n);

		if (_m < KernelCalcT::MR || _n < KernelCalcT::NR)
		{
			Dtype Td[KernelCalcT::MR * KernelCalcT::NR]{ 0.0 };
			Block<Dtype> TB{ Td, _n, _m, _n };
			TB = TB.Slice(0, 0, _m, _n);

			KernelCalcT::apply(kd, TAB.data, BB.data, TB.data, TAB.Stride, BB.Stride, TB.Stride);
			TB.CopyTo(CB, false, false, true);
		}
		else
		{
			KernelCalcT::apply(kd, TAB.data, BB.data, CB.data, TAB.Stride, BB.Stride, CB.Stride);
		}
	}
	{
		int32_t kd = K - ks;

		Block<Dtype> AB = _InputA.Slice(ai_s, ks, _m, kd);
		Dtype TAd[KernelCalcT::MR * _sk]{};
		Block<Dtype> TAB{ TAd, _m, kd, _m };
		AB.CopyTo(TAB, false, true);

		Block<Dtype> BB = _InputB.Slice(ks, bj_s, kd, _n);
		Block<Dtype> CB = _OutputC.Slice(ai_s, bj_s, _m, _n);

		if (_m < KernelCalcT::MR || _n < KernelCalcT::NR)
		{
			Dtype Td[KernelCalcT::MR * KernelCalcT::NR]{ 0.0 };
			Block<Dtype> TB{ Td, _n, _m, _n };
			TB = TB.Slice(0, 0, _m, _n);

			KernelCalcT::apply(kd, TAB.data, BB.data, TB.data, TAB.Stride, BB.Stride, TB.Stride);
			TB.CopyTo(CB, false, false, true);
		}
		else
		{
			KernelCalcT::apply(kd, TAB.data, BB.data, CB.data, TAB.Stride, BB.Stride, CB.Stride);
		}
	}
}
