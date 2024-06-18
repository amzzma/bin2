#include "test_bin.cpp"
#include "Kernel\include\Gemm.h"


int main()
{
    weight data;
    constexpr int32_t m = 11, n = 19, k = 45141;
    testbin2struct(data);
    float* a = reinterpret_cast<float*>(data.layer_1_data.get()),
        * b = reinterpret_cast<float*>(data.bXXX_data.get()),
        c[m * n]{ 0. };
    
    MatMul<float, Float32AVX2Gemm>(a, b, c, k, n, n, m, n, k);

    //for (int i = 0; i < m * k; ++i)
    //    std::cout << a[i] << " ";
    //pln
    //for (int i = 0; i < n * k; ++i)
    //    std::cout << b[i] << " ";
    //pln
    for(int i=0; i<m; ++i)
    {
        for(int j=0; j<n; ++j)
        {
            std::cout << c[i*n+j] << " ";
        }
        pln
    }
    pln
    return 0;
}