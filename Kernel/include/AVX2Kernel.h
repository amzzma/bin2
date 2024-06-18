#include <stdint.h>

void SGemm6x16Avx2(int32_t kc, float* a, float* b, float* c, int32_t rs_a, int32_t rs_b, int32_t rs_c);
void SGemm6x16Avx2(int32_t kc, double* a, double* b, double* c, int32_t rs_a, int32_t rs_b, int32_t rs_c);
void SGemm6x16Avx2(int32_t kc, int32_t* a, int32_t* b, int32_t* c, int32_t rs_a, int32_t rs_b, int32_t rs_c);