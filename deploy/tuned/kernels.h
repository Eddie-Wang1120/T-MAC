#include "stdint.h"
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m256_k800_n1_b2(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m6400_k800_n1_b2(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m256_k3200_n1_b2(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m6400_k3200_n1_b2(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m256_k10240_n1_b2(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m6400_k10240_n1_b2(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m20480_k3200_n1_b2(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m320_k3200_n1_b2(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m1600_k3200_n1_b2(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);inline int qgemm_lut_int8(int m, int k, int n, int b, void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C) {

    if (m == 256 && k == 800 && n == 1 && b == 2) return qgemm_lut_t1_int8_m256_k800_n1_b2(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    if (m == 256 && k == 3200 && n == 1 && b == 2) return qgemm_lut_t1_int8_m256_k3200_n1_b2(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    if (m == 256 && k == 10240 && n == 1 && b == 2) return qgemm_lut_t1_int8_m256_k10240_n1_b2(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    if (m == 320 && k == 3200 && n == 1 && b == 2) return qgemm_lut_t1_int8_m320_k3200_n1_b2(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    return -1;
}
inline int preprocessor_int8(int m, int k, int n, int b, void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT) {

    if (m == 6400 && k == 800 && n == 1 && b == 2) return preprocessor_t1_int8_m6400_k800_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);

    if (m == 6400 && k == 3200 && n == 1 && b == 2) return preprocessor_t1_int8_m6400_k3200_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);

    if (m == 6400 && k == 10240 && n == 1 && b == 2) return preprocessor_t1_int8_m6400_k10240_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);

    if (m == 20480 && k == 3200 && n == 1 && b == 2) return preprocessor_t1_int8_m20480_k3200_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);

    if (m == 1600 && k == 3200 && n == 1 && b == 2) return preprocessor_t1_int8_m1600_k3200_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);

    return -1;
}
