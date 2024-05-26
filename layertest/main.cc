#include "t-mac/kernels.h"
#include <stdint.h>
#include <cstdlib>
#include <memory>
#include <cstring>

using namespace std;
typedef uint16_t ggml_fp16_t;
typedef ggml_fp16_t half;

static void * aligned_malloc(size_t size) {
    void * ptr = nullptr;
    posix_memalign(&ptr, 64, size);
    return ptr;
}

typedef float tmac_float_type;

template <int Bits, int SIMD_LEN = 16>
struct BlockQTypeAccessor {
    static constexpr int simd_n_elem = SIMD_LEN * 8 / Bits;

    static uint8_t get_q(uint8_t * data, int idx, int group_size) {
        const uint8_t * qs = (const uint8_t *) (data);
        int internal_idx = idx % group_size;
        const uint8_t * simd_qs = qs + internal_idx / simd_n_elem * SIMD_LEN;
        int simd_idx = internal_idx % simd_n_elem;
        return simd_qs[simd_idx % SIMD_LEN] >> (simd_idx / SIMD_LEN * Bits);
    }

    // static tmac_float_type get_scale(const void * data, int idx) {
    //     ggml_fp16_t d = ((const block_t *) data)[idx / group_size].d;
    //     if (sizeof(tmac_float_type) == 2) {
    //         tmac_float_type * fp16dp = reinterpret_cast<tmac_float_type *>(&d);
    //         return *fp16dp;
    //     } else {
    //         return ggml_fp16_to_fp32(((const block_t *) data)[idx / group_size].d);
    //     }
    // }
};


void ggml_tmac_transform_tensor(uint8_t * tensor, int k, int m) {

    const int bits = 4;
    const int g = 4;
    const int ngroups_per_elem = 2;

    // int k = 4096;
    // int m = 4096;  // `n` in llama.cpp

    // TMAC::TMACGeMMConfig kcfg = wrapper->get_kcfg(m, k, 1, bits);
    const int bm              = 256;
    const int simd_n_in       = 16;
    const int simd_n_out      = 8;
    const int kfactor         = 8;
    const int group_size      = 2;  // could be different from block size in llama.cpp
    const int lut_scales_size = 128;
    const int scales_size     = 524288;
    const int n_tile_num      = 64;

    const int mgroup = ngroups_per_elem * simd_n_in;
    m = m * bits;

    uint8_t * qweights = (uint8_t *) aligned_malloc(k * m / 8);
    tmac_float_type * scales = (tmac_float_type *) aligned_malloc(scales_size * sizeof(tmac_float_type));

// for fast testing
// #define TMAC_EMPTY_WEIGHTS
// #ifndef TMAC_EMPTY_WEIGHTS
    // TODO: optimize to accelerate weights loading
    uint8_t * buf1 = new uint8_t[m * k];
    uint8_t * buf2 = new uint8_t[m * k / g];

    // # (M // bits, K, bits)
    // w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
    for (int im = 0; im < m / bits; im++) {
        for (int ik = 0; ik < k; ik++) {
            for (int ib = 0; ib < bits; ib++) {
                uint8_t v = tensor[im*m / bits + ik];
                // if (bits == 2) {
                //     v = BlockQTypeAccessor<2>::get_q(tensor, im * k + ik, group_size);
                // } else if (bits == 3) {
                //     v = BlockQTypeAccessor<3>::get_q(tensor, im * k + ik, group_size);
                // } else if (bits == 4) {
                //     v = BlockQTypeAccessor<4>::get_q(tensor, im * k + ik, group_size);
                // }
                buf1[im * k * bits + ik * bits + ib] = (v >> ib) & 1;
            }
        }
    }

    for (int i=0; i<m*k; i++) {
        printf("%d ", buf1[i]);
    }
    printf("\n");

    // # (M // bits, K, bits) -> (M // bits, bits, K) -> (M // bits, bits, K // g, g) -> (M // bits, bits, K // g)
    // w = w.transpose(0, 2, 1).reshape(M // bits, bits, K // g, g)
    // w = sum([(w[:, :, :, ig] << ig) for ig in range(g)])
    memset(buf2, 0, m * k / g);
    for (int im = 0; im < m / bits; im++) {
        for (int ik = 0; ik < k; ik++) {
            for (int ib = 0; ib < bits; ib++) {
                int new_im = im;
                int new_ib = ib;
                int new_ik = ik / g;
                int new_ig = ik % g;
                buf2[new_im * bits * k / g + new_ib * k / g + new_ik] += buf1[im * k * bits + ik * bits + ib] << new_ig;
            }
        }
    }

    for (int i=0; i<m*k / g; i++) {
        printf("%d ", buf2[i]);
    }
    printf("\n");

//     // # 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
//     // # for bits=3
//     // # bit0: [0, 8), bit1: [8, 16), bit2: [16, 24), bit0: [24, 32)
//     // # (M // bits // simd_n_float16, bits, simd_n_float16, K // g)
//     // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
//     // mgroup = ngroups_per_elem * simd_n_in
//     // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
//     // #             0        1             2             3                 4                  5
//     // w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
//     // w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
//     memset(qweights, 0, m * k / g / ngroups_per_elem);
//     for (int im = 0; im < m / bits; im++) {
//         for (int ib = 0; ib < bits; ib++) {
//             for (int ik = 0; ik < k / g; ik++) {
//                 int new_im = im / simd_n_out;
//                 int new_isno = im % simd_n_out;
//                 int new_ib = ib;
//                 int new_ik = ik;
//                 // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
//                 int new_idx = new_im * bits * simd_n_out * k / g + new_ib * simd_n_out * k / g + new_isno * k / g + new_ik;
//                 // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
//                 int nb2 = k / g;
//                 int nb1 = simd_n_in * nb2;
//                 int nb0 = ngroups_per_elem * nb1;
//                 new_im = new_idx / nb0;
//                 int new_ing = (new_idx % nb0) / nb1;
//                 int new_isni = (new_idx % nb1) / nb2;
//                 new_ik = (new_idx % nb2);
//                 new_idx = new_im * ngroups_per_elem * simd_n_in * k / g + new_isni * ngroups_per_elem * k / g + new_ing * k / g + new_ik;
//                 // #             0        1             2             3                 4                  5
//                 // w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
//                 int nb4 = kfactor;
//                 int nb3 = k / g / kfactor * nb4;
//                 nb2 = ngroups_per_elem * nb3;
//                 nb1 = simd_n_in * nb2;
//                 nb0 = bm / mgroup * nb1;
//                 new_im = new_idx / nb0;
//                 int new_ibm = (new_idx % nb0) / nb1;
//                 new_isni = (new_idx % nb1) / nb2;
//                 new_ing = (new_idx % nb2) / nb3;
//                 new_ik = (new_idx % nb3) / nb4;
//                 int new_ikf = (new_idx % nb4);
//                 new_idx = new_im * k / g / kfactor * bm / mgroup * kfactor * simd_n_in * ngroups_per_elem +
//                           new_ik * bm / mgroup * kfactor * simd_n_in * ngroups_per_elem +
//                           new_ibm * kfactor * simd_n_in * ngroups_per_elem +
//                           new_ikf * simd_n_in * ngroups_per_elem +
//                           new_isni * ngroups_per_elem +
//                           new_ing;
//                 new_idx = new_idx / ngroups_per_elem;
//                 // w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
//                 qweights[new_idx] += buf2[im * bits * k / g + ib * k / g + ik] << (new_ing * g);
//             }
//         }
//     }

//     int m_group_size, k_group_size;
//     if (scales_size < m / bits) {  // BitNet-like scale (m_groups,)
//         m_group_size = m / bits / scales_size;
//         k_group_size = k;
//     } else {  // GPTQ-like scale (m / bits, k / group_size)
//         GGML_ASSERT(scales_size == m / bits * k / group_size);
//         m_group_size = 1;
//         k_group_size = group_size;
//     }

//     // scales = scales.reshape(M // bm, bm // bits, K // group_size).transpose(0, 2, 1)
//     for (int im = 0; im < m / bits; im += m_group_size) {
//         for (int ik = 0; ik < k; ik += k_group_size) {
//             tmac_float_type scale;
//             int idx = im * k + ik;
//             if (bits == 2) {
//                 scale = BlockQTypeAccessor<2>::get_scale(tensor->data, idx);
//             } else if (bits == 3) {
//                 scale = BlockQTypeAccessor<3>::get_scale(tensor->data, idx);
//             } else if (bits == 4) {
//                 scale = BlockQTypeAccessor<4>::get_scale(tensor->data, idx);
//             }
//             int new_idx;
//             if (scales_size < m / bits) {
//                 new_idx = im / m_group_size;
//             } else {
//                 idx = idx / group_size;
//                 int new_im = idx / (bm / bits * k / group_size);
//                 int new_ibm = (idx % (bm / bits * k / group_size)) / (k / group_size);
//                 int new_ik = (idx % (k / group_size));
//                 new_idx = new_im * k / group_size * bm / bits + new_ik * bm / bits + new_ibm;
//             }
//             scales[new_idx] = scale;
//         }
//     }

//     delete[] buf1;
//     delete[] buf2;
// #else
//     memset(qweights, 0x88, k * m / 8);
//     for (int i = 0; i < scales_size; i++) {
//         scales[i] = 1.0f;
//     }

// #endif
}

int main() {

    int m = 4;
    int k = 4;

    float* lut_scale   = (float*)malloc(64*sizeof(float));
    float* lut_bias   = (float*)malloc(64*sizeof(float));
    float* q_lut   = (float*)malloc(1024 * 16*sizeof(float));
    uint8_t* A = (uint8_t*)malloc(m*k*sizeof(uint8_t));
    float* B   = (float*)malloc(4096*sizeof(float));
    
    for (int i=0; i<m*k; i++) {
        A[i] = i % 16;
    }

    for (int i=0; i<m*k; i++) {
        printf("%d ", A[i]);
    }
    printf("\n");

    ggml_tmac_transform_tensor(A, k, m);

    // preprocessor_int8(16384, 4096, 1, 4, (void*)B, (void*)lut_scale, (void*)lut_bias, (void*)q_lut);

    return 0;
}