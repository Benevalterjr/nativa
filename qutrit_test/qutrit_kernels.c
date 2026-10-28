#include "qutrit_kernels.h"

/* ------------------------------------------------------------------------- */
/* 1. INT4 Group-wise (32 weights per block)                                 */
/* ------------------------------------------------------------------------- */

QK_API void nat_quantize_int4_g32(const float* W, BlockQ4_32* out_blocks, int64_t M, int64_t K) {
    int64_t num_blocks_per_row = (K + 31) / 32;
    int64_t i;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        const float* row_in = W + i * K;
        BlockQ4_32* row_out = out_blocks + i * num_blocks_per_row;

        for (int64_t b = 0; b < num_blocks_per_row; b++) {
            int64_t base = b * 32;
            float max_abs = 0.0f;
            for (int j = 0; j < 32; j++) {
                float val = (base + j < K) ? fabsf(row_in[base + j]) : 0.0f;
                if (val > max_abs) max_abs = val;
            }

            float scale = max_abs / 7.0f + 1e-7f;
            float inv_scale = 1.0f / scale;
            row_out[b].scale = scale;

            for (int j = 0; j < 16; j++) {
                int64_t idx0 = base + 2 * j;
                int64_t idx1 = base + 2 * j + 1;

                float v0 = (idx0 < K) ? row_in[idx0] : 0.0f;
                float v1 = (idx1 < K) ? row_in[idx1] : 0.0f;

                int q0 = (int)roundf(v0 * inv_scale);
                int q1 = (int)roundf(v1 * inv_scale);

                if (q0 < -8) q0 = -8; else if (q0 > 7) q0 = 7;
                if (q1 < -8) q1 = -8; else if (q1 > 7) q1 = 7;

                uint8_t nib0 = (uint8_t)(q0 & 0x0F);
                uint8_t nib1 = (uint8_t)(q1 & 0x0F);
                row_out[b].qs[j] = nib0 | (nib1 << 4);
            }
        }
    }
}

QK_API void nat_gemv_int4_g32(const BlockQ4_32* W, const float* x, float* y, int64_t M, int64_t K) {
    int64_t num_blocks_per_row = (K + 31) / 32;
    int64_t i;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        const BlockQ4_32* row_blocks = W + i * num_blocks_per_row;
        float total_acc = 0.0f;

        for (int64_t b = 0; b < num_blocks_per_row; b++) {
            float scale = row_blocks[b].scale;
            const uint8_t* qs = row_blocks[b].qs;
            int64_t base = b * 32;
            float block_sum = 0.0f;

            for (int j = 0; j < 16; j++) {
                uint8_t byte_val = qs[j];
                int8_t q0 = (int8_t)(byte_val & 0x0F);
                int8_t q1 = (int8_t)((byte_val >> 4) & 0x0F);

                /* Sign-extend 4-bit two's complement */
                if (q0 >= 8) q0 -= 16;
                if (q1 >= 8) q1 -= 16;

                int64_t idx0 = base + 2 * j;
                int64_t idx1 = base + 2 * j + 1;

                if (idx0 < K) block_sum += (float)q0 * x[idx0];
                if (idx1 < K) block_sum += (float)q1 * x[idx1];
            }
            total_acc += block_sum * scale;
        }
        y[i] = total_acc;
    }
}

/* ------------------------------------------------------------------------- */
/* 2. Ternary and 5-trits Kernels                                            */
/* ------------------------------------------------------------------------- */

QK_API void nat_gemv_ternary_int8(const int8_t* W, const float* x, float* y, int64_t M, int64_t K, float gamma) {
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        float acc = 0.0f;
        const int8_t* row = W + i * K;
        for (int64_t k = 0; k < K; k++) {
            int8_t w = row[k];
            if (w == 1) acc += x[k];
            else if (w == -1) acc -= x[k];
        }
        y[i] = acc * gamma;
    }
}

QK_API void nat_pack_5trits_2d(const int8_t* in, uint8_t* out_bytes, int64_t M, int64_t K) {
    int64_t bpr = (K + 4) / 5;
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        const int8_t* row_in = in + i * K;
        uint8_t* row_out = out_bytes + i * bpr;
        for (int64_t g = 0; g < bpr; g++) {
            int64_t base = g * 5;
            uint16_t b_val = 0;
            uint16_t mul = 1;
            for (int j = 0; j < 5; j++) {
                int8_t t = (base + j < K) ? (row_in[base + j] + 1) : 1;
                b_val += (uint16_t)(t * mul);
                mul *= 3;
            }
            row_out[g] = (uint8_t)b_val;
        }
    }
}

QK_API void nat_gemv_veclut_5trits(const uint8_t* W_packed, const float* x, float* y, int64_t M, int64_t K, float gamma) {
    int64_t num_groups = (K + 4) / 5;
    float* lut_full = (float*)malloc(num_groups * 256 * sizeof(float));
    if (!lut_full) return;

    for (int64_t g = 0; g < num_groups; g++) {
        int64_t base = g * 5;
        float x0 = (base < K) ? x[base] : 0.0f;
        float x1 = (base + 1 < K) ? x[base + 1] : 0.0f;
        float x2 = (base + 2 < K) ? x[base + 2] : 0.0f;
        float x3 = (base + 3 < K) ? x[base + 3] : 0.0f;
        float x4 = (base + 4 < K) ? x[base + 4] : 0.0f;

        float sub01[9];
        float sub23[9];
        float sub4[3];
        for (int t1 = 0; t1 < 3; t1++) {
            for (int t0 = 0; t0 < 3; t0++) {
                sub01[t0 + 3 * t1] = (float)(t0 - 1) * x0 + (float)(t1 - 1) * x1;
                sub23[t0 + 3 * t1] = (float)(t0 - 1) * x2 + (float)(t1 - 1) * x3;
            }
        }
        for (int t4 = 0; t4 < 3; t4++) {
            sub4[t4] = (float)(t4 - 1) * x4;
        }

        float* g_lut = lut_full + g * 256;
        for (int b = 0; b < 243; b++) {
            /* Verified little-endian index arithmetic */
            g_lut[b] = sub01[b % 9] + sub23[(b / 9) % 9] + sub4[b / 81];
        }
        for (int b = 243; b < 256; b++) {
            g_lut[b] = 0.0f;
        }
    }

    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        const uint8_t* row = W_packed + i * num_groups;
        const float* lut_ptr = lut_full;
        int64_t g = 0;

        for (; g + 3 < num_groups; g += 4) {
            acc0 += lut_ptr[row[g]];
            acc1 += (lut_ptr + 256)[row[g + 1]];
            acc2 += (lut_ptr + 512)[row[g + 2]];
            acc3 += (lut_ptr + 768)[row[g + 3]];
            lut_ptr += 1024;
        }
        for (; g < num_groups; g++) {
            acc0 += lut_ptr[row[g]];
            lut_ptr += 256;
        }
        y[i] = (acc0 + acc1 + acc2 + acc3) * gamma;
    }

    free(lut_full);
}

/* ------------------------------------------------------------------------- */
/* 3. Qutrit Packing and Popcount GEMV Kernels                                */
/* ------------------------------------------------------------------------- */

QK_API void nat_pack_qutrit_baseline(const float* W, uint64_t* W_pos, uint64_t* W_neg, int64_t M, int64_t K, float gamma) {
    int64_t num_blocks = (K + 63) / 64;
    float th = 0.5f * gamma;
    int64_t i;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        const float* row_in = W + i * K;
        uint64_t* r_pos = W_pos + i * num_blocks;
        uint64_t* r_neg = W_neg + i * num_blocks;

        for (int64_t blk = 0; blk < num_blocks; blk++) {
            uint64_t p_pos = 0;
            uint64_t p_neg = 0;
            int64_t base = blk * 64;

            for (int b = 0; b < 64; b++) {
                int64_t k = base + b;
                if (k < K) {
                    float val = row_in[k];
                    if (val > th) {
                        p_pos |= (1ULL << b);
                    } else if (val < -th) {
                        p_neg |= (1ULL << b);
                    }
                }
            }
            r_pos[blk] = p_pos;
            r_neg[blk] = p_neg;
        }
    }
}

QK_API void nat_pack_qutrit_subquantum(const float* W, uint64_t* W_pos, uint64_t* W_neg, int64_t M, int64_t K, float gamma, float low_ratio) {
    int64_t num_blocks = (K + 63) / 64;
    float th_high = 0.5f * gamma;
    float th_low = low_ratio * gamma;
    int64_t i;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        const float* row_in = W + i * K;
        uint64_t* r_pos = W_pos + i * num_blocks;
        uint64_t* r_neg = W_neg + i * num_blocks;

        for (int64_t blk = 0; blk < num_blocks; blk++) {
            uint64_t p_pos = 0;
            uint64_t p_neg = 0;
            int64_t base = blk * 64;

            for (int b = 0; b < 64; b++) {
                int64_t k = base + b;
                if (k < K) {
                    float val = row_in[k];
                    if (val >= th_high) {
                        p_pos |= (1ULL << b);
                    } else if (val <= -th_high) {
                        p_neg |= (1ULL << b);
                    } else if (fabsf(val) >= th_low) {
                        /* Subquantum state '11' assigned when sign matches chessboard phase (-1)^(i+k) */
                        int phase = ((i + k) & 1) ? -1 : 1;
                        if ((val > 0.0f && phase > 0) || (val < 0.0f && phase < 0)) {
                            p_pos |= (1ULL << b);
                            p_neg |= (1ULL << b);
                        }
                    }
                }
            }
            r_pos[blk] = p_pos;
            r_neg[blk] = p_neg;
        }
    }
}

QK_API void nat_pack_qutrit_prob(const float* W, uint64_t* W_pos, uint64_t* W_neg, int64_t M, int64_t K, float gamma, float low_th, float high_th) {
    int64_t num_blocks = (K + 63) / 64;
    float th_high = 0.5f * gamma;
    float th_z_low = low_th * gamma;
    float th_z_high = high_th * gamma;
    int64_t i;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        const float* row_in = W + i * K;
        uint64_t* r_pos = W_pos + i * num_blocks;
        uint64_t* r_neg = W_neg + i * num_blocks;

        for (int64_t blk = 0; blk < num_blocks; blk++) {
            uint64_t p_pos = 0;
            uint64_t p_neg = 0;
            int64_t base = blk * 64;

            for (int b = 0; b < 64; b++) {
                int64_t k = base + b;
                if (k < K) {
                    float val = row_in[k];
                    if (val >= th_high) {
                        p_pos |= (1ULL << b);
                    } else if (val <= -th_high) {
                        p_neg |= (1ULL << b);
                    } else {
                        float abs_v = fabsf(val);
                        if (abs_v >= th_z_low && abs_v < th_z_high) {
                            /* Probabilistic uncertainty zone marked with 11 */
                            p_pos |= (1ULL << b);
                            p_neg |= (1ULL << b);
                        }
                    }
                }
            }
            r_pos[blk] = p_pos;
            r_neg[blk] = p_neg;
        }
    }
}

QK_API void nat_quantize_activations_popcnt(const float* x, uint64_t* x_sign, uint64_t* x_mag, int64_t K, float* act_scale) {
    int64_t num_blocks = (K + 63) / 64;
    float max_abs = 0.0f;
    for (int64_t k = 0; k < K; k++) {
        float a = fabsf(x[k]);
        if (a > max_abs) max_abs = a;
    }
    float scale = (max_abs > 1e-7f) ? (max_abs / 127.0f) : 1e-7f;
    float inv_scale = 1.0f / scale;
    *act_scale = scale;

    for (int64_t blk = 0; blk < num_blocks; blk++) {
        uint64_t s_mask = 0;
        uint64_t m_planes[7] = {0, 0, 0, 0, 0, 0, 0};
        int64_t base = blk * 64;

        for (int b = 0; b < 64; b++) {
            int64_t k = base + b;
            if (k < K) {
                float val = x[k];
                if (val < 0.0f) {
                    s_mask |= (1ULL << b);
                }
                int mag = (int)roundf(fabsf(val) * inv_scale);
                if (mag > 127) mag = 127;
                for (int bit = 0; bit < 7; bit++) {
                    if ((mag >> bit) & 1) {
                        m_planes[bit] |= (1ULL << b);
                    }
                }
            }
        }
        x_sign[blk] = s_mask;
        for (int bit = 0; bit < 7; bit++) {
            x_mag[blk * 7 + bit] = m_planes[bit];
        }
    }
}

QK_API void nat_qutrit_dot64_baseline_gemv(
    const uint64_t* W_pos,
    const uint64_t* W_neg,
    const uint64_t* x_sign,
    const uint64_t* x_mag,
    float* y,
    int64_t M,
    int64_t num_blocks,
    float gamma,
    float act_scale
) {
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        int64_t raw_sum = 0;
        const uint64_t* row_pos = W_pos + i * num_blocks;
        const uint64_t* row_neg = W_neg + i * num_blocks;

        for (int64_t blk = 0; blk < num_blocks; blk++) {
            uint64_t p_pos = row_pos[blk];
            uint64_t p_neg = row_neg[blk];
            /* In baseline, cancel superposition */
            uint64_t m_super = p_pos & p_neg;
            p_pos ^= m_super;
            p_neg ^= m_super;

            uint64_t xs = x_sign[blk];
            uint64_t not_xs = ~xs;

            uint64_t net_pos = (p_pos & not_xs) | (p_neg & xs);
            uint64_t net_neg = (p_neg & not_xs) | (p_pos & xs);

            const uint64_t* mp = x_mag + blk * 7;
            for (int b = 0; b < 7; b++) {
                int64_t diff = popcnt64(mp[b] & net_pos) - popcnt64(mp[b] & net_neg);
                raw_sum += diff * ((int64_t)1 << b);
            }
        }
        y[i] = (gamma * act_scale) * (float)raw_sum;
    }
}

QK_API void nat_qutrit_dot64_subquantum_gemv(
    const uint64_t* W_pos,
    const uint64_t* W_neg,
    const uint64_t* x_sign,
    const uint64_t* x_mag,
    float* y,
    int64_t M,
    int64_t num_blocks,
    float gamma,
    float beta,
    float act_scale
) {
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        int64_t raw_main = 0;
        int64_t raw_sub = 0;
        const uint64_t* row_pos = W_pos + i * num_blocks;
        const uint64_t* row_neg = W_neg + i * num_blocks;

        uint64_t neg_phase_mask = (i & 1) ? 0x5555555555555555ULL : 0xAAAAAAAAAAAAAAAAULL;
        uint64_t pos_phase_mask = ~neg_phase_mask;

        for (int64_t blk = 0; blk < num_blocks; blk++) {
            uint64_t p_pos = row_pos[blk];
            uint64_t p_neg = row_neg[blk];
            uint64_t m_super = p_pos & p_neg;
            uint64_t p_pos_pure = p_pos ^ m_super;
            uint64_t p_neg_pure = p_neg ^ m_super;

            uint64_t xs = x_sign[blk];
            uint64_t not_xs = ~xs;

            uint64_t net_pos_main = (p_pos_pure & not_xs) | (p_neg_pure & xs);
            uint64_t net_neg_main = (p_neg_pure & not_xs) | (p_pos_pure & xs);

            uint64_t sub_pos = m_super & pos_phase_mask;
            uint64_t sub_neg = m_super & neg_phase_mask;

            uint64_t net_pos_sub = (sub_pos & not_xs) | (sub_neg & xs);
            uint64_t net_neg_sub = (sub_neg & not_xs) | (sub_pos & xs);

            const uint64_t* mp = x_mag + blk * 7;
            for (int b = 0; b < 7; b++) {
                int64_t diff_m = popcnt64(mp[b] & net_pos_main) - popcnt64(mp[b] & net_neg_main);
                int64_t diff_s = popcnt64(mp[b] & net_pos_sub)  - popcnt64(mp[b] & net_neg_sub);
                raw_main += diff_m * ((int64_t)1 << b);
                raw_sub  += diff_s * ((int64_t)1 << b);
            }
        }
        y[i] = (gamma * act_scale) * ((float)raw_main + beta * (float)raw_sub);
    }
}

QK_API void nat_qutrit_dot64_ctx_gemv(
    const uint64_t* W_pos,
    const uint64_t* W_neg,
    const uint64_t* x_sign,
    const uint64_t* x_mag,
    float* y,
    int64_t M,
    int64_t num_blocks,
    float gamma,
    float act_scale,
    uint64_t c_mask
) {
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        int64_t raw_sum = 0;
        const uint64_t* row_pos = W_pos + i * num_blocks;
        const uint64_t* row_neg = W_neg + i * num_blocks;

        for (int64_t blk = 0; blk < num_blocks; blk++) {
            uint64_t p_pos = row_pos[blk];
            uint64_t p_neg = row_neg[blk];
            uint64_t m_super = p_pos & p_neg;

            /* Contextual switching: state '11' inverts sign when c_mask bit is 1 */
            uint64_t p_pos_eff = (p_pos ^ m_super) | (m_super & ~c_mask);
            uint64_t p_neg_eff = (p_neg ^ m_super) | (m_super & c_mask);

            uint64_t xs = x_sign[blk];
            uint64_t not_xs = ~xs;

            uint64_t net_pos = (p_pos_eff & not_xs) | (p_neg_eff & xs);
            uint64_t net_neg = (p_neg_eff & not_xs) | (p_pos_eff & xs);

            const uint64_t* mp = x_mag + blk * 7;
            for (int b = 0; b < 7; b++) {
                int64_t diff = popcnt64(mp[b] & net_pos) - popcnt64(mp[b] & net_neg);
                raw_sum += diff * ((int64_t)1 << b);
            }
        }
        y[i] = (gamma * act_scale) * (float)raw_sum;
    }
}

QK_API void nat_qutrit_dot64_subquantum_gemv_f32(
    const uint64_t* W_pos,
    const uint64_t* W_neg,
    const float* x,
    float* y,
    int64_t M,
    int64_t K,
    float gamma,
    float beta
) {
    int64_t num_blocks = (K + 63) / 64;
    int64_t i;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        float acc_main = 0.0f;
        float acc_sub = 0.0f;
        const uint64_t* row_pos = W_pos + i * num_blocks;
        const uint64_t* row_neg = W_neg + i * num_blocks;

        uint64_t neg_phase_mask = (i & 1) ? 0x5555555555555555ULL : 0xAAAAAAAAAAAAAAAAULL;
        uint64_t pos_phase_mask = ~neg_phase_mask;

        for (int64_t blk = 0; blk < num_blocks; blk++) {
            uint64_t p_pos = row_pos[blk];
            uint64_t p_neg = row_neg[blk];
            uint64_t m_super = p_pos & p_neg;
            uint64_t p_pos_pure = p_pos ^ m_super;
            uint64_t p_neg_pure = p_neg ^ m_super;

            uint64_t sub_pos = m_super & pos_phase_mask;
            uint64_t sub_neg = m_super & neg_phase_mask;

            int64_t base = blk * 64;
            int limit = ((base + 64) <= K) ? 64 : (int)(K - base);

            for (int b = 0; b < limit; b++) {
                uint64_t bit = (1ULL << b);
                float xv = x[base + b];
                if (p_pos_pure & bit) acc_main += xv;
                else if (p_neg_pure & bit) acc_main -= xv;

                if (sub_pos & bit) acc_sub += xv;
                else if (sub_neg & bit) acc_sub -= xv;
            }
        }
        y[i] = gamma * (acc_main + beta * acc_sub);
    }
}

/* ------------------------------------------------------------------------- */
/* 4. veclut_4qutrits (LUT256 for 4 qutrits per byte = 2.0 bits/param)       */
/* ------------------------------------------------------------------------- */

QK_API void nat_pack_4qutrits(const float* W, uint8_t* out_bytes, int64_t M, int64_t K, float gamma, float low_ratio) {
    int64_t bpr = (K + 3) / 4;
    float th_high = 0.5f * gamma;
    float th_low = low_ratio * gamma;
    int64_t i;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        const float* row_in = W + i * K;
        uint8_t* row_out = out_bytes + i * bpr;

        for (int64_t g = 0; g < bpr; g++) {
            int64_t base = g * 4;
            uint8_t byte_val = 0;

            for (int j = 0; j < 4; j++) {
                int64_t k = base + j;
                uint8_t code = 0; /* 00 = 0 */
                if (k < K) {
                    float val = row_in[k];
                    if (val >= th_high) {
                        code = 1; /* 01 = +1 */
                    } else if (val <= -th_high) {
                        code = 2; /* 10 = -1 */
                    } else if (fabsf(val) >= th_low) {
                        int phase = ((i + k) & 1) ? -1 : 1;
                        if ((val > 0.0f && phase > 0) || (val < 0.0f && phase < 0)) {
                            code = 3; /* 11 = subquantum */
                        }
                    }
                }
                byte_val |= (code << (2 * j));
            }
            row_out[g] = byte_val;
        }
    }
}

QK_API void nat_gemv_veclut_4qutrits(
    const uint8_t* W_packed,
    const float* x,
    float* y,
    int64_t M,
    int64_t K,
    float gamma,
    float beta
) {
    int64_t num_groups = (K + 3) / 4;
    float* lut_full = (float*)malloc(num_groups * 256 * sizeof(float));
    if (!lut_full) return;

    /* Precompute LUT256 per 4-weight group */
    for (int64_t g = 0; g < num_groups; g++) {
        int64_t base = g * 4;
        float x0 = (base < K) ? x[base] : 0.0f;
        float x1 = (base + 1 < K) ? x[base + 1] : 0.0f;
        float x2 = (base + 2 < K) ? x[base + 2] : 0.0f;
        float x3 = (base + 3 < K) ? x[base + 3] : 0.0f;

        float* g_lut = lut_full + g * 256;
        for (int b = 0; b < 256; b++) {
            int c0 = (b) & 3;
            int c1 = (b >> 2) & 3;
            int c2 = (b >> 4) & 3;
            int c3 = (b >> 6) & 3;

            /* In veclut float, code 3 uses neutral subquantum magnitude beta */
            float w0 = (c0 == 1) ? 1.0f : ((c0 == 2) ? -1.0f : ((c0 == 3) ? beta : 0.0f));
            float w1 = (c1 == 1) ? 1.0f : ((c1 == 2) ? -1.0f : ((c1 == 3) ? beta : 0.0f));
            float w2 = (c2 == 1) ? 1.0f : ((c2 == 2) ? -1.0f : ((c2 == 3) ? beta : 0.0f));
            float w3 = (c3 == 1) ? 1.0f : ((c3 == 2) ? -1.0f : ((c3 == 3) ? beta : 0.0f));

            g_lut[b] = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
        }
    }

    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        const uint8_t* row = W_packed + i * num_groups;
        const float* lut_ptr = lut_full;
        int64_t g = 0;

        for (; g + 3 < num_groups; g += 4) {
            acc0 += lut_ptr[row[g]];
            acc1 += (lut_ptr + 256)[row[g + 1]];
            acc2 += (lut_ptr + 512)[row[g + 2]];
            acc3 += (lut_ptr + 768)[row[g + 3]];
            lut_ptr += 1024;
        }
        for (; g < num_groups; g++) {
            acc0 += lut_ptr[row[g]];
            lut_ptr += 256;
        }
        y[i] = (acc0 + acc1 + acc2 + acc3) * gamma;
    }

    free(lut_full);
}
