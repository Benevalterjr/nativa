// Adicionando kernel Tiled Lean (sem unroll 2x, estritamente <= 15 registradores)
void mlp_gate_up_swiglu_tiled_lean(
    float* restrict h_mlp,
    const BlockQ4x4* restrict Wgate,
    const BlockQ4x4* restrict Wup,
    const BlockQ8* restrict x_q8,
    int rows, int cols,
    bool use_fast_exp
) {
    int nb = cols / 32;
    int groups = rows / 4;
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();

    int g;
    #pragma omp parallel for schedule(static)
    for (g = 0; g < groups; g++) {
        const BlockQ4x4* g_blk_row = Wgate + (size_t)g * nb;
        const BlockQ4x4* u_blk_row = Wup   + (size_t)g * nb;

        __m128 acc_g = _mm_setzero_ps();
        __m128 acc_u = _mm_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const BlockQ8* xb = &x_q8[b];
            __m128i x_lo = _mm_loadu_si128((const __m128i*)xb->qs);
            __m128i x_hi = _mm_loadu_si128((const __m128i*)(xb->qs + 16));
            __m128 x_sc = _mm_set1_ps(xb->scale);
            __m128i adj = _mm_set1_epi32(xb->sum_adj);

            // 1. Gate (reusa x_lo, x_hi)
            {
                const BlockQ4x4* gblk = &g_blk_row[b];
                __m128i gr0 = _mm_loadu_si128((const __m128i*)(gblk->qs));
                __m128i gr1 = _mm_loadu_si128((const __m128i*)(gblk->qs + 16));
                __m128i gr2 = _mm_loadu_si128((const __m128i*)(gblk->qs + 32));
                __m128i gr3 = _mm_loadu_si128((const __m128i*)(gblk->qs + 48));

                __m128i gt0 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr0, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr0, 4), mask), x_hi)), ones);
                __m128i gt1 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr1, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr1, 4), mask), x_hi)), ones);
                __m128i gt2 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr2, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr2, 4), mask), x_hi)), ones);
                __m128i gt3 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr3, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr3, 4), mask), x_hi)), ones);

                __m128i gh = _mm_hadd_epi32(_mm_hadd_epi32(gt0, gt1), _mm_hadd_epi32(gt2, gt3));
                __m128 g_fd = _mm_cvtepi32_ps(_mm_sub_epi32(gh, adj));
                __m128 g_sc = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, _mm_loadl_epi64((const __m128i*)gblk->scale)));
                acc_g = _mm_add_ps(acc_g, _mm_mul_ps(g_fd, _mm_mul_ps(g_sc, x_sc)));
            }

            // 2. Up (reusa exatamente os mesmos x_lo e x_hi dos registradores!)
            {
                const BlockQ4x4* ublk = &u_blk_row[b];
                __m128i ur0 = _mm_loadu_si128((const __m128i*)(ublk->qs));
                __m128i ur1 = _mm_loadu_si128((const __m128i*)(ublk->qs + 16));
                __m128i ur2 = _mm_loadu_si128((const __m128i*)(ublk->qs + 32));
                __m128i ur3 = _mm_loadu_si128((const __m128i*)(ublk->qs + 48));

                __m128i ut0 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur0, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur0, 4), mask), x_hi)), ones);
                __m128i ut1 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur1, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur1, 4), mask), x_hi)), ones);
                __m128i ut2 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur2, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur2, 4), mask), x_hi)), ones);
                __m128i ut3 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur3, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur3, 4), mask), x_hi)), ones);

                __m128i uh = _mm_hadd_epi32(_mm_hadd_epi32(ut0, ut1), _mm_hadd_epi32(ut2, ut3));
                __m128 u_fd = _mm_cvtepi32_ps(_mm_sub_epi32(uh, adj));
                __m128 u_sc = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, _mm_loadl_epi64((const __m128i*)ublk->scale)));
                acc_u = _mm_add_ps(acc_u, _mm_mul_ps(u_fd, _mm_mul_ps(u_sc, x_sc)));
            }
        }

        if (use_fast_exp) {
            __m128 out = fast_swiglu_ps(acc_g, acc_u);
            _mm_storeu_ps(&h_mlp[g * 4], out);
        } else {
            float garr[4], uarr[4];
            _mm_storeu_ps(garr, acc_g);
            _mm_storeu_ps(uarr, acc_u);
            for (int j = 0; j < 4; j++) {
                h_mlp[g * 4 + j] = (garr[j] / (1.0f + expf(-garr[j]))) * uarr[j];
            }
        }
    }

    int rem_rows = rows % 4;
    if (rem_rows > 0) {
        int g = groups;
        const BlockQ4x4* g_blk_row = Wgate + (size_t)g * nb;
        const BlockQ4x4* u_blk_row = Wup   + (size_t)g * nb;
        for (int r = 0; r < rem_rows; r++) {
            float g_acc = 0.0f, u_acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                const BlockQ8* xb = &x_q8[b];
                const BlockQ4x4* gb = &g_blk_row[b];
                const BlockQ4x4* ub = &u_blk_row[b];
                float gw_sc = bf16_to_fp32(gb->scale[r]);
                float uw_sc = bf16_to_fp32(ub->scale[r]);
                const uint8_t* g_qs = gb->qs + r * 16;
                const uint8_t* u_qs = ub->qs + r * 16;
                int32_t g_dot = 0, u_dot = 0;
                for (int i = 0; i < 16; i++) {
                    int8_t glo = (int8_t)(g_qs[i] & 0x0F) - 8;
                    int8_t ghi = (int8_t)(g_qs[i] >> 4) - 8;
                    int8_t ulo = (int8_t)(u_qs[i] & 0x0F) - 8;
                    int8_t uhi = (int8_t)(u_qs[i] >> 4) - 8;
                    g_dot += (int32_t)glo * (int32_t)xb->qs[i] + (int32_t)ghi * (int32_t)xb->qs[i + 16];
                    u_dot += (int32_t)ulo * (int32_t)xb->qs[i] + (int32_t)uhi * (int32_t)xb->qs[i + 16];
                }
                g_acc += (float)g_dot * (gw_sc * xb->scale);
                u_acc += (float)u_dot * (uw_sc * xb->scale);
            }
            float silu_g = g_acc / (1.0f + expf(-g_acc));
            h_mlp[g * 4 + r] = silu_g * u_acc;
        }
    }
}
