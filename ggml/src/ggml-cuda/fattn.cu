#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        if (use_gqa_opt && gqa_ratio > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ2_0, GGML_TYPE_TQ2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ3_0, GGML_TYPE_TQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ4_0, GGML_TYPE_TQ4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ4_0, GGML_TYPE_TQ2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ3_0, GGML_TYPE_TQ2_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO3_0, GGML_TYPE_ISO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO4_0, GGML_TYPE_ISO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO3_0, GGML_TYPE_ISO4_0)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ2_0, GGML_TYPE_TQ2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ3_0, GGML_TYPE_TQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ4_0, GGML_TYPE_TQ4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ4_0, GGML_TYPE_TQ2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ3_0, GGML_TYPE_TQ2_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO3_0, GGML_TYPE_ISO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO4_0, GGML_TYPE_ISO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO3_0, GGML_TYPE_ISO4_0)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        const bool ok_asym =
            (K->type == GGML_TYPE_TQ4_0  && V->type == GGML_TYPE_TQ2_0)  ||
            (K->type == GGML_TYPE_TQ3_0  && V->type == GGML_TYPE_TQ2_0)  ||
            (K->type == GGML_TYPE_ISO3_0 && V->type == GGML_TYPE_ISO4_0);
        if (!ok_asym) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            break;
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_TQ3_0:
        case GGML_TYPE_TQ4_0:
            // TQ types are only supported by the vec kernel which requires D <= 256:
            if (Q->ne[0] > 256) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case GGML_TYPE_ISO3_0:
        case GGML_TYPE_ISO4_0:
            // ISO D<=256: use vec kernel. ISO D>256: dequant to F16 then use tile kernel.
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    // V type: TQ vec-only check (relevant when GGML_CUDA_FA_ALL_QUANTS allows mixed K/V):
    if (Q->ne[0] > 256) {
        switch (V->type) {
            case GGML_TYPE_TQ2_0:
            case GGML_TYPE_TQ3_0:
            case GGML_TYPE_TQ4_0:
                return BEST_FATTN_KERNEL_NONE;
            default:
                break;
        }
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // TQ/ISO types are only natively supported by the vec kernel (direct quantized-KV dot product).
    // For decode (Q->ne[1] <= 2) force VEC here; launch_fattn cannot dequant these types to F16
    // via ggml_get_to_fp16_cuda (would NULL-pointer-deref). For prefill (Q->ne[1] > 2) the dispatcher
    // may choose WMMA/MMA/tile — that path is only reachable via ggml_cuda_flash_attn_ext's
    // tq/iso dequant wrapper, which materializes K/V as F16 before re-entering dispatch.
    const bool is_tq_or_iso = (K->type == GGML_TYPE_TQ2_0 || V->type == GGML_TYPE_TQ2_0 ||
                               K->type == GGML_TYPE_TQ3_0 || V->type == GGML_TYPE_TQ3_0 ||
                               K->type == GGML_TYPE_TQ4_0 || V->type == GGML_TYPE_TQ4_0 ||
                               K->type == GGML_TYPE_ISO3_0 || V->type == GGML_TYPE_ISO3_0 ||
                               K->type == GGML_TYPE_ISO4_0 || V->type == GGML_TYPE_ISO4_0);
    if (can_use_vector_kernel && is_tq_or_iso && Q->ne[1] <= 2) {
        return BEST_FATTN_KERNEL_VEC;
    }

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 192 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_WMMA_F16:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}
// ---- ISO dequant-to-F16 wrapper for D>256 (e.g. Kimi-Linear D=576) ----
// When ISO types are used with head dimensions too large for the vec kernel,
// we dequantize the ISO KV cache to F16 in a temporary buffer and then
// run the standard tile/mma kernel on the F16 data.

template <typename block_type, int QK, bool is_3bit>
__global__ void kernel_dequant_iso_to_f16(const char * __restrict__ src, half * __restrict__ dst_half,
                                           const int64_t blocks_per_row,
                                           const int64_t nb1, const int64_t nb2, const int64_t nb3,
                                           const int64_t ne1, const int64_t ne2, const int64_t ne3,
                                           const int64_t dst_stride_elems) {
    const int64_t global_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_rows = ne1 * ne2 * ne3;
    const int64_t flat_row = global_idx / blocks_per_row;
    const int64_t blk_in_row = global_idx % blocks_per_row;

    if (flat_row >= total_rows) return;

    // Decompose flat row index into (i1, i2, i3)
    const int64_t i3 = flat_row / (ne1 * ne2);
    const int64_t i2 = (flat_row / ne1) % ne2;
    const int64_t i1 = flat_row % ne1;

    const block_type * blk = (const block_type *)(src + i3*nb3 + i2*nb2 + i1*nb1) + blk_in_row;
    half * out = dst_half + flat_row * dst_stride_elems + blk_in_row * QK;

    const float d = __half2float(blk->d);

    // Unpack indices → centroid lookup
    float cent[QK];
    if constexpr (is_3bit) {
        for (int g = 0; g < QK / 8; g++) {
            const uint8_t * qs = &blk->qs[g * 3];
            const uint32_t packed = qs[0] | (qs[1] << 8) | (qs[2] << 16);
            for (int j = 0; j < 8; j++) {
                cent[g * 8 + j] = ISO3_0_CENTROIDS_FA[(packed >> (j * 3)) & 0x7];
            }
        }
    } else {
        for (int j = 0; j < QK / 2; j++) {
            const uint8_t packed = blk->qs[j];
            cent[2 * j]     = ISO4_0_CENTROIDS_FA[packed & 0xF];
            cent[2 * j + 1] = ISO4_0_CENTROIDS_FA[packed >> 4];
        }
    }

    // Inverse quaternion rotation per 4D group + scale → F16
    for (int g = 0; g < QK / 4; g++) {
        const float * q = ISO_QUAT_L_FA[g];
        const float * v = &cent[g * 4];
        // r = conj(q_L) * v (Hamilton product of quaternions)
        // conj(q) = (w, -x, -y, -z)
        const float cw =  q[0], cx = -q[1], cy = -q[2], cz = -q[3];
        float r0 = cw*v[0] - cx*v[1] - cy*v[2] - cz*v[3];
        float r1 = cw*v[1] + cx*v[0] + cy*v[3] - cz*v[2];
        float r2 = cw*v[2] - cx*v[3] + cy*v[0] + cz*v[1];
        float r3 = cw*v[3] + cx*v[2] - cy*v[1] + cz*v[0];
        out[g*4 + 0] = __float2half(r0 * d);
        out[g*4 + 1] = __float2half(r1 * d);
        out[g*4 + 2] = __float2half(r2 * d);
        out[g*4 + 3] = __float2half(r3 * d);
    }
}

// ---- TurboQuant dequant-to-F16 (for prefill WMMA/MMA/tile path) ----
// Shared block size QK=128 across TQ2/TQ3/TQ4. Same TQ3_0_SIGNS_FA pattern, same WHT inverse.
// QBITS selects the codebook and unpacking layout.
template <typename block_type, int QK, int QBITS>
__global__ void kernel_dequant_tq_to_f16(const char * __restrict__ src, half * __restrict__ dst_half,
                                          const int64_t blocks_per_row,
                                          const int64_t nb1, const int64_t nb2, const int64_t nb3,
                                          const int64_t ne1, const int64_t ne2, const int64_t ne3,
                                          const int64_t dst_stride_elems) {
    const int64_t global_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_rows = ne1 * ne2 * ne3;
    const int64_t flat_row   = global_idx / blocks_per_row;
    const int64_t blk_in_row = global_idx % blocks_per_row;

    if (flat_row >= total_rows) return;

    const int64_t i3 = flat_row / (ne1 * ne2);
    const int64_t i2 = (flat_row / ne1) % ne2;
    const int64_t i1 = flat_row % ne1;

    const block_type * blk = (const block_type *)(src + i3*nb3 + i2*nb2 + i1*nb1) + blk_in_row;
    half * out = dst_half + flat_row * dst_stride_elems + blk_in_row * QK;

    const float d = __half2float(blk->d);

    // Unpack indices → centroid lookup into rotated[] (WHT-domain values)
    float rotated[QK];
    if constexpr (QBITS == 2) {
        for (int j = 0; j < QK / 4; j++) {
            const uint8_t packed = blk->qs[j];
            rotated[4*j + 0] = TQ2_0_CENTROIDS_FA[ packed       & 3];
            rotated[4*j + 1] = TQ2_0_CENTROIDS_FA[(packed >> 2) & 3];
            rotated[4*j + 2] = TQ2_0_CENTROIDS_FA[(packed >> 4) & 3];
            rotated[4*j + 3] = TQ2_0_CENTROIDS_FA[(packed >> 6) & 3];
        }
    } else if constexpr (QBITS == 3) {
        for (int g = 0; g < QK / 8; g++) {
            const uint8_t * qp = &blk->qs[g * 3];
            rotated[g*8 + 0] = TQ3_0_CENTROIDS_FA[ qp[0]       & 7];
            rotated[g*8 + 1] = TQ3_0_CENTROIDS_FA[(qp[0] >> 3) & 7];
            rotated[g*8 + 2] = TQ3_0_CENTROIDS_FA[((qp[0] >> 6) | (qp[1] << 2)) & 7];
            rotated[g*8 + 3] = TQ3_0_CENTROIDS_FA[(qp[1] >> 1) & 7];
            rotated[g*8 + 4] = TQ3_0_CENTROIDS_FA[(qp[1] >> 4) & 7];
            rotated[g*8 + 5] = TQ3_0_CENTROIDS_FA[((qp[1] >> 7) | (qp[2] << 1)) & 7];
            rotated[g*8 + 6] = TQ3_0_CENTROIDS_FA[(qp[2] >> 2) & 7];
            rotated[g*8 + 7] = TQ3_0_CENTROIDS_FA[(qp[2] >> 5) & 7];
        }
    } else {  // QBITS == 4
        for (int j = 0; j < QK / 2; j++) {
            const uint8_t packed = blk->qs[j];
            rotated[2*j + 0] = TQ4_0_CENTROIDS_FA[packed & 0xF];
            rotated[2*j + 1] = TQ4_0_CENTROIDS_FA[packed >> 4];
        }
    }

    // In-place WHT butterfly (self-inverse up to 1/sqrt(QK) scale, folded into norm_d below)
    for (int step = 1; step < QK; step <<= 1) {
        for (int i = 0; i < QK; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                const float a = rotated[j];
                const float b = rotated[j + step];
                rotated[j]        = a + b;
                rotated[j + step] = a - b;
            }
        }
    }

    // Apply sign flips + scale, emit half
    const float norm_d = d * rsqrtf((float)QK);
    for (int j = 0; j < QK; j++) {
        out[j] = __float2half(rotated[j] * norm_d * TQ3_0_SIGNS_FA[j]);
    }
}

static void dequant_tq_tensor_to_f16(const ggml_tensor * src,
                                      half * dst, int64_t dst_stride_elems, cudaStream_t stream) {
    const int64_t ne0 = src->ne[0];
    const int64_t total_rows = src->ne[1] * src->ne[2] * src->ne[3];
    const int threads = 128;

    auto launch = [&](auto block_tag, auto qk_tag, auto qbits_tag) {
        using BT = typename decltype(block_tag)::type;
        constexpr int QK    = decltype(qk_tag)::value;
        constexpr int QBITS = decltype(qbits_tag)::value;
        const int64_t blocks_per_row = ne0 / QK;
        const int64_t total_work     = blocks_per_row * total_rows;
        const int64_t nblocks        = (total_work + threads - 1) / threads;
        kernel_dequant_tq_to_f16<BT, QK, QBITS>
            <<<nblocks, threads, 0, stream>>>((const char *)src->data, dst, blocks_per_row,
                                              src->nb[1], src->nb[2], src->nb[3],
                                              src->ne[1], src->ne[2], src->ne[3],
                                              dst_stride_elems);
    };

    // struct-wrappers to pass types/constants to the lambda
    struct TQ2 { using type = block_tq2_0; };
    struct TQ3 { using type = block_tq3_0; };
    struct TQ4 { using type = block_tq4_0; };

    if (src->type == GGML_TYPE_TQ2_0) {
        launch(TQ2{}, std::integral_constant<int, QK_TQ2_0>{}, std::integral_constant<int, 2>{});
    } else if (src->type == GGML_TYPE_TQ3_0) {
        launch(TQ3{}, std::integral_constant<int, QK_TQ3_0>{}, std::integral_constant<int, 3>{});
    } else if (src->type == GGML_TYPE_TQ4_0) {
        launch(TQ4{}, std::integral_constant<int, QK_TQ4_0>{}, std::integral_constant<int, 4>{});
    }
}

static void dequant_iso_tensor_to_f16(const ggml_tensor * src,
                                       half * dst, int64_t dst_stride_elems, cudaStream_t stream) {
    const int64_t ne0 = src->ne[0]; // head dim
    const int64_t total_rows = src->ne[1] * src->ne[2] * src->ne[3];

    if (src->type == GGML_TYPE_ISO3_0) {
        const int64_t blocks_per_row = ne0 / QK_ISO3_0;
        const int64_t total_work = blocks_per_row * total_rows;
        const int threads = 256;
        const int nblocks = (total_work + threads - 1) / threads;
        kernel_dequant_iso_to_f16<block_iso3_0, QK_ISO3_0, true>
            <<<nblocks, threads, 0, stream>>>((const char *)src->data, dst, blocks_per_row,
                                               src->nb[1], src->nb[2], src->nb[3],
                                               src->ne[1], src->ne[2], src->ne[3],
                                               dst_stride_elems);
    } else if (src->type == GGML_TYPE_ISO4_0) {
        const int64_t blocks_per_row = ne0 / QK_ISO4_0;
        const int64_t total_work = blocks_per_row * total_rows;
        const int threads = 256;
        const int nblocks = (total_work + threads - 1) / threads;
        kernel_dequant_iso_to_f16<block_iso4_0, QK_ISO4_0, false>
            <<<nblocks, threads, 0, stream>>>((const char *)src->data, dst, blocks_per_row,
                                               src->nb[1], src->nb[2], src->nb[3],
                                               src->ne[1], src->ne[2], src->ne[3],
                                               dst_stride_elems);
    }
}

// Wrapper: dequant TQ/ISO K/V to F16, then call the tile/mma/wmma kernel.
// Used for (a) ISO K/V with D>256 where vec kernel can't cover, (b) TQ/ISO prefill where we
// want WMMA/MMA throughput instead of serial vec.
static void ggml_cuda_flash_attn_ext_quant_dequant(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    auto is_iso = [](ggml_type t) { return t == GGML_TYPE_ISO3_0 || t == GGML_TYPE_ISO4_0; };
    auto is_tq  = [](ggml_type t) { return t == GGML_TYPE_TQ2_0  || t == GGML_TYPE_TQ3_0  || t == GGML_TYPE_TQ4_0; };

    const bool K_is_iso = is_iso(K->type);
    const bool V_is_iso = is_iso(V->type);
    const bool K_is_tq  = is_tq(K->type);
    const bool V_is_tq  = is_tq(V->type);

    cudaStream_t stream = ctx.stream();

    const int64_t K_ne0 = K->ne[0]; // head dim K
    const int64_t V_ne0 = V->ne[0]; // head dim V
    const int64_t K_total_rows = K->ne[1] * K->ne[2] * K->ne[3];
    const int64_t V_total_rows = V->ne[1] * V->ne[2] * V->ne[3];

    ggml_cuda_pool_alloc<half> K_f16_alloc;
    ggml_cuda_pool_alloc<half> V_f16_alloc;

    struct tensor_state {
        void *   data;
        ggml_type type;
        size_t   nb[GGML_MAX_DIMS];
    };
    tensor_state K_orig = { K->data, K->type, { K->nb[0], K->nb[1], K->nb[2], K->nb[3] } };
    tensor_state V_orig = { V->data, V->type, { V->nb[0], V->nb[1], V->nb[2], V->nb[3] } };

    auto patch_to_f16 = [](ggml_tensor * t, half * p, int64_t ne0) {
        t->data  = p;
        t->type  = GGML_TYPE_F16;
        t->nb[0] = sizeof(half);
        t->nb[1] = ne0 * sizeof(half);
        t->nb[2] = t->nb[1] * t->ne[1];
        t->nb[3] = t->nb[2] * t->ne[2];
    };

    if (K_is_iso) {
        K_f16_alloc.alloc(ctx.pool(), K_ne0 * K_total_rows);
        dequant_iso_tensor_to_f16(K, K_f16_alloc.get(), K_ne0, stream);
        patch_to_f16(K, K_f16_alloc.get(), K_ne0);
    } else if (K_is_tq) {
        K_f16_alloc.alloc(ctx.pool(), K_ne0 * K_total_rows);
        dequant_tq_tensor_to_f16(K, K_f16_alloc.get(), K_ne0, stream);
        patch_to_f16(K, K_f16_alloc.get(), K_ne0);
    }

    if (V_is_iso) {
        V_f16_alloc.alloc(ctx.pool(), V_ne0 * V_total_rows);
        dequant_iso_tensor_to_f16(V, V_f16_alloc.get(), V_ne0, stream);
        patch_to_f16(V, V_f16_alloc.get(), V_ne0);
    } else if (V_is_tq) {
        V_f16_alloc.alloc(ctx.pool(), V_ne0 * V_total_rows);
        dequant_tq_tensor_to_f16(V, V_f16_alloc.get(), V_ne0, stream);
        patch_to_f16(V, V_f16_alloc.get(), V_ne0);
    }

    // Now dispatch as if K/V are F16 — tile/mma/wmma handle it
    const int best = ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst);
    switch (best) {
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            // Possible if dispatcher preferred VEC for the patched F16 KV (e.g. small gqa / head-dim mix)
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        default:
            GGML_ABORT("TQ/ISO dequant wrapper: unexpected kernel type");
    }

    // Restore original tensor state
    K->data  = K_orig.data;
    K->type  = K_orig.type;
    K->nb[0] = K_orig.nb[0];
    K->nb[1] = K_orig.nb[1];
    K->nb[2] = K_orig.nb[2];
    K->nb[3] = K_orig.nb[3];
    V->data  = V_orig.data;
    V->type  = V_orig.type;
    V->nb[0] = V_orig.nb[0];
    V->nb[1] = V_orig.nb[1];
    V->nb[2] = V_orig.nb[2];
    V->nb[3] = V_orig.nb[3];
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool is_iso = (K->type == GGML_TYPE_ISO3_0 || K->type == GGML_TYPE_ISO4_0 ||
                         V->type == GGML_TYPE_ISO3_0 || V->type == GGML_TYPE_ISO4_0);
    const bool is_tq  = (K->type == GGML_TYPE_TQ2_0  || K->type == GGML_TYPE_TQ3_0  || K->type == GGML_TYPE_TQ4_0 ||
                         V->type == GGML_TYPE_TQ2_0  || V->type == GGML_TYPE_TQ3_0  || V->type == GGML_TYPE_TQ4_0);

    // ISO with D>256: always use the dequant-to-F16 wrapper (vec kernel doesn't cover D>256).
    // TQ/ISO prefill (Q->ne[1] > 2): dequant to F16 so WMMA/MMA/tile can take over, avoiding
    // the serial vec kernel's per-Q-token bottleneck. Dispatcher picks VEC for Q->ne[1] <= 2
    // (decode), which takes the native-quant path below.
    if (is_iso && Q->ne[0] > 256) {
        ggml_cuda_flash_attn_ext_quant_dequant(ctx, dst);
        return;
    }
    if ((is_tq || is_iso) && Q->ne[1] > 2) {
        ggml_cuda_flash_attn_ext_quant_dequant(ctx, dst);
        return;
    }

    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
