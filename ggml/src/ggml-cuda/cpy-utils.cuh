#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// TurboQuant tq3_0 constants for GPU
// Lloyd-Max optimal 8-level centroids for N(0,1)
__device__ static const float TQ3_0_BOUNDARIES_D[7] = {
    -1.7479f, -1.0500f, -0.5005f, 0.0f, 0.5005f, 1.0500f, 1.7479f
};

// Deterministic sign flip pattern: sign[i] = ((i * 0x9E3779B9) >> 31) ? -1.0 : +1.0
__device__ static const float TQ3_0_SIGNS_D[128] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f,
    -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f,
    -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f, -1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f, -1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f,
    +1.0f, -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f,
    +1.0f, -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f,
    +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f,
    +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f,
};

static __device__ void quantize_f32_tq3_0_block(const float * __restrict__ x, block_tq3_0 * __restrict__ y) {
    float buf[QK_TQ3_0];

    // 1. Compute RMS scale
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_TQ3_0; j++) {
        sum_sq += x[j] * x[j];
    }
    float rms = sqrtf(sum_sq / QK_TQ3_0);
    if (rms < 1e-10f) { rms = 1.0f; }

    y->d = rms;
    const float inv_rms = 1.0f / rms;

    // 2. Normalize and apply sign flips
    for (int j = 0; j < QK_TQ3_0; j++) {
        buf[j] = x[j] * inv_rms * TQ3_0_SIGNS_D[j];
    }

    // 3. In-place Walsh-Hadamard Transform (log2(QK_TQ3_0) butterfly stages)
    for (int step = 1; step < QK_TQ3_0; step <<= 1) {
        for (int i = 0; i < QK_TQ3_0; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j];
                float b = buf[j + step];
                buf[j]        = a + b;
                buf[j + step] = a - b;
            }
        }
    }

    // Normalize by 1/sqrt(QK_TQ3_0)
    const float norm = 1.0f / sqrtf((float)QK_TQ3_0);

    // 4. Quantize to 3-bit Lloyd-Max indices and pack
    uint8_t indices[QK_TQ3_0];
    for (int j = 0; j < QK_TQ3_0; j++) {
        float v = buf[j] * norm;
        uint8_t idx = 0;
        for (int b = 0; b < 7; b++) {
            if (v > TQ3_0_BOUNDARIES_D[b]) { idx = b + 1; }
        }
        indices[j] = idx;
    }

    // 5. Pack 3-bit indices: groups of 8 indices -> 3 bytes
    for (int g = 0; g < QK_TQ3_0 / 8; g++) {
        const uint8_t * idx = indices + g * 8;
        uint8_t * qp = y->qs + g * 3;
        qp[0] = (idx[0])      | (idx[1] << 3) | (idx[2] << 6);
        qp[1] = (idx[2] >> 2) | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7);
        qp[2] = (idx[5] >> 1) | (idx[6] << 2) | (idx[7] << 5);
    }
}

// TurboQuant tq4_0 4-bit Lloyd-Max 16-level boundaries
__device__ static const float TQ4_0_BOUNDARIES_D[15] = {
    -2.4013f, -1.8441f, -1.4377f, -1.0998f, -0.8000f, -0.5227f, -0.2584f, 0.0f,
     0.2584f,  0.5227f,  0.8000f,  1.0998f,  1.4377f,  1.8441f,  2.4013f
};

static __device__ void quantize_f32_tq4_0_block(const float * __restrict__ x, block_tq4_0 * __restrict__ y) {
    float buf[QK_TQ4_0];

    // 1. Compute RMS scale
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_TQ4_0; j++) {
        sum_sq += x[j] * x[j];
    }
    float rms = sqrtf(sum_sq / QK_TQ4_0);
    if (rms < 1e-10f) { rms = 1.0f; }

    y->d = rms;
    const float inv_rms = 1.0f / rms;

    // 2. Normalize and apply sign flips (same signs as TQ3_0)
    for (int j = 0; j < QK_TQ4_0; j++) {
        buf[j] = x[j] * inv_rms * TQ3_0_SIGNS_D[j];
    }

    // 3. In-place Walsh-Hadamard Transform
    for (int step = 1; step < QK_TQ4_0; step <<= 1) {
        for (int i = 0; i < QK_TQ4_0; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j];
                float b = buf[j + step];
                buf[j]        = a + b;
                buf[j + step] = a - b;
            }
        }
    }

    const float norm = 1.0f / sqrtf((float)QK_TQ4_0);

    // 4. Quantize to 4-bit and pack (two indices per byte)
    for (int j = 0; j < QK_TQ4_0 / 2; j++) {
        float v0 = buf[2 * j] * norm;
        float v1 = buf[2 * j + 1] * norm;

        uint8_t idx0 = 0;
        for (int b = 0; b < 15; b++) {
            if (v0 > TQ4_0_BOUNDARIES_D[b]) { idx0 = b + 1; }
        }
        uint8_t idx1 = 0;
        for (int b = 0; b < 15; b++) {
            if (v1 > TQ4_0_BOUNDARIES_D[b]) { idx1 = b + 1; }
        }

        y->qs[j] = idx0 | (idx1 << 4);
    }
}

// TurboQuant tq2_0 2-bit Lloyd-Max 4-level boundaries
__device__ static const float TQ2_0_BOUNDARIES_D[3] = {
    -0.9816f, 0.0f, 0.9816f
};

static __device__ void quantize_f32_tq2_0_block(const float * __restrict__ x, block_tq2_0 * __restrict__ y) {
    float buf[QK_TQ2_0];

    // 1. Compute RMS scale
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_TQ2_0; j++) {
        sum_sq += x[j] * x[j];
    }
    float rms = sqrtf(sum_sq / QK_TQ2_0);
    if (rms < 1e-10f) { rms = 1.0f; }

    y->d = rms;
    const float inv_rms = 1.0f / rms;

    // 2. Normalize and apply sign flips (same signs as TQ3_0)
    for (int j = 0; j < QK_TQ2_0; j++) {
        buf[j] = x[j] * inv_rms * TQ3_0_SIGNS_D[j];
    }

    // 3. In-place Walsh-Hadamard Transform
    for (int step = 1; step < QK_TQ2_0; step <<= 1) {
        for (int i = 0; i < QK_TQ2_0; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j];
                float b = buf[j + step];
                buf[j]        = a + b;
                buf[j + step] = a - b;
            }
        }
    }

    const float norm = 1.0f / sqrtf((float)QK_TQ2_0);

    // 4. Quantize to 2-bit and pack (four indices per byte)
    for (int j = 0; j < QK_TQ2_0 / 4; j++) {
        uint8_t packed = 0;
        for (int k = 0; k < 4; k++) {
            float v = buf[4 * j + k] * norm;
            uint8_t idx = 0;
            for (int b = 0; b < 3; b++) {
                if (v > TQ2_0_BOUNDARIES_D[b]) { idx = b + 1; }
            }
            packed |= (idx << (2 * k));
        }
        y->qs[j] = packed;
    }
}

static __device__ void cpy_blck_f32_tq3_0(const char * cxi, char * cdsti) {
    quantize_f32_tq3_0_block((const float *)cxi, (block_tq3_0 *)cdsti);
}

static __device__ void cpy_blck_f32_tq4_0(const char * cxi, char * cdsti) {
    quantize_f32_tq4_0_block((const float *)cxi, (block_tq4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_tq2_0(const char * cxi, char * cdsti) {
    quantize_f32_tq2_0_block((const float *)cxi, (block_tq2_0 *)cdsti);
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
