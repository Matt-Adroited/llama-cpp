/*
 * TriAttention GPU scoring kernel implementation
 *
 * Computes importance scores for KV cache entries directly on GPU memory.
 * Each thread block processes one (position) with all freq dimensions in
 * parallel. The kernel handles dequantization, inverse WHT rotation (for
 * TurboQuant types), inverse RoPE, and the full TriAttention scoring formula.
 *
 * Grid:  (n_cells, 1, 1)
 * Block: (freq_count, 1, 1)  — one thread per frequency pair
 *
 * Supports: F32, F16, Q8_0 (dense); TQ3_0, TQ4_0 (TurboQuant, head_dim=128 only).
 * IsoQuant (quaternion inverse) and MLA-partial-RoPE deferred.
 *
 * Reference: "TriAttention: Trigonometric KV Cache Eviction" (arXiv 2604.04921)
 */

#include "triattention-score.cuh"
#include "dequantize.cuh"

#include <cstdio>
#include <cstring>

// ============================================================================
// TurboQuant constants (single-stage rotation: FWHT → · SIGNS)
// NOTE: Our fork's TQ format differs from the upstream reference's TURBO*
// format (which uses SIGNS1 · FWHT · SIGNS2). Keep these local — must match
// the tables used by dequantize_tq3_0_block in fattn-common.cuh.
// ============================================================================

__device__ static const float TRIATT_TQ3_CENTROIDS[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f
};

__device__ static const float TRIATT_TQ4_CENTROIDS[16] = {
    -2.7331f, -2.0696f, -1.6186f, -1.2568f, -0.9428f, -0.6571f, -0.3883f, -0.1285f,
     0.1285f,  0.3883f,  0.6571f,  0.9428f,  1.2568f,  1.6186f,  2.0696f,  2.7331f
};

__device__ static const float TRIATT_TQ_SIGNS[128] = {
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

// ============================================================================
// Device helper: cooperative Walsh-Hadamard Transform in shared memory
// n = 128, freq_count = 64 threads (one butterfly pair per thread per stage).
// Includes 1/sqrt(128) normalization.
// ============================================================================

static __device__ void cooperative_fwht_128(float * smem, int tid) {
    for (int h = 1; h < 128; h *= 2) {
        int block_half = h;
        int block_size = h * 2;
        int i = (tid / block_half) * block_size + (tid % block_half);
        float a = smem[i];
        float b = smem[i + block_half];
        __syncthreads();
        smem[i]              = a + b;
        smem[i + block_half] = a - b;
        __syncthreads();
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    smem[tid * 2]     *= inv_sqrt_128;
    smem[tid * 2 + 1] *= inv_sqrt_128;
    __syncthreads();
}

// ============================================================================
// GPU-resident state
// ============================================================================

struct triattention_gpu_state {
    // Device arrays: calibration per sampled head
    float * d_q_mean_real;     // [n_sampled * freq_count]
    float * d_q_mean_imag;     // [n_sampled * freq_count]
    float * d_q_mean_abs;      // [n_sampled * freq_count]
    float * d_extra_weight;    // [n_sampled * freq_count]

    // Device arrays: precomputed
    float * d_omega;           // [freq_count]
    float * d_freq_scale_sq;   // [freq_count]
    float * d_offsets;         // [n_offsets]

    // Configuration
    triattention_gpu_config cfg;
};

// ============================================================================
// Device helper: dequantize one KV head row to shared memory
// Each thread loads 2 elements (its freq pair)
// ============================================================================

// Per-element TQ3_0 index extraction. Returns the 3-bit centroid index for
// element `el` within a QK_TQ3_0=128 block packed as 16 groups of 8 indices
// into 48 bytes (3 bytes per group of 8).
static __device__ __forceinline__ uint8_t tq3_extract_index(const uint8_t * qs, int el) {
    const int g = el >> 3;            // el / 8
    const int w = el & 7;             // el % 8
    const uint8_t * qp = qs + g * 3;
    switch (w) {
        case 0: return  qp[0]                         & 7;
        case 1: return (qp[0] >> 3)                   & 7;
        case 2: return ((qp[0] >> 6) | (qp[1] << 2))  & 7;
        case 3: return (qp[1] >> 1)                   & 7;
        case 4: return (qp[1] >> 4)                   & 7;
        case 5: return ((qp[1] >> 7) | (qp[2] << 1))  & 7;
        case 6: return (qp[2] >> 2)                   & 7;
        default:return (qp[2] >> 5)                   & 7;
    }
}

template <enum ggml_type K_TYPE>
static __device__ void dequant_head_to_smem(
    float * smem,
    const void * k_row_ptr,    // device pointer to quantized head data
    int tid,                   // thread id = freq index
    int /*head_dim*/)
{
    if constexpr (K_TYPE == GGML_TYPE_F32) {
        const float * src = (const float *)k_row_ptr;
        smem[tid * 2]     = src[tid * 2];
        smem[tid * 2 + 1] = src[tid * 2 + 1];
    } else if constexpr (K_TYPE == GGML_TYPE_F16) {
        const half * src = (const half *)k_row_ptr;
        smem[tid * 2]     = __half2float(src[tid * 2]);
        smem[tid * 2 + 1] = __half2float(src[tid * 2 + 1]);
    } else if constexpr (K_TYPE == GGML_TYPE_Q8_0) {
        // Q8_0: block_size=32, each block has d (fp16) + 32 qs (int8)
        const int el0 = tid * 2;
        const int el1 = tid * 2 + 1;
        const int blk0 = el0 / QK8_0;
        const int blk1 = el1 / QK8_0;
        const int off0 = el0 % QK8_0;
        const int off1 = el1 % QK8_0;
        const block_q8_0 * x = (const block_q8_0 *)k_row_ptr;
        smem[el0] = (float)x[blk0].qs[off0] * __half2float(x[blk0].d);
        smem[el1] = (float)x[blk1].qs[off1] * __half2float(x[blk1].d);
    } else if constexpr (K_TYPE == GGML_TYPE_TQ3_0) {
        // TQ3_0: block_size=128, head_dim=128 → one block per head.
        // Load raw centroid values only; FWHT + (d * SIGNS) applied later.
        const block_tq3_0 * x = (const block_tq3_0 *)k_row_ptr;
        const int el0 = tid * 2;
        const int el1 = tid * 2 + 1;
        smem[el0] = TRIATT_TQ3_CENTROIDS[tq3_extract_index(x[0].qs, el0)];
        smem[el1] = TRIATT_TQ3_CENTROIDS[tq3_extract_index(x[0].qs, el1)];
    } else if constexpr (K_TYPE == GGML_TYPE_TQ4_0) {
        // TQ4_0: block_size=128, qs packed as nibbles (2 indices per byte).
        const block_tq4_0 * x = (const block_tq4_0 *)k_row_ptr;
        const int el0 = tid * 2;      // always even
        const int el1 = tid * 2 + 1;  // always odd
        const uint8_t packed = x[0].qs[el0 >> 1];
        smem[el0] = TRIATT_TQ4_CENTROIDS[packed & 0xF];
        smem[el1] = TRIATT_TQ4_CENTROIDS[packed >> 4];
    }
}

// ============================================================================
// Main scoring kernel
//
// One block per cell (position). freq_count threads per block.
// Each thread handles one frequency pair [f, f+freq_count] (half layout).
//
// Pipeline per block:
//   1. Dequant K head row → shared mem [head_dim]
//   2. Inverse RoPE → get pre-RoPE K in "half" layout
//   3. Compute per-freq score contributions
//   4. Block reduction → one score per position
//   5. Write to output
// ============================================================================

template <enum ggml_type K_TYPE, bool NEED_WHT_INV, bool DISABLE_TRIG>
static __global__ void triattention_score_kernel(
    float       * __restrict__ scores_out,       // [n_cells]
    const void  * __restrict__ k_data,           // device ptr to full K tensor
    const uint64_t              /*n_embd_k_gqa*/,
    const size_t                row_bytes,
    const size_t                head_offset_bytes,
    const uint32_t              padded_hd,
    const uint32_t * __restrict__ cell_indices,  // [n_cells]
    const int32_t  * __restrict__ positions,     // [n_cells]
    const uint32_t              n_cells,
    const int64_t               round_start,
    const float  * __restrict__ omega,           // [freq_count]
    const float  * __restrict__ freq_scale_sq,   // [freq_count]
    const float  * __restrict__ offsets,         // [n_offsets]
    const uint32_t              n_offsets,
    const float  * __restrict__ q_mean_real,     // [freq_count] for this head
    const float  * __restrict__ q_mean_imag,     // [freq_count]
    const float  * __restrict__ q_mean_abs,      // [freq_count]
    const float  * __restrict__ extra_weight,    // [freq_count]
    const uint32_t              freq_count,
    const int                   agg_mode)        // 0=mean, 1=max
{
    const int cell_idx_local = blockIdx.x;
    const int f = threadIdx.x;

    if (cell_idx_local >= (int)n_cells || f >= (int)freq_count) return;

    extern __shared__ float smem[];
    // smem[0..head_dim-1]                     = dequantized K
    // smem[head_dim..head_dim+freq_count-1]   = partial scores for reduction
    float * k_smem = smem;
    float * score_smem = smem + padded_hd;

    // ---- Step 1: Dequant K head row to shared memory ----
    const uint32_t cell_global = cell_indices[cell_idx_local];
    const char * k_row_ptr = (const char *)k_data + (size_t)cell_global * row_bytes + head_offset_bytes;

    dequant_head_to_smem<K_TYPE>(k_smem, k_row_ptr, f, padded_hd);
    __syncthreads();

    // ---- Step 1b: Inverse WHT rotation for TurboQuant ----
    // Our fork stores: FWHT(x * inv_rms) → quantize to centroid index,
    // with a single post-FWHT sign flip at quantize time.
    // Inverse: centroid → FWHT (self-inverse, 1/sqrt(128) norm) → * d * SIGNS.
    // NOTE: This is single-stage, NOT the reference's two-stage
    // (SIGNS1 · FWHT · SIGNS2) rotation. Requires padded_hd == 128.
    if constexpr (NEED_WHT_INV) {
        if (padded_hd == 128 && f < 64) {
            cooperative_fwht_128(k_smem, f);
        }
        __syncthreads();

        // Load per-head scale `d` (same across 128-el block)
        float d_scale;
        if constexpr (K_TYPE == GGML_TYPE_TQ3_0) {
            const block_tq3_0 * x = (const block_tq3_0 *)k_row_ptr;
            d_scale = __half2float(x[0].d);
        } else if constexpr (K_TYPE == GGML_TYPE_TQ4_0) {
            const block_tq4_0 * x = (const block_tq4_0 *)k_row_ptr;
            d_scale = __half2float(x[0].d);
        } else {
            d_scale = 1.0f;  // unreachable if NEED_WHT_INV implies TQ*
        }

        const int el0 = f * 2;
        const int el1 = f * 2 + 1;
        k_smem[el0] *= d_scale * TRIATT_TQ_SIGNS[el0];
        k_smem[el1] *= d_scale * TRIATT_TQ_SIGNS[el1];
        __syncthreads();
    }

    // ---- Step 2: Inverse RoPE ----
    // Half layout: k_re = K[f], k_im = K[f + freq_count]
    {
        const int32_t pos = positions[cell_idx_local];
        const float w = omega[f];
        const float theta = w * (float)pos;
        const float cos_t = cosf(theta);
        const float sin_t = sinf(theta);

        float k_re = k_smem[f];
        float k_im = k_smem[f + freq_count];

        // Inverse rotation (angle = -theta):
        // k_re' =  k_re * cos(θ) + k_im * sin(θ)
        // k_im' = -k_re * sin(θ) + k_im * cos(θ)
        float pre_re =  k_re * cos_t + k_im * sin_t;
        float pre_im = -k_re * sin_t + k_im * cos_t;

        k_smem[f]              = pre_re;
        k_smem[f + freq_count] = pre_im;
    }
    __syncthreads();

    // ---- Step 3: Compute score ----
    const float k_re = k_smem[f];
    const float k_im = k_smem[f + freq_count];
    const float k_mag = sqrtf(k_re * k_re + k_im * k_im);

    float total_score;

    if constexpr (!DISABLE_TRIG) {
        const float base_delta = (float)(round_start - positions[cell_idx_local]);
        const float amp = q_mean_abs[f] * k_mag;
        const float w = omega[f];
        const float fscale_sq = freq_scale_sq[f];

        // Phase from conjugate multiply: E[q] * conj(k)
        const float conj_re = q_mean_real[f] * k_re + q_mean_imag[f] * k_im;
        const float conj_im = q_mean_imag[f] * k_re - q_mean_real[f] * k_im;
        const float phi = atan2f(conj_im, conj_re);

        const float ew = extra_weight[f] * fscale_sq * k_mag;

        if (agg_mode == 1) {
            float max_score = -1e30f;
            for (uint32_t d = 0; d < n_offsets; d++) {
                float delta = base_delta + offsets[d];
                float phase = w * delta + phi;
                float s = amp * fscale_sq * cosf(phase) + ew;
                max_score = fmaxf(max_score, s);
            }
            total_score = max_score;
        } else {
            float sum = 0.0f;
            for (uint32_t d = 0; d < n_offsets; d++) {
                float delta = base_delta + offsets[d];
                float phase = w * delta + phi;
                sum += amp * fscale_sq * cosf(phase) + ew;
            }
            total_score = sum / (float)n_offsets;
        }
    } else {
        // Ablation: norm-only scoring
        total_score = extra_weight[f] * freq_scale_sq[f] * k_mag;
    }

    // ---- Step 4: Block reduction ----
    score_smem[f] = total_score;
    __syncthreads();

    for (int stride = freq_count / 2; stride > 0; stride >>= 1) {
        if (f < stride) {
            score_smem[f] += score_smem[f + stride];
        }
        __syncthreads();
    }

    // ---- Step 5: Write result ----
    if (f == 0) {
        scores_out[cell_idx_local] = score_smem[0];
    }
}

// ============================================================================
// Kernel launch dispatcher
// ============================================================================

static void launch_score_kernel(
    triattention_gpu_state * state,
    float       * scores_out,
    const void  * k_data,
    uint64_t     n_embd_k_gqa,
    size_t       row_bytes,
    uint32_t     kv_head_idx,
    uint32_t     head_calib_idx,
    const uint32_t * cell_indices,
    const int32_t  * positions,
    uint32_t     n_cells,
    int64_t      round_start,
    int          agg_mode,
    cudaStream_t stream)
{
    const auto & cfg = state->cfg;
    const uint32_t fc = cfg.freq_count;
    const uint32_t hd = cfg.head_dim;

    const float * qmr = state->d_q_mean_real  + (size_t)head_calib_idx * fc;
    const float * qmi = state->d_q_mean_imag  + (size_t)head_calib_idx * fc;
    const float * qma = state->d_q_mean_abs   + (size_t)head_calib_idx * fc;
    const float * ew  = state->d_extra_weight + (size_t)head_calib_idx * fc;

    const dim3 grid(n_cells, 1, 1);
    const dim3 block(fc, 1, 1);
    const size_t smem_bytes = (hd + fc) * sizeof(float);

    // Host-side head offset in bytes (ggml_row_size is host-only)
    const size_t head_off = ggml_row_size(cfg.k_type, (uint64_t)kv_head_idx * hd);

    #define LAUNCH_KERNEL(KTYPE, WHT, TRIG) \
        triattention_score_kernel<KTYPE, WHT, TRIG><<<grid, block, smem_bytes, stream>>>( \
            scores_out, k_data, n_embd_k_gqa, row_bytes, head_off, hd, \
            cell_indices, positions, n_cells, round_start, \
            state->d_omega, state->d_freq_scale_sq, state->d_offsets, cfg.n_offsets, \
            qmr, qmi, qma, ew, fc, agg_mode)

    // TQ3_0/TQ4_0 require head_dim=128 (one FWHT block per head).
    if ((cfg.k_type == GGML_TYPE_TQ3_0 || cfg.k_type == GGML_TYPE_TQ4_0) && hd != 128) {
        fprintf(stderr,
            "[TriAttention GPU] TQ3_0/TQ4_0 require head_dim=128, got %u\n", hd);
        return;
    }

    if (cfg.disable_trig) {
        switch (cfg.k_type) {
            case GGML_TYPE_TQ3_0: LAUNCH_KERNEL(GGML_TYPE_TQ3_0, true,  true); break;
            case GGML_TYPE_TQ4_0: LAUNCH_KERNEL(GGML_TYPE_TQ4_0, true,  true); break;
            case GGML_TYPE_Q8_0:  LAUNCH_KERNEL(GGML_TYPE_Q8_0,  false, true); break;
            case GGML_TYPE_F16:   LAUNCH_KERNEL(GGML_TYPE_F16,   false, true); break;
            case GGML_TYPE_F32:   LAUNCH_KERNEL(GGML_TYPE_F32,   false, true); break;
            default:
                fprintf(stderr, "[TriAttention GPU] unsupported K type %d\n", cfg.k_type);
                break;
        }
    } else {
        switch (cfg.k_type) {
            case GGML_TYPE_TQ3_0: LAUNCH_KERNEL(GGML_TYPE_TQ3_0, true,  false); break;
            case GGML_TYPE_TQ4_0: LAUNCH_KERNEL(GGML_TYPE_TQ4_0, true,  false); break;
            case GGML_TYPE_Q8_0:  LAUNCH_KERNEL(GGML_TYPE_Q8_0,  false, false); break;
            case GGML_TYPE_F16:   LAUNCH_KERNEL(GGML_TYPE_F16,   false, false); break;
            case GGML_TYPE_F32:   LAUNCH_KERNEL(GGML_TYPE_F32,   false, false); break;
            default:
                fprintf(stderr, "[TriAttention GPU] unsupported K type %d\n", cfg.k_type);
                break;
        }
    }

    #undef LAUNCH_KERNEL
}

// ============================================================================
// Host API: Init
// ============================================================================

triattention_gpu_state * triattention_gpu_init(
    const triattention_gpu_config * config,
    const triattention_gpu_head_calib * head_calibs,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets,
    void * stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    triattention_gpu_state * state = new triattention_gpu_state();
    state->cfg = *config;

    const uint32_t fc = config->freq_count;
    const uint32_t ns = config->n_sampled;
    const size_t calib_bytes = (size_t)ns * fc * sizeof(float);

    CUDA_CHECK(cudaMalloc(&state->d_q_mean_real,  calib_bytes));
    CUDA_CHECK(cudaMalloc(&state->d_q_mean_imag,  calib_bytes));
    CUDA_CHECK(cudaMalloc(&state->d_q_mean_abs,   calib_bytes));
    CUDA_CHECK(cudaMalloc(&state->d_extra_weight, calib_bytes));

    for (uint32_t h = 0; h < ns; h++) {
        const size_t off = (size_t)h * fc * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(
            (char *)state->d_q_mean_real + off, head_calibs[h].q_mean_real,
            fc * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            (char *)state->d_q_mean_imag + off, head_calibs[h].q_mean_imag,
            fc * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            (char *)state->d_q_mean_abs + off, head_calibs[h].q_mean_abs,
            fc * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            (char *)state->d_extra_weight + off, head_calibs[h].extra_weight,
            fc * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMalloc(&state->d_omega,         fc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_freq_scale_sq, fc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_offsets,       config->n_offsets * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(state->d_omega,         omega,
        fc * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(state->d_freq_scale_sq, freq_scale_sq,
        fc * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(state->d_offsets,      offsets,
        config->n_offsets * sizeof(float), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    return state;
}

// ============================================================================
// Host API: Score head
// ============================================================================

void triattention_gpu_score_head(
    triattention_gpu_state * state,
    const void   * k_data_dev,
    uint64_t       n_embd_k_gqa,
    size_t         row_bytes,
    uint32_t       kv_head_idx,
    uint32_t       head_calib_idx,
    const uint32_t * cell_indices_dev,
    const int32_t  * positions_dev,
    uint32_t       n_cells,
    int64_t        round_start,
    int            agg_mode,
    float        * scores_dev,
    void * stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (n_cells == 0) return;

    launch_score_kernel(state, scores_dev, k_data_dev,
                        n_embd_k_gqa, row_bytes, kv_head_idx,
                        head_calib_idx, cell_indices_dev, positions_dev,
                        n_cells, round_start, agg_mode, stream);
}

// ============================================================================
// Host API: Utility functions
// ============================================================================

void triattention_gpu_scores_to_host(
    float * scores_host,
    const float * scores_dev,
    uint32_t n_cells,
    void * stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    CUDA_CHECK(cudaMemcpyAsync(scores_host, scores_dev,
        n_cells * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void triattention_gpu_upload_cells(
    uint32_t     ** cell_indices_dev,
    int32_t      ** positions_dev,
    const uint32_t * cell_indices_host,
    const int32_t  * positions_host,
    uint32_t        n_cells,
    void * stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    CUDA_CHECK(cudaMalloc(cell_indices_dev, n_cells * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(positions_dev,    n_cells * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpyAsync(*cell_indices_dev, cell_indices_host,
        n_cells * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(*positions_dev, positions_host,
        n_cells * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
}

float * triattention_gpu_alloc_scores(uint32_t n_cells, void * /* stream_ptr */) {
    float * ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n_cells * sizeof(float)));
    return ptr;
}

void triattention_gpu_free_dev(void * ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// ============================================================================
// Host API: Cleanup
// ============================================================================

void triattention_gpu_free(triattention_gpu_state * state) {
    if (!state) return;

    cudaFree(state->d_q_mean_real);
    cudaFree(state->d_q_mean_imag);
    cudaFree(state->d_q_mean_abs);
    cudaFree(state->d_extra_weight);
    cudaFree(state->d_omega);
    cudaFree(state->d_freq_scale_sq);
    cudaFree(state->d_offsets);

    delete state;
}
