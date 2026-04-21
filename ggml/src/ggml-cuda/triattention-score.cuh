/*
 * TriAttention GPU scoring kernel (CUDA/HIP)
 *
 * Computes TriAttention importance scores for KV cache entries directly on
 * the GPU, avoiding the costly GPU→CPU transfer of the full K tensor.
 * Only the resulting score array (one float per position) is copied back.
 *
 * v1 scope: dense K types only (F32, F16, Q8_0). TurboQuant (WHT-rotated)
 * and IsoQuant (quaternion-rotated) paths are deferred to v2.
 *
 * Reference: "TriAttention: Trigonometric KV Cache Eviction" (arXiv 2604.04921)
 */

#pragma once

#include "common.cuh"

// TriAttention GPU scoring API declarations are in ggml-cuda.h (included via common.cuh).
// Only the CUDA kernel implementation details are declared here.
