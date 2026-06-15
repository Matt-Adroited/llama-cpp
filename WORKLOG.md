# WORKLOG — llama-turboquant fork

**Repo:** `/home/matt/Projects/random/llama-cpp`  
**Branch:** `prefix-block-cache`  
**Origin:** `github.com/Matt-Adroited/llama-turboquant`  
**Upstream:** `ggml-org/llama.cpp` (tracking master)  
**Target:** AMD Strix Halo, gfx1151, ROCm 7.13 / HIP 7.13.26154, 128.5 GB unified LPDDR5X-8000  
**Primary build:** `build-hip-kda-opt` (LTO on, rocWMMA FATTN on, HIP graphs on)

Deeper per-topic notes live in `~/.claude/projects/-home-matt-Projects-random/memory/project_*.md`.  
This file is the chronological index — jump to a memory entry for details.

---

## Timeline

### 2026-04-04 — TurboQuant ported to upstream
- Rebased TQ2_0 / TQ3_0 / TQ4_0 KV-cache quantization onto latest upstream master (0 commits behind).
- Covers CPU + CUDA/HIP; per-block 128 WHT rotation + centroid codebook.
- MoE + TQ interaction validated; grub IOMMU fix needed on boot.
- → `project_turboquant_upstream.md`

### 2026-04-05 — Block prefix caching working
- vLLM-style block-level prefix cache for `llama-server`.
- KV-only: 3.4× prefill speedup on repeat prompts. Hybrid (recurrent + KV): 3.7×.
- Branch pushed.
- → `project_block_prefix_caching.md`

### 2026-04-15 — KDA integration + commits landed
- Kimi Delta Attention (Moonshot AI) ported for AMD HIP; Kimi-Linear-48B decoding.
- State quantization CLI flag `-cts` added.
- 4 commits landed on `prefix-block-cache`:
  - `d910b8177` Port TurboQuant KV quantization to latest upstream
  - `4d44a776a` Fix TQ3_0/TQ4_0 vec_dot linker errors on x86
  - `c22d1f5d9` Add KDA kernel, state quantization (`-cts`), QUICKSTART.md
  - `efe322390` Add vLLM-style block-level prefix caching for server
- **This is the last committed state.** Everything below is uncommitted work.
- → `project_kda_integration.md`

### 2026-04-19 — IsoQuant-Fast validated
- IsoQuant-Fast: quaternion rotation + Lloyd–Max centroids, `block=32`, ISO3_0 (3-bit) and ISO4_0 (4-bit).
- First full validation on Kimi-Linear MLA (576-dim head), 51.2 tg t/s.
- Files: `ggml-quants.c/h`, `fattn-vec.cuh`, `set-rows.cu`, `cpy-utils.cuh`, new `fattn-vec-instance-iso{3,4}_0-iso{3,4}_0.cu` template instances.
- → `project_isoquant_fast.md`

### 2026-04-19 — Asymmetric K≠V flash-attention guard
- Blanket `K->type != V->type` rejection at `fattn.cu:394` was silently forcing CPU fallback for any TQ3/TQ2 or ISO3/ISO4 asymmetric KV config.
- Replaced with allowlist for the legitimate asymmetric combinations.
- TQ3_0/TQ2_0 decode: 16.4 t/s (was CPU fallback). ISO3_0/ISO4_0 decode: 18.7 t/s.
- → `project_asymmetric_kv_guard.md`

### 2026-04-19 — TriAttention v1 (dense K)
- Ported pre-RoPE Q/K trig scoring KV eviction (arXiv 2604.04921) from `atomicmilkshake/llama-cpp-turboquant@feature/triattention` into our fork.
- GPU scoring kernel in HIP + CPU glue + 13 CLI flags + calibration script (`scripts/calibrate-triattention.py`, runs on native ROCm PyTorch, no CPU calibration).
- Dense K types only: F32 / F16 / Q8_0. Standard RoPE models (Llama, Qwen, Mistral, Gemma).
- Validated on Qwen2.5-1.5B + 7B. Prune event: 2–10 ms on gfx1151. +13% decode at 19K ctx on 7B.
- New files: `src/llama-triattention.{cpp,h}`, `ggml/src/ggml-cuda/triattention-score.{cu,cuh}`.
- → `project_triattention_v1.md`

### 2026-04-20 — TriAttention v2 (TQ3_0 K)
- v2 scoring kernel adds `NEED_WHT_INV` template path: dequant TQ3_0 K block → in-place WHT inverse → trig scoring.
- End-to-end validated on Llama-3.1-8B with `-ctk tq3_0`.
- Finding: eviction is query-blind, so needle-in-haystack retrieval requires `budget ≥ prompt length`; v1's default `budget=2048` evicts the needle on 100K prompts.
- → `project_triattention_v2_tq3.md`

### 2026-04-20 — Rotated-KV model compatibility table
- Tested TQ/ISO on a matrix of models.
- **Works:** Llama (all), Qwen3 (all). Full quality.
- **Breaks:** Qwen2.x (biased QKV projections — TQ/ISO rotation assumes zero-mean K), speculative draft models (draft+target KV formats don't match).
- → `project_rotated_kv_model_compat.md`

### 2026-04-20 — rocWMMA FATTN enabled on gfx1151
- Upstream `rocWMMA` only recognizes gfx1100/1101/1102/1200/1201 for RDNA.
- Patched **system** `/usr/include/rocwmma/internal/config.hpp` to add `__gfx1150__` and `__gfx1151__` → alias to `ROCWMMA_ARCH_GFX1100` (same WMMA ISA). Backup at `.bak`. **Risk:** ROCm package update will overwrite; re-apply if so.
- Flipped `GGML_HIP_ROCWMMA_FATTN=ON` and `GGML_LTO=ON` in `build-hip-kda-opt`.
- Fixed **latent FA dispatch bug** surfaced by enabling rocWMMA: WMMA path selected for TQ/ISO K/V, then `ggml_get_to_fp16_cuda()` returned `nullptr` (TQ/ISO have no F16 converter) → NULL ptr call, segfault. Fix: hoist TQ/ISO → `BEST_FATTN_KERNEL_VEC` dispatch **before** any tensor-core branch. Added defensive `GGML_ASSERT(to_fp16 != nullptr)` in `launch_fattn`.
- HIP graphs "experimental, slow" label is stale on ROCm 7.x — decode identical with/without (25.4 t/s both).
- Short-ctx (8k) decode: F16/F16 25.4 t/s, TQ3/TQ2 22.3 t/s, ISO3/ISO4 23.5 t/s.
- Files: `ggml/src/ggml-cuda/fattn.cu`, `fattn-common.cuh`. System file: `/usr/include/rocwmma/internal/config.hpp`.
- → `project_rocwmma_fattn_enabled.md`

### 2026-04-20/21 — Speculative decoding validated (no code changes)
- Llama-3.1-8B-Instruct-Q8_0 (target) + Llama-3.2-1B-Instruct-Q8_0 (draft), F16/F16 KV. Works day-1 on this branch: llama-server's spec plumbing untouched by our KDA/TriAttention/prefix-cache work, and narrow-gate FA dispatch (`Q->ne[1] <= 2` → VEC) means target's batch=N verify auto-routes to WMMA — free speedup, no kernel changes.
- `--draft-max` sweep on Llama (F16 KV, warmup + measured, T=0.0 seed=0):
  | dm | CODE t/s | HAIKU t/s | TRANSFORMER t/s | Notes |
  |---|---|---|---|---|
  | 8  | 180 | 172 | 64 | yesterday's config |
  | 10 | 217 | 79* | 58 | *haiku variance, small N |
  | **12** | **248** | **240** | 58 | **peak for code / structured text** |
  | 16 | 202 | 196 | 56 | verify cost dominates |
  At 100% draft acceptance on code / haiku, dm=12 packs 12 accepted tokens per verify pass → effective decode 248 t/s. Baseline 25.3 t/s → **9.8× speedup** on code (vs 5.07× at dm=8 yesterday). Transformer (unstructured prose, ~81% accept) peaks at dm=8 — longer drafts waste more verify work when accept < 85%.
- **Measurement protocol gotcha:** HIP graphs need warmup. First run of each prompt is ~2× slower than subsequent runs (kernel / graph capture cost). Always discard the first response. Yesterday's 133 t/s code figure was a warmed-up 2nd run; cold-start measurements look like ~64 t/s regression.
- TriAttention composes cleanly — same spec numbers with `--triattention-stats` + `--triattention-budget 2048`.
- MoE target did **not** generalize. Qwen3-Coder-30B-A3B-Instruct-Q4_K_M with jukofyork DRAFT-0.75B and Qwen3-0.6B Q8 drafts both regress on short/long text; best result is 1.13× on code with Qwen3-0.6B (82% accept). Two structural reasons: (1) MoE baseline already at 60 t/s, ~50% of 256 GB/s LPDDR5X roof — little headroom; (2) MoE expert routing produces next-token distributions no small dense draft can model well, acceptance seems capped ~82% on family-match drafts vs 96% on Llama dense.
- Files: none.
- → `project_speculative_decoding.md`, `project_speculative_decoding_qwen_moe.md`

### 2026-04-20 — TQ/ISO prefill via dequant-to-F16 wrapper
- Problem: any quantized K/V forced the FA dispatch into `VEC` for both decode *and* prefill. Vec is serial across the query axis — fine for batch=1 decode, catastrophic for prefill. F16/F16 prefill was 1141 t/s @ 3226 tok; TQ3/TQ2 sat at 142 t/s, ISO3/ISO4 at 162 t/s.
- Fix, all in `fattn.cu`:
  1. Narrow the VEC-force gate to `Q->ne[1] <= 2` (decode / tiny batch) — prefill falls through.
  2. New `kernel_dequant_tq_to_f16<block, QK, QBITS>` covering TQ2_0/TQ3_0/TQ4_0 — one thread per block, in-place WHT butterfly, sign flip via `TQ3_0_SIGNS_FA[128]`, scale `d * rsqrtf(QK)`.
  3. Generalize the ISO-only `ggml_cuda_flash_attn_ext_iso_dequant` → `ggml_cuda_flash_attn_ext_quant_dequant` handling both TQ and ISO; extended dispatch switch to WMMA_F16 + VEC in addition to TILE + MMA.
  4. Trigger: `(is_tq || is_iso) && Q->ne[1] > 2` routes through the wrapper; kept the prior ISO D>256 path.
- Results @ 3226-tok prefill:
  | Config | Prefill (t/s) | Decode (t/s) |
  |---|---|---|
  | F16/F16 | 1157.7 | 22.6 |
  | TQ3_0/TQ2_0 | **1135.2** (was 142, +8×) | 22.4 |
  | ISO3_0/ISO4_0 | **1142.1** (was 162, +7×) | 23.6 |
- Quality verified: identical degenerate loop on F16/TQ/ISO for the physics-words prompt, coherent on normal prompts → dequant kernel numerically correct.
- Files: `ggml/src/ggml-cuda/fattn.cu`.
- → `project_tq_iso_prefill_wmma.md`

### 2026-06-15 — Fresh-base re-port onto upstream master (`turboquant-sync-jun26`)
- Upstream had drifted ~834 commits. Rather than merge, re-ported the whole fork
  feature-by-feature onto a clean worktree branched off upstream tip `6e9007ae6`
  (`/home/matt/Projects/random/llama-turboquant-sync`, branch `turboquant-sync-jun26`).
- **Strategic call:** dropped our bespoke KDA op in favor of upstream's now-native
  `GGML_OP_GATED_DELTA_NET` (multi-backend). Kept everything else: TurboQuant,
  IsoQuant, TriAttention, the vLLM block-prefix cache, and the `-cts` flag (re-wired
  onto upstream's recurrent memory instead of our old KDA path).
- 9 commits, each build-validated on the gfx1151 HIP tree; final full-tree build green
  (124 targets, only cosmetic warnings):
  - `e8dbd10aa` TurboQuant base · `af3ee8180` x86 fix · `3e0967aba` IsoQuant
  - `049e75ab7` fattn dispatch · `ee666631f` q1_0 collision fix · `5a80370aa` server cache
  - `66e127a0b` TriAttention · `be2d7c56a` `-cts` state-quant · `f91d3d917` spec/bench/server polish
- **Conflicts of note:** upstream renamed the server context member to `ctx_tgt`/`ctx_dft`
  (spec-decode context split) — 11 bare `ctx` refs rewired. Upstream added a native x86
  `q1_0` vec_dot, colliding with our fallback `#define` (removed ours). The
  `params.speculative` → `params.speculative.draft` struct refactor changed the draft
  KV-type field path; spec-simple re-wired accordingly.
- Speculative decoding confirmed to need **no fork code** on the new base (upstream's
  parallel-draft + checkpoint rewrite supersedes the old single-draft loop).
- **Not yet merged** into `prefix-block-cache` — sits on its own branch for review.
- ⚠️ Tooling: the `rtk` git/grep/sed output filter fabricated diffs during this session
  (showed phantom "uncommitted" changes that didn't exist on disk). Ground-truthed every
  change with the native Read/Edit tools and `rtk proxy <cmd>` (raw, unfiltered) instead.

---

## Stacked wins on `prefix-block-cache` vs upstream

| Area | Delta |
|---|---|
| TurboQuant (TQ2/3/4) KV quant | new |
| IsoQuant-Fast (ISO3/4) KV quant | new |
| KDA attention (Moonshot) | new |
| `-cts` state quantization flag | new |
| Block prefix cache for server | 3.4–3.7× prefill on repeats |
| Asymmetric K/V guard fix | TQ/ISO GPU path restored |
| rocWMMA FATTN on gfx1151 | F16 prefill 1141 t/s |
| TQ/ISO prefill dequant-to-F16 | quantized KV at F16 prefill speed |
| TriAttention v1 + v2 | KV eviction (dense + TQ3 K) |
| Speculative decoding (built-in) | up to 9.8× decode on dense Llama code gen (dm=12); flat on MoE |

## Known risks / re-apply items

- System rocWMMA config patch (`/usr/include/rocwmma/internal/config.hpp`) is wiped by ROCm package updates. Backup at `.bak`.
- Speculative-decoding DRAFT models + TQ/ISO incompatible (format mismatch) — spec needs F16 KV on both target and draft.
- MoE targets have a low spec-decoding ceiling (~1.1× on code, regression on prose) regardless of draft choice — expert routing makes distribution hard to predict with small dense drafts.
- Qwen2.x models (biased QKV) don't tolerate rotated-KV quantization.

## Verification commands

```bash
# Long prefill bench (TQ3/TQ2)
./build-hip-kda-opt/bin/llama-server \
  -m /home/matt/Projects/ai/llama-3.1-8b-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  -c 32768 -ngl 99 -fa on -ctk tq3_0 -ctv tq2_0 --port 8099 &
curl -s -X POST http://127.0.0.1:8099/completion -H "Content-Type: application/json" \
  -d @/tmp/long-req.json | jq .timings
```

## Pending / deferred

- TriAttention v2 MLA adaptation (Kimi-Linear, partial RoPE) — deferred from v1.
- `--parallel N` + spec decoding interaction: does multi-slot share the WMMA verify pass, or is each slot isolated? (Throughput question for serving.)
- Better MoE draft options — either distill a purpose-built draft from Qwen3-Coder-30B-A3B, or test dense Qwen3 targets (14B/32B class) where the spec break-even math is more favorable.
- Draft-max sweep on Llama-3.1-8B code gen — 96% accept at dm=8 suggests we're under-drafting; raising to 12-16 may push past 133 t/s.
