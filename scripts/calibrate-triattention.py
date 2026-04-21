#!/usr/bin/env python3
"""
TriAttention calibration — collects pre-RoPE query statistics from a HF model.

Writes a .triattention binary file matching the format in
src/llama-triattention.h (magic 0x54524941 / "TRIA", version 1).

Algorithm (paper arXiv 2604.04921, Section 3):
  For each attention head h in each layer l:
    1. Forward calibration tokens through the model.
    2. Capture Q projection output (pre-RoPE).
    3. Reshape to (..., head_dim) and split into (freq_count, 2) pairs
       representing complex numbers q_f = q_real + i*q_imag.
    4. Compute E[q_f], E[|q_f|] across all tokens and the corpus.
    5. Store per-(layer, head) means for use at runtime.

Compatible with native PyTorch on ROCm 7.13 (torch.device("cuda") maps to HIP).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TRIATTENTION_MAGIC = 0x54524941
TRIATTENTION_VERSION = 1

ROPE_STYLE_HALF = 0          # Llama, Qwen, Mistral — [real_half | imag_half]
ROPE_STYLE_INTERLEAVED = 1   # GPT-NeoX variant — [r0, i0, r1, i1, ...]


@dataclass
class HeadStats:
    layer_idx: int
    head_idx: int
    q_mean_real: np.ndarray   # [freq_count]
    q_mean_imag: np.ndarray
    q_abs_mean:  np.ndarray
    r_f:         np.ndarray   # ||E[q_f]|| / E[|q_f|], validation metric


def split_complex_half(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Half layout: first half = real, second half = imag."""
    head_dim = q.shape[-1]
    fc = head_dim // 2
    return q[..., :fc], q[..., fc:]


def split_complex_interleaved(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved layout: (r0, i0, r1, i1, ...)."""
    head_dim = q.shape[-1]
    q2 = q.reshape(*q.shape[:-1], head_dim // 2, 2)
    return q2[..., 0], q2[..., 1]


def detect_rope_style(model) -> int:
    """Best-effort detection — Llama/Qwen/Mistral/Gemma use half layout in HF."""
    arch = type(model).__name__.lower()
    if any(k in arch for k in ("llama", "qwen", "mistral", "gemma", "phi")):
        return ROPE_STYLE_HALF
    return ROPE_STYLE_HALF


def install_q_hooks(model, head_dim: int, num_heads: int, rope_style: int):
    """Find q_proj layers and install hooks that accumulate stats per head."""
    fc = head_dim // 2

    sums_real: dict[int, torch.Tensor] = {}
    sums_imag: dict[int, torch.Tensor] = {}
    sums_abs:  dict[int, torch.Tensor] = {}
    counts:    dict[int, int]          = {}

    split = split_complex_half if rope_style == ROPE_STYLE_HALF else split_complex_interleaved

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            # output: (B, T, num_heads * head_dim) — Linear layer output
            B, T, H = output.shape
            q = output.view(B, T, num_heads, head_dim).float()
            q_r, q_i = split(q)   # (B, T, num_heads, fc) each
            q_abs = torch.sqrt(q_r * q_r + q_i * q_i)

            # Accumulate over (B, T) → per-head per-freq sums
            s_r = q_r.sum(dim=(0, 1))   # (num_heads, fc)
            s_i = q_i.sum(dim=(0, 1))
            s_a = q_abs.sum(dim=(0, 1))

            if layer_idx not in sums_real:
                sums_real[layer_idx] = s_r.detach().cpu()
                sums_imag[layer_idx] = s_i.detach().cpu()
                sums_abs[layer_idx]  = s_a.detach().cpu()
                counts[layer_idx]    = B * T
            else:
                sums_real[layer_idx] += s_r.detach().cpu()
                sums_imag[layer_idx] += s_i.detach().cpu()
                sums_abs[layer_idx]  += s_a.detach().cpu()
                counts[layer_idx]    += B * T
        return hook

    handles = []
    layer_modules = []

    # Locate decoder layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_modules = list(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer_modules = list(model.transformer.h)
    else:
        raise RuntimeError("Could not locate decoder layers (expected model.model.layers or transformer.h)")

    for li, layer in enumerate(layer_modules):
        # Llama/Qwen/Mistral/Gemma: layer.self_attn.q_proj
        q_proj = None
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
            q_proj = layer.self_attn.q_proj
        elif hasattr(layer, "attention") and hasattr(layer.attention, "q_proj"):
            q_proj = layer.attention.q_proj
        if q_proj is None:
            raise RuntimeError(f"Layer {li}: could not find q_proj")
        handles.append(q_proj.register_forward_hook(make_hook(li)))

    return handles, sums_real, sums_imag, sums_abs, counts, len(layer_modules)


def load_calibration_text(path: str | None, n_chars_min: int) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    # Fallback: small built-in corpus (good enough for smoke tests).
    return (
        "The quick brown fox jumps over the lazy dog. " * 200
        + "Attention is all you need. Transformer architectures have revolutionized NLP. " * 200
        + "In recent years, large language models have demonstrated emergent capabilities. " * 200
    )


def write_triattention_file(
    path: str,
    head_dim: int,
    num_layers: int,
    num_attn_heads: int,
    num_kv_heads: int,
    rope_theta: float,
    rope_style: int,
    model_name: str,
    per_head_stats: list[HeadStats],
):
    fc = head_dim // 2
    name_bytes = model_name.encode("utf-8") + b"\0"

    with open(path, "wb") as f:
        f.write(struct.pack("<I", TRIATTENTION_MAGIC))
        f.write(struct.pack("<I", TRIATTENTION_VERSION))
        f.write(struct.pack("<I", head_dim))
        f.write(struct.pack("<I", num_layers))
        f.write(struct.pack("<I", num_attn_heads))
        f.write(struct.pack("<I", num_kv_heads))
        f.write(struct.pack("<d", rope_theta))
        f.write(struct.pack("<I", rope_style))
        f.write(struct.pack("<I", len(per_head_stats)))
        f.write(struct.pack("<I", fc))
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)

        for s in per_head_stats:
            f.write(struct.pack("<I", s.layer_idx))
            f.write(struct.pack("<I", s.head_idx))
            f.write(s.q_mean_real.astype(np.float32).tobytes())
            f.write(s.q_mean_imag.astype(np.float32).tobytes())
            f.write(s.q_abs_mean .astype(np.float32).tobytes())
            f.write(s.r_f        .astype(np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--n-tokens", type=int, default=2048)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--output", required=True, help="Output .triattention file")
    ap.add_argument("--corpus", default=None, help="Optional plain-text corpus; otherwise uses a small built-in fallback")
    ap.add_argument("--device", default="cuda", help="torch device (default cuda — maps to HIP on ROCm)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--rope-style", default="auto", choices=["auto", "half", "interleaved"])
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    print(f"[calibrate] loading tokenizer/model {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(device).eval()

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
    num_heads  = getattr(cfg, "num_attention_heads", None) or cfg.n_head
    head_dim   = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads)
    num_kv     = getattr(cfg, "num_key_value_heads", num_heads)

    # rope_theta can live in several places across transformers versions:
    #   old: cfg.rope_theta (scalar)
    #   new: cfg.rope_parameters["rope_theta"]  or  cfg.rope_scaling["rope_theta"]
    rope_theta = None
    for src in (getattr(cfg, "rope_parameters", None), getattr(cfg, "rope_scaling", None)):
        if isinstance(src, dict) and "rope_theta" in src:
            rope_theta = src["rope_theta"]
            break
    if rope_theta is None:
        rope_theta = getattr(cfg, "rope_theta", None)
    if rope_theta is None:
        rope_theta = 10000.0
    rope_theta = float(rope_theta)

    if args.rope_style == "auto":
        rope_style = detect_rope_style(model)
    else:
        rope_style = ROPE_STYLE_HALF if args.rope_style == "half" else ROPE_STYLE_INTERLEAVED

    print(f"[calibrate] layers={num_layers} heads={num_heads} kv_heads={num_kv} "
          f"head_dim={head_dim} rope_theta={rope_theta} rope_style={rope_style}", flush=True)

    handles, sums_r, sums_i, sums_a, counts, _ = install_q_hooks(
        model, head_dim, num_heads, rope_style)

    text = load_calibration_text(args.corpus, args.n_tokens * 8)
    ids = tok(text, return_tensors="pt", truncation=False).input_ids[0]
    if ids.numel() < args.n_tokens:
        print(f"[calibrate] WARNING: corpus only has {ids.numel()} tokens, wanted {args.n_tokens}", flush=True)

    ids = ids[: args.n_tokens]

    # Chunk into max_seq_len windows for memory safety
    chunks = [ids[i : i + args.max_seq_len] for i in range(0, ids.numel(), args.max_seq_len)]
    print(f"[calibrate] running {len(chunks)} chunks ({ids.numel()} tokens total) ...", flush=True)

    with torch.no_grad():
        for ci, chunk in enumerate(chunks):
            inp = chunk.unsqueeze(0).to(device)
            model(inp)
            if (ci + 1) % 4 == 0 or ci == len(chunks) - 1:
                print(f"[calibrate]   chunk {ci+1}/{len(chunks)}", flush=True)

    for h in handles:
        h.remove()

    # Convert sums → per-(layer, head) stats
    per_head: list[HeadStats] = []
    fc = head_dim // 2
    for li in range(num_layers):
        if li not in counts or counts[li] == 0:
            continue
        N = float(counts[li])
        mean_r = (sums_r[li] / N).numpy()    # (num_heads, fc)
        mean_i = (sums_i[li] / N).numpy()
        mean_a = (sums_a[li] / N).numpy()
        for h in range(num_heads):
            qr = mean_r[h]
            qi = mean_i[h]
            qa = mean_a[h]
            norm_e = np.sqrt(qr * qr + qi * qi)
            r_f = np.where(qa > 1e-12, norm_e / np.maximum(qa, 1e-12), np.zeros_like(qa))
            per_head.append(HeadStats(li, h, qr, qi, qa, r_f))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_triattention_file(
        args.output,
        head_dim=head_dim,
        num_layers=num_layers,
        num_attn_heads=num_heads,
        num_kv_heads=num_kv,
        rope_theta=rope_theta,
        rope_style=rope_style,
        model_name=args.model,
        per_head_stats=per_head,
    )

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"[calibrate] wrote {args.output}  ({len(per_head)} head entries, {size_mb:.2f} MiB)", flush=True)
    # Sanity: average r_f across heads — should be > 0 and < 1 for trained models.
    all_rf = np.concatenate([s.r_f for s in per_head])
    print(f"[calibrate] r_f stats: mean={all_rf.mean():.4f} min={all_rf.min():.4f} max={all_rf.max():.4f}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
