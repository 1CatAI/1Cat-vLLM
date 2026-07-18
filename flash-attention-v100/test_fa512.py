#!/usr/bin/env python3
"""Parity test for flash-attention-v100 head_dim=512 kernels (prefill + decode).

Data source: by default, real token vectors are harvested from a gemma-4
checkpoint's embed_tokens (--model-dir), sliced per head and RMS-normalized
to approximate the post-q_norm/k_norm/v_norm distribution seen in the model.
If no checkpoint is available, falls back to seeded random data (deterministic,
but with a less realistic distribution).

Usage:
    python test_fa512.py                            # all cases, random data
    python test_fa512.py --model-dir /path/to/gemma-4-31B-it-qat-w4a16-ct
    python test_fa512.py --bench                    # with perf measurement
"""

import argparse
import os

import torch
import torch.nn.functional as F

from flash_attn_v100 import (
    flash_attn_decode_paged,
    flash_attn_func,
    flash_attn_prefill_paged,
)

EMB_KEY = "model.language_model.embed_tokens.weight"
EMB_DIM = 5376
BLOCK_SIZE = 16
ATOL = 2e-2
RTOL = 2e-2

_MODEL_DIR = None
_EMB_CACHE = None


def harvest_embeddings(n_rows: int) -> torch.Tensor:
    """Harvest real embedding rows (bf16->fp16, CPU).

    Consecutive rows of embed_tokens are trained vectors whose heavy-tailed
    distribution with outliers is far more representative than gaussian noise;
    per-head slices stand in for post-norm Q/K/V. Falls back to seeded randn.
    """
    global _EMB_CACHE
    if _EMB_CACHE is not None and _EMB_CACHE.shape[0] >= n_rows:
        return _EMB_CACHE
    safetensors = None
    if _MODEL_DIR:
        candidate = os.path.join(_MODEL_DIR, "model.safetensors")
        if os.path.exists(candidate):
            safetensors = candidate
    if safetensors is not None:
        from safetensors import safe_open

        with safe_open(safetensors, framework="pt", device="cpu") as f:
            rows = f.get_slice(EMB_KEY)[:n_rows, :]  # [n_rows, 5376]
        _EMB_CACHE = rows.to(torch.float16)
        src = f"real embeddings from {safetensors}"
    else:
        g = torch.Generator().manual_seed(20260718)
        _EMB_CACHE = torch.randn(n_rows, EMB_DIM, generator=g).to(torch.float16)
        src = "seeded random data (--model-dir not given)"
    print(f"[harvest] {n_rows} rows, {src}, std={_EMB_CACHE.std().item():.4f}")
    return _EMB_CACHE


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()


def make_real_qkv(B, H, H_kv, M, S, D, device="cuda"):
    """Build q and paged k/v cache from harvested embeddings.

    Head h takes the D-dim slice starting at (h*D)%EMB_DIM; every (b,h,m/s)
    uses a different token row; finally RMS-normalized per head.

    Note: the prefill kernel's q layout is [B, M, H, D] (seq-major, see
    fused_mha_forward_paged.cu: q_ptr = Q + q_head_linear*M*D; the vLLM
    backend passes query[start:end].unsqueeze(0), same layout).
    """
    n_q = B * H * M
    n_kv = B * H_kv * S
    emb = harvest_embeddings(n_q + n_kv)
    q_rows, kv_rows = emb[:n_q], emb[n_q:n_q + n_kv]

    q = torch.empty(B, M, H, D, dtype=torch.float16)  # [B,M,H,D]
    kc = torch.empty(B, H_kv, S, D, dtype=torch.float16)
    vc = torch.empty(B, H_kv, S, D, dtype=torch.float16)
    for b in range(B):
        for h in range(H):
            base = (b * H + h) * M
            off = (h * D) % EMB_DIM
            seg = q_rows[base:base + M]
            if off + D <= EMB_DIM:
                q[b, :, h, :] = seg[:, off:off + D]
            else:
                q[b, :, h, :] = torch.cat(
                    [seg[:, off:], seg[:, :D - (EMB_DIM - off)]], dim=-1)
        for h in range(H_kv):
            base = (b * H_kv + h) * S
            off = (h * D) % EMB_DIM
            off_v = ((H_kv - 1 - h) * D) % EMB_DIM
            seg = kv_rows[base:base + S]
            if off + D <= EMB_DIM:
                kc[b, h] = seg[:, off:off + D]
                vc[b, h] = seg[:, off_v:off_v + D]
            else:
                kc[b, h] = torch.cat(
                    [seg[:, off:], seg[:, :D - (EMB_DIM - off)]], dim=-1)
                vc[b, h] = torch.cat(
                    [seg[:, off_v:], seg[:, :D - (EMB_DIM - off_v)]], dim=-1)
    q = rms_norm(q.float()).to(torch.float16)
    kc = rms_norm(kc.float()).to(torch.float16)
    vc = rms_norm(vc.float()).to(torch.float16)
    return q.to(device), kc.to(device), vc.to(device)


def pack_paged_kv(k, v, block=BLOCK_SIZE):
    """[B,H_kv,S,D] -> paged cache [nblocks, block, H_kv, D] + block_table."""
    B, H_kv, S, D = k.shape
    nblk = (S + block - 1) // block
    total = B * nblk
    kc = torch.zeros(total, block, H_kv, D, dtype=k.dtype, device=k.device)
    vc = torch.zeros_like(kc)
    table = torch.arange(total, dtype=torch.int32, device=k.device).view(B, nblk)
    for b in range(B):
        for i in range(S):
            blk, off = i // block, i % block
            kc[table[b, blk], off, :] = k[b, :, i, :]
            vc[table[b, blk], off, :] = v[b, :, i, :]
    return kc, vc, table


def ref_attention(q, k, v, causal=True):
    """fp32 reference. q [B,M,H,D] (seq-major), k/v [B,H_kv,S,D].

    When chunked, q holds the last M positions of the sequence.
    """
    q = q.transpose(1, 2)  # -> [B,H,M,D]
    B, H, M, D = q.shape
    H_kv, S = k.shape[1], k.shape[2]
    rep = H // H_kv
    k = k.float().repeat_interleave(rep, dim=1)
    v = v.float().repeat_interleave(rep, dim=1)
    scores = torch.einsum("bhmd,bhsd->bhms", q.float(), k) * (D ** -0.5)
    if causal:
        qi = torch.arange(S - M, S, device=q.device).view(M, 1)
        kj = torch.arange(S, device=q.device).view(1, S)
        mask = kj > qi
        scores = scores.masked_fill(mask, float("-inf"))
    p = scores.softmax(dim=-1)
    out = torch.einsum("bhms,bhsd->bhmd", p, v)  # [B,H,M,D]
    return out.transpose(1, 2).to(torch.float16)  # -> [B,M,H,D]


def compare(name, got, ref):
    got, ref = got.float().cpu(), ref.float().cpu()
    if got.shape != ref.shape:
        got = got.reshape(ref.shape)
    diff = (got - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-3)
    ok = torch.allclose(got, ref, atol=ATOL, rtol=RTOL)
    print(f"  {name}: max_abs={diff.max().item():.5f} "
          f"mean_abs={diff.mean().item():.6f} "
          f"max_rel={rel.max().item():.4f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def case_prefill(D, B, H, H_kv, M, S, expect_fail=False, bench=False):
    print(f"[prefill] D={D} B={B} H={H} H_kv={H_kv} M={M} S={S}")
    q, k, v = make_real_qkv(B, H, H_kv, M, S, D)
    kc, vc, table = pack_paged_kv(k, v)
    seq_lens = torch.full((B,), S, dtype=torch.int32, device=q.device)
    try:
        if bench:
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
            for _ in range(5):
                out = flash_attn_prefill_paged(
                    q, kc, vc, table, seq_lens, causal=True)
            ev1.record()
            torch.cuda.synchronize()
            ms = ev0.elapsed_time(ev1) / 5
            print(f"  bench: {ms:.2f} ms/iter, {B * M / (ms / 1000):.1f} tok/s")
        else:
            out = flash_attn_prefill_paged(
                q, kc, vc, table, seq_lens, causal=True)
    except RuntimeError as e:
        print(f"  call failed: {str(e).splitlines()[0][:150]}")
        return "EXPECTED_FAIL" if expect_fail else False
    if expect_fail:
        print("  expected to fail but succeeded (kernel already adapted?)")
    ref = ref_attention(q, k, v, causal=True)
    return compare("prefill", out, ref)


def case_dense_prefill(D, B, H, H_kv, M, S, expect_fail=False):
    """Dense prefill path (flash_attn_func, fused_mha_forward.cu).

    The vLLM backend's no-prefix prefill goes here (flash_v100_dense_prefill),
    q layout [B, T, H, D], M==S.
    """
    print(f"[dense  ] D={D} B={B} H={H} H_kv={H_kv} M={M} S={S}")
    q, k, v = make_real_qkv(B, H, H_kv, M, S, D)  # q [B,M,H,D]; k/v [B,H_kv,S,D]
    kd = k.transpose(1, 2).contiguous()  # [B,S,H_kv,D]
    vd = v.transpose(1, 2).contiguous()
    try:
        out = flash_attn_func(q, kd, vd, causal=True)
    except RuntimeError as e:
        print(f"  call failed: {str(e).splitlines()[0][:150]}")
        return "EXPECTED_FAIL" if expect_fail else False
    ref = ref_attention(q, k, v, causal=True)
    return compare("dense", out, ref)


def case_decode(D, B, H, H_kv, S, expect_fail=False):
    print(f"[decode ] D={D} B={B} H={H} H_kv={H_kv} S={S}")
    q_full, k, v = make_real_qkv(B, H, H_kv, 1, S, D)  # [B,1,H,D]
    q = q_full[:, 0]  # [B, H, D]
    kc, vc, table = pack_paged_kv(k, v)
    seq_lens = torch.full((B,), S, dtype=torch.int32, device=q.device)
    try:
        out = flash_attn_decode_paged(q, kc, vc, table, seq_lens)
    except RuntimeError as e:
        print(f"  call failed: {str(e).splitlines()[0][:150]}")
        return "EXPECTED_FAIL" if expect_fail else False
    ref = ref_attention(q_full, k, v, causal=False)[:, 0]
    return compare("decode", out, ref)


def main():
    global _MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument(
        "--model-dir",
        default=os.environ.get("FA512_MODEL_DIR"),
        help="gemma-4 checkpoint dir for real-embedding harvest "
             "(falls back to seeded random data when absent)",
    )
    args = ap.parse_args()
    _MODEL_DIR = args.model_dir

    assert torch.cuda.is_available()
    results = {}

    # A. Regression on the in-service paths (harness self-check, must PASS)
    results["prefill_D256"] = case_prefill(256, 2, 32, 16, 256, 256)
    results["decode_D256"] = case_decode(256, 2, 32, 16, 373)
    results["dense_D256"] = case_dense_prefill(256, 2, 32, 16, 256, 256)

    # B. Target: D=512 (must all PASS after the kernel adaptation)
    results["prefill_D512"] = case_prefill(512, 2, 32, 4, 256, 256)
    results["prefill_D512_chunked"] = case_prefill(512, 1, 32, 4, 64, 1024)
    results["decode_D512"] = case_decode(512, 2, 32, 4, 373)
    results["dense_D512"] = case_dense_prefill(512, 2, 32, 4, 256, 256)

    if args.bench:
        print("\n[perf] baseline record only (re-run to compare after changes)")
        case_prefill(256, 2, 32, 16, 512, 512, bench=True)

    print("\n===== summary =====")
    bad = 0
    for k, r in results.items():
        if r == "EXPECTED_FAIL":
            print(f"  {k}: EXPECTED_FAIL (kernel not adapted yet)")
        elif r is True:
            print(f"  {k}: PASS")
        else:
            print(f"  {k}: FAIL  <-- unexpected")
            bad += 1
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
