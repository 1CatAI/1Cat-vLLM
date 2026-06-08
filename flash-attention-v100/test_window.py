"""Standalone correctness test for sliding-window FLASH_ATTN_V100 (Phase 1).

Compares the windowed paged decode + prefill kernels against an fp32 torch
reference. Run on a V100 (SM70). No vLLM, no model load.
"""
import torch
from flash_attn_v100 import (flash_attn_decode_paged, flash_attn_prefill_paged,
                             flash_attn_func)

torch.manual_seed(0)
DEV = "cuda"


def build_paged(k_cont, v_cont, block_size):
    """k_cont/v_cont: [S, Hkv, D] -> paged [num_blocks, block_size, Hkv, D] + block_table [1, nb]."""
    S, Hkv, D = k_cont.shape
    nb = (S + block_size - 1) // block_size
    k_cache = torch.zeros((nb, block_size, Hkv, D), dtype=k_cont.dtype, device=DEV)
    v_cache = torch.zeros((nb, block_size, Hkv, D), dtype=v_cont.dtype, device=DEV)
    for b in range(nb):
        s = b * block_size
        e = min(s + block_size, S)
        k_cache[b, : e - s] = k_cont[s:e]
        v_cache[b, : e - s] = v_cont[s:e]
    block_table = torch.arange(nb, dtype=torch.int32, device=DEV).view(1, nb)
    return k_cache, v_cache, block_table


def ref_attn(q, k, v, scale, window, causal_qpos=None):
    """q:[Hq,D] (single decode query), k/v:[S,Hkv,D]. Returns [Hq,D] fp32.
    Decode query sits at position S-1. window<0 = full."""
    Hq, D = q.shape
    S, Hkv, _ = k.shape
    qpk = Hq // Hkv
    out = torch.zeros((Hq, D), dtype=torch.float32, device=DEV)
    qpos = S - 1
    for h in range(Hq):
        kh = h // qpk
        scores = (q[h].float() @ k[:, kh].float().T) * scale  # [S]
        mask = torch.ones(S, dtype=torch.bool, device=DEV)
        if window >= 0:
            mask &= torch.arange(S, device=DEV) >= (qpos - window + 1)
        scores = scores.masked_fill(~mask, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        out[h] = p @ v[:, kh].float()
    return out


def test_decode(D, S, window, block_size=16, Hq=8, Hkv=2):
    scale = D ** -0.5
    q = torch.randn(1, Hq, D, dtype=torch.float16, device=DEV)
    k = torch.randn(S, Hkv, D, dtype=torch.float16, device=DEV)
    v = torch.randn(S, Hkv, D, dtype=torch.float16, device=DEV)
    k_cache, v_cache, block_table = build_paged(k, v, block_size)
    seq_lens = torch.tensor([S], dtype=torch.int32, device=DEV)
    out = flash_attn_decode_paged(
        q, k_cache, v_cache, block_table, seq_lens,
        softmax_scale=scale, kv_cache_dtype="auto", window=window,
    )
    ref = ref_attn(q[0], k, v, scale, window)
    got = out[0].float()
    err = (got - ref).abs().max().item()
    print(f"  decode D={D} S={S} win={window:>5}: max_abs_err={err:.5f}  {'OK' if err < 2e-2 else 'FAIL'}")
    return err < 2e-2


def test_prefill(D, S, window, block_size=16, Hq=8, Hkv=2):
    """Prefill: M=S queries, causal + window. Compare last-row + a mid row."""
    scale = D ** -0.5
    q = torch.randn(1, S, Hq, D, dtype=torch.float16, device=DEV)  # [B,M,H,D]
    k = torch.randn(S, Hkv, D, dtype=torch.float16, device=DEV)
    v = torch.randn(S, Hkv, D, dtype=torch.float16, device=DEV)
    k_cache, v_cache, block_table = build_paged(k, v, block_size)
    seq_lens = torch.tensor([S], dtype=torch.int32, device=DEV)
    out = flash_attn_prefill_paged(
        q, k_cache, v_cache, block_table, seq_lens,
        softmax_scale=scale, kv_cache_dtype="auto", causal=True, window=window,
    )  # [B,M,H,D]
    qpk = Hq // Hkv
    worst = 0.0
    for qi in (S - 1, S // 2, min(window + 3, S - 1)):
        for h in range(Hq):
            kh = h // qpk
            scores = (q[0, qi, h].float() @ k[:, kh].float().T) * scale
            idx = torch.arange(S, device=DEV)
            mask = idx <= qi
            if window >= 0:
                mask &= idx >= (qi - window + 1)
            scores = scores.masked_fill(~mask, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            ref = p @ v[:, kh].float()
            err = (out[0, qi, h].float() - ref).abs().max().item()
            worst = max(worst, err)
    print(f"  prefill D={D} S={S} win={window:>5}: max_abs_err={worst:.5f}  {'OK' if worst < 3e-2 else 'FAIL'}")
    return worst < 3e-2


def test_dense(D, S, window, Hq=8, Hkv=2):
    """Dense (non-paged) flash_attn_func, causal + window. q/k/v: [B,M,H,D]."""
    scale = D ** -0.5
    q = torch.randn(1, S, Hq, D, dtype=torch.float16, device=DEV)
    k = torch.randn(1, S, Hkv, D, dtype=torch.float16, device=DEV)
    v = torch.randn(1, S, Hkv, D, dtype=torch.float16, device=DEV)
    ws = (-1, -1) if window < 0 else (window - 1, 0)
    out = flash_attn_func(q, k, v, causal=True, softmax_scale=scale, window_size=ws)  # [B,M,H,D]
    qpk = Hq // Hkv
    worst = 0.0
    for qi in (S - 1, S // 2, min(window + 3, S - 1) if window > 0 else S // 3):
        for h in range(Hq):
            kh = h // qpk
            scores = (q[0, qi, h].float() @ k[0, :, kh].float().T) * scale
            idx = torch.arange(S, device=DEV)
            mask = idx <= qi
            if window >= 0:
                mask &= idx >= (qi - window + 1)
            scores = scores.masked_fill(~mask, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            ref = p @ v[0, :, kh].float()
            err = (out[0, qi, h].float() - ref).abs().max().item()
            worst = max(worst, err)
    print(f"  dense   D={D} S={S} win={window:>5}: max_abs_err={worst:.5f}  {'OK' if worst < 3e-2 else 'FAIL'}")
    return worst < 3e-2


if __name__ == "__main__":
    ok = True
    print("== full attention (window=-1) regression ==")
    for D in (128, 256):
        ok &= test_decode(D, 100, -1)
        ok &= test_prefill(D, 100, -1)
        ok &= test_dense(D, 100, -1)
    print("== sliding window ==")
    for D in (128, 256):
        for S, W in [(100, 32), (100, 64), (50, 64), (200, 48), (33, 32)]:
            ok &= test_decode(D, S, W)
            ok &= test_prefill(D, S, W)
            ok &= test_dense(D, S, W)
    print("== D=512 decode (global layers: full + windowed) ==")
    for S, W in [(100, -1), (8192, -1), (100, 32), (200, 48), (8192, 1024)]:
        ok &= test_decode(512, S, W)
    print("ALL PASS" if ok else "SOME FAILED")
