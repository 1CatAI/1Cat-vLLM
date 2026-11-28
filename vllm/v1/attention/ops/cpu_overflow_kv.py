# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-first 2-tier KV block pool with automatic emergency pinned-RAM overflow."""
from __future__ import annotations

import torch

_OVERFLOW_STATE: dict[str, object] = {
    "n_gpu": 0,
    "n_cpu": 0,
    "step_has_overflow": False,
}


def compute_cpu_overflow_split(
    num_blocks: int, total_bytes: int, overflow_bytes: int
) -> tuple[int, int]:
    """Split num_blocks into (n_gpu, n_cpu) deterministically."""
    if overflow_bytes <= 0 or num_blocks <= 4 or total_bytes <= 0:
        return num_blocks, 0
    bytes_per_block = total_bytes // num_blocks
    if bytes_per_block <= 0:
        return num_blocks, 0
    n_cpu = min(overflow_bytes // bytes_per_block, max(0, num_blocks - 2))
    return num_blocks - n_cpu, n_cpu


def _gather_chunk_kv(
    gpu_kv: torch.Tensor,
    cpu_kv: torch.Tensor,
    n_gpu: int,
    b_ids_gpu: torch.Tensor,
    b_ids_cpu_list: list[int],
    dtype_view: torch.dtype | None,
    out_dtype: torch.dtype,
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if all(b < n_gpu for b in b_ids_cpu_list):
        sub = gpu_kv[b_ids_gpu]
    elif all(b >= n_gpu for b in b_ids_cpu_list):
        c_idx = torch.tensor([b - n_gpu for b in b_ids_cpu_list], dtype=torch.long)
        sub = cpu_kv[c_idx].to(dev, non_blocking=True)
    else:
        sub = torch.empty(
            (len(b_ids_cpu_list), *gpu_kv.shape[1:]), dtype=gpu_kv.dtype, device=dev
        )
        g_pos = [i for i, b in enumerate(b_ids_cpu_list) if b < n_gpu]
        c_pos = [i for i, b in enumerate(b_ids_cpu_list) if b >= n_gpu]
        if g_pos:
            gp = torch.tensor(g_pos, dtype=torch.long, device=dev)
            gb = torch.tensor(
                [b_ids_cpu_list[i] for i in g_pos], dtype=torch.long, device=dev
            )
            sub[gp] = gpu_kv[gb]
        if c_pos:
            cp = torch.tensor(c_pos, dtype=torch.long, device=dev)
            cb = torch.tensor(
                [b_ids_cpu_list[i] - n_gpu for i in c_pos], dtype=torch.long
            )
            sub[cp] = cpu_kv[cb].to(dev, non_blocking=True)
    k, v = sub[:, 0], sub[:, 1]
    if dtype_view is not None and k.dtype != dtype_view:
        k = k.view(dtype_view)
        v = v.view(dtype_view)
    return k.to(out_dtype), v.to(out_dtype)


def overflow_attention_req(
    q: torch.Tensor,
    gpu_kv: torch.Tensor,
    cpu_kv: torch.Tensor,
    n_gpu: int,
    out: torch.Tensor,
    seq_len: int,
    scale: float,
    window: int,
    block_table_1d: torch.Tensor,
    softcap: float = 0.0,
    kb: int = 1024,
    dtype_view: torch.dtype | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> None:
    """Chunked online-softmax attention reading from GPU VRAM + pinned CPU overflow."""
    M, Hq, D = q.shape
    BS, G = gpu_kv.shape[2], gpu_kv.shape[3]
    n = Hq // G
    dev = q.device
    comp_dtype = (
        torch.float16
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        else q.dtype
    )
    Q = q.to(comp_dtype).view(M, G, n, D).permute(1, 2, 0, 3).reshape(G, n * M, D)
    q0 = seq_len - M
    lo = max(0, q0 - window + 1) if window > 0 else 0
    lo -= lo % BS
    kb = max(BS, kb - (kb % BS))
    m = torch.full((G, n * M), float("-inf"), device=dev, dtype=torch.float32)
    l = torch.zeros((G, n * M), device=dev, dtype=torch.float32)
    acc = torch.zeros((G, n * M, D), device=dev, dtype=torch.float32)
    qpos = q0 + torch.arange(M, device=dev)
    ks_t = k_descale.float().reshape(-1, 1, 1) if k_descale is not None else None
    vs_t = v_descale.float().reshape(-1, 1, 1) if v_descale is not None else None

    n_active_blks = (seq_len + BS - 1) // BS
    bt_cpu = block_table_1d[:n_active_blks].cpu().tolist()

    for s in range(lo, seq_len, kb):
        e = min(seq_len, s + kb)
        b0, b1 = s // BS, (e + BS - 1) // BS
        K_blk, V_blk = _gather_chunk_kv(
            gpu_kv,
            cpu_kv,
            n_gpu,
            block_table_1d[b0:b1].long(),
            bt_cpu[b0:b1],
            dtype_view,
            comp_dtype,
            dev,
        )
        K = K_blk.reshape(-1, G, D)[: e - s].transpose(0, 1)
        V = V_blk.reshape(-1, G, D)[: e - s].transpose(0, 1)
        S = torch.bmm(Q, K.transpose(1, 2)).float().mul_(scale)
        if ks_t is not None:
            S.mul_(ks_t)
        if softcap and softcap > 0:
            S = torch.tanh_(S.div_(softcap)).mul_(softcap)
        if e > q0 + 1 or (window > 0 and s < q0 + M - window):
            kpos = torch.arange(s, e, device=dev)
            bad = kpos[None, :] > qpos[:, None]
            if window > 0:
                bad |= (qpos[:, None] - kpos[None, :]) >= window
            S.view(G, n, M, e - s).masked_fill_(bad, float("-inf"))
        mn = torch.maximum(m, S.amax(2))
        a = torch.exp(m - mn).nan_to_num_(nan=0.0)
        S.sub_(mn[:, :, None]).exp_().nan_to_num_(nan=0.0)
        l.mul_(a).add_(S.sum(2))
        acc.mul_(a[:, :, None]).add_(torch.bmm(S.to(comp_dtype), V))
        m = mn
    acc.div_(l[:, :, None])
    if vs_t is not None:
        acc.mul_(vs_t)
    out.copy_(acc.view(G, n, M, D).permute(2, 0, 1, 3).reshape(M, Hq, D).to(out.dtype))
