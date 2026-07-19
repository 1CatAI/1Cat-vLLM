# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import tilelang

from flash_qla.utils import tensor_cache

from .hopper import get_warmup_chunks, fused_gdr_h, correct_initial_states


MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count


@tensor_cache
def _create_cu_seqlens(
    batch_size: int,
    num_tokens: int,
    device_idx: int,
):
    return (
        torch.arange((batch_size + 1), dtype=torch.int32, device=f"cuda:{device_idx}")
        * num_tokens
    )


@tensor_cache
def _calc_cp_seqs(
    raw_cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_v_heads: int,
):
    # TODO: tilelang kernel
    device = raw_cu_seqlens.device
    seqlen_dtype = raw_cu_seqlens.dtype
    raw_cu_seqlens = raw_cu_seqlens.tolist()
    raw_batch_size = len(raw_cu_seqlens) - 1
    seqlens = [raw_cu_seqlens[i + 1] - raw_cu_seqlens[i] for i in range(raw_batch_size)]
    num_chunks = [tilelang.cdiv(x, chunk_size) for x in seqlens]

    # autocp
    H = num_v_heads
    # Latency model: T = a·L_cp + b·(B·H·Lc/P) / L_cp + c
    # Minimizing T yields the theoretical optimum: L_cp* ∝ √(B·H·Lc / P), where P = MULTI_PROCESSOR_COUNT, L_cp = max_local_chunks
    # Scaled by empirical factor (3) and aligned to the nearest power of 2 for optimal SM scheduling & memory alignment.

    max_local_chunks = 2 ** round(
        math.log2(math.sqrt(H * sum(num_chunks) / MULTI_PROCESSOR_COUNT) * 3)
    )

    # Set min to 4 to ensure multi-stage pipelining in fused_gdr;
    max_local_chunks = max(max_local_chunks, 4)

    use_cp = False
    cp_cu_seqlens = []
    ht_mask = []
    seq_map_c2r = []
    seq_map_r2c = [0]
    max_local_tokens = max_local_chunks * chunk_size
    for i, c in enumerate(num_chunks):
        s = raw_cu_seqlens[i]
        e = raw_cu_seqlens[i + 1]
        if c > max_local_chunks:
            while s < e:
                cp_cu_seqlens.append(s)
                ht_mask.append(False)
                seq_map_c2r.append(i)
                s += max_local_tokens
            ht_mask[-1] = True
        else:
            cp_cu_seqlens.append(s)
            ht_mask.append(True)
            seq_map_c2r.append(i)
        seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(raw_cu_seqlens[-1])

    # Disable CP when B * H naturally saturates SM occupancy.
    # For varlen inputs, use `total_chunks / max_seq_chunks` as effective B,
    # since CP helps accelerate highly uneven sequence lengths.

    Be = sum(num_chunks) / max(num_chunks)
    use_cp = Be * H <= 40 or (Be * H <= 56 and max(num_chunks) >= 128)

    if use_cp:
        cp_cu_seqlens = torch.tensor(
            cp_cu_seqlens, dtype=seqlen_dtype, device=device, requires_grad=False
        )
        seq_map_c2r = torch.tensor(seq_map_c2r, dtype=seqlen_dtype, device=device)
        seq_map_r2c = torch.tensor(
            seq_map_r2c, dtype=seqlen_dtype, device=device, requires_grad=False
        )
        ht_mask = torch.tensor(
            ht_mask, dtype=torch.bool, device=device, requires_grad=False
        )
    else:
        cp_cu_seqlens, seq_map_r2c, ht_mask = None, None, None

    return use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask


def intra_card_cp_preprocess(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    raw_h0: torch.Tensor,
    raw_cu_seqlens: torch.Tensor,
    warmup_threshold: float = -10.0,
):
    batch_size, num_tokens, num_k_heads, k_head_dim = k.shape
    _, _, num_v_heads, v_head_dim = v.shape
    chunk_size = a.shape[-1]
    device = k.device

    if batch_size > 1:
        return raw_h0, raw_cu_seqlens, None, None

    if raw_cu_seqlens is None:
        raw_cu_seqlens = _create_cu_seqlens(batch_size, num_tokens, device.index)

    use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask = _calc_cp_seqs(
        raw_cu_seqlens,
        chunk_size,
        num_v_heads,
    )

    if not use_cp:
        return raw_h0, raw_cu_seqlens, None, None

    num_warmup_chunks, fallback_mask = get_warmup_chunks(
        g=g,
        cu_seqlens=cp_cu_seqlens,
        ht_mask=ht_mask,
        chunk_size=chunk_size,
        threshold=warmup_threshold,
    )  # [cp_batch_size, num_v_heads]
    _, ht, mt = fused_gdr_h(
        k=k,
        v=v,
        a=a,
        g=g,
        b=b,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        cu_seqlens=cp_cu_seqlens,
        num_warmup_chunks=num_warmup_chunks,
    )  # [cp_batch_size, num_v_heads, k_head_dim, v_head_dim]
    cp_h0 = correct_initial_states(
        raw_h0=raw_h0,
        ht_buffer=ht,
        mt_buffer=mt,
        fallback_mask=fallback_mask,
        seq_map_r2c=seq_map_r2c,
    )

    return cp_h0, cp_cu_seqlens, seq_map_c2r, raw_cu_seqlens


def intra_card_cp_preprocess_sm70(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    raw_h0: torch.Tensor | None,
    raw_cu_seqlens: torch.Tensor,
    warmup_threshold: float = -10.0,
    state_layout_vlk: bool = False,
):
    batch_size, num_tokens, num_k_heads, _ = k.shape
    _, _, num_v_heads, _ = v.shape
    chunk_size = a.shape[-1]
    device = k.device

    if batch_size > 1:
        return raw_h0, raw_cu_seqlens, None, None

    if raw_cu_seqlens is None:
        raw_cu_seqlens = _create_cu_seqlens(batch_size, num_tokens, device.index)

    if raw_cu_seqlens.shape[0] != 2:
        return raw_h0, raw_cu_seqlens, None, None

    use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask = _calc_cp_seqs(
        raw_cu_seqlens,
        chunk_size,
        num_v_heads,
    )

    if not use_cp:
        return raw_h0, raw_cu_seqlens, None, None

    assert cp_cu_seqlens is not None
    assert seq_map_c2r is not None
    assert ht_mask is not None

    _num_warmup_chunks, fallback_mask = get_warmup_chunks(
        g=g,
        cu_seqlens=cp_cu_seqlens,
        ht_mask=ht_mask,
        chunk_size=chunk_size,
        threshold=warmup_threshold,
    )
    if bool(fallback_mask.any().item()):
        return raw_h0, raw_cu_seqlens, None, None

    from .hopper.fused_fwd import fused_gdr_fwd

    _, _, ht = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=a,
        g=g,
        b=b,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        output_o=False,
        cu_seqlens=cp_cu_seqlens,
        state_layout_vlk=state_layout_vlk,
        sm70_original=True,
    )
    assert ht is not None

    cp_batch_size = cp_cu_seqlens.shape[0] - 1
    cp_h0_dtype = raw_h0.dtype if raw_h0 is not None else torch.float32
    cp_h0 = torch.empty(
        (cp_batch_size, num_v_heads, k.shape[-1], v.shape[-1]),
        dtype=cp_h0_dtype,
        device=device,
    )
    if raw_h0 is None:
        cp_h0[0].zero_()
    else:
        cp_h0[0].copy_(raw_h0[0])
    if cp_batch_size > 1:
        cp_h0[1:].copy_(ht[:-1].to(cp_h0.dtype))

    return cp_h0, cp_cu_seqlens, seq_map_c2r, raw_cu_seqlens
