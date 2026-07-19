# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError("SM70 FlashQLA backend requires CUDA.")

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5")
    src = Path(__file__).with_name("csrc") / "gdn_forward.cu"
    _EXT = load(
        name="flash_qla_sm70_gdn_strided",
        sources=[str(src)],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("FLASH_QLA_SM70_VERBOSE_BUILD", "0"))),
    )
    return _EXT


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> None:
    tensors = [q, k, v, g, beta]
    if initial_state is not None:
        tensors.append(initial_state)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("SM70 GDN tensors must be CUDA tensors.")
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("SM70 GDN tensors must be on the same CUDA device.")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("SM70 GDN tensors must be contiguous.")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("SM70 GDN backend supports fp16, bf16, and fp32 tensors.")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have the same dtype.")
    if g.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("g must be fp16, bf16, or fp32.")
    if beta.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("beta must be fp16, bf16, or fp32.")
    if initial_state is not None and initial_state.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("initial_state must be fp16, bf16, or fp32.")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, T, H, D].")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g and beta must have shape [B, T, Hv].")
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape.")

    batch, tokens, q_heads, k_dim = q.shape
    _, _, v_heads, v_dim = v.shape
    if v.shape[0] != batch or v.shape[1] != tokens:
        raise ValueError("v must have shape [B, T, Hv, V] matching q/k.")
    if g.shape != beta.shape or g.shape != v.shape[:3]:
        raise ValueError("g and beta must have shape [B, T, Hv].")
    if v_heads % q_heads != 0:
        raise ValueError("Hv must be divisible by Hq.")
    if k_dim != 128 or v_dim != 128:
        raise ValueError("SM70 FlashQLA backend currently supports K=V=128.")
    if initial_state is not None and initial_state.shape != (
        batch,
        v_heads,
        k_dim,
        v_dim,
    ):
        raise ValueError("initial_state must have shape [B, Hv, K, V].")


def _check_vlk_varlen_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor | None = None,
    validate_cu_seqlens: bool = True,
) -> None:
    tensors = [q, k, v, g, beta, cu_seqlens]
    if initial_state is not None:
        tensors.append(initial_state)
    if output is not None:
        tensors.append(output)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("SM70 GDN tensors must be CUDA tensors.")
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("SM70 GDN tensors must be on the same CUDA device.")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("SM70 GDN tensors must be contiguous.")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("SM70 GDN backend supports fp16, bf16, and fp32 tensors.")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have the same dtype.")
    if g.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("g must be fp16, bf16, or fp32.")
    if beta.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("beta must be fp16, bf16, or fp32.")
    if initial_state is not None and initial_state.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("initial_state must be fp16, bf16, or fp32.")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be int32.")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [1, T, H, D].")
    if q.shape[0] != 1:
        raise ValueError("SM70 varlen GDN expects flattened q/k/v with batch=1.")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g and beta must have shape [1, T, Hv].")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [N + 1].")
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape.")

    _, tokens, q_heads, k_dim = q.shape
    _, _, v_heads, v_dim = v.shape
    num_sequences = cu_seqlens.numel() - 1
    if v.shape[0] != 1 or v.shape[1] != tokens:
        raise ValueError("v must have shape [1, T, Hv, V] matching q/k.")
    if g.shape != beta.shape or g.shape != v.shape[:3]:
        raise ValueError("g and beta must have shape [1, T, Hv].")
    if v_heads % q_heads != 0:
        raise ValueError("Hv must be divisible by Hq.")
    if k_dim != 128 or v_dim != 128:
        raise ValueError("SM70 FlashQLA backend currently supports K=V=128.")
    if initial_state is not None and initial_state.shape != (
        num_sequences,
        v_heads,
        v_dim,
        k_dim,
    ):
        raise ValueError("initial_state must have shape [N, Hv, V, K].")
    if output is not None:
        if output.dtype != v.dtype:
            raise ValueError("output must match v dtype.")
        if output.shape != (1, tokens, v_heads, v_dim):
            raise ValueError("output must have shape [1, T, Hv, V].")
    if validate_cu_seqlens:
        cu_cpu = cu_seqlens.detach().cpu()
        if int(cu_cpu[0]) != 0:
            raise ValueError("cu_seqlens must start at 0.")
        if int(cu_cpu[-1]) != tokens:
            raise ValueError("cu_seqlens must end at the flattened token count.")
        if not bool((cu_cpu[1:] >= cu_cpu[:-1]).all()):
            raise ValueError("cu_seqlens must be non-decreasing.")


def chunk_gated_delta_rule_fwd_sm70(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    gate_is_exp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the experimental SM70/SM75 forward GDN backend.

    This keeps the public FlashQLA tensor contract:
    q/k: [B, T, Hq, K], v/o: [B, T, Hv, V], state: [B, Hv, K, V].
    """

    _check_inputs(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    ext = _load_ext()
    output, final_state = ext.gdn_forward(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        float(scale),
        output_final_state,
        gate_is_exp,
    )
    if not output_final_state:
        final_state = None
    return output, final_state


def chunk_gated_delta_rule_fwd_sm70_vlk_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    validate_cu_seqlens: bool = True,
    gate_is_exp: bool = False,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run SM70/SM75 forward with vLLM-only state layout [N, Hv, V, K].

    This is not a public FlashQLA varlen drop-in: K and V are both 128 for
    Qwen GDN, so the vLLM layout cannot be shape-distinguished from the public
    [N, Hv, K, V] contract.
    """

    _check_vlk_varlen_inputs(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        output,
        validate_cu_seqlens=validate_cu_seqlens,
    )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    ext = _load_ext()
    output, final_state = ext.gdn_forward_vlk_varlen(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        float(scale),
        output_final_state,
        validate_cu_seqlens,
        gate_is_exp,
        output,
    )
    if not output_final_state:
        final_state = None
    return output, final_state


def gdn_decode_mixed_qkv_global_state_sm70(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Run fused SM70 mixed-QKV decode against vLLM global state slots."""

    tensors = [mixed_qkv, a, b, A_log, dt_bias, state, state_indices, output]
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("SM70 GDN decode tensors must be CUDA tensors.")
    if any(tensor.device != mixed_qkv.device for tensor in tensors):
        raise ValueError("SM70 GDN decode tensors must be on the same CUDA device.")
    contiguous_tensors = {
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state_indices": state_indices,
        "output": output,
    }
    non_contiguous = [
        name
        for name, tensor in contiguous_tensors.items()
        if not tensor.is_contiguous()
    ]
    if non_contiguous:
        raise ValueError(
            "SM70 GDN decode tensors must be contiguous except mixed_qkv/state; "
            f"non-contiguous={non_contiguous}"
        )
    if mixed_qkv.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("mixed_qkv must be fp16, bf16, or fp32.")
    if a.dtype != mixed_qkv.dtype or b.dtype != mixed_qkv.dtype:
        raise ValueError("a and b must match mixed_qkv dtype.")
    if output.dtype != mixed_qkv.dtype:
        raise ValueError("output must match mixed_qkv dtype.")
    if A_log.dtype != torch.float32:
        raise ValueError("A_log must be float32.")
    if dt_bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("dt_bias must be fp16, bf16, or fp32.")
    if state_indices.dtype != torch.int32:
        raise ValueError("state_indices must be int32.")
    if mixed_qkv.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("mixed_qkv, a, and b must be rank-2 tensors.")
    if mixed_qkv.stride(1) != 1 or mixed_qkv.stride(0) < mixed_qkv.shape[1]:
        raise ValueError(
            "mixed_qkv must have dense columns and row stride >= logical width; "
            f"shape={tuple(mixed_qkv.shape)} stride={tuple(mixed_qkv.stride())}"
        )
    if state.ndim != 4 or output.ndim != 3:
        raise ValueError("state must be [slots,Hv,V,K], output [T,Hv,V].")
    tokens = mixed_qkv.shape[0]
    _, v_heads, v_dim, k_dim = state.shape
    if k_dim != 128 or v_dim != 128:
        raise ValueError("SM70 FlashQLA decode currently supports K=V=128.")
    if state.stride()[1:] != (v_dim * k_dim, k_dim, 1):
        raise ValueError(
            "state inner layout must be [slots,Hv,V,K] with contiguous [Hv,V,K] "
            f"pages; got stride={tuple(state.stride())}"
        )
    if a.shape != (tokens, v_heads) or b.shape != (tokens, v_heads):
        raise ValueError("a/b must have shape [T,Hv].")
    if A_log.shape != (v_heads,) or dt_bias.shape != (v_heads,):
        raise ValueError("A_log/dt_bias must have shape [Hv].")
    if state_indices.shape != (tokens,):
        raise ValueError("state_indices must have shape [T].")
    if output.shape != (tokens, v_heads, v_dim):
        raise ValueError("output must have shape [T,Hv,V].")
    if scale is None:
        scale = k_dim**-0.5
    ext = _load_ext()
    ext.gdn_decode_mixed_qkv_global_state(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        state_indices,
        output,
        float(scale),
        bool(use_qk_l2norm_in_kernel),
    )
    return output


def gdn_decode_mixed_qkv_ddtree_state_sm70(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    parent_ids: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Run parent-aware DDTree mixed-QKV decode against vLLM global state."""

    tensors = [
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        state_indices,
        parent_ids,
        num_accepted_tokens,
        cu_seqlens,
        output,
    ]
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("SM70 DDTree GDN tensors must be CUDA tensors.")
    if any(tensor.device != mixed_qkv.device for tensor in tensors):
        raise ValueError("SM70 DDTree GDN tensors must be on the same CUDA device.")
    contiguous_tensors = {
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state_indices": state_indices,
        "parent_ids": parent_ids,
        "num_accepted_tokens": num_accepted_tokens,
        "cu_seqlens": cu_seqlens,
        "output": output,
    }
    non_contiguous = [
        name
        for name, tensor in contiguous_tensors.items()
        if not tensor.is_contiguous()
    ]
    if non_contiguous:
        raise ValueError(
            "SM70 DDTree GDN tensors must be contiguous except mixed_qkv/state; "
            f"non-contiguous={non_contiguous}"
        )
    if mixed_qkv.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("mixed_qkv must be fp16, bf16, or fp32.")
    if a.dtype != mixed_qkv.dtype or b.dtype != mixed_qkv.dtype:
        raise ValueError("a and b must match mixed_qkv dtype.")
    if output.dtype != mixed_qkv.dtype:
        raise ValueError("output must match mixed_qkv dtype.")
    if A_log.dtype != torch.float32:
        raise ValueError("A_log must be float32.")
    if dt_bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("dt_bias must be fp16, bf16, or fp32.")
    if state_indices.dtype != torch.int32:
        raise ValueError("state_indices must be int32.")
    if parent_ids.dtype != torch.int32:
        raise ValueError("parent_ids must be int32.")
    if num_accepted_tokens.dtype != torch.int32:
        raise ValueError("num_accepted_tokens must be int32.")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be int32.")
    if mixed_qkv.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("mixed_qkv, a, and b must be rank-2 tensors.")
    if mixed_qkv.stride(1) != 1 or mixed_qkv.stride(0) < mixed_qkv.shape[1]:
        raise ValueError(
            "mixed_qkv must have dense columns and row stride >= logical width; "
            f"shape={tuple(mixed_qkv.shape)} stride={tuple(mixed_qkv.stride())}"
        )
    if state.ndim != 4 or output.ndim != 3:
        raise ValueError("state must be [slots,Hv,V,K], output [T,Hv,V].")
    if state_indices.ndim != 2 or parent_ids.ndim != 2:
        raise ValueError("state_indices and parent_ids must be rank-2 tensors.")
    if parent_ids.shape != state_indices.shape:
        raise ValueError("parent_ids must match state_indices shape.")
    tokens = mixed_qkv.shape[0]
    num_sequences = state_indices.shape[0]
    _, v_heads, v_dim, k_dim = state.shape
    if k_dim != 128 or v_dim != 128:
        raise ValueError("SM70 FlashQLA DDTree decode currently supports K=V=128.")
    if state.stride()[1:] != (v_dim * k_dim, k_dim, 1):
        raise ValueError(
            "state inner layout must be [slots,Hv,V,K] with contiguous [Hv,V,K] "
            f"pages; got stride={tuple(state.stride())}"
        )
    if a.shape != (tokens, v_heads) or b.shape != (tokens, v_heads):
        raise ValueError("a/b must have shape [T,Hv].")
    if A_log.shape != (v_heads,) or dt_bias.shape != (v_heads,):
        raise ValueError("A_log/dt_bias must have shape [Hv].")
    if num_accepted_tokens.shape != (num_sequences,):
        raise ValueError("num_accepted_tokens must have shape [N].")
    if cu_seqlens.shape != (num_sequences + 1,):
        raise ValueError("cu_seqlens must have shape [N + 1].")
    if output.shape != (tokens, v_heads, v_dim):
        raise ValueError("output must have shape [T,Hv,V].")
    if scale is None:
        scale = k_dim**-0.5
    ext = _load_ext()
    ext.gdn_decode_mixed_qkv_ddtree_state(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        state_indices,
        parent_ids,
        num_accepted_tokens,
        cu_seqlens,
        output,
        float(scale),
        bool(use_qk_l2norm_in_kernel),
    )
    return output


def resolve_column_groups_per_block_sm70(
    tokens: int,
    q_heads: int,
    v_heads: int,
) -> int:
    ext = _load_ext()
    return int(ext.resolve_column_groups_per_block(tokens, q_heads, v_heads))
