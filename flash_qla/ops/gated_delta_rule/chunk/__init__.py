# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch

from flash_qla.utils import l2norm

from .sm70 import chunk_gated_delta_rule_fwd_sm70

_HOPPER_BACKEND = None


def _get_tensor_compute_version(tensor: torch.Tensor) -> str:
    if not tensor.is_cuda:
        raise ValueError("FlashQLA tensors must be CUDA tensors.")
    major, minor = torch.cuda.get_device_capability(tensor.device)
    return f"{major}.{minor}"


def _load_hopper_backend():
    global _HOPPER_BACKEND
    if _HOPPER_BACKEND is None:
        from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector

        from .hopper import fused_gdr_bwd, fused_gdr_fwd, fused_gdr_h, kkt_solve

        try:
            from .cp_context import intra_card_cp_preprocess
        except ValueError:
            intra_card_cp_preprocess = None

        _HOPPER_BACKEND = (
            chunk_local_cumsum,
            group_reduce_vector,
            fused_gdr_fwd,
            fused_gdr_bwd,
            fused_gdr_h,
            kkt_solve,
            intra_card_cp_preprocess,
        )
    return _HOPPER_BACKEND


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    auto_cp: bool = True,
):
    target_compute_version = _get_tensor_compute_version(q)
    if target_compute_version in ("7.0", "7.5"):
        raise ValueError(
            "SM70 FlashQLA backend currently supports only the public "
            "chunk_gated_delta_rule forward API, not low-level A/h outputs."
        )
    if target_compute_version != "9.0":
        raise ValueError("FlashQLA supports SM90 and experimental SM70/SM75 only.")

    (
        chunk_local_cumsum,
        _,
        fused_gdr_fwd,
        _,
        _,
        kkt_solve,
        intra_card_cp_preprocess,
    ) = _load_hopper_backend()

    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    A = kkt_solve(
        k=k,
        b=beta,
        cu_seqlens=cu_seqlens,
    )
    if auto_cp:
        if intra_card_cp_preprocess is None:
            raise ValueError("FlashQLA auto_cp is only available on SM90.")
        initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
            intra_card_cp_preprocess(
                k=k,
                v=v,
                a=A,
                g=g,
                b=beta,
                raw_h0=initial_state,
                raw_cu_seqlens=cu_seqlens,
            )
        )
    else:
        cp_seq_map = None
        raw_cu_seqlens = None
    o, h, final_state = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_h=output_h,
        output_o=True,
        cu_seqlens=cu_seqlens,
        cp_seq_map=cp_seq_map,
        raw_cu_seqlens=raw_cu_seqlens,
    )
    return g, A, o, h, final_state


def chunk_gated_delta_rule_fwd_sm70_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    auto_cp: bool = False,
    state_layout_vlk: bool = False,
    output: torch.Tensor | None = None,
    inplace_final_state: bool = False,
):
    target_compute_version = _get_tensor_compute_version(q)
    if target_compute_version not in ("7.0", "7.5"):
        raise ValueError("SM70 TileLang FlashQLA path is only for SM70/SM75.")
    if auto_cp and state_indices is not None:
        raise ValueError(
            "SM70 TileLang FlashQLA auto_cp does not support indexed state yet."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    (
        chunk_local_cumsum,
        _,
        fused_gdr_fwd,
        _,
        _,
        kkt_solve,
        _,
    ) = _load_hopper_backend()

    g = chunk_local_cumsum(
        g,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    A = kkt_solve(
        k=k,
        b=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if auto_cp:
        from .cp_context import intra_card_cp_preprocess_sm70

        initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
            intra_card_cp_preprocess_sm70(
                q=q,
                k=k,
                v=v,
                a=A,
                g=g,
                b=beta,
                raw_h0=initial_state,
                raw_cu_seqlens=cu_seqlens,
                state_layout_vlk=state_layout_vlk,
            )
        )
        chunk_offsets = None
    else:
        cp_seq_map = None
        raw_cu_seqlens = None
    o, h, final_state = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_h=output_h,
        output_o=True,
        cu_seqlens=cu_seqlens,
        cp_seq_map=cp_seq_map,
        raw_cu_seqlens=raw_cu_seqlens,
        chunk_offsets=chunk_offsets,
        state_layout_vlk=state_layout_vlk,
        sm70_original=True,
        output=output,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        inplace_final_state=inplace_final_state,
    )
    return g, A, o, h, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    target_compute_version = _get_tensor_compute_version(q)
    if target_compute_version in ("7.0", "7.5"):
        raise ValueError("SM70 FlashQLA backend is forward-only.")
    if target_compute_version != "9.0":
        raise ValueError("FlashQLA backward supports SM90 only.")

    (
        chunk_local_cumsum,
        group_reduce_vector,
        _,
        fused_gdr_bwd,
        fused_gdr_h,
        _,
        _,
    ) = _load_hopper_backend()

    h, _, _ = fused_gdr_h(
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        initial_state=initial_state,
        output_final_state=False,
        output_h=True,
        cu_seqlens=cu_seqlens,
    )
    dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        do=do,
        dht=dht,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H:
        dq = group_reduce_vector(dq, Hg)
        dk = group_reduce_vector(dk, Hg)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        q_orig = q
        k_orig = k

        g, A, o, _, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_h=False,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        return o.to(q.dtype), final_state

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors

        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q_orig,
            k=k_orig,
            v=v,
            g=g,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        return (
            dq.to(q_orig),
            dk.to(k_orig),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
        )


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
):
    assert q.dtype == k.dtype == v.dtype
    target_compute_version = _get_tensor_compute_version(q)
    if target_compute_version not in ("7.0", "7.5"):
        assert q.dtype != torch.float32, (
            "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16 or float16."
        )
    assert not head_first, "head_first=True is not supported."
    assert v.shape[2] % k.shape[2] == 0, (
        "num_qk_heads must be divisible to num_v_heads."
    )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)

    if target_compute_version in ("7.0", "7.5"):
        if cu_seqlens is not None:
            raise ValueError("SM70 FlashQLA backend does not support cu_seqlens yet.")
        return chunk_gated_delta_rule_fwd_sm70(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=scale,
            initial_state=initial_state.contiguous()
            if initial_state is not None
            else None,
            output_final_state=output_final_state,
        )
    if target_compute_version != "9.0":
        raise ValueError("FlashQLA supports SM90 and experimental SM70/SM75 only.")

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    )

    return o, final_state
