# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2023, Tri Dao.
# ruff: noqa: E501


import importlib.util
import os
import sys

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# isort: off
# FA2 ships one library per architecture: the CMake build for the target list
# and, on mixed Volta/Turing rigs, a separate Turing build next to it. Both
# export the same module and op namespace, so a process can hold only one of
# them, and the choice has to follow the GPU the ops run on. Importing here
# would decide before a worker has selected its device, so the library loads
# on first use instead (load_fa2_library).
_FA2_MODULE = f"{__package__}._vllm_fa2_C"
_FA2_TURING_CAPABILITY = (7, 5)
_FA2_TURING_PATH = os.path.join(os.path.dirname(__file__), "_vllm_fa2_C_sm75.abi3.so")
_FA2_DEFAULT_SPEC = importlib.util.find_spec(_FA2_MODULE)
_fa2_loaded_capability: tuple[int, int] | None = None

if _FA2_DEFAULT_SPEC is not None or os.path.exists(_FA2_TURING_PATH):
    FA2_UNAVAILABLE_REASON = None
    FA2_AVAILABLE = True
else:
    FA2_UNAVAILABLE_REASON = f"no {_FA2_MODULE} library is installed"
    FA2_AVAILABLE = False


def _fa2_library_path(capability: tuple[int, int]) -> str | None:
    """Return the FA2 library installed for ``capability``, or None."""
    if capability == _FA2_TURING_CAPABILITY:
        return _FA2_TURING_PATH if os.path.exists(_FA2_TURING_PATH) else None
    return _FA2_DEFAULT_SPEC.origin if _FA2_DEFAULT_SPEC is not None else None


def load_fa2_library(device: torch.device) -> None:
    """Load the FA2 library built for ``device``'s architecture.

    A process holds one FA2 library; the first call decides which.
    """
    global _fa2_loaded_capability
    if _fa2_loaded_capability is not None:
        return
    capability = torch.cuda.get_device_capability(device)
    path = _fa2_library_path(capability)
    if path is None:
        raise ImportError(
            f"No {_FA2_MODULE} library is installed for compute capability "
            f"{capability[0]}.{capability[1]}"
        )
    spec = importlib.util.spec_from_file_location(_FA2_MODULE, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_FA2_MODULE] = module
    spec.loader.exec_module(module)
    _fa2_loaded_capability = capability
    logger.info(
        "Loaded FA2 library %s for compute capability %d.%d.",
        os.path.basename(path),
        *capability,
    )


def ensure_fa2_library_loaded() -> None:
    """Load the FA2 library for this process's current device, once.

    For code that resolves operators from ``torch.ops._vllm_fa2_C`` before the
    first attention call goes through this module (the SM70 backend looks its
    prefill and tail operators up at initialisation). Importing this module no
    longer loads a library, so those lookups ask here first.
    """
    if _fa2_loaded_capability is None:
        load_fa2_library(torch.device("cuda", torch.accelerator.current_device_index()))


try:
    from . import _vllm_fa3_C  # type: ignore[attr-defined]  # noqa: F401

    FA3_UNAVAILABLE_REASON = None
    FA3_AVAILABLE = True
except ImportError as e:
    FA3_UNAVAILABLE_REASON = str(e)
    FA3_AVAILABLE = False


try:
    import os

    _cute_interface_path = os.path.join(
        os.path.dirname(__file__), "cute", "interface.py"
    )
    if not os.path.exists(_cute_interface_path):
        raise ImportError("vllm.vllm_flash_attn.cute.interface not found")

    FA4_UNAVAILABLE_REASON = None
    FA4_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    FA4_UNAVAILABLE_REASON = str(e)
    FA4_AVAILABLE = False

# isort: on

DEFAULT_FA_VERSION = 2


def _is_fa2_supported() -> tuple[bool, str | None]:
    if not FA2_AVAILABLE:
        return False, f"FA2 is unavailable due to: {FA2_UNAVAILABLE_REASON}"
    from vllm.platforms import current_platform

    # SM75 enablement: Turing runs the fp16-only FA2 build; bf16 inputs are
    # rejected by the C++ entry points.
    # Mixed rigs: ask this worker's own GPU, not device 0 -- otherwise the
    # weakest card in the grid decides for every stage.
    device = torch.accelerator.current_device_index()
    if not current_platform.has_device_capability(75, device):
        return False, "FA2 is only supported on devices with compute capability >= 7.5"
    capability = current_platform.get_device_capability(device)
    if (
        capability is None
        or _fa2_library_path((capability.major, capability.minor)) is None
    ):
        return False, "no FA2 library is installed for this GPU's architecture"
    return True, None


def _is_fa3_supported() -> tuple[bool, str | None]:
    if not FA3_AVAILABLE:
        return False, f"FA3 is unavailable due to: {FA3_UNAVAILABLE_REASON}"
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(90):
        return False, "FA3 is only supported on devices with compute capability 9.x"
    return True, None


def _is_fa4_supported() -> tuple[bool, str | None]:
    if not FA4_AVAILABLE:
        return False, f"FA4 is unavailable due to: {FA4_UNAVAILABLE_REASON}"
    from vllm.platforms import current_platform

    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(110)
    ):
        return (
            False,
            "FA4 is only supported on devices with compute capability 9.x, 10.x, or 11.x",
        )
    return True, None


def is_fa_version_supported(fa_version: int) -> bool:
    if fa_version == 2:
        return _is_fa2_supported()[0]
    elif fa_version == 3:
        return _is_fa3_supported()[0]
    elif fa_version == 4:
        return _is_fa4_supported()[0]
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")


def fa_version_unsupported_reason(fa_version: int) -> str | None:
    if fa_version == 2:
        return _is_fa2_supported()[1]
    elif fa_version == 3:
        return _is_fa3_supported()[1]
    elif fa_version == 4:
        return _is_fa4_supported()[1]
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")


#
#  For vLLM we only care about `flash_attn_varlen_func` and
#   `flash_attn_with_kvcache` so we only maintain wrappers for these two.
#


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


# NOTE only used in FA3
def get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k_new: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    page_size: int | None = None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    has_softcap=False,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
):
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = torch.ops._vllm_fa3_C.get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0],
        window_size[1],
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )

    return scheduler_metadata


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: list[int] | None = None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert cu_seqlens_k is not None or seqused_k is not None, (
        "cu_seqlens_k or seqused_k must be provided"
    )
    assert cu_seqlens_k is None or seqused_k is None, (
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    )
    assert block_table is None or seqused_k is not None, (
        "seqused_k must be provided if block_table is provided"
    )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # custom op does not support non-tuple input
    real_window_size: tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)

    if fa_version == 2:
        if (
            scheduler_metadata is not None
            and q_descale is not None
            and k_descale is not None
            and v_descale is not None
        ):
            raise NotImplementedError(
                "FA2 does not support scheduler_metadata, q_descale, "
                "k_descale, v_descale"
            )
        if s_aux is not None:
            raise NotImplementedError("FA2 does not support s_aux")
        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")
        load_fa2_library(q.device)
        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            return_softmax_lse and dropout_p > 0,
            num_splits,
            None,
        )
    elif fa_version == 3:
        assert alibi_slopes is None, "Alibi is not supported in FA3"
        out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            q_v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,  # cu_seqlens_k
            None,  # cu_seqlens_k_new
            None,
            seqused_k,  # seqused_q, seqused_k
            max_seqlen_q,
            max_seqlen_k,
            block_table,
            None,  # kv_batch_idx
            None,  # leftpad_k
            None,
            None,
            None,  # rotary_cos, rotary_sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            True,  # rotary_interleaved
            scheduler_metadata,
            num_splits,
            None,  # pack_gqa
            0,  # sm_margin
            s_aux,  # s_aux
            cp_world_size,
            cp_rank,
            cp_tot_seqused_k,
        )
    elif fa_version == 4:
        assert alibi_slopes is None, "Alibi is not supported in FA4"

        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

        out, softmax_lse = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=block_table,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size_left=real_window_size[0] if real_window_size[0] >= 0 else None,
            window_size_right=real_window_size[1] if real_window_size[1] >= 0 else None,
            num_splits=num_splits,
            return_lse=return_softmax_lse,
            out=out,
            learnable_sink=s_aux,
        )
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")
    return (out, softmax_lse) if return_softmax_lse else out


def sparse_attn_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    load_fa2_library(q.device)
    out, softmax_lse = torch.ops._vllm_fa2_C.fwd_sparse(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out


def sparse_attn_varlen_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_varlen_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    load_fa2_library(q.device)
    out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd_sparse(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out
