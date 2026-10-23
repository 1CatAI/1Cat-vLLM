# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared LM-head provider. No model, quant method or runner callbacks."""

from dataclasses import dataclass
from types import ModuleType

import torch

from vllm import _sm70_ops as sm70_ops
from vllm._sm70.policy import NativeBindings
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear_io import (
    flatten_linear_input,
    restore_linear_output,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


@dataclass(frozen=True)
class VocabShard:
    org_vocab_start_index: int = 0
    num_org_vocab_padding: int = 0


class Sm70LMHeadState(torch.nn.Module):
    """Single owner of disposable packs and scratch; weight registration stays
    on the generic embedding. Binding another weight discards this state.
    """

    def __init__(self, weight, shard, *, is_lm_head, policy, dflash, trace):
        super().__init__()
        object.__setattr__(self, "weight", weight)
        self.shard_indices = shard
        self.is_lm_head = is_lm_head
        self.policy = policy
        self.dflash = dflash
        self.trace = trace
        self.native_ops: ModuleType | NativeBindings = sm70_ops
        self.fp8_ops: ModuleType | NativeBindings = sm70_ops


@dataclass
class RerankShadow:
    x: torch.Tensor
    output_shape: tuple[int, ...]
    selector_k: int
    values: torch.Tensor
    ids: torch.Tensor
    support_ids: torch.Tensor


_SM70_DFLASH2_QPN8_VOCAB_CHUNK = 62080
_SM70_DFLASH2_QPN8_MAX_ROWS = 8
_SM70_DFLASH2_QPN8_CANDIDATES = 64
_SM70_DFLASH2_QPN8_SPLIT_K = 8
_SM70_DFLASH2_QPN8_ACCUMULATOR_CHAINS = 1
_SM70_DFLASH2_RERANK_CTA_N = 128
_SM70_DFLASH2_RERANK_SPLIT_K = 10


def _sm70_dflash2_option(field: str, state: Sm70LMHeadState) -> bool:
    return bool(getattr(state.dflash, field))


def _sm70_dflash2_qpn8_rerank_enabled(state: Sm70LMHeadState) -> bool:
    return _sm70_dflash2_option("qpn8_rerank", state)


def _sm70_dflash2_qpn8_rerank_requested(state: Sm70LMHeadState) -> bool:
    return (
        _sm70_dflash2_qpn8_rerank_enabled(state=state)
        or state.dflash.qpn8_rerank_shadow
    )


def _sm70_lm_head_packed_layout_requested(
    fp32_logits: bool = False, *, state: Sm70LMHeadState
) -> bool:
    # FP32 dense logits and candidate rerank read the original FP16 parameter.
    # Only an explicitly enabled packed top1 still consumes this layout.
    return state.policy.lm_head_top1_tc or (
        not fp32_logits
        and (
            state.policy.lm_head_dense
            or _sm70_dflash2_qpn8_rerank_requested(state=state)
        )
    )


def _sm70_dflash2_use_dense_order(state: Sm70LMHeadState | None = None) -> bool:
    """Dense vocabulary tie order is the validated selector contract."""
    return True


def _trace_sm70_lm_head_skip(state: Sm70LMHeadState, reason: str) -> None:
    if state.trace.greedy_token_trace or state.trace.profile_trace:
        logger.warning_once("SM70 LM head fast path not prepared: %s", reason)


def _is_sm70_lm_head_fastpath_eligible(state: Sm70LMHeadState) -> bool:
    if not state.is_lm_head:
        return False
    if not (
        state.policy.lm_head_dense
        or bool(state.policy.lm_head_top1)
        or state.policy.lm_head_top1_tc
        or _sm70_dflash2_option("fp32_logits", state)
        or _sm70_dflash2_qpn8_rerank_requested(state=state)
    ):
        _trace_sm70_lm_head_skip(state, "disabled")
        return False
    if not current_platform.is_cuda_alike():
        _trace_sm70_lm_head_skip(state, "non_cuda_platform")
        return False
    if state.weight.dtype != torch.float16:
        _trace_sm70_lm_head_skip(state, f"weight_dtype={state.weight.dtype}")
        return False
    if not state.weight.is_cuda:
        _trace_sm70_lm_head_skip(state, "weight_not_cuda")
        return False
    if torch.cuda.get_device_capability(state.weight.device) != (7, 0):
        _trace_sm70_lm_head_skip(
            state, f"capability={torch.cuda.get_device_capability(state.weight.device)}"
        )
        return False
    if state.weight.ndim != 2:
        _trace_sm70_lm_head_skip(state, f"weight_ndim={state.weight.ndim}")
        return False
    if state.weight.shape[1] % 16 != 0 or state.weight.shape[0] % 32 != 0:
        _trace_sm70_lm_head_skip(state, f"weight_shape={tuple(state.weight.shape)}")
        return False
    return True


def _is_sm70_dflash2_qpn8_rerank_eligible(state: Sm70LMHeadState) -> bool:
    if not _sm70_dflash2_qpn8_rerank_requested(state=state):
        return False
    rows, hidden = state.weight.shape
    if rows < 64 or rows % 32 or hidden <= 0 or hidden % 128:
        logger.warning_once(
            "SM70 DFlash2 QPN8 rerank requires N>=64, N%%32=0 and K%%128=0; got %s. "
            "Using the dense LM head.",
            tuple(state.weight.shape),
        )
        return False
    if not _sm70_dflash2_option("fp32_logits", state) and (
        rows > _SM70_DFLASH2_QPN8_VOCAB_CHUNK or hidden != 5120
    ):
        # The legacy packed FP16 reranker requires exactly 64 candidates and
        # K=5120. The FP32 indexed reranker supports wider candidate sets.
        return False
    if state.shard_indices.num_org_vocab_padding != 0:
        logger.warning_once(
            "SM70 DFlash2 QPN8 rerank requires an unpadded local vocabulary; "
            "using the dense LM head."
        )
        return False
    required_ops = (
        "fp8_qpn8_prepare_sm70",
        "fp8_qpn8_gemm_sm70_out",
        "sm70_f16_indexed_rerank_packed_out",
        "sm70_f16_rerank_keys_out",
        "sm70_f16_rerank_topk_out",
    )
    missing = [name for name in required_ops if not hasattr(torch.ops._C, name)]
    if missing:
        logger.warning_once(
            "SM70 DFlash2 QPN8 rerank operators are unavailable (%s); using "
            "the dense LM head.",
            ", ".join(missing),
        )
        return False
    return True


@torch.inference_mode()
def _prepare_sm70_dflash2_qpn8_rerank(state: Sm70LMHeadState) -> bool:
    if getattr(state, "_sm70_dflash2_qpn8_rerank_prepared", False):
        return True
    if not _is_sm70_dflash2_qpn8_rerank_eligible(state):
        return False
    from vllm.config.sm70_native import capture_linear_native_config

    state.fp8_ops = NativeBindings(capture_linear_native_config("fp8").values)
    weight = state.weight
    rows, hidden = weight.shape
    qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    channel_scales = torch.empty((rows, 1), dtype=torch.float32, device=weight.device)
    # Quantize in bounded row chunks so startup never materializes another
    # full-size FP32 LM head.  The original FP16 parameter remains the oracle
    # used by the exact candidate rerank.
    chunk_rows = 4096
    for begin in range(0, rows, chunk_rows):
        end = min(begin + chunk_rows, rows)
        weight_f32 = weight[begin:end].float()
        scales = weight_f32.abs().amax(dim=1, keepdim=True).div_(448.0)
        scales.clamp_(min=torch.finfo(torch.float32).tiny)
        channel_scales[begin:end].copy_(scales)
        qweight[begin:end].copy_(
            weight_f32.div_(scales).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
        )

    codes, packed_scales = state.fp8_ops.fp8_qpn8_prepare_sm70(qweight, channel_scales)
    del qweight, channel_scales, weight_f32, scales
    torch.accelerator.empty_cache()

    device = weight.device
    fp32_logits = _sm70_dflash2_option("fp32_logits", state)
    state._sm70_dflash2_fp32_logits = fp32_logits
    rerank_dtype = torch.float32 if fp32_logits else torch.float16
    max_rows = _SM70_DFLASH2_QPN8_MAX_ROWS
    # Keep the accepted support density when a worker owns more vocabulary.
    # A wider shard screens 64 candidates per original-sized vocabulary chunk,
    # preserving every candidate that separate TP4 shards would have retained.
    groups = tuple(
        (begin, min(begin + _SM70_DFLASH2_QPN8_VOCAB_CHUNK, rows))
        for begin in range(0, rows, _SM70_DFLASH2_QPN8_VOCAB_CHUNK)
    )
    candidates = sum(
        min(_SM70_DFLASH2_QPN8_CANDIDATES, end - begin) for begin, end in groups
    )
    state._sm70_dflash2_qpn8_vocab_groups = groups
    state.register_buffer("_sm70_dflash2_qpn8_codes", codes, persistent=False)
    state.register_buffer("_sm70_dflash2_qpn8_scales", packed_scales, persistent=False)
    state.register_buffer(
        "_sm70_dflash2_qpn8_logits",
        torch.empty((max_rows, rows), dtype=torch.float16, device=device),
        persistent=False,
    )
    state.register_buffer(
        "_sm70_dflash2_qpn8_values",
        torch.empty((max_rows, candidates), dtype=torch.float16, device=device),
        persistent=False,
    )
    state.register_buffer(
        "_sm70_dflash2_qpn8_ids",
        torch.empty((max_rows, candidates), dtype=torch.int64, device=device),
        persistent=False,
    )
    state.register_buffer(
        "_sm70_dflash2_rerank_logits",
        torch.empty((max_rows, candidates), dtype=rerank_dtype, device=device),
        persistent=False,
    )
    if not fp32_logits:
        selected_rows = max_rows * candidates
        state.register_buffer(
            "_sm70_dflash2_rerank_selected_raw",
            torch.empty((selected_rows, hidden), dtype=torch.float16, device=device),
            persistent=False,
        )
        state.register_buffer(
            "_sm70_dflash2_rerank_selected_packed",
            torch.empty((selected_rows, hidden), dtype=torch.float16, device=device),
            persistent=False,
        )
        state.register_buffer(
            "_sm70_dflash2_rerank_expanded",
            torch.empty((max_rows, selected_rows), dtype=torch.float16, device=device),
            persistent=False,
        )
        state.register_buffer(
            "_sm70_dflash2_rerank_partials",
            torch.empty((max_rows, selected_rows), dtype=torch.float32, device=device),
            persistent=False,
        )
        state.register_buffer(
            "_sm70_dflash2_rerank_barriers",
            torch.zeros(64, dtype=torch.int32, device=device),
            persistent=False,
        )
    state.register_buffer(
        "_sm70_dflash2_rerank_dense_logits",
        torch.empty((max_rows, rows), dtype=rerank_dtype, device=device),
        persistent=False,
    )
    state.register_buffer(
        "_sm70_dflash2_rerank_keys",
        torch.empty((max_rows, candidates), dtype=torch.int64, device=device),
        persistent=False,
    )
    # Keep distinct top-16, top-20 and top-21 outputs. Slicing the columns of one
    # [max_rows, 20] allocation for top-16 leaves a row stride of 20 and makes
    # the result non-contiguous.  The TP all-gather requires contiguous inputs,
    # and inserting a runtime contiguous() copy would add work to both graphs.
    for selector_k in (16, 20, 21):
        state.register_buffer(
            f"_sm70_dflash2_rerank_values_{selector_k}",
            torch.empty((max_rows, selector_k), dtype=rerank_dtype, device=device),
            persistent=False,
        )
        state.register_buffer(
            f"_sm70_dflash2_rerank_positions_{selector_k}",
            torch.empty((max_rows, selector_k), dtype=torch.int64, device=device),
            persistent=False,
        )
        state.register_buffer(
            f"_sm70_dflash2_rerank_ids_{selector_k}",
            torch.empty((max_rows, selector_k), dtype=torch.int64, device=device),
            persistent=False,
        )
        state.register_buffer(
            f"_sm70_dflash2_rerank_key_values_{selector_k}",
            torch.empty((max_rows, selector_k), dtype=torch.int64, device=device),
            persistent=False,
        )
    state._sm70_dflash2_qpn8_rerank_prepared = True
    logger.info_once(
        "SM70 DFlash2 QPN8 rerank layout prepared: %d vocabulary chunks, "
        "%d candidates (%s logits).",
        len(groups),
        candidates,
        "FP32" if fp32_logits else "FP16",
    )
    return True


def _sm70_dflash2_rerank_output_buffers(
    state: Sm70LMHeadState,
    num_rows: int,
    selector_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select contiguous rerank outputs, including the target tie sentinel."""
    if selector_k == 16:
        values = state._sm70_dflash2_rerank_values_16[:num_rows]
        positions = state._sm70_dflash2_rerank_positions_16[:num_rows]
        ids = state._sm70_dflash2_rerank_ids_16[:num_rows]
    elif selector_k == 20:
        values = state._sm70_dflash2_rerank_values_20[:num_rows]
        positions = state._sm70_dflash2_rerank_positions_20[:num_rows]
        ids = state._sm70_dflash2_rerank_ids_20[:num_rows]
    elif selector_k == 21:
        values = state._sm70_dflash2_rerank_values_21[:num_rows]
        positions = state._sm70_dflash2_rerank_positions_21[:num_rows]
        ids = state._sm70_dflash2_rerank_ids_21[:num_rows]
    else:
        raise ValueError(f"Unsupported DFlash2 rerank top-k: {selector_k}")
    return values, positions, ids


def _sm70_dflash2_dense_order_topk(
    sparse_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    values: torch.Tensor,
    ids: torch.Tensor,
    selector_k: int,
    vocab_start_index: int,
) -> None:
    """Restore dense-vocabulary tie order after sparse candidate reranking."""
    sparse_logits.fill_(-float("inf"))
    sparse_logits.scatter_(1, candidate_ids, candidate_logits)
    torch.topk(
        sparse_logits,
        selector_k,
        dim=-1,
        sorted=True,
        out=(values, ids),
    )
    ids.add_(vocab_start_index)


def _sm70_dflash2_candidate_order_topk(
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    values: torch.Tensor,
    ids: torch.Tensor,
    selector_k: int,
    vocab_start_index: int,
) -> None:
    """Select exact values with original-vocabulary tie precedence."""
    if values.size(1) != selector_k:
        raise ValueError(
            f"DFlash2 rerank output width {values.size(1)} != top-k {selector_k}"
        )
    sm70_ops.sm70_f16_rerank_topk_out(
        values,
        ids,
        candidate_logits,
        candidate_ids,
        vocab_start_index,
    )


def maybe_prepare_sm70_lm_head_top1(state: Sm70LMHeadState) -> bool:
    if not _is_sm70_lm_head_fastpath_eligible(state):
        return False

    if (
        _sm70_dflash2_option("fp32_logits", state)
        and len(state.weight.shape) == 2
        and state.weight.shape[0] > 0
        and state.weight.shape[1] % 16 == 0
    ):
        # Dense FP32 output does not depend on candidate-layout availability
        # or the number of devices participating in tensor parallelism.
        state._sm70_dflash2_fp32_logits = True

    raw_top1_requested = bool(state.policy.lm_head_top1)
    packed_layout_requested = _sm70_lm_head_packed_layout_requested(
        getattr(state, "_sm70_dflash2_fp32_logits", False), state=state
    )
    if raw_top1_requested:
        state._sm70_f16_raw_top1_ready = True

    if not packed_layout_requested:
        _prepare_sm70_dflash2_qpn8_rerank(state)
        logger.info_once("SM70 original-weight LM head path prepared.")
        return True

    if not hasattr(torch.ops._C, "sm70_f16_prepare"):
        _trace_sm70_lm_head_skip(state, "missing_sm70_f16_prepare_op")
        return raw_top1_requested
    if getattr(state, "_sm70_f16_prepared", False):
        return True
    state.native_ops = NativeBindings(state.policy.native.values)
    prepared = state.native_ops.sm70_f16_prepare(state.weight)
    state.register_buffer("_sm70_f16_tm_weight", prepared[0], persistent=False)
    state._sm70_f16_k_ld = int(prepared[1][0].item())
    state._sm70_f16_prepared = True
    if _prepare_sm70_dflash2_qpn8_rerank(state):
        pass
    else:
        if state.policy.dense_log_error is not None:
            raise ValueError(state.policy.dense_log_error)
        if state.policy.dense_log_enabled:
            logger.info_once("SM70 dense fp16 fast path enabled for LM head.")
        else:
            logger.info_once("SM70 LM head top1 layout prepared.")
    return True


def _maybe_sm70_lm_head_forward(
    state: Sm70LMHeadState,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if getattr(state, "_sm70_dflash2_fp32_logits", False):
        x_2d = flatten_linear_input(x).contiguous()
        out = torch.mm(x_2d, state.weight.t(), out_dtype=torch.float32)
        if bias is not None:
            out = out + bias.float()
        return restore_linear_output(out, x)
    if not state.policy.lm_head_dense:
        return None
    if not getattr(state, "_sm70_f16_prepared", False):
        return None
    if not hasattr(torch.ops._C, "sm70_f16_gemm"):
        return None

    x_2d = flatten_linear_input(x)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    tm_weight = getattr(state, "_sm70_f16_tm_weight", None)
    k_ld = getattr(state, "_sm70_f16_k_ld", None)
    if tm_weight is not None and k_ld is not None:
        out = torch.empty(
            (x_2d.size(0), tm_weight.shape[0]),
            dtype=x_2d.dtype,
            device=x_2d.device,
        )
        state.native_ops.sm70_f16_gemm_out(out, x_2d, tm_weight, k_ld, False)
    else:
        out = state.native_ops.sm70_f16_gemm(x_2d, state.weight)

    if bias is not None:
        out = out + bias
    return restore_linear_output(out, x)


def _maybe_sm70_lm_head_top1(
    state: Sm70LMHeadState,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    lm_head_top1 = bool(state.policy.lm_head_top1)
    lm_head_top1_tc = state.policy.lm_head_top1_tc
    if not (lm_head_top1 or lm_head_top1_tc):
        return None
    if bias is not None:
        return None
    raw_top1_ready = lm_head_top1 and getattr(state, "_sm70_f16_raw_top1_ready", False)
    packed_top1_ready = lm_head_top1_tc and getattr(state, "_sm70_f16_prepared", False)
    if not (raw_top1_ready or packed_top1_ready):
        _trace_sm70_lm_head_skip(state, "top1_not_prepared")
        return None
    if not (
        hasattr(torch.ops._C, "sm70_f16_lm_head_top1_out")
        or hasattr(torch.ops._C, "sm70_f16_lm_head_top1_tc_out")
    ):
        logger.warning_once(
            "SM70 LM head top1 requested, but no top1 op is available; falling back."
        )
        return None
    if x.dtype != torch.float16 or not x.is_cuda:
        return None
    if torch.cuda.get_device_capability(x.device) != (7, 0):
        return None

    x_2d = flatten_linear_input(x)
    num_rows = x_2d.size(0)
    if num_rows <= 0:
        return None
    if num_rows != 1 and not (
        lm_head_top1_tc
        and num_rows <= 17
        and hasattr(torch.ops._C, "sm70_f16_lm_head_top1_tc_out")
    ):
        return None

    weight = state.weight
    if weight.dtype != torch.float16 or not weight.is_cuda or weight.stride(1) != 1:
        return None

    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    values = torch.empty((x_2d.size(0),), dtype=torch.float32, device=x_2d.device)
    indices = torch.empty((x_2d.size(0),), dtype=torch.int64, device=x_2d.device)
    if packed_top1_ready and hasattr(torch.ops._C, "sm70_f16_lm_head_top1_tc_out"):
        tm_weight = getattr(state, "_sm70_f16_tm_weight", None)
        k_ld = getattr(state, "_sm70_f16_k_ld", None)
        if tm_weight is not None and k_ld is not None:
            sm70_ops.sm70_f16_lm_head_top1_tc_out(
                values,
                indices,
                x_2d,
                tm_weight,
                int(k_ld),
                state.shard_indices.org_vocab_start_index,
                state.shard_indices.num_org_vocab_padding,
            )
            logger.info_once("SM70 Tensor Core LM head top1 epilogue path enabled.")
            return values.reshape(*x.shape[:-1]), indices.reshape(*x.shape[:-1])
        if num_rows != 1:
            return None

    if num_rows != 1:
        return None

    if not raw_top1_ready or not hasattr(torch.ops._C, "sm70_f16_lm_head_top1_out"):
        return None

    sm70_ops.sm70_f16_lm_head_top1_out(
        values,
        indices,
        x_2d,
        weight,
        int(weight.stride(0)),
        state.shard_indices.org_vocab_start_index,
        state.shard_indices.num_org_vocab_padding,
    )
    logger.info_once("SM70 fused LM head top1 path enabled.")
    return values.reshape(*x.shape[:-1]), indices.reshape(*x.shape[:-1])


def _maybe_sm70_dflash2_qpn8_rerank(
    state: Sm70LMHeadState,
    x: torch.Tensor,
    selector_k: int,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | RerankShadow | None:
    """Return shard-local top-k after QPN8 support search and FP16 rerank."""
    if not _sm70_dflash2_qpn8_rerank_requested(state=state):
        return None
    if not getattr(state, "_sm70_dflash2_qpn8_rerank_prepared", False):
        return None
    if selector_k not in (16, 20, 21) or bias is not None:
        return None
    if (
        selector_k == 21
        and not _sm70_dflash2_use_dense_order(state=state)
        and not getattr(state, "_sm70_dflash2_fp32_logits", False)
    ):
        return None
    if x.dtype != torch.float16 or not x.is_cuda:
        return None
    if torch.cuda.get_device_capability(x.device) != (7, 0):
        return None

    x_2d = flatten_linear_input(x)
    num_rows = x_2d.size(0)
    if not 1 <= num_rows <= _SM70_DFLASH2_QPN8_MAX_ROWS:
        return None
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    qpn8_logits = state._sm70_dflash2_qpn8_logits[:num_rows]
    qpn8_values = state._sm70_dflash2_qpn8_values[:num_rows]
    qpn8_ids = state._sm70_dflash2_qpn8_ids[:num_rows]
    state.fp8_ops.fp8_qpn8_gemm_sm70_out(
        qpn8_logits,
        x_2d,
        state._sm70_dflash2_qpn8_codes,
        state._sm70_dflash2_qpn8_scales,
        _SM70_DFLASH2_QPN8_SPLIT_K,
        _SM70_DFLASH2_QPN8_ACCUMULATOR_CHAINS,
        False,
        False,
    )
    # The exact FP16 rerank is permutation-invariant over this approximate
    # support, so skip the unnecessary 64-element result sort. This keeps the
    # official PyTorch multiblock selector while avoiding its final bitonic
    # kernel and leaves candidate quality unchanged.
    candidate_offset = 0
    for begin, end in state._sm70_dflash2_qpn8_vocab_groups:
        count = min(_SM70_DFLASH2_QPN8_CANDIDATES, end - begin)
        group_values = qpn8_values[:, candidate_offset : candidate_offset + count]
        group_ids = qpn8_ids[:, candidate_offset : candidate_offset + count]
        torch.topk(
            qpn8_logits[:, begin:end],
            count,
            dim=-1,
            sorted=False,
            out=(group_values, group_ids),
        )
        if begin:
            group_ids.add_(begin)
        candidate_offset += count

    fp32_logits = getattr(state, "_sm70_dflash2_fp32_logits", False)
    if fp32_logits:
        from vllm.model_executor.kernels.lm_head.fp32 import indexed_fp32_logits

        indexed_fp32_logits(
            x_2d,
            state.weight,
            qpn8_ids,
            state._sm70_dflash2_rerank_logits[:num_rows],
        )
        logger.info_once("SM70 DFlash2 FP32 candidate logits enabled.")
    else:
        sm70_ops.sm70_f16_indexed_rerank_packed_out(
            state._sm70_dflash2_rerank_logits[:num_rows],
            x_2d,
            state._sm70_f16_tm_weight,
            qpn8_ids,
            state._sm70_dflash2_rerank_selected_packed,
            state._sm70_dflash2_rerank_expanded,
            state._sm70_dflash2_rerank_partials,
            state._sm70_dflash2_rerank_barriers,
            _SM70_DFLASH2_RERANK_CTA_N,
            _SM70_DFLASH2_RERANK_SPLIT_K,
        )
    rerank_logits = state._sm70_dflash2_rerank_logits[:num_rows]
    values, _positions, ids = _sm70_dflash2_rerank_output_buffers(
        state, num_rows, selector_k
    )
    use_dense_order = fp32_logits or _sm70_dflash2_use_dense_order(state=state)
    if use_dense_order:
        _sm70_dflash2_dense_order_topk(
            state._sm70_dflash2_rerank_dense_logits[:num_rows],
            qpn8_ids,
            rerank_logits,
            values,
            ids,
            selector_k,
            state.shard_indices.org_vocab_start_index,
        )
    else:
        _sm70_dflash2_candidate_order_topk(
            qpn8_ids,
            rerank_logits,
            values,
            ids,
            selector_k,
            state.shard_indices.org_vocab_start_index,
        )

    if state.dflash.qpn8_rerank_shadow:
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW is eager-only; disable "
                "CUDA Graph/torch.compile for the real-hidden coverage audit."
            )
        return RerankShadow(
            x_2d, (*x.shape[:-1], selector_k), selector_k, values, ids, qpn8_ids
        )

    logger.info_once(
        "SM70 DFlash2 QPN8 top-64 per vocabulary chunk plus %s rerank path "
        "enabled (chunks=%d, dense_order=%s).",
        "FP32 candidate" if fp32_logits else "packed TurboMind FP16",
        len(state._sm70_dflash2_qpn8_vocab_groups),
        use_dense_order,
    )
    output_shape = (*x.shape[:-1], selector_k)
    return values.reshape(output_shape), ids.reshape(output_shape)


def _maybe_sm70_dflash2_lm_head_top20(
    state: Sm70LMHeadState,
    x: torch.Tensor,
    selector_k: int,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | RerankShadow | None:
    """Return selector candidates from the opt-in QPN8 plus FP16 rerank."""
    return _maybe_sm70_dflash2_qpn8_rerank(state, x, selector_k, bias)


def complete_shadow(state: Sm70LMHeadState, request: RerankShadow, dense_logits):
    selector_k = request.selector_k
    values, ids, qpn8_ids = request.values, request.ids, request.support_ids
    num_rows = request.x.shape[0]
    dense_values, dense_ids = torch.topk(
        dense_logits,
        selector_k,
        dim=-1,
        sorted=True,
    )
    support_matches = (dense_ids[:, :, None] == qpn8_ids[:, None, :]).any(dim=-1)
    dense_global_ids = dense_ids + state.shard_indices.org_vocab_start_index
    rerank_matches = (dense_global_ids[:, :, None] == ids[:, None, :]).any(dim=-1)
    call = int(getattr(state, "_sm70_dflash2_qpn8_shadow_calls", 0)) + 1
    state._sm70_dflash2_qpn8_shadow_calls = call
    logger.info(
        "SM70_DFLASH2_QPN8_SHADOW call=%d rows=%d top_k=%d "
        "support_missing=%d exact_set_rows=%d top1_match_rows=%d "
        "ordered_value_max_abs=%.6g",
        call,
        num_rows,
        selector_k,
        int((~support_matches).sum().item()),
        int(rerank_matches.all(dim=-1).sum().item()),
        int((dense_global_ids[:, 0] == ids[:, 0]).sum().item()),
        float((dense_values.float() - values.float()).abs().max().item()),
    )
    output_shape = request.output_shape
    return dense_values.reshape(output_shape), dense_global_ids.reshape(output_shape)
