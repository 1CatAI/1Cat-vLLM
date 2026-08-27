# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter

import vllm.envs as envs
from vllm import _sm70_ops as sm70_ops
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.batch_invariant import (
    linear_batch_invariant,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

DEFAULT_VOCAB_PADDING_SIZE = 64
logger = init_logger(__name__)

_SM70_DFLASH2_QPN8_LM_HEAD_SHAPE = (62080, 5120)
_SM70_DFLASH2_QPN8_MAX_ROWS = 8
_SM70_DFLASH2_QPN8_CANDIDATES = 64
_SM70_DFLASH2_QPN8_SPLIT_K = 8
_SM70_DFLASH2_QPN8_ACCUMULATOR_CHAINS = 1
_SM70_DFLASH2_RERANK_CTA_N = 128
_SM70_DFLASH2_RERANK_SPLIT_K = 10


def _sm70_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _sm70_lm_head_top1_default() -> bool:
    return not _sm70_env_bool("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", False)


def _sm70_dflash2_qpn8_rerank_enabled() -> bool:
    return envs.VLLM_SM70_DFLASH2_QPN8_RERANK


def _sm70_dflash2_qpn8_rerank_requested() -> bool:
    return (
        _sm70_dflash2_qpn8_rerank_enabled() or envs.VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW
    )


def _sm70_dflash2_use_dense_order() -> bool:
    """Keep production on the scored full-vocabulary tie-order contract."""
    if envs.VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER:
        return True
    if envs.VLLM_SM70_DFLASH2_QPN8_ALLOW_CANDIDATE_ORDER:
        logger.warning_once(
            "Using experimental SM70 DFlash2 QPN8 candidate-order top-k. "
            "This path is not authorized for quality-sensitive serving."
        )
        return False
    logger.warning_once(
        "Ignoring VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER=0 without the explicit "
        "benchmark-only VLLM_SM70_DFLASH2_QPN8_ALLOW_CANDIDATE_ORDER=1; "
        "using the scored dense-order path."
    )
    return True


def _trace_sm70_lm_head_skip(reason: str) -> None:
    if envs.VLLM_SM70_GREEDY_TOKEN_FASTPATH_TRACE or envs.VLLM_SM70_PROFILE_TRACE:
        logger.warning_once("SM70 LM head fast path not prepared: %s", reason)


def _is_sm70_lm_head_fastpath_eligible(layer: torch.nn.Module) -> bool:
    prefix = getattr(layer, "prefix", "")
    is_lm_head = (
        bool(prefix) and prefix.rsplit(".", 1)[-1] == "lm_head"
    ) or layer.__class__.__name__ == "ParallelLMHead"
    if not is_lm_head:
        return False
    if not (
        _sm70_env_bool("VLLM_SM70_ENABLE_LM_HEAD_FASTPATH", False)
        or _sm70_env_bool("VLLM_SM70_LM_HEAD_TOP1", _sm70_lm_head_top1_default())
        or _sm70_env_bool("VLLM_SM70_LM_HEAD_TOP1_TC", False)
        or _sm70_dflash2_qpn8_rerank_requested()
    ):
        _trace_sm70_lm_head_skip("disabled")
        return False
    if not current_platform.is_cuda_alike():
        _trace_sm70_lm_head_skip("non_cuda_platform")
        return False
    if not hasattr(torch.ops._C, "sm70_f16_prepare"):
        _trace_sm70_lm_head_skip("missing_sm70_f16_prepare_op")
        return False
    if layer.weight.dtype != torch.float16:
        _trace_sm70_lm_head_skip(f"weight_dtype={layer.weight.dtype}")
        return False
    if not layer.weight.is_cuda:
        _trace_sm70_lm_head_skip("weight_not_cuda")
        return False
    if torch.cuda.get_device_capability(layer.weight.device) != (7, 0):
        _trace_sm70_lm_head_skip(
            f"capability={torch.cuda.get_device_capability(layer.weight.device)}"
        )
        return False
    if layer.weight.ndim != 2:
        _trace_sm70_lm_head_skip(f"weight_ndim={layer.weight.ndim}")
        return False
    if layer.weight.shape[1] % 16 != 0 or layer.weight.shape[0] % 32 != 0:
        _trace_sm70_lm_head_skip(f"weight_shape={tuple(layer.weight.shape)}")
        return False
    return True


def _is_sm70_dflash2_qpn8_rerank_eligible(layer: torch.nn.Module) -> bool:
    if not _sm70_dflash2_qpn8_rerank_requested():
        return False
    if tuple(layer.weight.shape) != _SM70_DFLASH2_QPN8_LM_HEAD_SHAPE:
        logger.warning_once(
            "SM70 DFlash2 QPN8 rerank requires LM-head shape %s; got %s. "
            "Using the dense LM head.",
            _SM70_DFLASH2_QPN8_LM_HEAD_SHAPE,
            tuple(layer.weight.shape),
        )
        return False
    if getattr(layer, "tp_size", 1) != 4:
        logger.warning_once(
            "SM70 DFlash2 QPN8 rerank requires TP4; using the dense LM head."
        )
        return False
    if layer.shard_indices.num_org_vocab_padding != 0:
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
def _prepare_sm70_dflash2_qpn8_rerank(layer: torch.nn.Module) -> bool:
    if not _is_sm70_dflash2_qpn8_rerank_eligible(layer):
        return False

    weight = layer.weight
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

    codes, packed_scales = sm70_ops.fp8_qpn8_prepare_sm70(qweight, channel_scales)
    del qweight, channel_scales, weight_f32, scales
    torch.accelerator.empty_cache()

    device = weight.device
    max_rows = _SM70_DFLASH2_QPN8_MAX_ROWS
    candidates = _SM70_DFLASH2_QPN8_CANDIDATES
    layer.register_buffer("_sm70_dflash2_qpn8_codes", codes, persistent=False)
    layer.register_buffer("_sm70_dflash2_qpn8_scales", packed_scales, persistent=False)
    layer.register_buffer(
        "_sm70_dflash2_qpn8_logits",
        torch.empty((max_rows, rows), dtype=torch.float16, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_sm70_dflash2_qpn8_values",
        torch.empty((max_rows, candidates), dtype=torch.float16, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_sm70_dflash2_qpn8_ids",
        torch.empty((max_rows, candidates), dtype=torch.int64, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_sm70_dflash2_rerank_logits",
        torch.empty((max_rows, candidates), dtype=torch.float16, device=device),
        persistent=False,
    )
    selected_rows = max_rows * candidates
    layer.register_buffer(
        "_sm70_dflash2_rerank_selected_raw",
        torch.empty((selected_rows, hidden), dtype=torch.float16, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_sm70_dflash2_rerank_selected_packed",
        torch.empty((selected_rows, hidden), dtype=torch.float16, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_sm70_dflash2_rerank_expanded",
        torch.empty((max_rows, selected_rows), dtype=torch.float16, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_sm70_dflash2_rerank_partials",
        torch.empty((max_rows, selected_rows), dtype=torch.float32, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_sm70_dflash2_rerank_barriers",
        torch.zeros(64, dtype=torch.int32, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_sm70_dflash2_rerank_dense_logits",
        torch.empty((max_rows, rows), dtype=torch.float16, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_sm70_dflash2_rerank_keys",
        torch.empty((max_rows, candidates), dtype=torch.int64, device=device),
        persistent=False,
    )
    # Keep distinct top-16 and top-20 outputs.  Slicing the columns of one
    # [max_rows, 20] allocation for top-16 leaves a row stride of 20 and makes
    # the result non-contiguous.  The TP all-gather requires contiguous inputs,
    # and inserting a runtime contiguous() copy would add work to both graphs.
    for selector_k in (16, 20):
        layer.register_buffer(
            f"_sm70_dflash2_rerank_values_{selector_k}",
            torch.empty((max_rows, selector_k), dtype=torch.float16, device=device),
            persistent=False,
        )
        layer.register_buffer(
            f"_sm70_dflash2_rerank_positions_{selector_k}",
            torch.empty((max_rows, selector_k), dtype=torch.int64, device=device),
            persistent=False,
        )
        layer.register_buffer(
            f"_sm70_dflash2_rerank_ids_{selector_k}",
            torch.empty((max_rows, selector_k), dtype=torch.int64, device=device),
            persistent=False,
        )
        layer.register_buffer(
            f"_sm70_dflash2_rerank_key_values_{selector_k}",
            torch.empty((max_rows, selector_k), dtype=torch.int64, device=device),
            persistent=False,
        )
    layer._sm70_dflash2_qpn8_rerank_prepared = True
    logger.info_once(
        "SM70 DFlash2 QPN8 top-64 plus exact TurboMind FP16 rerank layout prepared."
    )
    return True


def _sm70_dflash2_rerank_output_buffers(
    layer: torch.nn.Module,
    num_rows: int,
    selector_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select graph-stable, contiguous rerank outputs for top-16 or top-20."""
    if selector_k == 16:
        values = layer._sm70_dflash2_rerank_values_16[:num_rows]
        positions = layer._sm70_dflash2_rerank_positions_16[:num_rows]
        ids = layer._sm70_dflash2_rerank_ids_16[:num_rows]
    elif selector_k == 20:
        values = layer._sm70_dflash2_rerank_values_20[:num_rows]
        positions = layer._sm70_dflash2_rerank_positions_20[:num_rows]
        ids = layer._sm70_dflash2_rerank_ids_20[:num_rows]
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


def maybe_prepare_sm70_lm_head_top1(layer: torch.nn.Module) -> bool:
    if getattr(layer, "_sm70_f16_prepared", False):
        return True
    if not _is_sm70_lm_head_fastpath_eligible(layer):
        return False
    prepared = sm70_ops.sm70_f16_prepare(layer.weight)
    layer._sm70_f16_tm_weight = prepared[0]
    layer._sm70_f16_k_ld = int(prepared[1][0].item())
    layer._sm70_f16_prepared = True
    if _prepare_sm70_dflash2_qpn8_rerank(layer):
        pass
    elif envs.VLLM_SM70_ENABLE_LM_HEAD_FASTPATH:
        logger.info_once("SM70 dense fp16 fast path enabled for LM head.")
    else:
        logger.info_once("SM70 LM head top1 layout prepared.")
    return True


def _maybe_sm70_lm_head_forward(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not _sm70_env_bool("VLLM_SM70_ENABLE_LM_HEAD_FASTPATH", False):
        return None
    if not getattr(layer, "_sm70_f16_prepared", False):
        return None
    if not hasattr(torch.ops._C, "sm70_f16_gemm"):
        return None

    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    tm_weight = getattr(layer, "_sm70_f16_tm_weight", None)
    k_ld = getattr(layer, "_sm70_f16_k_ld", None)
    if tm_weight is not None and k_ld is not None:
        out = torch.empty(
            (x_2d.size(0), tm_weight.shape[0]),
            dtype=x_2d.dtype,
            device=x_2d.device,
        )
        sm70_ops.sm70_f16_gemm_out(out, x_2d, tm_weight, k_ld, False)
    else:
        out = sm70_ops.sm70_f16_gemm(x_2d, layer.weight)

    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], out.shape[-1])


def _maybe_sm70_lm_head_top1(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    lm_head_top1 = _sm70_env_bool(
        "VLLM_SM70_LM_HEAD_TOP1", _sm70_lm_head_top1_default()
    )
    lm_head_top1_tc = _sm70_env_bool("VLLM_SM70_LM_HEAD_TOP1_TC", False)
    if not (lm_head_top1 or lm_head_top1_tc):
        return None
    if bias is not None:
        return None
    if not getattr(layer, "_sm70_f16_prepared", False):
        _trace_sm70_lm_head_skip("top1_not_prepared")
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

    x_2d = x.reshape(-1, x.shape[-1])
    num_rows = x_2d.size(0)
    if num_rows <= 0:
        return None
    if num_rows != 1 and not (
        lm_head_top1_tc
        and num_rows <= 17
        and hasattr(torch.ops._C, "sm70_f16_lm_head_top1_tc_out")
    ):
        return None

    weight = layer.weight
    if weight.dtype != torch.float16 or not weight.is_cuda or weight.stride(1) != 1:
        return None

    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    values = torch.empty((x_2d.size(0),), dtype=torch.float32, device=x_2d.device)
    indices = torch.empty((x_2d.size(0),), dtype=torch.int64, device=x_2d.device)
    if lm_head_top1_tc and hasattr(torch.ops._C, "sm70_f16_lm_head_top1_tc_out"):
        tm_weight = getattr(layer, "_sm70_f16_tm_weight", None)
        k_ld = getattr(layer, "_sm70_f16_k_ld", None)
        if tm_weight is not None and k_ld is not None:
            sm70_ops.sm70_f16_lm_head_top1_tc_out(
                values,
                indices,
                x_2d,
                tm_weight,
                int(k_ld),
                layer.shard_indices.org_vocab_start_index,
                layer.shard_indices.num_org_vocab_padding,
            )
            logger.info_once("SM70 Tensor Core LM head top1 epilogue path enabled.")
            return values.reshape(*x.shape[:-1]), indices.reshape(*x.shape[:-1])
        if num_rows != 1:
            return None

    if num_rows != 1:
        return None

    if not lm_head_top1 or not hasattr(torch.ops._C, "sm70_f16_lm_head_top1_out"):
        return None

    sm70_ops.sm70_f16_lm_head_top1_out(
        values,
        indices,
        x_2d,
        weight,
        int(weight.stride(0)),
        layer.shard_indices.org_vocab_start_index,
        layer.shard_indices.num_org_vocab_padding,
    )
    logger.info_once("SM70 fused LM head top1 path enabled.")
    return values.reshape(*x.shape[:-1]), indices.reshape(*x.shape[:-1])


def _maybe_sm70_dflash2_qpn8_rerank(
    layer: torch.nn.Module,
    x: torch.Tensor,
    selector_k: int,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return shard-local top-k after QPN8 support search and FP16 rerank."""
    if not _sm70_dflash2_qpn8_rerank_requested():
        return None
    if not getattr(layer, "_sm70_dflash2_qpn8_rerank_prepared", False):
        return None
    if selector_k not in (16, 20) or bias is not None:
        return None
    if x.dtype != torch.float16 or not x.is_cuda:
        return None
    if torch.cuda.get_device_capability(x.device) != (7, 0):
        return None

    x_2d = x.reshape(-1, x.shape[-1])
    num_rows = x_2d.size(0)
    if not 1 <= num_rows <= _SM70_DFLASH2_QPN8_MAX_ROWS:
        return None
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    qpn8_logits = layer._sm70_dflash2_qpn8_logits[:num_rows]
    qpn8_values = layer._sm70_dflash2_qpn8_values[:num_rows]
    qpn8_ids = layer._sm70_dflash2_qpn8_ids[:num_rows]
    sm70_ops.fp8_qpn8_gemm_sm70_out(
        qpn8_logits,
        x_2d,
        layer._sm70_dflash2_qpn8_codes,
        layer._sm70_dflash2_qpn8_scales,
        _SM70_DFLASH2_QPN8_SPLIT_K,
        _SM70_DFLASH2_QPN8_ACCUMULATOR_CHAINS,
        False,
        False,
    )
    # The exact FP16 rerank is permutation-invariant over this approximate
    # support, so skip the unnecessary 64-element result sort. This keeps the
    # official PyTorch multiblock selector while avoiding its final bitonic
    # kernel and leaves candidate quality unchanged.
    torch.topk(
        qpn8_logits,
        _SM70_DFLASH2_QPN8_CANDIDATES,
        dim=-1,
        sorted=False,
        out=(qpn8_values, qpn8_ids),
    )

    sm70_ops.sm70_f16_indexed_rerank_packed_out(
        layer._sm70_dflash2_rerank_logits[:num_rows],
        x_2d,
        layer._sm70_f16_tm_weight,
        qpn8_ids,
        layer._sm70_dflash2_rerank_selected_packed,
        layer._sm70_dflash2_rerank_expanded,
        layer._sm70_dflash2_rerank_partials,
        layer._sm70_dflash2_rerank_barriers,
        _SM70_DFLASH2_RERANK_CTA_N,
        _SM70_DFLASH2_RERANK_SPLIT_K,
    )
    rerank_logits = layer._sm70_dflash2_rerank_logits[:num_rows]
    values, _positions, ids = _sm70_dflash2_rerank_output_buffers(
        layer, num_rows, selector_k
    )
    use_dense_order = _sm70_dflash2_use_dense_order()
    if use_dense_order:
        _sm70_dflash2_dense_order_topk(
            layer._sm70_dflash2_rerank_dense_logits[:num_rows],
            qpn8_ids,
            rerank_logits,
            values,
            ids,
            selector_k,
            layer.shard_indices.org_vocab_start_index,
        )
    else:
        _sm70_dflash2_candidate_order_topk(
            qpn8_ids,
            rerank_logits,
            values,
            ids,
            selector_k,
            layer.shard_indices.org_vocab_start_index,
        )

    if envs.VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW:
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW is eager-only; disable "
                "CUDA Graph/torch.compile for the real-hidden coverage audit."
            )
        dense_logits = layer.quant_method.apply(layer, x_2d, bias=None)
        dense_values, dense_ids = torch.topk(
            dense_logits,
            selector_k,
            dim=-1,
            sorted=True,
        )
        support_matches = (dense_ids[:, :, None] == qpn8_ids[:, None, :]).any(dim=-1)
        dense_global_ids = dense_ids + layer.shard_indices.org_vocab_start_index
        rerank_matches = (dense_global_ids[:, :, None] == ids[:, None, :]).any(dim=-1)
        call = int(getattr(layer, "_sm70_dflash2_qpn8_shadow_calls", 0)) + 1
        layer._sm70_dflash2_qpn8_shadow_calls = call
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
        output_shape = (*x.shape[:-1], selector_k)
        return dense_values.reshape(output_shape), dense_global_ids.reshape(
            output_shape
        )

    logger.info_once(
        "SM70 DFlash2 QPN8 top-64 plus exact packed TurboMind FP16 rerank "
        "path enabled (dense_order=%s).",
        use_dense_order,
    )
    output_shape = (*x.shape[:-1], selector_k)
    return values.reshape(output_shape), ids.reshape(output_shape)


def _maybe_sm70_dflash2_lm_head_top20(
    layer: torch.nn.Module,
    x: torch.Tensor,
    selector_k: int,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return selector candidates from the opt-in QPN8 plus FP16 rerank."""
    return _maybe_sm70_dflash2_qpn8_rerank(layer, x, selector_k, bias)


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if current_platform.is_cpu():
            from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm

            dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
            return

        maybe_prepare_sm70_lm_head_top1(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sm70_out = _maybe_sm70_lm_head_forward(layer, x, bias)
        if sm70_out is not None:
            return sm70_out
        if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
            return linear_batch_invariant(x, layer.weight, bias)
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, offset=offset
    )


@dataclass
class VocabParallelEmbeddingShardIndices:
    """Indices for a shard of a vocab parallel embedding."""

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self):
        # sanity checks
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index

        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index

        assert self.org_vocab_start_index <= self.padded_org_vocab_start_index
        assert self.added_vocab_start_index <= self.padded_added_vocab_start_index
        assert self.org_vocab_end_index <= self.padded_org_vocab_end_index
        assert self.added_vocab_end_index <= self.padded_added_vocab_end_index

        assert self.num_org_elements <= self.num_org_elements_padded
        assert self.num_added_elements <= self.num_added_elements_padded


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


# --8<-- [start:vocab_parallel_embedding]
@PluggableLayer.register("vocab_parallel_embedding")
class VocabParallelEmbedding(PluggableLayer):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    In this example, we will have the original vocab size = 1010,
    added vocab size = 16 and padding to 64. Therefore, the total
    vocab size with padding will be 1088 (because we first pad 1010 to
    1024, add 16, and then pad to 1088).
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1025 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1010 | ... | 1025 |  -1  | ... |  -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  528 | ... | 543 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  | 528 | ... |  543 |

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
        quant_config: quant config for the layer
        prefix: full name of the layer in the state dict
    """  # noqa: E501

    # --8<-- [end:vocab_parallel_embedding]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        quant_method: QuantizeMethodBase | None = None,
    ):
        super().__init__()
        self.prefix = prefix

        # Keep the input dimensions.
        tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim

        # Model-specific embeddings can preselect a storage method. This is
        # required by Qwen4Exp PLE, whose table remains FP8 even when the
        # routed experts use a different checkpoint quantization config.
        if quant_method is None and quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = type(self) is VocabParallelEmbedding
        quant_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method)
        )
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        # Divide the weight matrix along the vocabulary dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size
        )
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index
            - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index
            - self.shard_indices.added_vocab_start_index
        )

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """Get start and end indices for vocab parallel embedding, following the
        layout outlined in the class docstring, based on the given tp_rank and
        tp_size."""
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        padded_org_vocab_start_index, padded_org_vocab_end_index = (
            vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank, tp_size)
        )
        padded_added_vocab_start_index, padded_added_vocab_end_index = (
            vocab_range_from_global_vocab_size(
                num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
            )
        )
        # remove padding
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    def get_sharded_to_full_mapping(self) -> list[int] | None:
        """Get a mapping that can be used to reindex the gathered
        logits for sampling.

        During sampling, we gather logits from all ranks. The relationship
        of index->token_id will follow the same format as outlined in the class
        docstring. However, after the gather, we want to reindex the final
        logits tensor to map index->token_id one-to-one (the index is always
        equal the token_id it corresponds to). The indices returned by this
        method allow us to do that.
        """
        if self.tp_size < 2:
            return None

        base_embeddings: list[int] = []
        added_embeddings: list[int] = []
        padding: list[int] = []
        for tp_rank in range(self.tp_size):
            shard_indices = self._get_indices(
                self.num_embeddings_padded,
                self.org_vocab_size_padded,
                self.num_embeddings,
                self.org_vocab_size,
                tp_rank,
                self.tp_size,
            )
            range_start = self.num_embeddings_per_partition * tp_rank
            range_end = self.num_embeddings_per_partition * (tp_rank + 1)
            base_embeddings.extend(
                range(range_start, range_start + shard_indices.num_org_elements)
            )
            padding.extend(
                range(
                    range_start + shard_indices.num_org_elements,
                    range_start + shard_indices.num_org_elements_padded,
                )
            )
            added_embeddings.extend(
                range(
                    range_start + shard_indices.num_org_elements_padded,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                )
            )
            padding.extend(
                range(
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements_padded,
                )
            )
            assert (
                range_start
                + shard_indices.num_org_elements_padded
                + shard_indices.num_added_elements_padded
                == range_end
            )
        ret = base_embeddings + added_embeddings + padding
        assert len(ret) == self.num_embeddings_padded
        return ret

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)

        # If the parameter is a gguf weight, then load it directly.
        if getattr(param, "is_gguf_weight_type", None):
            param.data.copy_(loaded_weight)
            param.weight_type = loaded_weight.item()
            return
        elif isinstance(param, UninitializedParameter):
            shape = list(loaded_weight.shape)
            if output_dim is not None:
                shape[output_dim] = self.num_embeddings_per_partition
            param.materialize(tuple(shape), dtype=loaded_weight.dtype)

        # If parameter does not have output dim, then it should
        # be copied onto all gpus (e.g. g_idx for act_order gptq).
        if output_dim is None:
            if (
                loaded_weight.ndim == 0
                and param.data.ndim == 1
                and param.data.numel() == 1
            ):
                loaded_weight = loaded_weight.reshape(1)
            assert param.data.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return

        # Shard indexes for loading the weight
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        # If param packed on the same dim we are sharding on, then
        # need to adjust offsets of loaded weight by pack_factor.
        if packed_dim is not None and packed_dim == output_dim:
            packed_factor = (
                param.packed_factor
                if isinstance(param, BasevLLMParameter)
                else param.pack_factor
            )
            assert loaded_weight.shape[output_dim] == (
                self.org_vocab_size // param.packed_factor
            )
            start_idx = start_idx // packed_factor
            shard_size = shard_size // packed_factor
        else:
            assert loaded_weight.shape[output_dim] == self.org_vocab_size

        # Copy the data. Select chunk corresponding to current shard.
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0] :].data.fill_(0)

    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        # Mask the output embedding.
        if self.tp_size > 1:
            if output_parallel.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ):
                # Each token row has exactly one TP owner. Communicate the raw
                # FP8 bytes as int8 because NCCL does not reduce FP8 directly.
                comm_output = output_parallel.view(torch.int8)
                comm_output.masked_fill_(input_mask.unsqueeze(-1), 0)
                output = tensor_model_parallel_all_reduce(comm_output)
                return output.view(output_parallel.dtype)
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output

    def maybe_get_sm70_lm_head_top1(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return _maybe_sm70_lm_head_top1(self, hidden_states, bias)

    def maybe_get_sm70_dflash2_top20(
        self,
        hidden_states: torch.Tensor,
        selector_k: int,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return _maybe_sm70_dflash2_lm_head_top20(self, hidden_states, selector_k, bias)

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings_per_partition}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", num_embeddings_padded={self.num_embeddings_padded}"
        s += f", tp_size={self.tp_size}"
        return s


# --8<-- [start:parallel_lm_head]
@PluggableLayer.register("parallel_lm_head")
class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    # --8<-- [end:parallel_lm_head]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
        )
        self.quant_config = quant_config
        if bias:
            self.bias = Parameter(
                torch.empty(self.num_embeddings_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        """Tie the weights with word embeddings."""
        # GGUF quantized embed_tokens.
        if self.quant_config and self.quant_config.get_name() == "gguf":
            return embed_tokens
        else:
            self.weight = embed_tokens.weight
            return self

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
