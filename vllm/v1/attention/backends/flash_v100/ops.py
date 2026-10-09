# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy loading of the Flash-V100 native operators."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100.runtime import (
    bind_attention_operation,
    prepare_attention_runtime,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


# Lazy imports: only resolve optional CUDA extensions when needed.
_flash_attn_func = None
_flash_attn_bhmd_func = None
_flash_attn_decode_paged = None
_flash_attn_decode_paged_xqa = None
_flash_attn_decode_paged_wmma = None
_flash_attn_grouped_verify_paged = None
_flash_attn_grouped_verify_max_query_tokens = 8
_flash_attn_grouped_verify_request_major_abi_version = 0
_flash_attn_grouped_verify_checked = False
_flash_attn_prefill_paged = None
_flash_attn_prefill_paged_bhmd = None
_flash_attn_prefill_paged_bfla = None
_flash_attn_prefill_paged_splitkv = None
_sm70_splitd_d256_ops = None
_sm70_splitd_d256_ops_checked = False
_sm70_d256_gqa_architecture_op = None
_sm70_d256_gqa_architecture_op_checked = False
_sm70_d256_gqa_architecture_q8192_op = None
_sm70_d256_gqa_architecture_q8192_op_checked = False
_fp8_e5m2_paged_kv_to_fp16 = None
_fp8_e5m2_paged_kv_to_fp16_checked = False
_flash_attn_turboquant_decode_paged = None
_flash_attn_turboquant_decode_checked = False
_paged_kv_utils = None


def callable_accepts_keyword(fn: object, name: str) -> bool:
    if not callable(fn):
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )


def get_flash_ops():
    """Lazy-load flash_attn_v100 ops if available."""
    global _flash_attn_func, _flash_attn_bhmd_func
    global _flash_attn_decode_paged, _flash_attn_decode_paged_xqa
    global _flash_attn_prefill_paged
    global _flash_attn_decode_paged_wmma, _flash_attn_prefill_paged_bhmd
    global _flash_attn_prefill_paged_bfla, _flash_attn_prefill_paged_splitkv
    if (
        _flash_attn_func is None
        or _flash_attn_decode_paged is None
        or _flash_attn_prefill_paged is None
    ):
        try:
            from flash_attn_v100 import (
                flash_attn_bhmd_func,
                flash_attn_decode_paged,
                flash_attn_func,
                flash_attn_prefill_paged,
            )

            _flash_attn_func = flash_attn_func
            _flash_attn_bhmd_func = flash_attn_bhmd_func
            _flash_attn_decode_paged = flash_attn_decode_paged
            _flash_attn_prefill_paged = flash_attn_prefill_paged
            try:
                from flash_attn_v100 import flash_attn_decode_paged_xqa

                _flash_attn_decode_paged_xqa = flash_attn_decode_paged_xqa
            except ImportError:
                _flash_attn_decode_paged_xqa = None
            try:
                from flash_attn_v100 import flash_attn_decode_paged_wmma

                _flash_attn_decode_paged_wmma = flash_attn_decode_paged_wmma
            except ImportError:
                _flash_attn_decode_paged_wmma = None
            try:
                from flash_attn_v100 import flash_attn_prefill_paged_bhmd

                _flash_attn_prefill_paged_bhmd = flash_attn_prefill_paged_bhmd
            except ImportError:
                _flash_attn_prefill_paged_bhmd = None
            try:
                from flash_attn_v100 import flash_attn_prefill_paged_bfla

                _flash_attn_prefill_paged_bfla = flash_attn_prefill_paged_bfla
            except ImportError:
                _flash_attn_prefill_paged_bfla = None
            try:
                from flash_attn_v100 import flash_attn_prefill_paged_splitkv

                _flash_attn_prefill_paged_splitkv = flash_attn_prefill_paged_splitkv
            except ImportError:
                _flash_attn_prefill_paged_splitkv = None
        except ImportError as exc:
            logger.warning_once(
                "Flash-V100 Python operators could not be imported (%s: %s).",
                type(exc).__name__,
                exc,
            )
            _flash_attn_func = None
            _flash_attn_bhmd_func = None
            _flash_attn_decode_paged = None
            _flash_attn_decode_paged_xqa = None
            _flash_attn_decode_paged_wmma = None
            _flash_attn_prefill_paged = None
            _flash_attn_prefill_paged_bhmd = None
            _flash_attn_prefill_paged_bfla = None
            _flash_attn_prefill_paged_splitkv = None
    if _flash_attn_func is not None:
        prepare_attention_runtime()
    return tuple(
        bind_attention_operation(operation)
        for operation in (
            _flash_attn_func,
            _flash_attn_bhmd_func,
            _flash_attn_decode_paged,
            _flash_attn_decode_paged_xqa,
            _flash_attn_decode_paged_wmma,
            _flash_attn_prefill_paged,
            _flash_attn_prefill_paged_bhmd,
            _flash_attn_prefill_paged_bfla,
            _flash_attn_prefill_paged_splitkv,
        )
    )


def get_flash_grouped_verify_op():
    """Load the optional exact SM70 grouped verifier."""
    global _flash_attn_grouped_verify_paged
    global _flash_attn_grouped_verify_max_query_tokens
    global _flash_attn_grouped_verify_request_major_abi_version
    global _flash_attn_grouped_verify_checked
    if _flash_attn_grouped_verify_checked:
        return bind_attention_operation(_flash_attn_grouped_verify_paged)

    _flash_attn_grouped_verify_checked = True
    try:
        from flash_attn_v100 import flash_attn_grouped_verify_paged

        _flash_attn_grouped_verify_paged = flash_attn_grouped_verify_paged
        try:
            from flash_attn_v100 import (
                flash_attn_grouped_verify_max_query_tokens,
            )

            _flash_attn_grouped_verify_max_query_tokens = int(
                flash_attn_grouped_verify_max_query_tokens()
            )
        except (ImportError, RuntimeError, TypeError, ValueError):
            _flash_attn_grouped_verify_max_query_tokens = 8
        try:
            from flash_attn_v100 import (
                flash_attn_grouped_verify_request_major_abi_version,
            )

            _flash_attn_grouped_verify_request_major_abi_version = int(
                flash_attn_grouped_verify_request_major_abi_version()
            )
        except (ImportError, RuntimeError, TypeError, ValueError):
            _flash_attn_grouped_verify_request_major_abi_version = 0
    except ImportError:
        _flash_attn_grouped_verify_paged = None
    return bind_attention_operation(_flash_attn_grouped_verify_paged)


def get_sm70_splitd_d256_ops():
    """Load the exact SM70 Split-D dense and paged prefill operators."""
    global _sm70_splitd_d256_ops
    global _sm70_splitd_d256_ops_checked
    if _sm70_splitd_d256_ops_checked:
        return _sm70_splitd_d256_ops

    _sm70_splitd_d256_ops_checked = True
    try:
        required_ops = (
            "sm70_d256_splitd_n32_dense_fwd",
            "sm70_d256_splitd_n32_paged_fwd",
        )
        with suppress(ImportError):
            # The FA2 library loads on first use, one per process and chosen
            # for the worker's device; make sure it is there before the
            # operators are resolved.
            from vllm.vllm_flash_attn.flash_attn_interface import (
                ensure_fa2_library_loaded,
            )

            ensure_fa2_library_loaded()

        namespace = getattr(torch.ops, "_vllm_fa2_C", None)
        if namespace is None or not all(
            hasattr(namespace, op_name) for op_name in required_ops
        ):
            # A partially cached Python interface can import successfully
            # without registering its native operators. Source-overlay
            # deployments can also intentionally keep the extension outside
            # the checkout. In both cases, load only an explicitly selected
            # sidecar and then validate the actual operator capability below.
            library_path = _config.raw("VLLM_SM70_FA2_D256_LIBRARY")
            if library_path is not None:
                torch.ops.load_library(library_path)
                logger.info(
                    "Loaded external SM70 D256 prefill library from %s.",
                    library_path,
                )

        dense = torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_dense_fwd
        paged = torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_paged_fwd
        splitkv3 = getattr(
            torch.ops._vllm_fa2_C,
            "sm70_d256_splitd_n32_dense_splitkv3_fwd",
            None,
        )
        _sm70_splitd_d256_ops = (dense, paged, splitkv3)
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        _sm70_splitd_d256_ops = None
        logger.warning_once(
            "SM70 D256 exact-prefill operators are unavailable (%s: %s). "
            "Long prefill will use a slower fallback. Verify that the active "
            "vllm package contains a loadable _vllm_fa2_C extension with the "
            "sm70_d256_splitd_n32_dense_fwd and "
            "sm70_d256_splitd_n32_paged_fwd operators.",
            type(exc).__name__,
            exc,
        )
    return _sm70_splitd_d256_ops


def _sm70_gqa_has_fp32_accumulation() -> bool:
    capability = getattr(torch.ops._vllm_fa2_C, "sm70_d256_gqa_accumulation_bits", None)
    if capability is not None and capability() == 32:
        return True
    logger.warning_once(
        "SM70 Q8000/Q8192 prefill requires rebuilt FA2 with FP32 QK and PV "
        "accumulation; using exact dense prefill until the library is updated."
    )
    return False


_sm70_gqa_capabilities: dict[str, Callable | None] = {}


def get_sm70_d256_gqa_architecture_op():
    """Select by engine policy; cache only immutable native capabilities."""
    global _sm70_d256_gqa_architecture_op_checked
    if not _sm70_d256_gqa_architecture_op_checked:
        _sm70_gqa_capabilities.clear()
        _sm70_d256_gqa_architecture_op_checked = True
    use_v37 = _config.options().value("prefill_d256_gqa_v37")
    op_name = "sm70_d256_gqa_v37_fwd" if use_v37 else "sm70_d256_gqa_architecture_fwd"
    if op_name in _sm70_gqa_capabilities:
        return _sm70_gqa_capabilities[op_name]
    operation = None
    try:
        if not hasattr(torch.ops._vllm_fa2_C, op_name):
            get_sm70_splitd_d256_ops()
        if not use_v37 and not _sm70_gqa_has_fp32_accumulation():
            _sm70_gqa_capabilities[op_name] = None
            return None
        operation = getattr(torch.ops._vllm_fa2_C, op_name, None)
        if operation is None:
            logger.warning_once(
                "Requested SM70 GQA operator %s is absent; rebuild FA2. "
                "Using exact dense prefill, not relabelling the old kernel.",
                op_name,
            )
    except (AttributeError, ImportError, RuntimeError) as exc:
        if _config.options().value("prefill_d256_gqa_arch_128k_experimental"):
            logger.warning_once(
                "SM70 D256 GQA architecture operator is unavailable "
                "(%s: %s); using the exact dense prefill kernel.",
                type(exc).__name__,
                exc,
            )
    _sm70_gqa_capabilities[op_name] = operation
    return operation


def get_sm70_d256_gqa_architecture_q8192_op():
    """Load the native Q8192 specialization when the extension provides it."""
    global _sm70_d256_gqa_architecture_q8192_op
    global _sm70_d256_gqa_architecture_q8192_op_checked
    if _sm70_d256_gqa_architecture_q8192_op_checked:
        return _sm70_d256_gqa_architecture_q8192_op

    _sm70_d256_gqa_architecture_q8192_op_checked = True
    op_name = "sm70_d256_gqa_architecture_q8192_fwd"
    try:
        if not hasattr(torch.ops._vllm_fa2_C, op_name):
            get_sm70_splitd_d256_ops()
        if not _sm70_gqa_has_fp32_accumulation():
            _sm70_d256_gqa_architecture_q8192_op = None
            return None
        _sm70_d256_gqa_architecture_q8192_op = getattr(
            torch.ops._vllm_fa2_C,
            op_name,
            None,
        )
    except (AttributeError, ImportError, RuntimeError):
        _sm70_d256_gqa_architecture_q8192_op = None
    return _sm70_d256_gqa_architecture_q8192_op


def get_sm70_v37_e4m3_bridge_op():
    """Resolve the format-specific bridge from the same FA2 runtime."""
    # E4M3 storage conversion is independent of the dense compute kernel.
    get_sm70_splitd_d256_ops()
    return getattr(torch.ops._vllm_fa2_C, "sm70_v37_e4m3_bridge", None)


def get_fp8_e5m2_paged_kv_bridge_op():
    global _fp8_e5m2_paged_kv_to_fp16
    global _fp8_e5m2_paged_kv_to_fp16_checked
    if not _fp8_e5m2_paged_kv_to_fp16_checked:
        _fp8_e5m2_paged_kv_to_fp16_checked = True
        try:
            from flash_attn_v100 import fp8_e5m2_paged_kv_to_fp16

            _fp8_e5m2_paged_kv_to_fp16 = fp8_e5m2_paged_kv_to_fp16
        except ImportError:
            _fp8_e5m2_paged_kv_to_fp16 = None
    return _fp8_e5m2_paged_kv_to_fp16


# MLA context-chunk prefill needs the LSE that the dense SM70 kernel already
# computes, plus independent Q and K sequence metadata for M != N attention.
_flash_attn_forward_lse: Callable[..., tuple[torch.Tensor, ...]] | None = None
_flash_attn_forward_lse_checked = False


def get_flash_dense_forward() -> Callable[..., tuple[torch.Tensor, ...]] | None:
    """Lazy-load the private LSE-capable forward entry of the FA-V100 wheel."""
    global _flash_attn_forward_lse, _flash_attn_forward_lse_checked
    if not _flash_attn_forward_lse_checked:
        _flash_attn_forward_lse_checked = True
        try:
            from flash_attn_v100.flash_attn_interface import _flash_attn_forward

            _flash_attn_forward_lse = _flash_attn_forward
        except (ImportError, AttributeError):
            _flash_attn_forward_lse = None
    return bind_attention_operation(_flash_attn_forward_lse)


def _get_flash_turboquant_decode_op():
    """Lazy-load the optional TurboQuant decode op without touching base ops."""
    global _flash_attn_turboquant_decode_checked
    global _flash_attn_turboquant_decode_paged
    if not _flash_attn_turboquant_decode_checked:
        try:
            from flash_attn_v100 import (
                flash_attn_turboquant_decode_paged,
                flash_attn_turboquant_decode_paged_available,
            )

            if flash_attn_turboquant_decode_paged_available():
                _flash_attn_turboquant_decode_paged = flash_attn_turboquant_decode_paged
            else:
                _flash_attn_turboquant_decode_paged = None
        except ImportError:
            _flash_attn_turboquant_decode_paged = None
        _flash_attn_turboquant_decode_checked = True
    return bind_attention_operation(_flash_attn_turboquant_decode_paged)


def flash_v100_turboquant_decode_available() -> bool:
    return _get_flash_turboquant_decode_op() is not None


def flash_v100_turboquant_decode(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    softmax_scale: float,
    mse_bits: int,
    value_quant_bits: int,
    norm_correction: bool,
    num_kv_splits: int,
) -> torch.Tensor:
    """Run Flash-V100 decode directly over TurboQuant packed paged cache."""
    op = _get_flash_turboquant_decode_op()
    if op is None:
        raise RuntimeError("flash_attn_v100 TurboQuant decode op is unavailable")
    return op(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        softmax_scale=softmax_scale,
        out=output,
        mse_bits=mse_bits,
        value_quant_bits=value_quant_bits,
        norm_correction=norm_correction,
        num_kv_splits=num_kv_splits,
    )


def get_paged_kv_utils():
    """Lazy-load paged KV extraction CUDA extension."""
    global _paged_kv_utils
    if _paged_kv_utils is None:
        try:
            from flash_attn_v100 import paged_kv_utils  # type: ignore[attr-defined]

            _paged_kv_utils = paged_kv_utils
        except ImportError:
            try:
                import paged_kv_utils

                _paged_kv_utils = paged_kv_utils
            except ImportError:
                _paged_kv_utils = None
    return _paged_kv_utils


# Public owner operations; legacy bindings are installed by package assembly.
LEGACY_ALIASES = {
    "_get_paged_kv_utils": "get_paged_kv_utils",
    "_get_fp8_e5m2_paged_kv_bridge_op": "get_fp8_e5m2_paged_kv_bridge_op",
    "_get_sm70_v37_e4m3_bridge_op": "get_sm70_v37_e4m3_bridge_op",
    "_get_flash_dense_forward": "get_flash_dense_forward",
    "_get_sm70_d256_gqa_architecture_op": "get_sm70_d256_gqa_architecture_op",
    "_get_sm70_splitd_d256_ops": "get_sm70_splitd_d256_ops",
    "_callable_accepts_keyword": "callable_accepts_keyword",
    "_get_sm70_d256_gqa_architecture_q8192_op": (
        "get_sm70_d256_gqa_architecture_q8192_op"
    ),
    "_get_flash_grouped_verify_op": "get_flash_grouped_verify_op",
    "_get_flash_ops": "get_flash_ops",
}


if TYPE_CHECKING:
    # Static compatibility only; runtime writes use live owner aliases.
    _get_paged_kv_utils = get_paged_kv_utils
    _get_fp8_e5m2_paged_kv_bridge_op = get_fp8_e5m2_paged_kv_bridge_op
    _get_sm70_v37_e4m3_bridge_op = get_sm70_v37_e4m3_bridge_op
    _get_flash_dense_forward = get_flash_dense_forward
    _get_sm70_d256_gqa_architecture_op = get_sm70_d256_gqa_architecture_op
    _get_sm70_splitd_d256_ops = get_sm70_splitd_d256_ops
    _callable_accepts_keyword = callable_accepts_keyword
    _get_sm70_d256_gqa_architecture_q8192_op = get_sm70_d256_gqa_architecture_q8192_op
    _get_flash_grouped_verify_op = get_flash_grouped_verify_op
    _get_flash_ops = get_flash_ops
