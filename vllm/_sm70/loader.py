# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native library discovery and historical load order."""

import os
from pathlib import Path

import torch

from vllm.platforms import current_platform

current_platform.import_kernels()


def _maybe_load_fp8_qpn8_library() -> None:
    """Load an explicitly selected source-built QPN8 extension.

    Production builds register these operators in ``vllm._C``. This opt-in
    path lets source experiments add only the QPN8 operators to an otherwise
    compatible installed build, including in spawned TP workers.
    """
    library_path = os.getenv("VLLM_SM70_FP8_QPN8_LIBRARY")
    if library_path is None:
        return
    generic_override = os.getenv("VLLM_SM70_FP8_QPN8")
    specific_override = os.getenv("VLLM_SM70_FP8_QPN8_PP2_TP4")
    online_override = os.getenv("VLLM_SM70_QWEN4_EXP_ONLINE_QPN8")
    generic_enabled = generic_override == "1"
    specific_enabled = generic_override != "0" and specific_override == "1"
    online_enabled = online_override == "1"
    if generic_enabled or specific_enabled or online_enabled:
        torch.ops.load_library(library_path)


_maybe_load_fp8_qpn8_library()


def _nvfp4_qpn2_prefill_library_path() -> str | None:
    library_path = os.getenv("VLLM_SM70_NVFP4_QPN2_PREFILL_LIBRARY")
    if library_path is not None:
        return library_path
    bundled = sorted(
        Path(__file__).resolve().parents[1].glob("_sm70_nvfp4_qpn2_prefill_C*.so")
    )
    return str(bundled[-1]) if bundled else None


def has_deferred_nvfp4_qpn2_prefill_library() -> bool:
    """Return whether a separate large-M prefill fragment is configured."""
    return _nvfp4_qpn2_prefill_library_path() is not None


def load_deferred_nvfp4_qpn2_prefill_library() -> bool:
    """Load an isolated large-M QPN2-packed prefill fragment.

    Source-overlay deployments can retain a previously validated decode
    extension while adding the newer, large-M-only QPN2-packed prefill op.
    The op owns its temporary dense workspace, so registration before AOT
    prefill compilation does not retain the large buffer during decode graph
    capture.
    Production wheels normally link the op into ``vllm._C``; a bundled
    fragment is only used when the main extension does not provide it.
    """
    if hasattr(torch.ops._C, "nvfp4_qpn4_prefill_sm70_out"):
        return True

    library_path = _nvfp4_qpn2_prefill_library_path()
    if library_path is not None:
        torch.ops.load_library(library_path)
    return hasattr(torch.ops._C, "nvfp4_qpn4_prefill_sm70_out")


load_deferred_nvfp4_qpn2_prefill_library()


def _maybe_load_nvfp4_qpn_m1_library() -> None:
    """Load the narrow Qwen3.8 NVFP4 experiment in spawned TP workers."""
    library_path = os.getenv("VLLM_SM70_NVFP4_QPN_M1_LIBRARY")
    m1_enabled = os.getenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE", "1") != "0"
    batch_enabled = os.getenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE", "1") != "0"
    mtp5_enabled = os.getenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE", "0") != "0"
    route_enabled = m1_enabled or batch_enabled or mtp5_enabled
    if library_path is not None and route_enabled:
        torch.ops.load_library(library_path)


_maybe_load_nvfp4_qpn_m1_library()


def _maybe_load_glm53_fp16_gemv_library() -> None:
    """Load the exact GLM-5.3 decode GEMV during source-side validation."""
    if hasattr(torch.ops._C, "sm70_glm53_fp16_gemv_out"):
        return
    library_path = os.getenv("VLLM_SM70_GLM53_FP16_GEMV_LIBRARY")
    if library_path is not None:
        torch.ops.load_library(library_path)


_maybe_load_glm53_fp16_gemv_library()


def _maybe_load_sm70_sampler_library() -> None:
    """Load the sampler fragment only when the main ``vllm._C`` lacks it."""
    if hasattr(torch.ops._C, "sm70_sample_chunked_top20_philox_token_out"):
        return

    library_path = os.getenv("VLLM_SM70_SAMPLER_LIBRARY")
    if library_path is None:
        bundled = sorted(
            Path(__file__).resolve().parents[1].glob("_sm70_sampler_C*.so")
        )
        if bundled:
            library_path = str(bundled[-1])
    if library_path is not None:
        torch.ops.load_library(library_path)


_maybe_load_sm70_sampler_library()
