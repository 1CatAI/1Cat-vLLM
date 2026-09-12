# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit compact E4M3 scalar attention for SM70 target tail graphs."""

from functools import lru_cache

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.v1.attention.ops.sm70_e4m3_long import (
    BUILTIN_MAX_CONTEXT,
    load_attention_library,
    long_attention_capability,
    resolve_long_attention,
)

logger = init_logger(__name__)


# The compact scalar tail op ships inside the FA2 extension, so it needs no
# manifest and no environment variable. The manifest stays as an explicit
# override for an unqualified experimental candidate.
BUILTIN_SCALAR_OP = "sm70_scalar_attention_fwd"
BUILTIN_SCALAR_MANIFEST = {
    "module_name": "_vllm_fa2_C",
    "library_sha256": BUILTIN_SCALAR_OP,
    "share_kv_six_heads": True,
    "compact_page_map": True,
    "e4m3_lut": True,
    # Capacity this kernel was qualified for, in the same units as the
    # long-context contract it is admitted against.
    "max_context": BUILTIN_MAX_CONTEXT,
}


@lru_cache(maxsize=1)
def builtin_scalar_tail_attention():
    try:
        return getattr(torch.ops._vllm_fa2_C, BUILTIN_SCALAR_OP)
    except AttributeError:
        return None


def scalar_tail_attention_available() -> bool:
    return builtin_scalar_tail_attention() is not None


@lru_cache(maxsize=4)
def load_scalar_tail_attention(manifest_name: str, device: torch.device):
    if manifest_name:
        module, manifest = load_attention_library(manifest_name)
    else:
        operator = builtin_scalar_tail_attention()
        if operator is None:
            raise ValueError("The compact scalar tail operator is not compiled in")
        builtin = type("_Builtin", (), {"run": operator})
        module, manifest = builtin, BUILTIN_SCALAR_MANIFEST
    # The tail rides the long-context graph variant, so it is only meaningful
    # while that route is on, and the kernel has to cover every bound the graph
    # can be captured at -- which never exceeds the long-context capability.
    _, long_manifest = resolve_long_attention()
    if long_manifest is None:
        logger.info_once(
            "SM70 compact scalar tail attention is inactive: the E4M3 "
            "long-context route is disabled.",
            scope="process",
        )
        return None
    capability = long_attention_capability(long_manifest)
    if not (
        manifest.get("share_kv_six_heads")
        and manifest.get("compact_page_map")
        and manifest.get("e4m3_lut")
        and manifest.get("max_context", 0) >= capability
    ):
        raise ValueError(
            "Scalar tails require the compact E4M3 manifest covering the "
            f"{capability}-token long-context capability"
        )
    # Allocate before memory profiling and graph capture. Model layers execute
    # serially on the worker stream; all target graphs retain this same storage.
    buffers = (
        torch.empty((1, 6, 256, 256), dtype=torch.float32, device=device),
        torch.empty((1, 6, 256), dtype=torch.float32, device=device),
        torch.empty((1, 6, 256), dtype=torch.float32, device=device),
        torch.full((1,), 256, dtype=torch.int32, device=device),
    )
    logger.info_once(
        "Loaded SM70 compact scalar tail attention: module=%s SHA256=%s; "
        "256 partitions, FP32 numerator/max/sum.",
        manifest["module_name"],
        manifest["library_sha256"],
        scope="process",
    )

    def run(
        q,
        k,
        v,
        table,
        lengths,
        *,
        out,
        softmax_scale,
        k_scale,
        v_scale,
        kv_cache_dtype,
        window_size,
        max_seq_len_hint,
        partition_size_hint,
        anchor_lens,
        anchored_window,
    ) -> bool:
        descriptor = (
            get_forward_context().batch_descriptor
            if is_forward_context_available()
            else None
        )
        bucket = getattr(descriptor, "attention_context_bucket", None)
        if not (
            bucket is not None
            and bucket <= capability
            and q.shape == (1, 6, 256)
            and q.dtype == torch.float16
            and q.device == device
            and q.is_contiguous()
            and k.ndim == 4
            and k.shape[1:] == (3296, 1, 256)
            and k.dtype == v.dtype == torch.uint8
            and v.shape == k.shape
            and kv_cache_dtype == "fp8_e4m3"
            and window_size == (-1, -1)
            and anchor_lens is None
            and anchored_window == 0
            and partition_size_hint in (None, 1024)
            and type(max_seq_len_hint) is int
            and 0 < max_seq_len_hint <= capability
        ):
            return False
        module.run(
            q,
            k,
            v,
            out,
            table,
            lengths,
            *buffers,
            float(softmax_scale),
            float(k_scale),
            float(v_scale),
        )
        return True

    return run
