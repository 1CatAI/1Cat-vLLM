# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compensated SM70 attention with explicit native capability contracts."""

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import torch

from vllm.config.execution_policy import flash_v100_options
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.v1.attention.ops.sm70_grouped import model_context_limit

logger = init_logger("vllm.v1.attention.ops.sm70_e4m3_long")

# Compatibility capacity of binaries predating the dynamic-context ABI.
BUILTIN_MAX_CONTEXT = 262144
MANIFEST_ENV = "VLLM_SM70_E4M3_LONG_ATTENTION_MANIFEST"
DISABLE_ENV = "VLLM_SM70_E4M3_LONG_ATTENTION"
_WORKSPACES: dict[tuple, list[tuple[torch.Tensor, torch.Tensor]]] = {}


# The grouped long-context route is compiled into the shipped FA2 extension, so
# it is available without any environment variable. The manifest variable stays
# as an explicit override for an unqualified experimental candidate.
BUILTIN_OP = "sm70_grouped_long_fwd"
# The compiled operator carries the probability-swizzle / early-store layout the
# route was qualified with. The source digest names that layout so a rebuilt
# kernel cannot silently inherit the workspace identity of a different one.
BUILTIN_SOURCE_SHA256 = (
    "4d1841d7d0578656d3bdc20d15f26f854a77acd5138d25998f01fe1016a24372"
)
# Every B1 verifier tail width uses the long-context graph contract at this
# capacity; q1 additionally has the compact scalar tail.
BUILTIN_QUERY_ROWS = (2, 3, 4, 5, 6, 7, 8)
BUILTIN_MANIFEST = {
    "module_name": "_vllm_fa2_C",
    "source_sha256": BUILTIN_SOURCE_SHA256,
    "splits": 80,
    "head_groups": 1,
    "max_context": BUILTIN_MAX_CONTEXT,
    "query_rows": list(BUILTIN_QUERY_ROWS),
}


@lru_cache(maxsize=1)
def builtin_long_attention():
    try:
        # The FA2 library loads on first use; resolve it before the lookup.
        from vllm.vllm_flash_attn.flash_attn_interface import (
            ensure_fa2_library_loaded,
        )

        ensure_fa2_library_loaded()
        return getattr(torch.ops._vllm_fa2_C, BUILTIN_OP)
    except (AttributeError, ImportError):
        return None


# Explicit opt-out. The accelerated route is on by default, so an operator needs
# a way back to the full-context route without rebuilding, and the paired A/B
# validation needs both arms from one build. An explicit off wins over a
# manifest override.
DISABLE_VALUES = {"0", "false", "no", "off"}


def long_attention_enabled(policy=None) -> bool:
    policy = flash_v100_options() if policy is None else policy
    if not policy.value("e4m3_long_enabled"):
        return False
    return bool(policy.e4m3_long_manifest) or builtin_long_attention() is not None


@lru_cache(maxsize=4)
def load_attention_library(manifest_name: str):
    manifest_path = Path(manifest_name).resolve()
    manifest = json.loads(manifest_path.read_text())
    library = Path(manifest["library"])
    if not library.is_absolute():
        library = manifest_path.parent / library
    library = library.resolve()
    if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["library_sha256"]:
        raise ValueError(
            "Long-attention native library SHA does not match its manifest"
        )
    name = library.name.split(".")[0]
    if name != manifest["module_name"]:
        raise ValueError(
            "Long-attention native module name does not match its manifest"
        )
    spec = importlib.util.spec_from_file_location(name, library)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load long-attention extension {library}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded_file = module.__file__
    if loaded_file is None or Path(loaded_file).resolve() != library:
        raise RuntimeError("Long-attention native extension module alias")
    return module, manifest


@lru_cache(maxsize=1)
def load_long_attention(manifest_name: str):
    module, manifest = load_attention_library(manifest_name)
    # Split counts change both arithmetic and workspace geometry.
    if manifest.get("splits", 80) != 80 or manifest["head_groups"] != 1:
        raise ValueError("The long-attention serving route requires 80 six-head splits")
    context_limit, query_rows = long_attention_contract(manifest)
    logger.info_once(
        "Loaded experimental SM70 E4M3 attention: module=%s SHA256=%s "
        "max_context=%d query_rows=%s; 80 splits, compensated FP32 state.",
        manifest["module_name"],
        manifest["library_sha256"],
        context_limit,
        query_rows,
        scope="process",
    )
    return module.run, manifest


def long_attention_capability(manifest) -> int:
    """The context the operator was qualified for, as its manifest declares it."""
    capability = manifest.get("max_context", BUILTIN_MAX_CONTEXT)
    if type(capability) is not int or capability <= 0:
        raise ValueError("Unsupported long-attention operator capacity")
    return capability


def builtin_dynamic_context(*, scalar: bool = False) -> bool:
    """The loaded DSO, including each independently built operator, owns its ABI."""
    name = (
        "sm70_scalar_attention_workspace_revision"
        if scalar
        else "sm70_grouped_long_context_revision"
    )
    revision = (
        getattr(torch.ops._vllm_fa2_C, name, None)
        if builtin_long_attention() is not None
        else None
    )
    return revision is not None and revision() >= 1


def builtin_context_limit(capacity: int | None = None, *, scalar: bool = False) -> int:
    """Resolve the served window against the loaded operator's workspace ABI."""
    capacity = model_context_limit(capacity, fallback=BUILTIN_MAX_CONTEXT)
    return (
        capacity
        if builtin_dynamic_context(scalar=scalar)
        else min(capacity, BUILTIN_MAX_CONTEXT)
    )


def context_table_supported(table, page_size: int, capacity: int) -> bool:
    """The graph's served window admits its page-rounded table, never wider."""
    return (
        table.ndim == 2
        and page_size > 0
        and 0 < table.shape[1] <= (capacity + page_size - 1) // page_size
    )


def long_attention_query_rows(manifest) -> tuple[int, ...]:
    """The query widths the route admits, mirroring the native [2, 8] check."""
    query_rows = tuple(manifest.get("query_rows", [8]))
    if not query_rows or any(type(q) is not int or not 2 <= q <= 8 for q in query_rows):
        raise ValueError("Unsupported long-attention query-row contract")
    return query_rows


def long_attention_contract(manifest, capacity: int | None = None):
    """The bound the long-context graph is captured at, and its query widths.

    ``capacity`` is the context the model is served with. The bound is the
    smaller of that and the operator's declared capability, so the served window
    decides the route and the operator is never driven past what it covers.
    ``None`` means no service window is known, which leaves the capability.
    """
    capability = long_attention_capability(manifest)
    if capacity is None:
        return capability, long_attention_query_rows(manifest)
    return (
        max(min(capability, int(capacity)), 1),
        long_attention_query_rows(manifest),
    )


def resolve_long_attention(policy=None, capacity: int | None = None):
    """The route in effect: an explicit manifest candidate, else the shipped one."""
    policy = flash_v100_options() if policy is None else policy
    if not long_attention_enabled(policy):
        return None, None
    manifest_name = policy.e4m3_long_manifest
    if manifest_name:
        return load_long_attention(manifest_name)
    operator = builtin_long_attention()
    if operator is None:
        return None, None
    capability = builtin_context_limit(capacity)
    manifest = dict(BUILTIN_MANIFEST, max_context=capability)
    logger.info_once(
        "Using the shipped SM70 E4M3 long-context route (%s): capability=%d "
        "query_rows=%s; 80 splits, compensated FP32 state.",
        BUILTIN_OP,
        capability,
        BUILTIN_QUERY_ROWS,
        scope="process",
    )
    revision = getattr(
        torch.ops._vllm_fa2_C, "sm70_grouped_long_query_rows_revision", None
    )
    if revision is not None:
        manifest = {**manifest, "query_rows_revision": int(revision())}
    kv_revision = getattr(
        torch.ops._vllm_fa2_C, "sm70_grouped_long_kv_heads_revision", None
    )
    if kv_revision is not None:
        manifest = {**manifest, "kv_heads_revision": int(kv_revision())}
    return operator, manifest


def long_attention_graph_contract(capacity: int | None = None, *, policy=None):
    """The captured bound for ``capacity``, or ``(None, ())`` when route is off."""
    _, manifest = resolve_long_attention(policy, capacity)
    if manifest is None:
        return None, ()
    return long_attention_contract(manifest, capacity)


def long_attention_max_batch_size(manifest=None, *, policy=None) -> int:
    """Batch capacity of the loaded operator; old libraries remain B1-only."""
    if manifest is None:
        _, manifest = resolve_long_attention(policy)
    if manifest is None:
        return 1
    if manifest["module_name"] == "_vllm_fa2_C":
        query = getattr(torch.ops._vllm_fa2_C, "sm70_grouped_long_max_batch_size", None)
        return int(query()) if query is not None else 1
    batch_size = manifest.get("max_batch_size", 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Unsupported long-attention batch contract")
    return batch_size


def long_attention_batch_query_rows(manifest=None, *, policy=None) -> tuple[int, ...]:
    if manifest is None:
        _, manifest = resolve_long_attention(policy)
    if manifest is None:
        return ()
    rows = long_attention_query_rows(manifest)
    if manifest.get("query_rows_revision", 1) >= 2:
        return rows
    return tuple(row for row in rows if row == 8)


# Keep the qualified fixed pages for older native libraries. Rebuilt libraries
# also expose the existing runtime-page kernel for positive 16-aligned pages.
ADMITTED_PAGE_SIZES = (1648, 3296)
_REPORTED_PAGE_SIZES: set[int] = set()


def long_attention_page_supported(page_size: int, manifest: dict) -> bool:
    if page_size in ADMITTED_PAGE_SIZES:
        return True
    if page_size <= 0 or page_size % 16 or manifest["module_name"] != "_vllm_fa2_C":
        return False
    revision = getattr(torch.ops._vllm_fa2_C, "sm70_grouped_long_page_revision", None)
    return revision is not None and revision() >= 1


def _report_unadmitted_page_size(page_size: int) -> None:
    if page_size in _REPORTED_PAGE_SIZES:
        return
    _REPORTED_PAGE_SIZES.add(page_size)
    logger.warning(
        "SM70 E4M3 long-context route declined: the paged-KV page size is %d "
        "tokens. Rebuild the shipped operator for positive 16-aligned pages; "
        "older libraries retain fixed pages %s. Using the ordinary path.",
        page_size,
        ADMITTED_PAGE_SIZES,
    )


def wrap_long_attention(fallback):
    from vllm.config import get_current_vllm_config_or_none

    operator, manifest = resolve_long_attention()
    if operator is None:
        return fallback
    capability = long_attention_capability(manifest)
    query_rows = long_attention_query_rows(manifest)
    max_batch_size = long_attention_max_batch_size(manifest)
    request_query_rows = long_attention_batch_query_rows(manifest)
    config = get_current_vllm_config_or_none()
    model = getattr(config, "model_config", None)
    model_capacity = getattr(model, "max_model_len", None)
    variable_request_rows = manifest.get("query_rows_revision", 1) >= 2
    parallel_kv_heads = manifest.get("kv_heads_revision", 0) >= 1

    def run(
        q, k, v, table, row_lengths, *, out, softmax_scale, k_scale=1.0, v_scale=1.0
    ):
        descriptor = (
            get_forward_context().batch_descriptor
            if is_forward_context_available()
            else None
        )
        bucket = getattr(descriptor, "attention_context_bucket", None)
        batch_size = int(table.shape[0]) if table.ndim == 2 else 0
        num_rows = q.shape[0] if q.ndim == 3 else 0
        if bucket is None and variable_request_rows and table.ndim == 2 and k.ndim == 4:
            # Bind the loaded model's window once. The physical table supplies
            # a conservative bound for direct callers without a model config.
            bucket = model_capacity or int(table.shape[1]) * int(k.shape[1])
        request_major_batch = (
            2 <= batch_size <= max_batch_size
            and num_rows % batch_size == 0
            and num_rows // batch_size in request_query_rows
        )
        single_request_tail = batch_size == 1 and num_rows in query_rows
        # The graph builder owns the served window: it stamps the bound it
        # captured the variant at onto the descriptor. The wrapper only has to
        # refuse a bound the operator was not qualified for.
        if not (
            bucket is not None
            and bucket <= capability
            and q.ndim == 3
            and (single_request_tail or request_major_batch)
            and q.shape[1] > 0
            and q.shape[2] == 256
            and k.ndim == 4
            and context_table_supported(table, k.shape[1], capability)
            and table.is_contiguous()
            and row_lengths.ndim == 1
            and row_lengths.shape[0] == q.shape[0]
            and long_attention_page_supported(k.shape[1], manifest)
            and k.shape[2] * 6 == q.shape[1]
            and k.shape[3] == 256
            and v.shape == k.shape
        ):
            if k.ndim == 4 and not long_attention_page_supported(k.shape[1], manifest):
                _report_unadmitted_page_size(int(k.shape[1]))
            return fallback(
                q,
                k,
                v,
                table,
                row_lengths,
                out=out,
                softmax_scale=softmax_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        # Allocate a fixed workspace once for each warmup/capture stream. Graph
        # replay never allocates. Layers reuse it in stream order; different
        # streams and versions never share the legacy 80-split buffers.
        if request_major_batch:
            logger.info_once(
                "SM70 E4M3 long request-major route selected: batch=%d "
                "query_rows=%d page=%d bound=%d; FP32 state.",
                batch_size,
                num_rows // batch_size,
                k.shape[1],
                bucket,
                scope="process",
            )
        stream = torch.cuda.current_stream(q.device).cuda_stream
        key = (manifest["source_sha256"], bucket, 80, q.device, stream)
        bank = _WORKSPACES.setdefault(key, [])
        panels = batch_size * (k.shape[2] if parallel_kv_heads else 1)
        # Full graphs replay in stream order, just like successive layers.
        # Reuse a larger request panel for smaller captures. Keep older panels
        # alive when growing: an earlier graph may still reference them.
        for partial, lse in bank:
            if partial.shape[0] >= panels:
                break
        else:
            partial = torch.empty(
                (panels, 80, 8, 6, 256), dtype=torch.float32, device=q.device
            )
            lse = torch.empty(
                (panels, 80, 8, 6, 2), dtype=torch.float32, device=q.device
            )
            bank.append((partial, lse))
        if panels == 1:
            partial, lse = partial[0], lse[0]
        else:
            partial, lse = partial[:panels], lse[:panels]
        return run_six_head_groups(
            operator,
            q,
            k,
            v,
            out,
            table,
            row_lengths,
            partial,
            lse,
            float(softmax_scale),
            float(k_scale),
            float(v_scale),
            parallel_kv_heads=parallel_kv_heads,
        )

    if variable_request_rows and (
        model_capacity is None or model_capacity <= capability
    ):
        run.__dict__.update(
            request_query_rows=request_query_rows,
            max_batch_size=max_batch_size,
        )
    return run


def run_six_head_groups(operator, q, k, v, out, *args, parallel_kv_heads=False):
    """Reuse the validated six-head kernel without copying the paged KV cache.

    Rebuilt operators parallelize KV heads in one grid and own a workspace
    panel per head/request. Older binaries consume one shared panel serially;
    graph capture retains their per-group query/output buffers.
    """
    if k.shape[2] == 1 or parallel_kv_heads:
        return operator(q, k, v, out, *args)
    for head in range(k.shape[2]):
        group_q = q[:, head * 6 : (head + 1) * 6].contiguous()
        group_out = torch.empty_like(group_q)
        operator(
            group_q,
            k[:, :, head : head + 1],
            v[:, :, head : head + 1],
            group_out,
            *args,
        )
        out[:, head * 6 : (head + 1) * 6].copy_(group_out)
    return out
