# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in compensated attention with explicit native build manifests."""

import hashlib
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)

# Capacity the shipped operator was qualified for. This is a claim about the
# compiled kernel, not a service limit: the graph bound the route is captured at
# is the smaller of this and the context the model is actually served with, so a
# deployment with a different window is routed on its own value and the operator
# is never asked for more context than it was admitted for.
BUILTIN_MAX_CONTEXT = 262144
MANIFEST_ENV = "VLLM_SM70_E4M3_LONG_ATTENTION_MANIFEST"
DISABLE_ENV = "VLLM_SM70_E4M3_LONG_ATTENTION"
_WORKSPACES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


# The grouped long-context route is compiled into the shipped FA2 extension, so
# it is available without any environment variable. The manifest variable stays
# as an explicit override for an unqualified experimental candidate.
BUILTIN_OP = "sm70_grouped_long_fwd"
# The compiled operator carries the probability-swizzle / early-store layout the
# route was qualified with. The source digest names that layout so a rebuilt
# kernel cannot silently inherit the workspace identity of a different one.
BUILTIN_SOURCE_SHA256 = (
    "eb7a85511f581fcd22cf13619c85ed2f42a8cbc8b216bb3e632bf448f6b820e1"
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
        return getattr(torch.ops._vllm_fa2_C, BUILTIN_OP)
    except AttributeError:
        return None


# Explicit opt-out. The accelerated route is on by default, so an operator needs
# a way back to the full-context route without rebuilding, and the paired A/B
# validation needs both arms from one build. An explicit off wins over a
# manifest override.
DISABLE_VALUES = {"0", "false", "no", "off"}


def long_attention_enabled() -> bool:
    if os.environ.get(DISABLE_ENV, "").strip().lower() in DISABLE_VALUES:
        return False
    return bool(os.environ.get(MANIFEST_ENV)) or builtin_long_attention() is not None


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
        "Loaded experimental SM70 E4M3 q8 attention: module=%s SHA256=%s "
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


def resolve_long_attention():
    """The route in effect: an explicit manifest candidate, else the shipped one."""
    if not long_attention_enabled():
        return None, None
    manifest_name = os.environ.get(MANIFEST_ENV)
    if manifest_name:
        return load_long_attention(manifest_name)
    operator = builtin_long_attention()
    if operator is None:
        return None, None
    logger.info_once(
        "Using the shipped SM70 E4M3 q8 long-context route (%s): capability=%d "
        "query_rows=%s; 80 splits, compensated FP32 state.",
        BUILTIN_OP,
        BUILTIN_MAX_CONTEXT,
        BUILTIN_QUERY_ROWS,
        scope="process",
    )
    return operator, BUILTIN_MANIFEST


def long_attention_graph_contract(capacity: int | None = None):
    """The captured bound for ``capacity``, or ``(None, ())`` when route is off."""
    _, manifest = resolve_long_attention()
    if manifest is None:
        return None, ()
    return long_attention_contract(manifest, capacity)


def wrap_long_attention(fallback):
    operator, manifest = resolve_long_attention()
    if operator is None:
        return fallback
    capability = long_attention_capability(manifest)
    query_rows = long_attention_query_rows(manifest)

    def run(
        q, k, v, table, row_lengths, *, out, softmax_scale, k_scale=1.0, v_scale=1.0
    ):
        descriptor = (
            get_forward_context().batch_descriptor
            if is_forward_context_available()
            else None
        )
        bucket = getattr(descriptor, "attention_context_bucket", None)
        # The graph builder owns the served window: it stamps the bound it
        # captured the variant at onto the descriptor. The wrapper only has to
        # refuse a bound the operator was not qualified for.
        if not (
            bucket is not None
            and bucket <= capability
            and q.ndim == 3
            and q.shape[0] in query_rows
            and q.shape[1:] == (6, 256)
            and k.ndim == 4
            and k.shape[1] in (1648, 3296)
            and k.shape[2:] == (1, 256)
            and v.shape == k.shape
        ):
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
        stream = torch.cuda.current_stream(q.device).cuda_stream
        key = (manifest["source_sha256"], bucket, 80, q.device, stream)
        if key not in _WORKSPACES:
            _WORKSPACES[key] = (
                torch.empty((80, 8, 6, 256), dtype=torch.float32, device=q.device),
                torch.empty((80, 8, 6, 2), dtype=torch.float32, device=q.device),
            )
        partial, lse = _WORKSPACES[key]
        return operator(
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
        )

    return run
