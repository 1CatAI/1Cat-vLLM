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

# Include generation headroom after a 128K prompt. Larger CPU upper bounds use
# the existing full-context graph; device row lengths remain authoritative.
MAX_CONTEXT = 132096
MANIFEST_ENV = "VLLM_SM70_E4M3_LONG_ATTENTION_MANIFEST"
_WORKSPACES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def long_attention_enabled() -> bool:
    return bool(os.environ.get(MANIFEST_ENV))


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


def long_attention_contract(manifest):
    context_limit = manifest.get("max_context", MAX_CONTEXT)
    query_rows = tuple(manifest.get("query_rows", [8]))
    if (
        type(context_limit) is not int
        or not 0 < context_limit <= 262144
        or not query_rows
        or any(type(q) is not int or not 2 <= q <= 8 for q in query_rows)
    ):
        raise ValueError("Unsupported long-attention context or query-row contract")
    return context_limit, query_rows


def long_attention_graph_contract():
    _, manifest = load_long_attention(os.environ[MANIFEST_ENV])
    return long_attention_contract(manifest)


def wrap_long_attention(fallback):
    manifest_name = os.environ.get(MANIFEST_ENV)
    if not manifest_name:
        return fallback
    operator, manifest = load_long_attention(manifest_name)
    context_limit, query_rows = long_attention_contract(manifest)

    def run(
        q, k, v, table, row_lengths, *, out, softmax_scale, k_scale=1.0, v_scale=1.0
    ):
        descriptor = (
            get_forward_context().batch_descriptor
            if is_forward_context_available()
            else None
        )
        if not (
            descriptor is not None
            and descriptor.attention_context_bucket == context_limit
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
        key = (manifest["source_sha256"], context_limit, 80, q.device, stream)
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
