# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit DFlash2 schedules independent of target weight quantization.

The manifest selects independently reversible GDN/context/attention/selection
routes. It never loads a QPN2 projection library or inspects quantization names.
Existing tensor, state and shape guards decide whether an operation qualifies.
Set VLLM_SM70_DFLASH2_COMMON_MANIFEST before worker startup and select
CommonDFlash2Extension to use this experimental entry point. Importing without
that setting changes no route. Model/precision admission remains separate.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch


def _library(entry: dict) -> Path:
    path = Path(entry["library"]).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"DFlash2 library digest mismatch: {path.name}")
    return path


def install_sparse_dense_order(select) -> None:
    from vllm.model_executor.layers import vocab_parallel_embedding as embedding

    original = embedding._sm70_dflash2_dense_order_topk
    seen = set()

    def run(
        sparse_logits,
        candidate_ids,
        candidate_logits,
        values,
        ids,
        selector_k,
        vocab_start_index,
    ):
        eligible = (
            candidate_logits.dtype == torch.float32
            and candidate_logits.shape[0] in (7, 8)
            and candidate_logits.shape[1] == 64
            and sparse_logits.shape[1] == 62080
            and selector_k in (16, 20, 21)
            and candidate_logits.is_contiguous()
            and candidate_ids.is_contiguous()
            and values.is_contiguous()
            and ids.is_contiguous()
        )
        if torch.compiler.is_compiling() or not eligible:
            return original(
                sparse_logits,
                candidate_ids,
                candidate_logits,
                values,
                ids,
                selector_k,
                vocab_start_index,
            )
        key = (candidate_logits.shape[0], selector_k)
        if key not in seen:
            seen.add(key)
            print(
                f"COMMON_DFLASH2_SORT_ROUTE rank={torch.distributed.get_rank()} "
                f"rows={key[0]} k={key[1]}",
                flush=True,
            )
        return select(candidate_ids, candidate_logits, values, ids, vocab_start_index)

    embedding._sm70_dflash2_dense_order_topk = run


def install_common_routes(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported common-route manifest version")
    allowed = {"schema_version", "gdn_value_tile", "context_probe", "attention", "sort"}
    if set(manifest) - allowed:
        raise ValueError("Unknown common-route manifest entries")
    # Validate every native dependency before modifying Python dispatch.
    libraries = {
        key: _library(manifest[key])
        for key in ("attention", "sort")
        if manifest.get(key) is not None
    }
    if "sort" in libraries and torch.__version__.split("+")[0] != "2.10.0":
        raise ValueError("Exact sparse tie ordering requires frozen PyTorch 2.10.0")
    if "sort" in libraries:
        from benchmarks.kernels.benchmark_sm70_sparse_dense_topk import select

        torch.ops.load_library(str(libraries["sort"]))
        install_sparse_dense_order(select)
    if "attention" in libraries:
        from benchmarks.kernels.sm70_grouped_attention_candidate_route import (
            install_grouped_attention_candidate,
        )

        path = libraries["attention"]
        name = manifest["attention"]["module_name"]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load grouped attention: {path.name}")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        logged = False

        def attention(q, *args, **kwargs):
            nonlocal logged
            if not logged:
                logged = True
                print(
                    "COMMON_DFLASH2_ATTENTION_CAPTURE "
                    f"rank={torch.distributed.get_rank()} shape={tuple(q.shape)}",
                    flush=True,
                )
            return native.run(q, *args, **kwargs)

        install_grouped_attention_candidate(attention)
    if manifest.get("context_probe", False):
        from benchmarks.kernels.sm70_context_probe_candidate_route import (
            install_context_probe_candidate,
        )

        install_context_probe_candidate()
    if manifest.get("gdn_value_tile", False):
        from benchmarks.kernels.sm70_gdn_value_tile_candidate_route import (
            install_gdn_value_tile_candidate,
        )

        install_gdn_value_tile_candidate()


class CommonDFlash2Extension:
    """Worker extension for the explicitly selected, quantization-free routes."""


if manifest_path := os.getenv("VLLM_SM70_DFLASH2_COMMON_MANIFEST"):
    install_common_routes(json.loads(Path(manifest_path).read_text()))
