# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 Gemma RMSNorm collective-fusion helpers and closed-residual analysis."""

from __future__ import annotations

import operator
from typing import Any

import torch
import torch.fx as fx


def _same_token_extent(lhs: object, rhs: object) -> bool:
    if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
        return False
    lhs_val = lhs.meta.get("val")
    rhs_val = rhs.meta.get("val")
    if lhs_val is None or rhs_val is None:
        return False
    return bool(
        lhs_val.ndim == 2
        and rhs_val.ndim == 2
        and lhs_val.shape[0] == rhs_val.shape[0]
        and lhs_val.shape[1] == rhs_val.shape[1]
    )


def mark_closed_residual_chain(
    graph: fx.Graph, *, allow_piecewise_output: bool = False
) -> int:
    """Enable sharded-residual RS+Norm+AG only when residual_out stays closed.

    A node in ``fused`` may omit All-Gathering ``residual_out`` iff every
    consumer of ``getitem(node, 1)`` is either ``args[1]`` of another node in
    ``safe`` with identical ``(M, N)`` extent (backward induction from dead
    ``residual_out``) or a piecewise subgraph output boundary of a verified
    closed-residual model (``pp_size == 1`` without middle-layer auxiliary
    hidden-state consumers).
    """
    collective = torch.ops.vllm.all_reduce_gemma_rms_norm.default
    fused = [node for node in graph.nodes if node.target == collective]
    if not fused:
        return 0

    if allow_piecewise_output:
        for node in graph.nodes:
            if node.op != "placeholder":
                continue
            val = node.meta.get("val")
            if (
                val is None
                or getattr(val, "dtype", None) != torch.float32
                or getattr(val, "ndim", 0) != 2
                or val.shape[-1] != 5120
            ):
                continue
            for user in node.users:
                if user.op == "output":
                    continue
                if not (
                    user.target == collective
                    and len(user.args) >= 2
                    and user.args[1] is node
                    and all(
                        arg is not node for idx, arg in enumerate(user.args) if idx != 1
                    )
                ):
                    allow_piecewise_output = False
                    break
            if not allow_piecewise_output:
                break

    safe: set[fx.Node] = set()
    for node in reversed(fused):
        node_ok = True
        for user in node.users:
            if user.target != operator.getitem or len(user.args) < 2:
                node_ok = False
                break
            if user.args[1] == 0:
                continue
            if user.args[1] != 1:
                node_ok = False
                break
            for consumer in user.users:
                if allow_piecewise_output and consumer.op == "output":
                    continue
                if not (
                    consumer in safe
                    and len(consumer.args) >= 2
                    and consumer.args[1] is user
                    and all(
                        arg is not user
                        for idx, arg in enumerate(consumer.args)
                        if idx != 1
                    )
                    and user not in consumer.kwargs.values()
                    and _same_token_extent(node.args[0], consumer.args[0])
                ):
                    node_ok = False
                    break
            if not node_ok:
                break
        if node_ok:
            safe.add(node)

    for node in fused:
        allow = node in safe
        if "group_name" in node.kwargs or "allow_shard_residual" in node.kwargs:
            node.kwargs = {**node.kwargs, "allow_shard_residual": allow}
        elif len(node.args) >= 5:
            node.args = (*node.args[:5], allow)
        node.meta.pop("eager_input_vals", None)
    return len(safe)


def is_closed_residual_model(config: Any, hidden_dim: int, model_dtype: Any) -> bool:
    return bool(
        hidden_dim == 5120
        and model_dtype == torch.float16
        and getattr(config.model_config.hf_text_config, "model_type", None)
        == "qwen3_5_text"
        and config.parallel_config.pipeline_parallel_size == 1
        and (
            config.speculative_config is None
            or getattr(config.speculative_config, "method", None) == "mtp"
        )
    )


def register_unified_gemma_patterns(fusion_pass: Any) -> bool:
    if not (fusion_pass.sm70_tp4_push_mode or fusion_pass.sm70_tp2_mode):
        return False
    # Import registers the norm boundary even when the model is traced later.
    import vllm.model_executor.layers.layernorm  # noqa: F401
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        Sm70AllReduceGemmaRMSNormPattern,
        Sm70PushGemmaRMSNormPattern,
        Sm70Tp4LongPrefillFusedNormPattern,
    )

    unified = torch.ops.vllm.all_reduce_gemma_rms_norm
    Sm70PushGemmaRMSNormPattern(
        fusion_pass.model_dtype,
        fusion_pass.device,
        fusion_pass.sm70_norm_epsilon,
        fusion_pass.tp_size,
    ).register(fusion_pass.patterns)
    if hasattr(
        torch.ops.vllm,
        "sm70_gemma_long_prefill_fused_add_rms_norm",
    ):
        Sm70Tp4LongPrefillFusedNormPattern(
            fusion_pass.sm70_norm_epsilon,
            fusion_pass.model_dtype,
            fusion_pass.device,
            collective=unified,
        ).register(fusion_pass.patterns)
    Sm70AllReduceGemmaRMSNormPattern(
        fusion_pass.sm70_norm_epsilon,
        fusion_pass.model_dtype,
        fusion_pass.device,
        torch.float32,
        collective=unified,
    ).register(fusion_pass.patterns)
    fusion_pass.disabled = False
    return True
