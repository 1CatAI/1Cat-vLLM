# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 packed-weight bindings for shared MoE stages.

The codec borrows layer-owned banks through a view: reloading/rebinding weights
remains visible, and this object owns no workspace or strategy configuration.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.fused_moe.sm70.declarations import (
    fp4_binding_mode,
    fp4_native_binding,
)
from vllm.model_executor.layers.quantization.sm70_moe_router import Sm70MoeRoutePlan
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    LayerWorkspaceView,
)


@dataclass(frozen=True)
class Fp4MoECodec:
    family: Literal["nvfp4", "mxfp4"]
    weights: LayerWorkspaceView
    dimensions: LayerWorkspaceView
    raw_scale: bool = False
    swiglu_limit: float | None = None
    bindings: Any = None

    @property
    def operators(self):
        return self.bindings if self.bindings is not None else ops

    @staticmethod
    def prepare_weights(
        family, packed, scales, group_size, *, interleave_gated_silu=False
    ):
        prepare = getattr(ops, family + "_sm70_prepare")
        if family == "nvfp4":
            return prepare(
                packed, scales, group_size, interleave_gated_silu=interleave_gated_silu
            )
        return prepare(packed, scales, group_size)

    def _op(self, stage: str, mode: str, *, qpn_mtp=False):
        mode = fp4_binding_mode(
            self.family, mode, raw_scale=self.raw_scale, qpn_mtp=qpn_mtp
        )
        return getattr(self.operators, fp4_native_binding(self.family, stage, mode))

    def _bank(self, stage: str) -> tuple[torch.Tensor, ...]:
        w = self.weights
        if self.raw_scale:
            return (
                getattr(w, stage + "_tm_weight"),
                getattr(w, stage + "_raw_scale_codes"),
                getattr(w, stage + "_raw_global_scales"),
            )
        return getattr(w, stage + "_tm_weight"), getattr(w, stage + "_tm_scales")

    def _groups(self):
        return tuple(
            getattr(self.weights, "_nvfp4_grouped_" + field)
            for field in ("rows", "experts", "sizes", "total")
        )

    def _dense(self, stage: str) -> tuple[Any, ...]:
        return (
            getattr(self.weights, stage + "_strided_ptrs_w"),
            getattr(self.weights, stage + "_strided_ptrs_s"),
        )

    def _shape(self, stage: str) -> tuple[int, int, int]:
        return (
            getattr(self.dimensions, stage + "_k_dim"),
            getattr(self.dimensions, stage + "_n_dim"),
            self.dimensions.group_size,
        )

    def _expand(self, stage: str, interleaved: bool) -> None:
        if self.raw_scale:
            _, codes, global_scales = self._bank(stage)
            self.operators.nvfp4_expand_raw_scales_sm70_out(
                getattr(self.weights, stage + "_tm_scales"),
                codes,
                global_scales,
                interleaved,
            )

    def _qpn(self, stage, out, x, ids, plan) -> None:
        w13 = stage == "w13"
        split_k = plan.w13_split_k if w13 else plan.w2_split_k
        args = (out, x, *self._bank(stage), ids, w13)
        if self.family == "mxfp4":
            self._op(stage, "qpn")(*args)
        elif self.raw_scale:
            self._op(stage, "qpn")(*args, plan.interleaved if w13 else False, split_k)
        else:
            op = self._op(stage, "qpn", qpn_mtp=plan.qpn_mtp)
            op(*args, split_k)

    def gemm_w13(
        self,
        plan: Sm70MoeRoutePlan,
        buffers,
        x,
        ids,
        offsets,
        experts,
        count,
        topk_ids,
    ) -> None:
        mode = plan.w13.value
        out = buffers["gate_up"]
        if mode == "active_grouped":
            self._op("w13", mode)(
                buffers["intermediate"],
                x,
                *self._bank("w13"),
                ids,
                *self._groups(),
                plan.w13_split_k,
                plan.interleaved,
            )
        elif mode == "qpn":
            self._qpn("w13", out, x, ids, plan)
        elif mode == "fused_qpn":
            self._op("w13", "fused_qpn")(
                buffers["intermediate"], x, *self._bank("w13"), ids
            )
        elif mode == "fused_batch_qpn":
            op = self._op("w13", mode)
            op(buffers["intermediate"], x, *self._bank("w13"), ids, plan.interleaved)
        elif mode == "prepare_w13":
            self._op("w13", "prepare_w13")(
                out,
                buffers["permuted_input"],
                x,
                topk_ids,
                *self._dense("w13"),
                buffers["compact_expert_offsets"],
                buffers["inv_permuted_idx"],
                buffers["permuted_experts_id"],
                *self._shape("w13"),
                self.dimensions.hidden_size,
            )
        else:
            self._expand("w13", plan.interleaved)
            if mode == "glm_qpn":
                self._op("w13", "glm_qpn")(
                    out, x, *self._bank("w13"), ids, buffers["sorted_row_idx"], True
                )
            elif mode == "indexed_split_fused":
                for middle, prefix, n in (
                    (buffers["intermediate"][:, :128], "w13_head", 256),
                    (buffers["intermediate"][:, 128:], "w13_tail", 64),
                ):
                    self._op("w13", mode)(
                        middle,
                        x,
                        buffers["input_row_indices"],
                        offsets,
                        experts,
                        *self._dense(prefix),
                        count,
                        self.dimensions.w13_k_dim,
                        n,
                        self.dimensions.group_size,
                    )
            elif mode in ("indexed_prefill", "indexed_fused"):
                op = (
                    self._op("w13", "indexed_fused")
                    if mode == "indexed_fused"
                    else self._op("w13", "indexed_prefill")
                )
                op(
                    buffers["intermediate"] if mode == "indexed_fused" else out,
                    x,
                    buffers["input_row_indices"],
                    offsets,
                    experts,
                    *self._dense("w13"),
                    count,
                    *self._shape("w13"),
                )
            else:
                self._op("w13", "dense")(
                    out,
                    x,
                    offsets,
                    experts,
                    *self._dense("w13"),
                    count,
                    *self._shape("w13"),
                )

    def gemm_w2(self, plan, buffers, ids, offsets, experts, count, topk_weights):
        mode = plan.w2.value
        x = buffers["intermediate"]
        if mode in ("active_grouped", "grouped_batch_reduce"):
            self._op("w2", mode)(
                buffers["output"],
                buffers["sorted_output"],
                x,
                *self._bank("w2"),
                topk_weights,
                *self._groups(),
            )
        elif mode == "qpn":
            self._qpn("w2", buffers["sorted_output"], x, ids, plan)
        elif mode == "direct_reduce":
            self._op("w2", "direct_reduce")(
                buffers["output"], x, *self._bank("w2"), ids, topk_weights
            )
        elif mode == "batch_reduce":
            op = self._op("w2", mode)
            op(buffers["output"], x, *self._bank("w2"), ids, topk_weights)
        else:
            self._expand("w2", False)
            self._op("w2", "dense")(
                buffers["sorted_output"],
                x,
                offsets,
                experts,
                *self._dense("w2"),
                count,
                *self._shape("w2"),
            )
