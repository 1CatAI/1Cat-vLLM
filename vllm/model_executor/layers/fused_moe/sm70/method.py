# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin adapter to the existing FusedMoEMethodBase lifecycle."""

from vllm._sm70.policy import NativeBindings
from vllm.config.sm70_moe import capture_sm70_moe_config
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.sm70.weight_codec import Sm70MoEWeightCodec
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    select_single_token_plan,
)


class Sm70MoEMethodBase(FusedMoEMethodBase):
    def _initialize_sm70_policy(self, family, layer, logger):
        self.sm70_moe_policy = capture_sm70_moe_config(family)
        self.native_ops = NativeBindings(self.sm70_moe_policy.native.values)
        self.weight_codec = Sm70MoEWeightCodec(
            family.upper(), logger, bindings=self.native_ops
        )
        policy = self.sm70_moe_policy
        self.use_batched_gemm = bool(policy.batched)
        layer.sm70_moe_diagnostics = policy.diagnostics
        layer.sm70_moe_policy = policy
        import torch

        # Native availability is static after registration, not a token decision.
        compact = "compact" in (policy.single_token_w13 or ()) and hasattr(
            torch.ops._C, f"{family}_moe_single_token_compact_dense_w13_sm70_out"
        )
        indexed13 = "indexed" in (policy.single_token_w13 or ()) and hasattr(
            torch.ops._C, f"{family}_moe_single_token_indexed_dense_w13_sm70_out"
        )
        indexed2 = policy.single_token_w2 == "indexed" and hasattr(
            torch.ops._C, f"{family}_moe_single_token_indexed_dense_stage_sm70_out"
        )
        weighted = policy.single_token_reduce == "weighted" and hasattr(
            torch.ops._C, "awq_moe_single_token_weighted_reduce_out"
        )
        self.single_token_indexed = indexed13 and indexed2
        self.single_token_plan = select_single_token_plan(
            compact_w13=compact,
            indexed_w13=indexed13,
            indexed_w2=indexed2,
            weighted_reduce=weighted,
        )
        self.strict_single_token_plan = select_single_token_plan(
            compact_w13=compact,
            indexed_w13=indexed13,
            indexed_w2=indexed2,
            weighted_reduce=weighted,
            strict=True,
        )
        self.batched_single_token_plan = select_single_token_plan(
            compact_w13=compact,
            indexed_w13=indexed13,
            indexed_w2=indexed2,
            weighted_reduce=weighted,
            batched_indexed=True,
        )
        self.legacy_single_token = policy.legacy_compact and hasattr(
            torch.ops._C, f"{family}_moe_single_token_sm70_out"
        )

    @property
    def supports_eplb(self) -> bool:
        return False

    def get_fused_moe_quant_config(self, layer):
        return None
