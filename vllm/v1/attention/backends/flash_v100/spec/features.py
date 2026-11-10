# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Method-specific verification metadata providers."""

from __future__ import annotations

from vllm.v1.spec_decode.attention_features import SpecFeatureRegistry


class DDTreeFeature:
    @staticmethod
    def prepare(state, attn_metadata, common_attn_metadata, tree_verify, prepared):
        if not tree_verify:
            # EAGER build path: cap the small-query decode workspace/launch grid
            # to the runtime max_seq_len, mirroring build_for_cudagraph_capture
            # (:2431-2445). Without a cap, _update_smallq_decode_metadata stores
            # the full block-table capacity (== max_model_len worth of blocks) at
            # smallq_decode_workspace_seq_capacity_hint (:2350), and the interface
            # then launches ceil(max_model_len/partition_size) partitions where
            # only ceil(max_seq_len/partition_size) do work (1024 vs 17 at
            # max_model_len=262144 / S=4224 / ps=256). The cap keeps launch equal
            # to the runtime coverage; the interface floors it at effective
            # max_seq_len (_get_decode_plan:165-169), so it can never under-cover.
            if prepared is not None:
                state._attach_prepared_dflash2_smallq_metadata(
                    attn_metadata,
                    prepared,
                )
            else:
                state._update_smallq_decode_metadata(
                    attn_metadata,
                    common_attn_metadata,
                    workspace_seq_capacity_cap=(
                        int(getattr(common_attn_metadata, "max_seq_len", 0) or 0)
                        or None
                    ),
                )


class DFlash2Feature:
    @staticmethod
    def prepare(state, attn_metadata, common_attn_metadata, tree_verify, prepared):
        # Explicit tree input keeps its historical precedence, even for callers
        # whose configured method does not agree with the supplied payload.
        if tree_verify:
            return
        if prepared is not None:
            state._attach_prepared_dflash2_smallq_metadata(attn_metadata, prepared)
        else:
            MTPFeature.prepare(state, attn_metadata, common_attn_metadata, False, None)


class MTPFeature:
    @staticmethod
    def prepare(state, attn_metadata, common_attn_metadata, tree_verify, prepared):
        if tree_verify or prepared is not None:
            DDTreeFeature.prepare(
                state, attn_metadata, common_attn_metadata, tree_verify, prepared
            )
            return
        state._update_smallq_decode_metadata(
            attn_metadata,
            common_attn_metadata,
            workspace_seq_capacity_cap=(
                int(getattr(common_attn_metadata, "max_seq_len", 0) or 0) or None
            ),
        )


# The proposer-side registry owns method lookup and validates registrations.
# Common attention assembly has no speculative method names.
FEATURES = SpecFeatureRegistry(
    {
        "dflash": DFlash2Feature,
        "dflash_ddtree": DDTreeFeature,
        "mtp": MTPFeature,
    },
    fallback=MTPFeature,
)


def prepare_verification(
    self,
    attn_metadata,
    common_attn_metadata,
    ddtree_tree_verify,
    prepared_dflash2_smallq_metadata,
):
    self.feature.prepare(
        self,
        attn_metadata,
        common_attn_metadata,
        ddtree_tree_verify,
        prepared_dflash2_smallq_metadata,
    )
