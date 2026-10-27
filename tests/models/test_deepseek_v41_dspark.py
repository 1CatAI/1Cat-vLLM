# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.models.deepseek_v41 import dspark
from vllm.models.deepseek_v41.cache import V41CacheBuilder, V41CacheMetadata
from vllm.models.deepseek_v41.dspark import V41DraftMoE, draft_weight_name
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer


def test_v41_metadata_carries_noncausal_contract_into_the_existing_proposer():
    common = SimpleNamespace(
        num_reqs=1,
        block_table_tensor=torch.zeros(1, 2, dtype=torch.int32),
        slot_mapping=torch.arange(5),
        is_prefilling=None,
        is_dummy_batch=False,
        causal=False,
    )
    builder = SimpleNamespace(
        prefill_flag=torch.ones(1, dtype=torch.bool),
        kv_cache_spec=SimpleNamespace(block_size=16),
    )
    metadata = V41CacheBuilder.build(
        cast(V41CacheBuilder, builder), 0, cast(CommonAttentionMetadata, common)
    )
    group = SimpleNamespace(
        kv_cache_group_id=0,
        layer_names=["dspark.model.layers.0.attn.window_cache"],
        get_metadata_builder=lambda: SimpleNamespace(
            build_for_drafting=lambda **kwargs: metadata
        ),
    )
    proposer = SimpleNamespace(
        _dflash_new_common_attn_metadata_by_gid={},
        draft_attn_groups=[group],
        dflash_causal=False,
    )
    _, per_layer = DFlashProposer.build_per_group_and_layer_attn_metadata(
        cast(DFlashProposer, proposer), cast(CommonAttentionMetadata, common)
    )
    assert cast(V41CacheMetadata, per_layer[group.layer_names[0]]).causal is False


@pytest.mark.parametrize(
    "source,target",
    [
        ("mtp.0.main_proj.weight", "model.main_proj.weight"),
        ("mtp.0.main_norm.weight", "model.main_norm.weight"),
        (
            "mtp.1.ffn.experts.127.w1.scale",
            "model.layers.1.ffn.experts.127.gate_proj.scale",
        ),
        (
            "mtp.1.ffn.shared_experts.w2.weight",
            "model.layers.1.ffn.shared_experts.down_proj.weight",
        ),
        ("mtp.2.norm.weight", "model.norm.weight"),
        ("mtp.2.markov_head.embed.weight", "model.markov_embed.weight"),
        ("mtp.2.markov_head.head.weight", "model.markov_bias.weight"),
        ("mtp.2.confidence_head.proj.weight", "model.confidence.weight"),
        ("mtp.0.ffn.gate.bias_vl", None),
        ("layers.0.attn.wkv.weight", None),
        ("embed.weight", None),
    ],
)
def test_draft_weight_mapping_keeps_target_and_vl_outside_the_text_draft(
    source, target
):
    assert draft_weight_name(source) == target


@pytest.mark.parametrize("source", ["mtp.1.main_proj.weight", "mtp.0.norm.weight"])
def test_dspark_heads_require_the_correct_stage(source):
    with pytest.raises(ValueError):
        draft_weight_name(source)


def test_draft_expert_addition_and_tp_sum_keep_wide_fp32_values(monkeypatch):
    class Partials(torch.nn.Module):
        def forward_partials(self, hidden_states, router_logits):
            assert hidden_states.dtype == torch.float16
            assert router_logits is hidden_states
            return torch.full_like(hidden_states, 16496), torch.full_like(
                hidden_states, 8
            )

    model = SimpleNamespace(gate=torch.nn.Identity(), experts=Partials())

    def reduce(partial):
        assert partial.dtype == torch.float32
        return partial * 4

    monkeypatch.setattr(dspark, "tensor_model_parallel_all_reduce", reduce)
    result = V41DraftMoE.forward(
        cast(V41DraftMoE, model), torch.zeros(5, 5120, dtype=torch.float16)
    )
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all() and torch.isinf(result.half()).all()
    torch.testing.assert_close(result, torch.full_like(result, 66016), rtol=0, atol=0)
