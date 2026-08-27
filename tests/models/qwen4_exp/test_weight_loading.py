# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    Qwen4ExpSparseMoeBlock,
    _remap_qsa_cache_scale_name,
)
from vllm.models.qwen4_exp.nvidia.mtp import _validate_mtp_expert_weights_loaded


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "q",
        ),
        (
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "k",
        ),
        (
            "layers.1.linear_attn.in_proj_qkv.weight",
            "layers.1.linear_attn.in_proj_qkvz.weight",
            (0, 1, 2),
        ),
        (
            "layers.1.linear_attn.in_proj_z.weight",
            "layers.1.linear_attn.in_proj_qkvz.weight",
            3,
        ),
        (
            "layers.1.linear_attn.in_proj_b.weight",
            "layers.1.linear_attn.in_proj_ba.weight",
            0,
        ),
        (
            "layers.1.mlp.gate_proj.weight",
            "layers.1.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.1.mlp.experts.0.gate_proj.weight",
            "layers.1.mlp.experts.0.gate_proj.weight",
            None,
        ),
        (
            "layers.0.self_attn.indexer.index_qk_proj.weight",
            "layers.0.self_attn.indexer.index_qk_proj.weight",
            None,
        ),
        (
            "layers.0.attn_hyper_connection.input_mix_weight_down.weight",
            "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
            0,
        ),
        (
            "layers.0.attn_hyper_connection.block_inject_weight.weight",
            "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
            1,
        ),
        (
            "hyper_connection_mixer.input_mix_weight_down.weight",
            "hyper_connection_mixer.input_mix_weight_down.weight",
            None,
        ),
        (
            "layers.1.ple.ple_embedding.layer_multipliers",
            "layers.1.ple.ple_embedding.layer_multipliers",
            None,
        ),
    ],
)
def test_text_checkpoint_mapper_preserves_qwen4_exp_specific_weights(
    checkpoint_name: str,
    model_name: str,
    shard_id: str | int | tuple[int, ...] | None,
) -> None:
    assert Qwen4ExpModel.hf_to_vllm_mapper._map_name_with_shard(checkpoint_name) == (
        model_name,
        shard_id,
    )


def test_outer_checkpoint_mapper_selects_language_model_only_paths() -> None:
    mapper = Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper

    assert (
        mapper._map_name("model.language_model.layers.0.ple.key_proj.weight")
        == "language_model.model.layers.0.ple.key_proj.weight"
    )
    assert mapper._map_name("lm_head.weight") == "language_model.lm_head.weight"
    assert mapper._map_name("model.visual.blocks.0.attn.qkv.weight") == (
        "visual.blocks.0.attn.qkv.weight"
    )


def test_sparse_moe_attaches_private_recursive_loader_mapping(monkeypatch) -> None:
    class FakeExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expert_mapping = None

    def fake_qwen3_next_init(self, vllm_config, prefix="") -> None:
        del vllm_config, prefix
        nn.Module.__init__(self)
        self.n_routed_experts = 2
        self.n_redundant_experts = 0
        self.experts = FakeExperts()

    monkeypatch.setattr(
        Qwen3NextSparseMoeBlock,
        "__init__",
        fake_qwen3_next_init,
    )
    vllm_config = type(
        "VllmConfigStub",
        (),
        {
            "parallel_config": type(
                "ParallelConfigStub", (), {"use_sequence_parallel_moe": False}
            )(),
            "model_config": type(
                "ModelConfigStub",
                (),
                {
                    "hf_text_config": type(
                        "TextConfigStub",
                        (),
                        {"shared_expert_intermediate_size": 640},
                    )()
                },
            )(),
        },
    )()

    block = Qwen4ExpSparseMoeBlock(vllm_config, prefix="model.layers.0.mlp")

    assert block.experts.expert_mapping == [
        ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
        ("experts.w13_weight", "experts.gate_up_proj", 1, "w3"),
        ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
        ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
        ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
        ("experts.w13_", "experts.1.gate_proj.", 1, "w1"),
        ("experts.w2_", "experts.1.down_proj.", 1, "w2"),
        ("experts.w13_", "experts.1.up_proj.", 1, "w3"),
    ]


def test_fused_mtp_expert_checkpoint_loads_every_expert() -> None:
    class FakeFusedExperts(nn.Module):
        load_weights = FusedMoE.load_weights

        def __init__(self) -> None:
            super().__init__()
            self.layer_name = "model.layers.0.mlp.experts"
            self.w13_weight = nn.Parameter(torch.empty(1))
            self.w2_weight = nn.Parameter(torch.empty(1))
            self.calls: list[tuple[str, str, int, torch.Tensor]] = []
            self.expert_mapping = FusedMoE.make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=2,
                include_fused=True,
            )

        def weight_loader(
            self,
            *,
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool,
        ) -> bool:
            assert return_success
            param_name = "w13" if param is self.w13_weight else "w2"
            self.calls.append((param_name, shard_id, expert_id, loaded_weight.clone()))
            return True

    experts = FakeFusedExperts()
    gate_up = torch.arange(2 * 6 * 2).reshape(2, 6, 2)
    down = torch.arange(2 * 2 * 3).reshape(2, 2, 3)

    loaded = list(
        experts.load_weights([("gate_up_proj", gate_up), ("down_proj", down)])
    )

    assert loaded == [
        "w13_weight",
        "w13_weight",
        "w13_weight",
        "w13_weight",
        "w2_weight",
        "w2_weight",
    ]
    assert [(name, shard, expert) for name, shard, expert, _ in experts.calls] == [
        ("w13", "w1", 0),
        ("w13", "w1", 1),
        ("w13", "w3", 0),
        ("w13", "w3", 1),
        ("w2", "w2", 0),
        ("w2", "w2", 1),
    ]
    torch.testing.assert_close(experts.calls[0][3], gate_up[0, :3])
    torch.testing.assert_close(experts.calls[1][3], gate_up[1, :3])
    torch.testing.assert_close(experts.calls[2][3], gate_up[0, 3:])
    torch.testing.assert_close(experts.calls[3][3], gate_up[1, 3:])
    torch.testing.assert_close(experts.calls[4][3], down[0])
    torch.testing.assert_close(experts.calls[5][3], down[1])


def test_mtp_expert_loading_fails_closed() -> None:
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = nn.Module()
    model.model.layers[0].mlp.experts = nn.Module()
    experts = model.model.layers[0].mlp.experts
    experts.w13_weight = nn.Parameter(torch.empty(1))
    experts.w2_weight = nn.Parameter(torch.empty(1))

    w13_name = "model.layers.0.mlp.experts.w13_weight"
    w2_name = "model.layers.0.mlp.experts.w2_weight"
    _validate_mtp_expert_weights_loaded(model, {w13_name, w2_name})

    with pytest.raises(ValueError, match="w2_weight"):
        _validate_mtp_expert_weights_loaded(model, {w13_name})


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name"),
    [
        (
            "layers.0.self_attn.k_proj.k_scale",
            "layers.0.self_attn._k_scale",
        ),
        (
            "layers.0.self_attn.v_proj.output_scale",
            "layers.0.self_attn._v_scale",
        ),
        (
            "language_model.model.layers.0.self_attn.attn.k_scale",
            "language_model.model.layers.0.self_attn._k_scale",
        ),
        (
            "layers.0.self_attn.indexer.index_qk_proj.weight_scale",
            "layers.0.self_attn.indexer.index_qk_proj.weight_scale",
        ),
        (
            "layers.1.self_attn.k_proj.k_scale",
            "layers.1.self_attn.k_proj.k_scale",
        ),
    ],
)
def test_only_qsa_main_cache_scales_move_to_the_merged_owner(
    checkpoint_name: str,
    model_name: str,
) -> None:
    assert _remap_qsa_cache_scale_name(checkpoint_name, frozenset({0})) == model_name
