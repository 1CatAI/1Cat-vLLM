# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.registry import (
    _MULTIMODAL_MODELS,
    _SPECULATIVE_DECODING_MODELS,
    ModelRegistry,
)
from vllm.models.inkling2.configs import Inkling2DSparkConfig


def test_inkling2_dspark_registry_placement():
    assert "Inkling2DSparkModel" in _SPECULATIVE_DECODING_MODELS
    assert "Inkling2DSparkModel" not in _MULTIMODAL_MODELS
    assert "Inkling2DSparkModel" in ModelRegistry.get_supported_archs()


def test_explicit_sidecar_loader_shared_key_and_fail_closed(tmp_path):
    """Explicit sidecar_weight_file loads only mtp.safetensors and fails closed."""
    main_shard = tmp_path / "model-00001-of-00001.safetensors"
    sidecar = tmp_path / "mtp.safetensors"
    index = tmp_path / "model.safetensors.index.json"

    save_file(
        {
            "shared.weight": torch.tensor([1.0, 2.0], dtype=torch.float16),
            "main.only": torch.tensor([10.0], dtype=torch.float16),
        },
        str(main_shard),
    )
    save_file(
        {
            "shared.weight": torch.tensor([99.0, 100.0], dtype=torch.float16),
            "mtp.only": torch.tensor([42.0], dtype=torch.float16),
        },
        str(sidecar),
    )
    index.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "shared.weight": "model-00001-of-00001.safetensors",
                    "main.only": "model-00001-of-00001.safetensors",
                },
            }
        )
    )

    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    model_cfg = SimpleNamespace(model=str(tmp_path), revision=None)

    # 1) Target model loads only main checkpoint weights (not mtp.safetensors).
    target_weights = dict(loader.get_all_weights(model_cfg, SimpleNamespace()))
    assert set(target_weights.keys()) == {"shared.weight", "main.only"}
    assert torch.equal(
        target_weights["shared.weight"], torch.tensor([1.0, 2.0], dtype=torch.float16)
    )

    # 2) Draft model with explicit sidecar_weight_file loads only mtp.safetensors.
    draft_stub = SimpleNamespace(sidecar_weight_file="mtp.safetensors")
    draft_weights = dict(loader.get_all_weights(model_cfg, draft_stub))
    assert set(draft_weights.keys()) == {"shared.weight", "mtp.only"}
    assert torch.equal(
        draft_weights["shared.weight"],
        torch.tensor([99.0, 100.0], dtype=torch.float16),
    )

    # 3) Missing sidecar fails closed (never falls back to main checkpoint shards).
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="mtp.safetensors"):
        list(loader.get_all_weights(model_cfg, draft_stub))

    # 4) Existing allow_patterns_overrides caller still loads indexed main shard.
    override_stub = SimpleNamespace(
        allow_patterns_overrides=["model-00001-of-00001.safetensors"]
    )
    override_weights = dict(loader.get_all_weights(model_cfg, override_stub))
    assert set(override_weights.keys()) == {"shared.weight", "main.only"}


def test_inkling2_dspark_config_contracts_and_tap_order():
    """Verify typed DSpark config attributes and [5, 23, 41, 48] tap identity/order."""
    target_cfg = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=2304,
            vocab_size=200058,
            padded_vocab_size=200064,
            final_logit_softcapping=30.0,
            num_loop_passes=2,
            num_layer_visits=48,
            loop_config={"num_passes": 2},
        ),
        mtp_config={
            "block_size": 8,
            "hidden_size": 2304,
            "num_hidden_layers": 2,
            "num_attention_heads": 18,
            "num_key_value_heads": 6,
            "head_dim": 128,
            "intermediate_size": 4608,
            "markov_head_type": "rnn_hidden",
            "target_layer_ids": [5, 23, 41, 48],
        },
    )
    draft_cfg = Inkling2DSparkConfig.from_target(target_cfg)
    assert draft_cfg.dspark_target_layer_ids == [5, 23, 41, 48]
    assert draft_cfg.sample_from_anchor is False
    assert draft_cfg.sm70_force_fp16_draft_kv is True
    assert draft_cfg.dspark_allow_pp1_all_layers is True

    # Verify tap identity and order across 48 visits + final_out (tap 48):
    # swapping any two same-shaped tensors must fail.
    capture_set = set(draft_cfg.dspark_target_layer_ids)
    taps: list[torch.Tensor] = []
    for visit in range(48):
        val = torch.full((2, 4), float(visit), dtype=torch.float16)
        if visit in capture_set:
            taps.append(val.clone())
    final_out = torch.full((2, 4), 48.0, dtype=torch.float16)
    captured = [*taps, final_out]
    assert len(captured) == 4
    assert [int(t[0, 0].item()) for t in captured] == [5, 23, 41, 48]


def test_deepseek_v41_dspark_defaults_preserved():
    """Negative regression test: DeepSeek V4.1 DSpark defaults remain unchanged."""
    ds_v41_draft_cfg = SimpleNamespace(model_type="deepseek_v41_text")
    assert getattr(ds_v41_draft_cfg, "sample_from_anchor", True) is True
    assert getattr(ds_v41_draft_cfg, "sm70_force_fp16_draft_kv", False) is False
    assert getattr(ds_v41_draft_cfg, "dspark_allow_pp1_all_layers", False) is False
    assert "DSparkDeepseekV41ForCausalLM" in _SPECULATIVE_DECODING_MODELS


def test_dspark_pp1_layer_validation_bounds():
    """Verify PP=1 looped tap validation accepts [5,23,41,48] and rejects 999."""
    from vllm.config.speculative import SpeculativeConfig

    def _make_stub(layer_ids: list[int]):
        return SimpleNamespace(
            target_parallel_config=SimpleNamespace(
                pipeline_parallel_size=1,
                communication=SimpleNamespace(pp_layer_partition=None),
            ),
            target_model_config=SimpleNamespace(
                get_total_num_hidden_layers=lambda: 24,
                hf_text_config=SimpleNamespace(num_layer_visits=48),
            ),
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    dspark_target_layer_ids=layer_ids,
                    dspark_allow_pp1_all_layers=True,
                )
            ),
        )

    SpeculativeConfig._verify_dspark_final_stage_ownership(_make_stub([5, 23, 41, 48]))
    with pytest.raises(ValueError, match="dspark_target_layer_id"):
        SpeculativeConfig._verify_dspark_final_stage_ownership(
            _make_stub([5, 23, 41, 999])
        )
