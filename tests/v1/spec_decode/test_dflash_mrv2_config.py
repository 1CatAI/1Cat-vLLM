# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id


def _config(method: str, num_speculative_tokens: int = 7) -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    config.method = method
    config.num_speculative_tokens = num_speculative_tokens
    config.parallel_drafting = True
    return config


def test_dflash_and_ddtree_routes_are_disjoint() -> None:
    mrv2 = _config("dflash")
    ddtree = _config("dflash_ddtree")

    assert mrv2.use_dflash()
    assert not mrv2.use_dflash_ddtree()
    assert mrv2.use_dflash_family()
    assert not ddtree.use_dflash()
    assert ddtree.use_dflash_ddtree()
    assert ddtree.use_dflash_family()


def test_mrv2_dflash_reserves_all_mask_slots() -> None:
    assert _config("dflash").max_num_new_slots_for_drafting == 7
    # The retained DDTree path keeps its existing flat-parallel slot contract.
    assert _config("dflash_ddtree").max_num_new_slots_for_drafting == 6


def test_dflash_forces_v2_and_rejects_explicit_v1(monkeypatch) -> None:
    fake = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
    )
    monkeypatch.setattr("vllm.config.vllm.envs.VLLM_USE_V2_MODEL_RUNNER", None)
    assert VllmConfig.use_v2_model_runner.fget(fake)

    monkeypatch.setattr("vllm.config.vllm.envs.VLLM_USE_V2_MODEL_RUNNER", False)
    with pytest.raises(ValueError, match="implemented only by Model Runner V2"):
        VllmConfig.use_v2_model_runner.fget(fake)


def test_dflash_target_layers_use_boundary_indices() -> None:
    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(target_layer_ids=[5, 19, 33, 47, 61])
        )
    )
    assert get_eagle3_aux_layers_from_config(spec_config) == (6, 20, 34, 48, 62)


def test_parallel_drafting_token_id_prefers_dflash_config() -> None:
    config = SimpleNamespace(
        dflash_config={"mask_token_id": 248070},
        pard_token=1,
    )
    assert get_parallel_drafting_token_id(config) == 248070


def test_ddtree_topk_adapter_uses_official_logits_processor() -> None:
    hidden_states = torch.randn(2, 4)
    expected_ids = torch.tensor([[3, 2], [1, 0]])
    expected_logprobs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])

    class Processor:
        def get_topk_tokens_and_logprobs(self, head, states, top_k):
            assert head == "head"
            assert states is hidden_states
            assert top_k == 2
            return expected_ids, expected_logprobs

    model = SimpleNamespace(
        draft_id_to_target_id=None,
        logits_processor=Processor(),
        lm_head="head",
    )
    actual = DFlashQwen3ForCausalLM.get_topk_tokens_and_logprobs(
        model, hidden_states, 2
    )
    assert actual is not None
    assert actual[0] is expected_ids
    assert actual[1] is expected_logprobs


def test_draft_kv_dtype_is_public_and_defaults_to_inherit() -> None:
    assert SpeculativeConfig.kv_cache_dtype is None


def test_ddtree_draft_config_combines_sm70_kv_and_non_causal(monkeypatch) -> None:
    @dataclass(frozen=True)
    class CacheConfig:
        cache_dtype: str

    @dataclass(frozen=True)
    class AttentionConfig:
        use_non_causal: bool

    @dataclass(frozen=True)
    class DraftConfig:
        cache_config: CacheConfig
        attention_config: AttentionConfig

    base = DraftConfig(
        cache_config=CacheConfig(cache_dtype="fp8_e5m2"),
        attention_config=AttentionConfig(use_non_causal=False),
    )
    proposer = object.__new__(DFlashProposer)
    proposer.speculative_config = SimpleNamespace(kv_cache_dtype=None)
    proposer.dflash_causal = False

    monkeypatch.setattr(
        SpecDecodeBaseProposer,
        "_create_draft_vllm_config",
        lambda _self: base,
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.dflash.current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability=lambda capability: capability == 70,
        ),
    )

    draft = DFlashProposer._create_draft_vllm_config(proposer)

    assert draft.cache_config.cache_dtype == "auto"
    assert draft.attention_config.use_non_causal is True


def test_draft_attention_metadata_resolves_causal_per_kv_group() -> None:
    class Builder:
        def build(self, common_prefix_len, common_attn_metadata, **_kwargs):
            assert common_prefix_len == 0
            return common_attn_metadata

    class Group:
        def __init__(self, layer_name: str):
            self.layer_names = [layer_name]
            self.builder = Builder()

        def get_metadata_builder(self, _index: int):
            return self.builder

    attn_metadata = build_attn_metadata(
        attn_groups=[[Group("layer.0")], [Group("layer.1")]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        block_tables=[
            torch.zeros((1, 1), dtype=torch.int32),
            torch.zeros((1, 1), dtype=torch.int32),
        ],
        slot_mappings=torch.zeros((2, 1), dtype=torch.int64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object(), object()]),
        causal={0: False, 1: True},
    )

    assert attn_metadata["layer.0"].causal is False
    assert attn_metadata["layer.1"].causal is True
