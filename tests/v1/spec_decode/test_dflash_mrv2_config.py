# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.config.scheduler import SchedulerConfig
from vllm.config.speculative import (
    SpeculativeConfig,
    get_dflash_model_draft_tokens,
    uses_adaptive_dflash_lookup,
)
from vllm.config.vllm import VllmConfig
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    _prepare_dflash_inputs_to_capture,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu.spec_decode.utils import (
    DraftTokensHandler,
    get_parallel_drafting_token_id,
)


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
    assert _config("dflash").max_num_new_target_slots_for_drafting == 0
    # The retained DDTree path keeps its existing flat-parallel slot contract.
    assert _config("dflash_ddtree").max_num_new_slots_for_drafting == 6
    assert _config("dflash_ddtree").max_num_new_target_slots_for_drafting == 6


def test_mrv2_dflash_preserves_target_scheduler_token_budget() -> None:
    scheduler = SimpleNamespace(
        max_num_seqs=512,
        max_num_batched_tokens=4096,
        max_num_scheduled_tokens=None,
    )
    config = SimpleNamespace(
        speculative_config=_config("dflash"),
        scheduler_config=scheduler,
    )

    VllmConfig._set_max_num_scheduled_tokens(config)

    assert scheduler.max_num_scheduled_tokens == 4096


@pytest.mark.parametrize("target_capacity", [8, 64])
def test_dflash_embedding_buffer_covers_expanded_query_capacity(
    monkeypatch, target_capacity
) -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as module

    def init_base(self, config, device):
        self.device = device
        self.dtype = torch.float16
        self.hidden_size = 3
        self.max_num_tokens = target_capacity
        self.max_num_reqs = 4
        self.num_speculative_steps = 7
        self.speculative_config = config.speculative_config
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={})
        )

    monkeypatch.setattr(module.DraftModelSpeculator, "__init__", init_base)
    monkeypatch.setattr(
        module, "InputBuffers", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(module, "get_dflash_model_draft_tokens", lambda config: 7)
    monkeypatch.setattr(module, "get_parallel_drafting_token_id", lambda config: 0)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda config: False,
    )
    config = SimpleNamespace(speculative_config=SimpleNamespace())
    speculator = module.DFlashSpeculator(config, torch.device("cpu"))
    capacity = max(target_capacity, 4 * 8)
    assert speculator.max_num_tokens == capacity
    assert speculator.hidden_states.shape == (capacity, 3)
    assert speculator.inputs_embeds.shape == (capacity, 3)


def test_dflash_capture_uses_its_own_persistent_slot_mapping(monkeypatch) -> None:
    query_slots = torch.zeros((2, 16), dtype=torch.int64)
    seen: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dflash.cudagraph.InputBatch.make_dummy",
        lambda *args: SimpleNamespace(),
    )

    def fake_build_slot_mappings(slot_mappings, kv_cache_config):
        del kv_cache_config
        seen["slot_mappings"] = slot_mappings
        return {"draft.layer": slot_mappings[1]}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dflash.cudagraph.build_slot_mappings_by_layer",
        fake_build_slot_mappings,
    )
    block_tables = SimpleNamespace(
        cp_size=1,
        get_dummy_block_tables=lambda num_reqs: (),
    )

    state = _prepare_dflash_inputs_to_capture(
        num_reqs=1,
        num_tokens=8,
        input_buffers=object(),
        block_tables=block_tables,
        query_slot_mappings=query_slots,
        attn_groups=[],
        kv_cache_config=object(),
        max_model_len=1024,
        skip_attn=True,
        causal=False,
    )

    captured = seen["slot_mappings"]
    assert captured.shape == (2, 8)
    assert captured.data_ptr() == query_slots.data_ptr()
    assert torch.all(query_slots == PAD_SLOT_ID)
    assert state.slot_mappings["draft.layer"].data_ptr() == query_slots[1].data_ptr()


@pytest.mark.parametrize(
    ("assist", "selector_top_k", "verify_tokens", "expected"),
    [
        (True, 16, 15, 7),
        (False, 16, 15, 15),
        (True, 0, 15, 15),
        (True, 16, 7, 7),
    ],
)
def test_lookup_decouples_checkpoint_and_verify_widths(
    assist: bool,
    selector_top_k: int,
    verify_tokens: int,
    expected: int,
) -> None:
    config = SimpleNamespace(
        method="dflash",
        ngram_assist=assist,
        num_speculative_tokens=verify_tokens,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                dflash_config={
                    "block_size": 8,
                    "selector_top_k": selector_top_k,
                }
            )
        ),
    )

    assert get_dflash_model_draft_tokens(config) == expected
    assert uses_adaptive_dflash_lookup(config) is (expected < verify_tokens)


def test_draft_token_handler_exposes_only_selected_verify_width() -> None:
    handler = object.__new__(DraftTokensHandler)
    handler.req_ids = []
    handler.draft_tokens_np = None
    handler.num_draft_tokens = 0
    batch = SimpleNamespace(
        req_ids=["req-0"],
        has_structured_output_reqs=False,
    )
    proposals = torch.zeros((1, 15), dtype=torch.int64)

    handler.set_draft_tokens(batch, proposals, num_draft_tokens=7)

    assert handler.num_draft_tokens == 7
    assert handler.req_ids == ["req-0"]
    with pytest.raises(ValueError, match="proposal tensor width"):
        handler.set_draft_tokens(batch, proposals, num_draft_tokens=16)


def _adaptive_lookup_config() -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    config.method = "dflash"
    config.ngram_assist = True
    config.num_speculative_tokens = 15
    config.ddtree_disable_tree_verify = False
    config.disable_padded_drafter_batch = False
    config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            dflash_config={"block_size": 8, "selector_top_k": 16}
        ),
        verify_with_parallel_config=lambda *_args: None,
        verify_with_model_config=lambda *_args: None,
    )
    config.verify_with_parallel_config = lambda *_args: None
    config.verify_with_model_config = lambda *_args: None
    return config


def test_adaptive_lookup_disables_async_scheduling_by_default(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_ADAPTIVE", "1")
    scheduler = SchedulerConfig(
        max_model_len=8192,
        is_encoder_decoder=False,
        async_scheduling=None,
    )

    config = VllmConfig(
        scheduler_config=scheduler,
        speculative_config=_adaptive_lookup_config(),
    )

    assert config.scheduler_config.async_scheduling is False


def test_adaptive_lookup_rejects_explicit_async_scheduling(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_ADAPTIVE", "1")
    scheduler = SchedulerConfig(
        max_model_len=8192,
        is_encoder_decoder=False,
        async_scheduling=True,
    )

    with pytest.raises(ValueError, match="adaptive q8/q16 lookup verification"):
        VllmConfig(
            scheduler_config=scheduler,
            speculative_config=_adaptive_lookup_config(),
        )


def test_dflash_forces_v2_and_rejects_explicit_v1(monkeypatch) -> None:
    fake = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
        model_config=None,
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


def test_dflash_aux_projection_matches_fc_weight_dtype() -> None:
    class FC:
        input_size = 4
        weight = torch.empty((4, 4), dtype=torch.float16)

        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            assert hidden_states.dtype == self.weight.dtype
            return hidden_states

    model = SimpleNamespace(
        model=SimpleNamespace(use_aux_hidden_state=True, fc=FC()),
    )
    hidden_states = torch.ones((2, 4), dtype=torch.float32)

    output = DFlashQwen3ForCausalLM.combine_hidden_states(model, hidden_states)

    assert output.dtype == torch.float16


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
