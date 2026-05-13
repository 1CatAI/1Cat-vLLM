# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.config.speculative import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    _get_dflash_per_layer_sliding_window,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.spec_decode.dflash import DFlashProposer


def test_dflash_speculative_helpers() -> None:
    spec = object.__new__(SpeculativeConfig)
    spec.method = "dflash"

    assert spec.use_dflash()
    assert spec.use_eagle()
    assert not spec.uses_draft_model()


def test_eagle_model_mixin_collects_aux_hidden_states() -> None:
    class DummyModel(EagleModelMixin):
        pass

    model = DummyModel()
    model._set_aux_hidden_state_layers((0, 2))

    hidden_states = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[3.0, 4.0]])
    aux_hidden_states: list[torch.Tensor] = []

    model._maybe_add_hidden_state(aux_hidden_states, 0, hidden_states, residual)
    model._maybe_add_hidden_state(aux_hidden_states, 1, hidden_states, residual)
    model._maybe_add_hidden_state(aux_hidden_states, 2, hidden_states, residual)

    assert len(aux_hidden_states) == 2
    assert torch.equal(aux_hidden_states[0], hidden_states + residual)
    assert torch.equal(aux_hidden_states[1], hidden_states + residual)


def test_eagle_model_mixin_snapshots_aux_hidden_states() -> None:
    class DummyModel(EagleModelMixin):
        pass

    model = DummyModel()
    model._set_aux_hidden_state_layers((0,))

    hidden_states = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[3.0, 4.0]])
    aux_hidden_states: list[torch.Tensor] = []

    model._maybe_add_hidden_state(aux_hidden_states, 0, hidden_states, residual)
    hidden_states.add_(10.0)
    residual.add_(10.0)

    assert len(aux_hidden_states) == 1
    assert torch.equal(aux_hidden_states[0], torch.tensor([[4.0, 6.0]]))


def test_qwen35_model_forward_returns_aux_hidden_states(monkeypatch) -> None:
    class DummyLayer(nn.Module):
        def __init__(self, hidden_delta: float, residual_delta: float):
            super().__init__()
            self.hidden_delta = hidden_delta
            self.residual_delta = residual_delta

        def forward(self, *, positions, hidden_states, residual):
            del positions
            if residual is None:
                residual = torch.zeros_like(hidden_states)
            return (
                hidden_states + self.hidden_delta,
                residual + self.residual_delta,
            )

    class DummyNorm(nn.Module):
        def forward(self, hidden_states, residual):
            return hidden_states + residual, residual

    model = object.__new__(Qwen3_5Model)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([DummyLayer(10.0, 100.0), DummyLayer(20.0, 200.0)])
    model.norm = DummyNorm()
    model._set_aux_hidden_state_layers((1, 2))

    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_5.get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    final_hidden, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.zeros(1, dtype=torch.int64),
        inputs_embeds=torch.tensor([[1.0, 2.0]]),
    )

    assert torch.equal(final_hidden, torch.tensor([[331.0, 332.0]]))
    assert len(aux_hidden_states) == 2
    assert torch.equal(aux_hidden_states[0], torch.tensor([[111.0, 112.0]]))
    assert torch.equal(aux_hidden_states[1], torch.tensor([[331.0, 332.0]]))


def test_dflash_model_registry_entry_present() -> None:
    assert _TEXT_GENERATION_MODELS["DFlashDraftModel"] == (
        "qwen3_dflash",
        "DFlashQwen3ForCausalLM",
    )


def test_dflash_uses_per_layer_sliding_window_for_qwen36_draft() -> None:
    config = SimpleNamespace(
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=2048,
    )

    assert _get_dflash_per_layer_sliding_window(config, 0) == 2048
    assert _get_dflash_per_layer_sliding_window(config, 1) == 2048
    assert _get_dflash_per_layer_sliding_window(config, 2) is None


def test_dflash_allows_mixed_kv_cache_groups() -> None:
    proposer = object.__new__(DFlashProposer)
    proposer._draft_attn_layer_names = {"draft.layers.0.attn", "draft.layers.4.attn"}

    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["draft.layers.0.attn"]),
            SimpleNamespace(layer_names=["draft.layers.4.attn"]),
        ]
    )

    DFlashProposer.validate_same_kv_cache_group(proposer, kv_cache_config)


def test_dflash_slot_mapping_uses_layer_kv_cache_group() -> None:
    proposer = object.__new__(DFlashProposer)
    proposer._draft_attn_layer_names = {"draft.layers.0.attn", "draft.layers.4.attn"}
    proposer.draft_layer_to_kv_cache_gid = {
        "draft.layers.0.attn": 0,
        "draft.layers.4.attn": 1,
    }
    proposer.kv_cache_gid = 0
    proposer._query_slot_mapping_buffers_by_gid = {
        0: torch.tensor([1, 2, 3], dtype=torch.int64),
        1: torch.tensor([4, 5, 6], dtype=torch.int64),
    }

    slot_mapping = DFlashProposer._get_slot_mapping(proposer, 2)

    assert torch.equal(
        slot_mapping["draft.layers.0.attn"], torch.tensor([1, 2], dtype=torch.int64)
    )
    assert torch.equal(
        slot_mapping["draft.layers.4.attn"], torch.tensor([4, 5], dtype=torch.int64)
    )


def test_dflash_combine_hidden_states_avoids_double_hidden_norm() -> None:
    class AddOne(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(4))

        def forward(self, x):
            return x + 1.0

    class AddHundred(nn.Module):
        def forward(self, x):
            return x + 100.0

    model = object.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(model)
    model.model = SimpleNamespace(
        use_aux_hidden_state=True,
        fc=AddOne(),
        hidden_norm=AddHundred(),
    )

    hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    combined = model.combine_hidden_states(hidden_states)

    assert torch.equal(combined, hidden_states + 1.0)


def test_gdn_builder_prefers_runner_spec_mask_for_padded_mixed_batch() -> None:
    device = torch.device("cpu")
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        cache_config=SimpleNamespace(mamba_cache_mode="all"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    kv_cache_spec = MambaSpec(
        block_size=4,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="all",
        num_speculative_blocks=2,
    )
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec,
        ["model.layers.0.linear_attn"],
        vllm_config,
        device,
    )

    query_start_loc = torch.tensor([0, 3, 5, 8, 8], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([13, 9, 7, 0], dtype=torch.int32, device=device)
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens.cpu(),
        _num_computed_tokens_cpu=torch.tensor([10, 7, 4, 0], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=8,
        max_query_len=3,
        max_seq_len=13,
        block_table_tensor=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [10, 11, 12, 13, 14, 15, 16, 17],
                [20, 21, 22, 23, 24, 25, 26, 27],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
            device=device,
        ),
        slot_mapping=torch.arange(8, dtype=torch.int64, device=device),
        causal=True,
    )

    num_accepted_tokens = torch.tensor([2, 1, 1, 1], dtype=torch.int32, device=device)
    # The explicit mask is the source of truth. This draft-count tensor is
    # intentionally misaligned to catch accidental mask recomputation.
    num_decode_draft_tokens_cpu = torch.tensor([2, -1, -1, 2], dtype=torch.int32)
    spec_sequence_masks_cpu = torch.tensor([True, False, True, False])

    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        spec_sequence_masks_cpu=spec_sequence_masks_cpu,
    )

    assert attn_metadata.num_spec_decodes == 2
    assert attn_metadata.num_prefills == 1
    assert attn_metadata.num_decodes == 0
    assert torch.equal(attn_metadata.spec_sequence_masks, spec_sequence_masks_cpu)
    assert torch.equal(
        attn_metadata.spec_query_start_loc,
        torch.tensor([0, 3, 6], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.non_spec_query_start_loc,
        torch.tensor([0, 2, 2], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.num_accepted_tokens,
        torch.tensor([2, 1], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.spec_state_indices_tensor,
        torch.tensor([[3, 4, 5], [21, 22, 23]], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.non_spec_state_indices_tensor,
        torch.tensor([12, -1], dtype=torch.int32),
    )
