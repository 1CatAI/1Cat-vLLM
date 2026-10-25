# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the GLM DFlash2 checkpoint and heterogeneous PP layout.

These tests exercise real model boundary/sharing methods with tiny arithmetic
layers. They do not substitute for real distributed execution or GPU parity.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.config.speculative import _get_dflash2_checkpoint_draft_tokens
from vllm.model_executor.model_loader.gguf_adapters import get_gguf_adapter
from vllm.model_executor.model_loader.gguf_adapters.dflash import DFlashAdapter
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
from vllm.model_executor.models.shared_weights import (
    DFLASH_DRAFT_WEIGHTS,
    share_embeddings,
    share_lm_head,
    validate_dflash_shared_weights,
)
from vllm.models.glm5next.nvidia.model import Glm5NextModel
from vllm.transformers_utils.configs.eagle import EAGLEConfig
from vllm.transformers_utils.gguf_config import gguf_config_from_metadata
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)

_BOUNDARIES = (6, 15, 25, 34, 43)


@pytest.fixture
def glm_draft_config():
    # Relevant fields from GLM-5.3-Flash-DFlash2-F16.gguf. Vocab size is
    # recovered from tokenizer.ggml.tokens in the actual checkpoint reader.
    metadata = {
        "general.architecture": "dflash",
        "dflash.block_count": 5,
        "dflash.context_length": 1048576,
        "dflash.embedding_length": 4096,
        "dflash.feed_forward_length": 12288,
        "dflash.attention.head_count": 32,
        "dflash.attention.head_count_kv": 8,
        "dflash.attention.causal": False,
        "dflash.rope.freq_base": 10000.0,
        "dflash.attention.layer_norm_rms_epsilon": 1e-5,
        "dflash.attention.key_length": 128,
        "dflash.attention.value_length": 128,
        "dflash.block_size": 8,
        "dflash.conv_kernel_size": 2,
        "dflash.conv_group_size": 16,
        "dflash.selector_rank": 256,
        "dflash.selector_top_k": 16,
        "dflash.target_layers": list(_BOUNDARIES),
        "dflash.attention.sliding_window": 2048,
        "dflash.attention.sliding_window_pattern": [True] * 5,
        "dflash.vocab_size": 154880,
        "tokenizer.ggml.mask_token_id": 154856,
    }
    return EAGLEConfig(model=gguf_config_from_metadata(metadata), method="dflash")


def test_glm_checkpoint_native_dflash_contract(glm_draft_config):
    config = glm_draft_config
    assert config.architectures == ["DFlash2DraftModel"]
    assert config.hidden_size == 4096
    assert config.dflash_config["target_layer_ids"] == [5, 14, 24, 33, 42]
    spec = SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=config))
    assert get_eagle3_aux_layers_from_config(spec) == _BOUNDARIES
    assert _get_dflash2_checkpoint_draft_tokens(config) == 7
    assert config.dflash_config["mask_token_id"] < config.vocab_size
    assert config.layer_types == ["sliding_attention"] * 5
    assert not config.is_causal
    assert isinstance(get_gguf_adapter(config, tp_size=1), DFlashAdapter)


class _IncrementLayer(nn.Module):
    def __init__(self, layer_idx, mhc):
        super().__init__()
        self.layer_idx = layer_idx
        self.mhc = mhc

    def hc_post(self, hidden, residual, post, comb):
        return residual + hidden.unsqueeze(1)

    def forward(self, positions, hidden, residual, post, comb):
        if not self.mhc:
            # Later layers overwrite the activation: aux captures must own a
            # snapshot instead of aliasing this working buffer.
            hidden.add_(1)
            return hidden, hidden, None, None
        if residual is not None:
            streams = self.hc_post(hidden, residual, post, comb)
        elif hidden.ndim == 3:
            streams = hidden
        else:
            streams = hidden[:, None].expand(-1, 4, -1).clone()
        shape = (hidden.shape[0], streams.shape[-1])
        return (
            torch.ones(shape, dtype=hidden.dtype),
            streams,
            torch.ones(hidden.shape[0], 4, 1),
            torch.eye(4).expand(hidden.shape[0], -1, -1),
        )


@pytest.mark.parametrize("num_tokens", [1, 8])
@pytest.mark.parametrize(
    "mhc,materialized", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize(
    "partition",
    [
        # Actual deployment: GPU6 is limited to two layers, final stage owns
        # the draft, and every selected aux boundary is inside a stage.
        (9, 5, 5, 2, 7, 7, 10),
        # Captures exactly at a PP boundary must be sent once and preserved.
        (6, 6, 7, 6, 9, 9, 2),
    ],
)
def test_glm_aux_snapshots_cross_all_seven_stages(
    monkeypatch, num_tokens, mhc, materialized, partition
):
    group = SimpleNamespace(is_first_rank=True, is_last_rank=False)
    monkeypatch.setattr("vllm.models.glm5next.nvidia.model.get_pp_group", lambda: group)
    config = SimpleNamespace(hidden_size=8, mhc=mhc, mhc_num_residual_streams=4)
    intermediate = None
    start = 0
    for rank, layer_count in enumerate(partition):
        group.is_first_rank = rank == 0
        group.is_last_rank = rank == 6
        model = Glm5NextModel.__new__(Glm5NextModel)
        nn.Module.__init__(model)
        model.config = config
        model.start_layer = start
        model.end_layer = start + layer_count
        model.is_sequence_parallel = False
        model._materialize_pp_mhc_boundary = materialized
        model._pp_aux_dump = SimpleNamespace(policy=SimpleNamespace(directory=None))
        model._active_layers = nn.ModuleList(
            _IncrementLayer(index, mhc) for index in range(start, model.end_layer)
        )
        model.norm = nn.Identity()
        model._set_aux_hidden_state_layers(_BOUNDARIES)
        if intermediate is not None:
            expected = model.make_empty_intermediate_tensors(
                num_tokens, torch.float16, torch.device("cpu")
            )
            assert tuple(intermediate.tensors) == tuple(expected.tensors)
            for key, value in intermediate.tensors.items():
                assert value.shape == expected[key].shape
                assert value.dtype == expected[key].dtype
        result = model(
            input_ids=None,
            positions=torch.arange(num_tokens),
            intermediate_tensors=intermediate,
            inputs_embeds=(
                torch.zeros(num_tokens, 8, dtype=torch.float16) if rank == 0 else None
            ),
        )
        start = model.end_layer
        if rank < 6:
            intermediate = result
    assert start == 45
    _, snapshots = result
    assert len(snapshots) == len(_BOUNDARIES)
    for snapshot, boundary in zip(snapshots, _BOUNDARIES):
        torch.testing.assert_close(snapshot, torch.full_like(snapshot, boundary))

    # Exercise the actual draft concatenation/projection boundary as well.
    class Projection(nn.Module):
        input_size = 5 * 8
        quant_method = SimpleNamespace(params_dtype=torch.float16)

        def forward(self, hidden):
            assert hidden.shape == (num_tokens, self.input_size)
            return hidden.reshape(num_tokens, 5, 8).mean(1)

    draft = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(draft)
    draft.model = SimpleNamespace(fc=Projection(), use_aux_hidden_state=True)
    draft._sampling_policy = SimpleNamespace(compact_aux_hidden=True)
    combined = draft.combine_aux_hidden_states(snapshots)
    torch.testing.assert_close(
        combined, torch.full_like(combined, sum(_BOUNDARIES) / 5)
    )


class _PackedVocab(nn.Module):
    def __init__(self):
        super().__init__()
        # Intentionally has no dense .weight. Sharing must retain the module,
        # including its packing metadata and quantization method.
        self.register_buffer("qweight", torch.ones(3, 8, dtype=torch.uint8))
        self.register_buffer("qweight_type", torch.tensor(14))


@pytest.mark.parametrize("replica_loaded", [False, True])
def test_dflash_shares_packed_last_stage_vocab_and_checks_replica(replica_loaded):
    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = _PackedVocab()
    target.model.embed_tokens._dflash_pp_replica_expected = True
    target.model.embed_tokens._dflash_pp_replica_loaded = replica_loaded
    target.lm_head = _PackedVocab()
    wrapper = SimpleNamespace(language_model=target)
    draft = nn.Module()
    draft.model = nn.Module()
    draft.has_own_embed_tokens = False
    draft.has_own_lm_head = False
    shared_embedding = share_embeddings(draft, target, DFLASH_DRAFT_WEIGHTS, pp_size=7)
    shared_head = share_lm_head(draft, wrapper, target, DFLASH_DRAFT_WEIGHTS)
    assert shared_embedding and shared_head
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head
    if replica_loaded:
        validate_dflash_shared_weights(draft, True, True, is_last_rank=True)
    else:
        with pytest.raises(RuntimeError, match="replica.*was not loaded"):
            validate_dflash_shared_weights(draft, True, True, is_last_rank=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_dflash_quantized_target_head_requires_explicit_fallback(monkeypatch, enabled):
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash2._flashinfer_topk", lambda: None
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash2.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0, 99.0, 100.0]])

    class QuantMethod:
        def apply(self, layer, hidden, bias):
            assert hidden.shape == (1, 8)
            assert bias is None
            return logits.clone()

    draft = DFlash2Qwen3ForCausalLM.__new__(DFlash2Qwen3ForCausalLM)
    nn.Module.__init__(draft)
    draft._sm70_dflash2_policy = Sm70DFlash2Config(quant_lm_head=enabled)
    draft._sm70_dflash2_policy.resolve(qualified=False)
    draft.lm_head = SimpleNamespace(
        quant_method=QuantMethod(),
        shard_indices=SimpleNamespace(num_org_vocab_padding=2, org_vocab_start_index=0),
    )
    draft.model = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=2))
    draft.output_multiplier = 0.5
    draft.final_logit_softcapping = None
    if not enabled:
        with pytest.raises(ValueError, match="dense-logit fallback"):
            draft.compute_candidates(torch.zeros(1, 8))
    else:
        ids, values = draft.compute_candidates(torch.zeros(1, 8))
        torch.testing.assert_close(ids, torch.tensor([[1, 2]]))
        torch.testing.assert_close(values, torch.tensor([[2.0, 1.5]]))
