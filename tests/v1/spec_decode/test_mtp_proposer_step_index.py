# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import pytest
import torch

from vllm.forward_context import BatchDescriptor
from vllm.v1.spec_decode import step3p5 as step3p5_module
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _MTPModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        self.calls.append(("top", spec_step_idx))
        return torch.full(
            (hidden_states.shape[0],),
            spec_step_idx,
            dtype=torch.int64,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        self.calls.append(("logits", spec_step_idx))
        logits = torch.zeros((hidden_states.shape[0], 8), dtype=torch.float32)
        logits[:, spec_step_idx] = 1
        return logits


class _PlainModel:
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.full((hidden_states.shape[0],), 7, dtype=torch.int64)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((hidden_states.shape[0], 8), dtype=torch.float32)
        logits[:, 6] = 1
        return logits


class _GreedySamplingMetadata:
    all_greedy = True


class _InputBatch:
    req_ids = ["current"]


class _PrevPositions:
    np = [0]


class _InputIds:
    def __init__(self) -> None:
        self.gpu = torch.zeros(2, dtype=torch.int32)

    def copy_to_gpu(self, num_tokens: int) -> None:
        raise AssertionError("input_ids CPU copy should not be needed")


class _SchedulerOutput:
    scheduled_spec_decode_tokens = {"r0": [123]}


class _Step3p5CommonAttentionMetadata:
    def __init__(self) -> None:
        self.num_actual_tokens = 1
        self.max_query_len = 1
        self.query_start_loc = torch.zeros(2, dtype=torch.int32)
        self.query_start_loc_cpu = torch.zeros(2, dtype=torch.int32)
        self.seq_lens = torch.ones(1, dtype=torch.int32)
        self._seq_lens_cpu = None
        self._num_computed_tokens_cpu = None
        self.slot_mapping = torch.zeros(1, dtype=torch.int64)

    def batch_size(self) -> int:
        return 1


class _Step3p5Model:
    def __init__(self) -> None:
        self.spec_step_indices: list[int] = []

    def __call__(self, **kwargs) -> torch.Tensor:
        self.spec_step_indices.append(kwargs["spec_step_idx"])
        return torch.zeros((1, 4), dtype=torch.float32)


def _proposer(method: str, model: object) -> SpecDecodeBaseProposer:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.method = method
    proposer.model = model
    proposer._enable_probabilistic_draft_probs = False
    proposer._static_draft_vocab = None
    proposer.use_local_argmax_reduction = True
    return proposer


def test_mtp_greedy_sample_passes_spec_step_idx_to_model():
    model = _MTPModel()
    proposer = _proposer("mtp", model)

    tokens = proposer._greedy_sample(torch.zeros((2, 4)), spec_step_idx=3)

    assert torch.equal(tokens, torch.tensor([3, 3]))
    assert model.calls == [("top", 3)]


def test_mtp_logits_sample_passes_spec_step_idx_to_model():
    model = _MTPModel()
    proposer = _proposer("mtp", model)
    proposer.use_local_argmax_reduction = False

    tokens, probs = proposer._sample_draft_tokens(
        torch.zeros((2, 4)),
        _GreedySamplingMetadata(),  # type: ignore[arg-type]
        spec_step_idx=2,
    )

    assert probs is None
    assert torch.equal(tokens, torch.tensor([2, 2]))
    assert model.calls == [("logits", 2)]


def test_non_mtp_does_not_pass_spec_step_idx_kwarg():
    proposer = _proposer("eagle", _PlainModel())

    tokens = proposer._greedy_sample(torch.zeros((2, 4)), spec_step_idx=3)

    assert torch.equal(tokens, torch.tensor([7, 7]))


def test_step3p5_proposer_uses_batch_descriptor_graph_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer = object.__new__(Step3p5MTPProposer)
    model = _Step3p5Model()
    metadata = _Step3p5CommonAttentionMetadata()
    context_descriptors: list[BatchDescriptor] = []

    proposer.model = model
    proposer._last_draft_probs = None
    proposer.num_speculative_tokens = 2
    proposer.parallel_drafting = False
    proposer.uses_mrope = False
    proposer.constant_draft_positions = True
    proposer.allowed_attn_types = None
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = False
    proposer.block_size = 1
    proposer.positions = torch.zeros(2, dtype=torch.int64)
    proposer.arange = torch.arange(2, dtype=torch.int32)
    proposer.token_arange_np = torch.arange(2, dtype=torch.int32).numpy()
    proposer.input_ids = torch.zeros(2, dtype=torch.int32)
    proposer.hidden_states = torch.zeros((2, 4), dtype=torch.float32)
    proposer.vllm_config = object()
    proposer.set_inputs_first_pass = lambda **kwargs: (
        1,
        torch.tensor([0], dtype=torch.int64),
        metadata,
    )
    proposer.build_per_group_and_layer_attn_metadata = (
        lambda common_attn_metadata, draft_index=0: ([], {})
    )
    proposer._determine_batch_execution_and_padding = lambda num_tokens: (
        None,
        1,
        None,
        BatchDescriptor(num_tokens=1),
    )
    proposer._batch_descriptor_for_spec_step = (
        lambda descriptor, spec_step_idx: BatchDescriptor(
            num_tokens=descriptor.num_tokens,
            graph_variant=17 + spec_step_idx,
        )
    )
    proposer.build_model_inputs_first_pass = lambda *args: ({}, 1)
    proposer.model_returns_tuple = lambda: False
    proposer._sample_draft_tokens_for_step = (
        lambda hidden_states, sampling_metadata, spec_step_idx: (
            torch.tensor([spec_step_idx + 1], dtype=torch.int64),
            None,
        )
    )
    proposer._get_positions = lambda num_tokens: proposer.positions[:num_tokens]
    proposer._get_slot_mapping = lambda *args, **kwargs: {}

    def fake_set_forward_context(*args, **kwargs):
        context_descriptors.append(kwargs["batch_descriptor"])
        return nullcontext()

    monkeypatch.setattr(step3p5_module, "set_forward_context", fake_set_forward_context)

    proposer.propose(
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros((1, 4), dtype=torch.float32),
        torch.zeros(1, dtype=torch.int64),
        None,
        metadata,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
    )

    assert model.spec_step_indices == [17, 18]
    assert [descriptor.graph_variant for descriptor in context_descriptors] == [17, 18]


def test_gemma4_non_centroid_greedy_sample_passes_spec_step_idx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer = object.__new__(Gemma4Proposer)
    proposer._centroids_sizes = []
    observed_spec_step_indices: list[int] = []

    def fake_greedy_sample(
        _proposer: SpecDecodeBaseProposer,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        observed_spec_step_indices.append(spec_step_idx)
        return torch.full((hidden_states.shape[0],), spec_step_idx, dtype=torch.int64)

    monkeypatch.setattr(SpecDecodeBaseProposer, "_greedy_sample", fake_greedy_sample)

    token_ids = proposer._greedy_sample(
        torch.zeros((2, 4), dtype=torch.float32), spec_step_idx=5
    )

    assert torch.equal(token_ids, torch.tensor([5, 5]))
    assert observed_spec_step_indices == [5]


def test_list_draft_tokens_use_generation_req_id_snapshot():
    runner = object.__new__(GPUModelRunner)
    runner._draft_token_ids = [[11, 12]]
    runner._draft_token_req_ids = ["generated"]
    runner.input_batch = _InputBatch()

    draft_token_ids, req_ids = runner._get_draft_token_ids_cpu()

    assert draft_token_ids == [[11, 12]]
    assert req_ids == ["generated"]


def test_prepare_input_ids_rejects_scheduled_spec_without_draft_tensor():
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = type(
        "InputBatch",
        (),
        {
            "req_ids": ["r0"],
            "prev_sampled_token_ids": torch.tensor([[55]], dtype=torch.int32),
        },
    )()
    runner.prev_positions = _PrevPositions()
    runner.input_ids = _InputIds()
    runner.enable_prompt_embeds = False
    runner.pin_memory = False
    runner.device = torch.device("cpu")
    runner._draft_token_ids = None
    runner.num_spec_tokens = 1

    with pytest.raises(RuntimeError, match="has no draft token tensor"):
        runner._prepare_input_ids(
            _SchedulerOutput(),  # type: ignore[arg-type]
            num_reqs=1,
            total_num_scheduled_tokens=2,
            cu_num_tokens=torch.tensor([2], dtype=torch.int32).numpy(),
        )
