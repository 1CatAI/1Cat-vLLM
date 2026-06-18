# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import pytest

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
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


def _proposer(method: str, model: object) -> SpecDecodeBaseProposer:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.method = method
    proposer.model = model
    proposer._enable_probabilistic_draft_probs = False
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
