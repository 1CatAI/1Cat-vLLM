# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.sample.sampler as sampler_module
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler


def _sampler() -> Sampler:
    sampler = object.__new__(Sampler)
    sampler.compute_nans = False
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.array([0.0, 0.0], dtype=np.float32)),
        max_num_logprobs=lambda _indices: -1,
    )
    sampler.logprob_token_ids_state = SimpleNamespace(
        max_num_token_ids=lambda _indices: 0
    )
    sampler.logit_bias_state = SimpleNamespace(use_logit_bias=np.array([False, False]))
    sampler.penalties_state = SimpleNamespace(use_penalty=np.array([False, False]))
    sampler.bad_words_state = SimpleNamespace(
        num_bad_words=SimpleNamespace(np=np.array([0, 0], dtype=np.int32))
    )
    return sampler


def test_sm70_v2_greedy_fastpath_accepts_plain_temperature_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sampler_module.envs, "VLLM_SM70_GREEDY_TOKEN_FASTPATH", True)
    sampler = _sampler()
    input_batch = SimpleNamespace(idx_mapping_np=np.array([1], dtype=np.int32))

    assert sampler.can_use_sm70_greedy_token_fastpath(input_batch)


def test_sm70_v2_decode_uses_model_top_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
            assert hidden_states.shape == (1, 4)
            return torch.tensor([42], dtype=torch.int64)

        def compute_logits(self, _hidden_states: torch.Tensor) -> torch.Tensor:
            raise AssertionError("full logits must not run on the greedy fastpath")

    runner = object.__new__(GPUModelRunner)
    runner.model = Model()
    runner.lora_config = None
    runner.sampler = SimpleNamespace(
        can_use_sm70_greedy_token_fastpath=lambda _input_batch: True
    )
    runner.device = torch.device("cuda")
    runner.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=None))
    input_batch = SimpleNamespace(
        logits_indices=torch.tensor([0]),
        num_draft_tokens=0,
        num_reqs=1,
        num_tokens=1,
        is_prefilling_np=np.array([False]),
        seq_lens=torch.tensor([17], dtype=torch.int32),
        cu_num_logits=None,
        idx_mapping=None,
    )
    monkeypatch.setattr(
        model_runner_module.current_platform,
        "is_device_capability",
        lambda _capability: True,
    )
    monkeypatch.setattr(
        model_runner_module,
        "get_num_sampled_and_rejected",
        lambda *_args: (
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        ),
    )

    output, num_sampled, num_rejected = GPUModelRunner.sample(
        runner, torch.zeros(1, 4), input_batch, None
    )

    assert output.sampled_token_ids.tolist() == [[42]]
    assert num_sampled.tolist() == [1]
    assert num_rejected.tolist() == [0]

    runner.lora_config = object()
    with pytest.raises(AssertionError, match="full logits must not run"):
        GPUModelRunner.sample(runner, torch.zeros(1, 4), input_batch, None)


@pytest.mark.parametrize(
    "blocker",
    [
        "disabled",
        "nan_count",
        "random",
        "logprobs",
        "logprob_token_ids",
        "logit_bias",
        "penalty",
        "bad_words",
    ],
)
def test_sm70_v2_greedy_fastpath_rejects_non_equivalent_sampling(
    monkeypatch: pytest.MonkeyPatch, blocker: str
) -> None:
    monkeypatch.setattr(
        sampler_module.envs,
        "VLLM_SM70_GREEDY_TOKEN_FASTPATH",
        blocker != "disabled",
    )
    sampler = _sampler()
    input_batch = SimpleNamespace(idx_mapping_np=np.array([1], dtype=np.int32))

    if blocker == "nan_count":
        sampler.compute_nans = True
    elif blocker == "random":
        sampler.sampling_states.temperature.np[1] = 0.5
    elif blocker == "logprobs":
        sampler.sampling_states.max_num_logprobs = lambda _indices: 1
    elif blocker == "logprob_token_ids":
        sampler.logprob_token_ids_state.max_num_token_ids = lambda _indices: 1
    elif blocker == "logit_bias":
        sampler.logit_bias_state.use_logit_bias[1] = True
    elif blocker == "penalty":
        sampler.penalties_state.use_penalty[1] = True
    elif blocker == "bad_words":
        sampler.bad_words_state.num_bad_words.np[1] = 1

    assert not sampler.can_use_sm70_greedy_token_fastpath(input_batch)
