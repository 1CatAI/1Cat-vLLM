# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test


def prepare(ids, computed, prompts, previous_rows, previous_samples, tokens, sizes):
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.pin_memory = False
    runner.num_spec_tokens = 0
    runner.enable_prompt_embeds = False
    runner.input_batch = SimpleNamespace(
        req_ids=ids,
        prev_req_id_to_index=previous_rows,
        prev_sampled_token_ids=torch.tensor(
            previous_samples, dtype=torch.int32
        ).reshape(-1, 1),
        num_computed_tokens_cpu=np.array(computed),
        num_prompt_tokens=np.array(prompts),
    )
    runner.prev_positions = SimpleNamespace(np=np.empty(len(ids), dtype=np.int32))
    cpu = torch.tensor(tokens, dtype=torch.int32)
    gpu = torch.empty_like(cpu)
    runner.input_ids = SimpleNamespace(cpu=cpu, gpu=gpu)
    runner._copy_buffer_to_gpu = lambda buffer, n: buffer.gpu[:n].copy_(buffer.cpu[:n])
    runner._compute_prev_positions(len(ids))
    runner._prepare_input_ids(
        SimpleNamespace(scheduled_spec_decode_tokens={}),
        len(ids),
        sum(sizes),
        np.cumsum(sizes),
    )
    return gpu


@pytest.mark.parametrize("computed", [0, 2])
def test_recomputed_prompt_does_not_reuse_discarded_sample(computed):
    prompt = [10, 11, 12, 13]
    tokens = prepare(
        ["replaying"],
        [computed],
        [4],
        {"replaying": 0},
        [-1],
        prompt[computed:],
        [4 - computed],
    )
    assert tokens.tolist() == prompt[computed:]
    # Reproduce the real embedding boundary without triggering a CUDA assert.
    assert torch.nn.functional.embedding(tokens.long(), torch.zeros(32, 2)).shape == (
        4 - computed,
        2,
    )


def test_reordered_decode_keeps_gpu_sample_while_prompt_replays_from_cpu():
    tokens = prepare(
        ["decoding", "replaying"],
        [4, 0],
        [4, 4],
        {"replaying": 0, "decoding": 1},
        [-1, 17],
        [-1, 10, 11, 12, 13],
        [1, 4],
    )
    assert tokens.tolist() == [17, 10, 11, 12, 13]


def test_decode_only_still_uses_previous_gpu_sample():
    assert prepare(["decode"], [4], [4], {"decode": 0}, [17], [-1], [1]).item() == 17
