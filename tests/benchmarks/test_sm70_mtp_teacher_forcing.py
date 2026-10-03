# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from benchmarks.sm70_mtp_teacher_forcing import flush, install


class GPUModelRunner:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
        self.speculator = SimpleNamespace(
            method="mtp",
            propose=lambda **kwargs: kwargs,
            run_model=lambda *args, **kwargs: None,
            _sample_draft=lambda *args: None,
            use_local_argmax_reduction=False,
            model=self.model,
            input_buffers=SimpleNamespace(
                positions=torch.tensor([7, 8]),
                input_ids=torch.zeros(2, dtype=torch.long),
            ),
        )
        self.prepare_inputs = lambda batch: batch
        self.sample: Callable[..., Any] = lambda *args: None


def test_forcing_aligns_target_and_shifted_draft(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    runner = GPUModelRunner()
    worker = SimpleNamespace(model_runner=runner)
    tape = list(range(32))
    install(worker, tape, 8, "frozen", str(tmp_path))
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=5,
        positions=torch.arange(7, 12),
        input_ids=torch.zeros(5, dtype=torch.long),
        logits_indices=torch.arange(5),
    )
    runner.prepare_inputs(batch)
    assert batch.input_ids.tolist() == tape[7:12]
    sampled, count, rejected = runner.sample(torch.zeros(5, 4), batch, None)
    assert sampled.sampled_token_ids.tolist() == [tape[8:13]]
    assert count.tolist() == [5] and rejected.tolist() == [0]
    runner.speculator.run_model(2)
    assert runner.speculator.input_buffers.input_ids.tolist() == tape[8:10]
    proposed = runner.speculator._sample_draft(
        torch.zeros(2, 4), None, torch.tensor([7, 8]), None, None
    )
    assert proposed.tolist() == tape[9:11]
    flush(worker)
    draft = torch.load(tmp_path / "draft.pt", weights_only=True)
    assert draft["position_ids"].tolist() == [8, 9]
    assert draft["token_ids"].tolist() == tape[8:10]
    assert not hasattr(runner, "_mtp15_forcing")


def test_failed_dump_restores_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    runner = GPUModelRunner()
    worker = SimpleNamespace(model_runner=runner)
    original = (
        runner.prepare_inputs,
        runner.sample,
        runner.speculator.propose,
        runner.speculator.run_model,
        runner.speculator._sample_draft,
    )
    install(worker, list(range(32)), 8, "frozen", str(tmp_path))
    with pytest.raises(RuntimeError, match="No target logits"):
        flush(worker)
    assert original == (
        runner.prepare_inputs,
        runner.sample,
        runner.speculator.propose,
        runner.speculator.run_model,
        runner.speculator._sample_draft,
    )
    assert not hasattr(runner, "_mtp15_forcing")
