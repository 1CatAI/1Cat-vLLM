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


def test_eager_forcing_keeps_decode_semantics_and_restores_context(
    tmp_path, monkeypatch
):
    from vllm.compilation.sm70_decode_graph import is_sm70_decode_graph_compiling

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    runner = GPUModelRunner()
    seen = []
    runner.speculator.run_model = lambda *args, **kwargs: seen.append(
        is_sm70_decode_graph_compiling()
    )
    install(
        SimpleNamespace(model_runner=runner), list(range(32)), 8, "x", str(tmp_path)
    )
    assert not is_sm70_decode_graph_compiling()
    runner.speculator.run_model(2)
    assert seen == [True]
    assert not is_sm70_decode_graph_compiling()
    flush(SimpleNamespace(model_runner=runner), discard=True)


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


def test_forcing_keeps_dynamic_vocab_tail_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    runner = GPUModelRunner()
    seen = []
    runner.speculator.greedy_draft_vocab = SimpleNamespace(
        observe_target_logits=lambda logits, prefill: seen.append(
            (logits.clone(), prefill)
        )
    )
    worker = SimpleNamespace(model_runner=runner)
    install(worker, list(range(32)), 8, "frozen", str(tmp_path))
    batch = SimpleNamespace(
        positions=torch.tensor([7]),
        logits_indices=torch.tensor([0]),
        num_draft_tokens=0,
    )
    runner.sample(torch.zeros(1, 4), batch, None)
    assert len(seen) == 1 and seen[0][1]
    flush(worker, discard=True)
