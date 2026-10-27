# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from benchmarks.deepseek_v41 import trace


def test_profile_installs_on_runner_without_a_drafter(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device: None)
    cache = SimpleNamespace(
        runtime=SimpleNamespace(statistics=torch.ones(1)),
        tm=SimpleNamespace(cold_transcodes=torch.ones(1), prepare=lambda: None),
        resolve=lambda: None,
    )
    target = SimpleNamespace(
        layers=[
            SimpleNamespace(
                ffn=SimpleNamespace(
                    experts=SimpleNamespace(
                        quant_method=SimpleNamespace(expert_cache=cache)
                    )
                )
            )
        ],
        engram_io=SimpleNamespace(
            runtime=SimpleNamespace(
                statistics=torch.ones(1),
                lookup=lambda: None,
            )
        ),
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(
            model=target,
            forward=lambda: None,
            compute_logits=lambda: None,
            commit_speculative_state=lambda counts: None,
        ),
        _model_forward=lambda: None,
        _sample=lambda: None,
    )
    worker = SimpleNamespace(model_runner=runner, device="cpu")
    trace.install(worker)
    assert worker._v41_trace == dict(
        events=[], commits=[], target_forwards=[], sampling=[]
    )
    assert not hasattr(runner, "drafter")
    assert cache.runtime.statistics.item() == cache.tm.cold_transcodes.item() == 0
    trace.set_draft_cap(worker, None)
    with pytest.raises(ValueError, match="requires DSpark"):
        trace.set_draft_cap(worker, 0)
    with pytest.raises(RuntimeError, match="already installed"):
        trace.install(worker)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    runner._sample()
    assert worker._v41_trace["events"] == []


def test_draft_cap_enables_runner_length_transfer_and_trimming():
    runner = SimpleNamespace(
        drafter=SimpleNamespace(),
        max_num_reqs=2,
        pin_memory=False,
        _dspark_verification_lengths_cpu=None,
    )
    worker = SimpleNamespace(model_runner=runner)
    for cap in (0, 1, 2, 3, 4, 5, None):
        trace.set_draft_cap(worker, cap)
        enabled = cap is not None and cap < 5
        assert runner.dspark_confidence_scheduling == enabled
        assert runner.drafter.confidence_scheduling_enabled == enabled
        assert runner.drafter._max_verification_tokens == (5 if cap is None else cap)
        assert runner._dspark_verification_lengths_cpu.shape == (2,)
        assert runner._dspark_verification_lengths_cpu.dtype == torch.int32
    with pytest.raises(ValueError, match="0..5"):
        trace.set_draft_cap(worker, 6)


def test_sampler_snapshot_survives_inplace_eos_or_output_budget_clipping():
    sampled = torch.tensor([[10, 11, -1, 777, 888, 999]])
    record: dict[str, list[tuple[list[int], torch.Tensor]]] = dict(sampling=[])
    counts = [5]
    trace.remember_sampling(record, counts, SimpleNamespace(sampled_token_ids=sampled))
    sampled[:, 1:] = -1
    counts[0] = 0
    proposed, actual = record["sampling"][0]
    assert proposed == [5]
    assert actual.tolist() == [[10, 11, -1, 777, 888, 999]]
