# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from benchmarks.sm70_mtp_phase_events import flush, install


def test_phase_events_preserve_graph_calls_without_synchronizing_rounds(monkeypatch):
    clock, synchronizations, calls = [0], [], []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            clock[0] += 1
            self.timestamp = clock[0]

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda: synchronizations.append(1)
    )

    def manager(label):
        return SimpleNamespace(run_fullgraph=lambda desc: calls.append((label, desc)))

    draft = SimpleNamespace(
        num_speculative_steps=4,
        prefill_cudagraph_manager=manager("prefill"),
        decode_cudagraph_manager=manager("decode"),
    )
    desc = SimpleNamespace(num_tokens=5)

    def propose():
        draft.prefill_cudagraph_manager.run_fullgraph(desc)
        for _ in range(3):
            draft.decode_cudagraph_manager.run_fullgraph(desc)
        return "unchanged"

    draft.propose = propose
    runner = SimpleNamespace(
        speculator=draft,
        cudagraph_manager=manager("target"),
        execute_model=lambda: None,
        prepare_inputs=lambda: None,
        sample_tokens=lambda: None,
        sample=lambda: None,
    )
    worker = SimpleNamespace(rank=0, model_runner=runner)
    install(worker)
    assert draft.propose() == draft.propose() == "unchanged"
    assert len(synchronizations) == 1
    result = flush(worker)
    assert len(synchronizations) == 2
    assert draft.propose is propose
    assert [r["label"] for r in result["records"] if "draft_step" in r["label"]] == [
        f"draft_step/{step}/M5" for _ in range(2) for step in range(4)
    ]
    assert calls == [
        ("prefill" if step == 0 else "decode", desc)
        for _ in range(2)
        for step in range(4)
    ]
    assert all(r["gpu_duration_ms"] > 0 for r in result["records"])
    assert not hasattr(worker, "_mtp_phase_events")
