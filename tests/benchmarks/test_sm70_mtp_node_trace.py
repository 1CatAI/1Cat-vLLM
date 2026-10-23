# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from benchmarks.sm70_mtp_node_trace import install, uninstall


def test_annotations_preserve_graph_calls_and_reset_draft_steps(monkeypatch, tmp_path):
    annotations, calls = [], []

    @contextmanager
    def nvtx(label):
        annotations.append(label)
        yield

    monkeypatch.setattr(torch.cuda.nvtx, "range", nvtx)
    desc = SimpleNamespace(num_tokens=5)

    def manager(label):
        return SimpleNamespace(run_fullgraph=lambda d: calls.append((label, d)))

    draft = SimpleNamespace(
        num_speculative_steps=4,
        model=torch.nn.Linear(3, 4),
        prefill_cudagraph_manager=manager("prefill"),
        decode_cudagraph_manager=manager("decode"),
    )

    def propose():
        draft.prefill_cudagraph_manager.run_fullgraph(desc)
        for _ in range(3):
            draft.decode_cudagraph_manager.run_fullgraph(desc)
        return "unchanged"

    draft.propose = propose
    runner = SimpleNamespace(
        speculator=draft,
        model=torch.nn.Linear(3, 4),
        cudagraph_manager=manager("target"),
        execute_model=lambda: None,
        prepare_inputs=lambda: None,
        sample_tokens=lambda: None,
        sample=lambda: None,
    )
    worker = SimpleNamespace(rank=0, model_runner=runner)
    install(worker, tmp_path)
    assert draft.propose() == draft.propose() == "unchanged"
    assert [label for label in annotations if "draft_step" in label] == [
        f"mtp15.draft_step/{step}/M5" for _ in range(2) for step in range(4)
    ]
    assert calls == [
        ("prefill" if step == 0 else "decode", desc)
        for _ in range(2)
        for step in range(4)
    ]
    uninstall(worker)
    assert draft.propose is propose
    draft.propose()
    assert len(annotations) == 10


def test_plan_snapshots_follow_replay_and_exclude_invalid_experts(
    monkeypatch, tmp_path
):
    @contextmanager
    def nvtx(label):
        yield

    monkeypatch.setattr(torch.cuda.nvtx, "range", nvtx)
    model = torch.nn.Module()
    model._nvfp4_grouped_total = torch.tensor([2], dtype=torch.int32)
    model._nvfp4_grouped_experts = torch.tensor([3, 512, -1, 999], dtype=torch.int32)
    model.w13_tm_weight = torch.empty(512, 1, 1)
    result = object()
    manager = lambda: SimpleNamespace(run_fullgraph=lambda desc: result)
    draft = SimpleNamespace(
        num_speculative_steps=4,
        model=torch.nn.Linear(3, 4),
        propose=lambda: None,
        prefill_cudagraph_manager=manager(),
        decode_cudagraph_manager=manager(),
    )
    runner = SimpleNamespace(
        model=model,
        speculator=draft,
        cudagraph_manager=manager(),
        execute_model=lambda: None,
        prepare_inputs=lambda: None,
        sample_tokens=lambda: None,
        sample=lambda: None,
    )
    worker = SimpleNamespace(rank=0, model_runner=runner)
    install(worker, tmp_path)
    assert (
        runner.cudagraph_manager.run_fullgraph(SimpleNamespace(num_tokens=5)) is result
    )
    model._nvfp4_grouped_total.fill_(1)
    model._nvfp4_grouped_experts[0] = -1
    runner.cudagraph_manager.run_fullgraph(SimpleNamespace(num_tokens=1))
    runner.cudagraph_manager.run_fullgraph(SimpleNamespace(num_tokens=5))
    uninstall(worker)
    data = json.loads((tmp_path / "rank0_expert_groups.json").read_text())
    assert data["groups_per_round"] == [[2], [1]]
    assert data["valid_groups_per_round"] == [[1], [0]]
    assert not hasattr(worker, "_mtp_node_trace_plans")
