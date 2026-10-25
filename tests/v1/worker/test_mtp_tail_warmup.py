# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol tests for actual tail warmup; no model/CUDA initialization.

Run with the checkout's existing .venv from the repository root. Real
SchedulerOutput/Request/SamplingParams types are used; only GPU callbacks,
collectives, logger, and allocation specs are replaced by CPU fakes.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request


def load_helpers():
    source = (
        Path(__file__).resolve().parents[3] / "vllm/v1/worker/gpu/warmup.py"
    ).read_text()
    tree = ast.parse(source)
    names = {"_mtp_tail_warmup_profiles", "_warmup_mtp_tail_shapes"}
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "__name__": __name__,
        "GPUModelRunner": SimpleNamespace,
        "Callable": __import__("collections.abc").abc.Callable,
        "Any": __import__("typing").Any,
        "SchedulerOutput": SchedulerOutput,
        "CachedRequestData": CachedRequestData,
        "NewRequestData": NewRequestData,
        "GrammarOutput": GrammarOutput,
        "Request": Request,
        "SamplingParams": SamplingParams,
        "torch": torch,
        "time": __import__("time"),
        "logger": logging.getLogger(__name__),
        # One ordinary block per 8 tokens; reserve 4 lookahead tokens, exactly
        # like the real helper's general attention rule. Two projected groups
        # make block accounting nontrivial without importing GPU-only modules.
        "_warmup_block_counter": lambda runner: (
            lambda tokens, spec: (tokens + 4 + spec - 1) // spec
        ),
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, "candidate", "exec"), namespace)
    return namespace


@pytest.fixture
def helpers():
    return load_helpers()


def make_runner(depth=4, enabled=True, capacity=8, num_blocks=512):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"mtp_tail_warmup": enabled},
            is_speculative_draft=False,
            max_concurrent_batches=depth,
        ),
        speculative_config=SimpleNamespace(method="mtp"),
        num_speculative_steps=4,
        is_pooling_model=False,
        max_model_len=215040,
        scheduler_config=SimpleNamespace(
            max_num_seqs=capacity, max_num_batched_tokens=8192
        ),
        kv_cache_config=SimpleNamespace(
            num_blocks=num_blocks,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=8),
                SimpleNamespace(kv_cache_spec=16),
            ],
        ),
    )


class FeedbackHarness:
    def __init__(self, depth, num_blocks):
        self.depth = depth
        self.num_blocks = num_blocks
        self.pending: list[tuple[str, ...] | None] = [None] * depth
        self.live = {}
        self.allocated = set()
        self.executions = []
        self.decode_shapes = []
        self.prefill_steps = {}
        self.decode_steps = {}
        self.committed = set()
        self.last: SchedulerOutput | None = None
        self.sample_calls = 0

    def allocate(self, groups):
        for blocks in groups:
            for block in blocks:
                assert 0 < block < self.num_blocks
                assert block not in self.allocated
                self.allocated.add(block)

    def execute(self, output):
        slot = self.pending.pop(0)
        self.pending.append(None)
        if slot is not None:
            for rid in slot:
                assert rid in self.live, "freed before delayed feedback commit"
                self.committed.add(rid)
        self.executions.append(output)
        self.last = output
        step = len(self.executions) - 1
        if output.finished_req_ids:
            assert output.finished_req_ids == set(self.live)
            assert not any(self.pending), "pending feedback during block reuse"
            for rid in output.finished_req_ids:
                assert rid in self.committed
                assert step - self.decode_steps[rid] >= self.depth
            self.live.clear()
            self.allocated.clear()
            self.committed.clear()
        for req in output.scheduled_new_reqs:
            assert req.req_id not in self.live
            assert len(req.prompt_token_ids) == 6
            assert req.sampling_params.max_tokens > 5
            self.live[req.req_id] = req
            self.prefill_steps[req.req_id] = step
            self.allocate(req.block_ids)
        cached = output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            assert rid in self.live and rid in self.committed
            assert step - self.prefill_steps[rid] == self.depth
            assert cached.num_computed_tokens[i] == 6
            assert cached.num_output_tokens[i] == 1
            if cached.new_block_ids[i] is not None:
                self.allocate(cached.new_block_ids[i])
            self.decode_steps[rid] = step
        if cached.req_ids:
            shape = tuple(output.num_scheduled_tokens.values())
            self.decode_shapes.append(shape)
            assert sum(shape) == output.total_num_scheduled_tokens
            for rid, count in output.num_scheduled_tokens.items():
                draft = output.scheduled_spec_decode_tokens.get(rid, ())
                assert len(draft) == count - 1
                if count == 1:
                    assert rid not in output.scheduled_spec_decode_tokens
            self.committed.clear()

    def sample(self, grammar):
        assert grammar is None
        assert self.last is not None
        assert self.last.total_num_scheduled_tokens > 0
        assert self.pending[-1] is None
        self.pending[-1] = tuple(self.last.num_scheduled_tokens)
        self.sample_calls += 1


def configure_cpu(monkeypatch, *, pp_size=7, reduced_capacity=None):
    import vllm.distributed.parallel_state as parallel_state

    pp = SimpleNamespace(world_size=pp_size, is_first_rank=True, cpu_group=object())
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp)
    reductions = []

    def reduce(tensor, *, op, group):
        assert tensor.device.type == "cpu"
        assert op == torch.distributed.ReduceOp.MIN
        assert group is pp.cpu_group
        reductions.append(int(tensor.item()))
        if reduced_capacity is not None:
            assert reduced_capacity <= tensor.item()
            tensor.fill_(reduced_capacity)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    return reductions


def test_complete_shape_plan(helpers):
    profiles = helpers["_mtp_tail_warmup_profiles"](8, 4)
    assert len(profiles) == 102
    assert len(set(profiles)) == 102
    for n in range(1, 9):
        for first in range(1, 5):
            assert (first,) + (5,) * (n - 1) in profiles
            if n >= 2:
                for second in range(first, 5):
                    assert (first, second) + (5,) * (n - 2) in profiles
    assert all(tuple(sorted(profile)) == profile for profile in profiles)


@pytest.mark.parametrize("depth", [1, 2, 4, 7])
def test_real_scheduler_output_feedback_protocol(helpers, monkeypatch, depth):
    reductions = configure_cpu(monkeypatch, pp_size=1 if depth == 1 else 7)
    runner = make_runner(depth)
    harness = FeedbackHarness(depth, runner.kv_cache_config.num_blocks)
    helpers["_warmup_mtp_tail_shapes"](runner, harness.execute, harness.sample)
    assert len(harness.decode_shapes) == 102
    assert harness.sample_calls == 204
    assert len(harness.executions) == depth + 102 * (2 * depth + 2)
    assert not harness.live and not harness.allocated and not any(harness.pending)
    assert reductions == ([] if depth == 1 else [8])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("capacity", [0, 1, 2, 3, 4, 8])
def test_minimum_rank_capacity_controls_same_plan(helpers, monkeypatch, capacity):
    configure_cpu(monkeypatch, reduced_capacity=capacity)
    runner = make_runner()
    harness = FeedbackHarness(4, runner.kv_cache_config.num_blocks)
    helpers["_warmup_mtp_tail_shapes"](runner, harness.execute, harness.sample)
    expected = helpers["_mtp_tail_warmup_profiles"](capacity, 4)
    assert tuple(harness.decode_shapes) == expected


def test_block_budget_limits_cohort_and_reserves_null(helpers, monkeypatch):
    reductions = configure_cpu(monkeypatch)
    runner = make_runner(num_blocks=10)
    # ceil((6+5+4)/8) + ceil((6+5+4)/16) = 3 blocks/request.
    harness = FeedbackHarness(4, runner.kv_cache_config.num_blocks)
    helpers["_warmup_mtp_tail_shapes"](runner, harness.execute, harness.sample)
    assert reductions == [3]
    assert max(map(len, harness.decode_shapes)) == 3


@pytest.mark.parametrize("mode", ["disabled", "draft", "non_dict"])
def test_no_collective_for_disabled_or_child_runner(helpers, monkeypatch, mode):
    import vllm.distributed.parallel_state as parallel_state

    get_group = Mock(side_effect=AssertionError("must not enter target PP group"))
    monkeypatch.setattr(parallel_state, "get_pp_group", get_group)
    runner = make_runner()
    if mode == "disabled":
        runner.vllm_config.additional_config.clear()
    elif mode == "draft":
        runner.vllm_config.is_speculative_draft = True
    else:
        runner.vllm_config.additional_config = object()
    execute = Mock()
    sample = Mock()
    helpers["_warmup_mtp_tail_shapes"](runner, execute, sample)
    get_group.assert_not_called()
    execute.assert_not_called()
    sample.assert_not_called()


@pytest.mark.parametrize("spec_steps", [0, 5, 64])
def test_reject_unbounded_or_non_spec_plan(helpers, spec_steps):
    with pytest.raises(ValueError, match="1 to 4"):
        helpers["_mtp_tail_warmup_profiles"](8, spec_steps)


def test_forward_failure_propagates_without_new_collective(helpers, monkeypatch):
    configure_cpu(monkeypatch)
    runner = make_runner()
    seen = []

    def execute(output):
        seen.append(output)
        if output.scheduled_new_reqs:
            raise RuntimeError("injected target forward failure")

    with pytest.raises(RuntimeError, match="injected target forward failure"):
        helpers["_warmup_mtp_tail_shapes"](runner, execute, Mock())
    # Once a rank's forward fails, do not attempt distributed recovery/drain:
    # startup fails instead of exposing a partially initialized service.
    assert len(seen) == 5
    assert seen[-1].scheduled_new_reqs


def test_hook_inside_existing_connector_finally():
    source = (
        Path(__file__).resolve().parents[3] / "vllm/v1/worker/gpu/warmup.py"
    ).read_text()
    tree = ast.parse(source)
    warmup = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "warmup_kernels"
    )
    try_node = next(node for node in warmup.body if isinstance(node, ast.Try))
    expression = try_node.body[-1]
    assert isinstance(expression, ast.Expr)
    call = expression.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "_warmup_mtp_tail_shapes"
    assert "set_disabled(False)" in ast.unparse(try_node.finalbody[0])


def test_outer_connector_restored_when_tail_warmup_fails(helpers):
    source = (
        Path(__file__).resolve().parents[3] / "vllm/v1/worker/gpu/warmup.py"
    ).read_text()
    tree = ast.parse(source)
    warmup = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "warmup_kernels"
    )
    warmup.decorator_list = []
    helpers["_kernel_prefill_warmup_token_counts"] = lambda runner, length: ()
    helpers["_warmup_mtp_tail_shapes"] = Mock(
        side_effect=RuntimeError("injected tail warmup failure")
    )
    module = ast.Module(body=[warmup], type_ignores=[])
    exec(compile(module, "candidate", "exec"), helpers)
    runner = make_runner()
    runner.kv_connector = SimpleNamespace(set_disabled=Mock())
    with pytest.raises(RuntimeError, match="injected tail warmup failure"):
        helpers["warmup_kernels"](runner, Mock(), Mock())
    assert runner.kv_connector.set_disabled.call_args_list == [
        __import__("unittest.mock").mock.call(True),
        __import__("unittest.mock").mock.call(False),
    ]
