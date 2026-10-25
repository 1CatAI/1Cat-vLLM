# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regressions for delayed feedback in the V2 runner, including real GPU copies."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.mamba_align import run_mamba_align_postprocess
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize("mapping", [[0], [2, 0], [-1, 0], [1, -1, 0]])
@pytest.mark.parametrize("mode", ["align", "all"])
def test_v2_delayed_feedback_regathers_request_rows(monkeypatch, mapping, mode):
    """Commit old feedback against request-owned rows, preserving -1 masks."""
    runner: Any = object.__new__(GPUModelRunner)
    canonical = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    latest_rows = canonical[[2, 1, 0]].clone()
    outputs = {
        "idx_mapping": torch.tensor(mapping, dtype=torch.int32),
        "sampled_tokens": torch.full((len(mapping), 5), 42, dtype=torch.int64),
        "num_sampled": torch.ones(len(mapping), dtype=torch.int32),
        "num_rejected": torch.zeros(len(mapping), dtype=torch.int32),
        "draft_tokens": torch.full((len(mapping), 4), 101, dtype=torch.int64),
    }
    calls = []

    def gather(indices, num_reqs_padded):
        assert indices.tolist() == [max(index, 0) for index in mapping]
        assert num_reqs_padded == len(mapping)
        latest_rows[: len(mapping)] = canonical[indices.long()]
        calls.append("gather")

    def commit(indices, sampled, accepted, rejected):
        assert indices is outputs["idx_mapping"]
        assert indices.tolist() == mapping
        assert sampled is outputs["sampled_tokens"]
        assert accepted is outputs["num_sampled"]
        assert rejected is outputs["num_rejected"]
        if mode == "align":
            for row, index in enumerate(mapping):
                if index >= 0:
                    torch.testing.assert_close(latest_rows[row], canonical[index])
        calls.append("commit")

    def scatter(destination, indices, drafts):
        assert destination is runner.req_states.draft_tokens
        assert indices is outputs["idx_mapping"]
        assert drafts is outputs["draft_tokens"]
        calls.append("scatter")

    runner.pp_handler = SimpleNamespace(get_prev_sampled_outputs=lambda: outputs)
    runner.cache_config = SimpleNamespace(mamba_cache_mode=mode)
    runner.block_tables = SimpleNamespace(gather_block_tables=gather)
    runner.req_states = SimpleNamespace(draft_tokens=torch.empty((3, 4)))
    runner.postprocess_sampled = commit
    monkeypatch.setattr(model_runner, "scatter_draft_tokens", scatter)

    runner.update_pp_decode_requests()
    assert calls == (["gather"] if mode == "align" else []) + ["commit", "scatter"]


@pytest.mark.parametrize("with_handler", [False, True])
def test_v2_empty_feedback_does_not_touch_state(with_handler):
    runner: Any = object.__new__(GPUModelRunner)
    runner.pp_handler = (
        SimpleNamespace(get_prev_sampled_outputs=lambda: None) if with_handler else None
    )
    # All state attributes are deliberately absent: an empty pipeline slot must
    # return before gathering cache rows or committing any recurrent state.
    runner.update_pp_decode_requests()


def test_v2_preemption_invalidates_feedback_once():
    runner: Any = object.__new__(GPUModelRunner)
    live = {"finished": 0, "preempted": 1}
    freed: list[int] = []
    removed: list[str] = []
    runner.req_states = SimpleNamespace(remove_request=lambda req: live.pop(req, None))
    runner.pp_handler = SimpleNamespace(on_req_idx_freed=freed.append)
    runner.encoder_cache = None
    runner.prompt_logprobs_worker = None
    runner.lora_state = SimpleNamespace(remove_request=removed.append)
    schedule = SimpleNamespace(
        finished_req_ids={"finished"}, preempted_req_ids={"finished", "preempted"}
    )
    runner.finish_requests(schedule)
    runner.finish_requests(schedule)
    assert sorted(freed) == [0, 1]
    assert sorted(removed) == ["finished", "preempted"]
    assert live == {}


@pytest.mark.parametrize("mapping", [[0], [2, 0], [-1, 0], [1, -1, 0]])
@pytest.mark.parametrize("crossing", [True, False])
def test_v2_delayed_mamba_actual_gpu_copy(mapping, crossing):
    """The real gather/copy/scatter kernels must write only the old live rows."""
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    device = torch.device("cuda:0")
    canonical = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        dtype=torch.int32,
        device=device,
    )
    current = canonical[[2, 1, 0]].clone()
    tables: Any = object.__new__(BlockTables)
    tables.num_kv_cache_groups = 1
    tables.input_block_tables = [current]
    tables.block_table_ptrs = torch.tensor(
        [canonical.data_ptr()], dtype=torch.int64, device=device
    )
    tables.input_block_table_ptrs = torch.tensor(
        [current.data_ptr()], dtype=torch.int64, device=device
    )
    tables.block_table_strides = torch.tensor([4], dtype=torch.int64, device=device)
    tables.num_blocks = SimpleNamespace(
        gpu=torch.full((1, 3), 4, dtype=torch.int32, device=device)
    )
    state = torch.arange(13 * 8, dtype=torch.int32, device=device).reshape(13, 8)
    before = state.clone()
    ctx: Any = SimpleNamespace(
        is_initialized=True,
        num_states=1,
        block_size=16,
        num_accepted_tokens_out=torch.zeros(3, dtype=torch.int32, device=device),
        block_table_ptrs=tables.input_block_table_ptrs,
        block_table_stride_req=4,
        state_base_addrs=torch.tensor(
            [state.data_ptr()], dtype=torch.int64, device=device
        ),
        state_block_strides=torch.tensor([32], dtype=torch.int64, device=device),
        state_elem_sizes=torch.tensor([4], dtype=torch.int32, device=device),
        state_inner_sizes=torch.tensor([8], dtype=torch.int64, device=device),
        state_conv_widths=torch.zeros(1, dtype=torch.int32, device=device),
        state_group_indices=torch.zeros(1, dtype=torch.int32, device=device),
    )
    idx = torch.tensor(mapping, dtype=torch.int32, device=device)
    accepted = torch.full((3,), 2 if crossing else 1, dtype=torch.int32, device=device)
    computed = torch.full(
        (3,), 16 if crossing else 15, dtype=torch.int32, device=device
    )
    state_idx = torch.ones(3, dtype=torch.int32, device=device)
    drafts = torch.arange(len(mapping) * 4, dtype=torch.int64, device=device).view(
        -1, 4
    )
    destination = torch.full((3, 4), -1, dtype=torch.int64, device=device)
    outputs = {
        "idx_mapping": idx,
        "sampled_tokens": torch.full(
            (len(mapping), 5), 42, dtype=torch.int64, device=device
        ),
        "num_sampled": torch.ones(len(mapping), dtype=torch.int32, device=device),
        "num_rejected": torch.zeros(len(mapping), dtype=torch.int32, device=device),
        "draft_tokens": drafts,
    }
    runner: Any = object.__new__(GPUModelRunner)
    runner.pp_handler = SimpleNamespace(get_prev_sampled_outputs=lambda: outputs)
    runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
    runner.block_tables = tables
    runner.req_states = SimpleNamespace(draft_tokens=destination)

    def commit(indices, *_):
        run_mamba_align_postprocess(
            ctx, len(mapping), accepted, state_idx, computed, indices
        )

    runner.postprocess_sampled = commit
    runner.update_pp_decode_requests()
    torch.accelerator.synchronize()
    expected_state = before.clone()
    expected_drafts = torch.full_like(destination, -1)
    for row, req_idx in enumerate(mapping):
        if req_idx < 0:
            continue
        if crossing:
            expected_state[req_idx * 4 + 1] = before[req_idx * 4 + 3]
        expected_drafts[req_idx] = drafts[row]
    torch.testing.assert_close(state, expected_state)
    torch.testing.assert_close(destination, expected_drafts)
