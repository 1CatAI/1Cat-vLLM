# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from benchmarks.compare_sm70_dflash2_state_audit import compare, tensor_difference
from benchmarks.sm70_dflash2_state_audit import gather_state, selected_ssm_slots


@pytest.fixture
def captures(tmp_path):
    directories = (tmp_path / "left", tmp_path / "right")
    for directory in directories:
        directory.mkdir()
        for rank in range(4):
            for step, phase in enumerate(("prefill", "verify")):
                states = {
                    f"{phase}/layer0/{name}:(1, 2)": torch.ones(1, 2)
                    for name in (
                        "conv/input",
                        "conv/output",
                        "recurrent/q",
                        "recurrent/input_state",
                        "recurrent/output",
                    )
                }
                data = {
                    "case": "test",
                    "rank": rank,
                    "step": step,
                    "phase": phase,
                    "num_draft_tokens": 7 if step else 0,
                    "positions": torch.tensor([step]),
                    "input_ids": torch.tensor([1]),
                    "hidden": torch.ones(1, 2),
                    "states": states,
                    "tensors": {},
                    "sampling": {"seeds": torch.tensor([0])},
                    "native_logits": torch.arange(32).float().reshape(1, -1)
                    if rank == 0
                    else None,
                    "expected_layers": [0],
                    "capture_epoch": step,
                }
                torch.save(data, directory / f"test-rank{rank}-step{step}.pt")
    return directories


def test_audit_comparator_requires_all_ranks(captures):
    left, right = captures
    assert compare(left, right)["summary"]["all_logits_bitwise_equal"]
    # Even equal, incomplete arms cannot be reported as a successful A/A.
    for directory in captures:
        (directory / "test-rank3-step1.pt").unlink()
    with pytest.raises(ValueError, match="four TP ranks"):
        compare(left, right)


def test_audit_comparator_requires_actual_candidate_route(captures):
    left, right = captures
    with pytest.raises(ValueError, match="missing packed route hit"):
        compare(left, right, right_verifier_route="packed")
    for path in right.glob("*-step1.pt"):
        data = torch.load(path, weights_only=True)
        data["verifier_routes"] = ["route/verify/layer0/packed"]
        torch.save(data, path)
    assert compare(left, right, right_verifier_route="packed")["summary"][
        "all_logits_bitwise_equal"
    ]


def test_audit_comparator_rejects_missing_layer_observations(captures):
    left, right = captures
    for directory in captures:
        path = directory / "test-rank0-step1.pt"
        data = torch.load(path, weights_only=True)
        data["expected_layers"] = [0, 1]
        torch.save(data, path)
    with pytest.raises(ValueError, match="missing verify/layer1"):
        compare(left, right)


def test_audit_comparator_rejects_nonfinite_logits(captures):
    left, right = captures
    path = right / "test-rank0-step1.pt"
    data = torch.load(path, weights_only=True)
    data["native_logits"][0, 0] = float("nan")
    torch.save(data, path)
    with pytest.raises(ValueError, match="nonfinite"):
        compare(left, right)


def test_state_audit_reads_accepted_slot_and_preserves_invalid_selectors():
    table = torch.tensor([[7, 0, 9, 2], [3, 5, 4, 8]], dtype=torch.int32)
    assert selected_ssm_slots(table, torch.tensor([2, 4])).tolist() == [0, 8]
    assert selected_ssm_slots(table, torch.tensor([0, 5])).tolist() == [-1, -1]


def test_audit_compares_singleton_strided_metadata():
    wide = torch.empty_strided((1,), (8,), dtype=torch.int64).fill_(7)
    assert wide.is_contiguous() and wide.stride() == (8,)
    assert tensor_difference(wide, torch.tensor([7]))["bitwise_equal"]


def test_state_audit_distinguishes_padding_from_live_slot_zero():
    pool = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    snapshot = gather_state(pool, torch.tensor([0, -1, 3, 4]))
    assert snapshot["indices"].tolist() == [0, -1, 3, 4]
    assert snapshot["valid"].tolist() == [True, False, True, False]
    torch.testing.assert_close(snapshot["values"][0], pool[0], atol=0, rtol=0)
    torch.testing.assert_close(snapshot["values"][2], pool[3], atol=0, rtol=0)
    assert not snapshot["values"][[1, 3]].count_nonzero()
    pool.fill_(100)
    assert snapshot["values"][0, 0, 0] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_state_audit_graph_replays_changed_slots_without_mutating_pool():
    pool = torch.arange(48, device="cuda", dtype=torch.float32).reshape(8, 2, 3)
    table = torch.tensor([[7, 0, 6, 2]], device="cuda", dtype=torch.int32)
    selector = torch.tensor([1], device="cuda", dtype=torch.int32)
    # Warm CUDA allocation and selection before graph capture.
    gather_state(pool, selected_ssm_slots(table, selector))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        snapshot = gather_state(pool, selected_ssm_slots(table, selector))
    for selected, expected in ((1, 7), (2, 0), (4, 2)):
        selector.fill_(selected)
        before = pool.clone()
        graph.replay()
        torch.testing.assert_close(
            snapshot["values"][0], pool[expected], atol=0, rtol=0
        )
        torch.testing.assert_close(pool, before, atol=0, rtol=0)
    selector.fill_(0)
    graph.replay()
    assert not snapshot["valid"].any()
    assert not snapshot["values"].count_nonzero()
