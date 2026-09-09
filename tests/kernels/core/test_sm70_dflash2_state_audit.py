# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from benchmarks.compare_sm70_dflash2_natural_audit import compare_natural
from benchmarks.compare_sm70_dflash2_state_audit import compare, tensor_difference
from benchmarks.sm70_dflash2_state_audit import (
    cpu_request_slots,
    gather_state,
    selected_ssm_slots,
)
from benchmarks.sm70_dflash2_state_layout import (
    check_slot_mapping,
    explain_state_difference,
)


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


def test_audit_tracks_eos_changes_without_a_top1_flip(captures):
    left, right = captures
    eos = (0, 30)
    assert (
        compare(left, right, eos_token_ids=eos)["summary"][
            "max_eos_probability_abs_difference"
        ]
        == 0
    )
    path = right / "test-rank0-step1.pt"
    data = torch.load(path, weights_only=True)
    data["native_logits"][0, 30] += 0.5
    torch.save(data, path)
    result = compare(left, right, eos_token_ids=eos)
    assert result["summary"]["top1_changed_rows"] == 0
    assert result["summary"]["max_eos_probability_abs_difference"] > 0
    row = result["logits"][-1]
    assert row["full_eos_probabilities_left"] != row["full_eos_probabilities_right"]
    assert (
        row["sampling_eos_probabilities_left"]
        != row["sampling_eos_probabilities_right"]
    )
    with pytest.raises(ValueError, match="outside the captured vocabulary"):
        compare(left, right, eos_token_ids=(32,))


def test_state_layout_requires_bijective_slots_and_preserves_padding():
    key = "verify/layer0/recurrent/slot_table:(1, 3)"
    left = {key: torch.tensor([[52, 53, -1]])}
    right = {key: torch.tensor([[154, 155, -1]])}
    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    check_slot_mapping(left, right, mapping, reverse)
    chosen = "verify/layer0/recurrent/input_state/indices:(1,)"
    # An accepted selector reading the wrong logical slot cannot be explained
    # as another physical renaming after the full table has been observed.
    with pytest.raises(ValueError, match="Inconsistent or aliased"):
        check_slot_mapping(
            {chosen: torch.tensor([52])},
            {chosen: torch.tensor([155])},
            mapping,
            reverse,
        )
    with pytest.raises(ValueError, match="Inconsistent or aliased"):
        check_slot_mapping(left, {key: torch.tensor([[154, 154, -1]])}, {}, {})
    with pytest.raises(ValueError, match="Padding slot"):
        check_slot_mapping(left, {key: torch.tensor([[154, 155, 0]])}, {}, {})


@pytest.mark.parametrize("phase", ["prefill", "verify"])
def test_state_layout_compares_only_proven_conv_input_window(phase):
    prefix = f"{phase}/layer0/conv"
    key = prefix + "/input_state/values:(1, 2, 10)"
    left = {
        key: torch.zeros(1, 2, 10),
        prefix + "/input_state/valid:(1,)": torch.tensor([True]),
        prefix + "/has_initial_state:(1,)": torch.tensor([True]),
        prefix + "/num_accepted_tokens:(1,)": torch.tensor([4]),
    }
    right = {k: v.clone() for k, v in left.items()}
    # Prefill reads columns 0..2. With selector 4, verify reads columns 3..5.
    right[key][..., 8] = 123
    assert explain_state_difference(key, left, right, 4) == (
        "unused_convolution_storage"
    )
    active_column = 1 if phase == "prefill" else 4
    right[key][..., active_column] = 1
    assert explain_state_difference(key, left, right, 4) is None
    del right[prefix + "/input_state/valid:(1,)"]
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        explain_state_difference(key, left, right, 4)


def test_state_layout_keeps_all_verifier_output_bytes():
    prefix = "verify/layer0/conv/output_state"
    key = prefix + "/values:(1, 2, 10)"
    left = {key: torch.zeros(1, 2, 10), prefix + "/valid:(1,)": torch.tensor([True])}
    right = {k: v.clone() for k, v in left.items()}
    right[key][..., 9] = 123
    assert explain_state_difference(key, left, right, 4) is None


@pytest.fixture
def natural_captures(captures):
    for directory in captures:
        for path in directory.glob("*-rank*-step*.pt"):
            row = torch.load(path, weights_only=True)
            row.update(
                control="natural_sampling",
                aux_hidden_states=[torch.ones(1, 2)],
                num_sampled=torch.tensor([1]),
                num_rejected=torch.tensor([7 if row["step"] else 0]),
                sampled_token_ids=torch.tensor([[3, -1]]),
            )
            torch.save(row, path)
            torch.save(
                {
                    **{k: row[k] for k in ("case", "rank", "step")},
                    "draft_tokens": torch.tensor([[4, 5, 6]]),
                    "projected_context": torch.ones(1, 2),
                },
                directory / f"proposal-test-tp{row['rank']}-forward{row['step']}.pt",
            )
    return captures


def test_natural_audit_locates_proposal_before_next_target(natural_captures):
    left, right = natural_captures
    assert compare_natural(left, right)["cases"][0]["all_logical_tensors_equal"]
    path = right / "proposal-test-tp2-forward0.pt"
    row = torch.load(path, weights_only=True)
    row["draft_tokens"][0, 0] += 1
    torch.save(row, path)
    path = right / "test-rank2-step1.pt"
    row = torch.load(path, weights_only=True)
    row["input_ids"][0] += 1
    torch.save(row, path)
    result = compare_natural(left, right)["cases"][0]
    assert not result["all_logical_tensors_equal"]
    first = result["first_observed_difference"]
    assert (first["step"], first["phase"]) == (0, "proposal")
    assert first["differences"][0]["name"] == "draft_tokens"


@pytest.mark.parametrize(
    "missing", ["test-rank3-step1.pt", "proposal-test-tp3-forward1.pt"]
)
def test_natural_audit_rejects_incomplete_equal_arms(natural_captures, missing):
    for directory in natural_captures:
        (directory / missing).unlink()
    with pytest.raises(ValueError, match="four TP ranks|missing proposal"):
        compare_natural(*natural_captures)


def test_natural_audit_ignores_unwritten_output_padding(natural_captures):
    left, right = natural_captures
    path = right / "test-rank0-step1.pt"
    row = torch.load(path, weights_only=True)
    row["sampled_token_ids"][0, 1] = 100
    torch.save(row, path)
    assert compare_natural(left, right)["cases"][0]["all_logical_tensors_equal"]


def test_natural_audit_aligns_request_slots(natural_captures):
    for index, directory in enumerate(natural_captures):
        for path in directory.glob("proposal-*.pt"):
            row = torch.load(path, weights_only=True)
            row["idx_mapping"] = torch.tensor([index + 1])
            row["seeds"] = torch.tensor([999, 888, 777, 666])
            row["seeds"][index + 1] = 0
            torch.save(row, path)
    result = compare_natural(*natural_captures)["cases"][0]
    assert result["all_logical_tensors_equal"]
    assert result["different_request_slot_mappings"]
    values = torch.tensor([999, 0, 777])
    actual = cpu_request_slots(values, torch.tensor([1]))
    values[1] = 100
    assert actual.tolist() == [0]


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
