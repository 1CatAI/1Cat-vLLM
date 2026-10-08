# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent sequence execution, including preparation and native decline."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import prefill_candidates as sequence
from vllm.v1.attention.backends.flash_v100.config import V100AttnConfig
from vllm.v1.attention.backends.flash_v100.workspace import V100Workspace

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "choice",
    [
        "bfla",
        "fa2",
        "bhmd",
        "dense",
        "bridge",
        "splitkv",
        "paged",
        "fa2_decline",
        "bridge_decline",
        "mask_decline",
    ],
)
def test_prefill_candidates_with_independent_operators(monkeypatch, choice):
    calls = []
    query = torch.zeros((1024, 6, 256), dtype=torch.float16)
    cache = torch.zeros((128, 8, 1, 256), dtype=torch.float16)
    output = torch.zeros_like(query)
    request = sequence.PrefillRequest(
        SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0),
        query,
        cache,
        cache,
        SimpleNamespace(
            block_table=torch.arange(128).unsqueeze(0), seq_lens=torch.tensor([1024])
        ),
        output,
        0,
        0,
        1024,
        1024,
        1024,
        query.unsqueeze(0),
        1,
        1,
        256,
        8,
        True,
        (-1, -1),
    )
    policy = cast(
        V100AttnConfig,
        SimpleNamespace(
            prefill_contig_dense_allow_copy=True,
            prefill_bfla_mask_block_n=64,
            prefill_split_kv_tokens=256,
        ),
    )

    def predicate(name, enabled):
        def decide(**kwargs):
            calls.append(name)
            return enabled

        return decide

    def native(name):
        def execute(query, *args, **kwargs):
            calls.append(name)
            if name == "fa2" and choice == "fa2_decline":
                return None
            result = kwargs.get("out")
            if result is None:
                result = torch.empty_like(query)
            result.fill_(7)
            return result

        return execute

    def run_paged(*, route, fn, **kwargs):
        calls.append("timed:" + route)
        return fn()

    def bridge(**kwargs):
        calls.append("bridge")
        if choice == "bridge_decline":
            return None
        kwargs["out"].fill_(7)
        return kwargs["out"], True

    def view(*args):
        calls.append("view")
        return (cache, cache) if choice in {"fa2", "dense"} else None

    def gather(*args):
        calls.append("gather")
        return cache, cache

    def mask(*args, **kwargs):
        calls.append("mask")
        return None if choice == "mask_decline" else torch.ones(1)

    monkeypatch.setattr(
        sequence._config, "registered", lambda name: choice.startswith("fa2")
    )
    monkeypatch.setattr(sequence._masks, "_build_bfla_block_mask_for_seq", mask)
    monkeypatch.setattr(sequence._kv_layout, "_contiguous_paged_kv_view", view)
    monkeypatch.setattr(sequence._kv_layout, "_gather_paged_kv_to_exact_dense", gather)
    monkeypatch.setattr(
        sequence._kv_layout,
        "_contiguous_paged_kv_bhmd",
        lambda *args: (cache, cache) if choice == "bhmd" else None,
    )
    monkeypatch.setattr(sequence._ops, "_get_sm70_splitd_d256_ops", lambda: object())
    monkeypatch.setattr(
        sequence._routing, "_record_route", lambda name: calls.append("route:" + name)
    )
    ops = sequence.PrefillOps(
        bridge=bridge,
        run_paged=run_paged,
        should_bridge=predicate("should_bridge", choice.startswith("bridge")),
        should_bfla=predicate("should_bfla", choice in {"bfla", "mask_decline"}),
        should_contig=predicate("should_contig", choice in {"bhmd", "dense"}),
        should_gather=predicate("should_gather", choice == "fa2_decline"),
        should_split=predicate("should_split", choice in {"splitkv", "bridge_decline"}),
        bhmd=native("bhmd"),
        dense=native("dense"),
        paged=native("paged"),
        bfla=native("bfla"),
        splitkv=native("splitkv"),
        uniform=lambda *args, **kwargs: (
            torch.tensor([0, 1024]),
            torch.tensor([0, 1024]),
        ),
        try_fa2=native("fa2"),
        log_bfla=lambda *args: None,
        log_fa2=lambda *args: None,
        log_contiguous_bhmd=lambda *args: None,
        log_contiguous_dense=lambda *args: None,
        log_dense_fa2=lambda *args: None,
        log_fp8_bridge=lambda *args: None,
        log_splitkv=lambda *args: None,
    )
    executor = sequence.PrefillExecutor(
        sequence.PrefillConfig(policy, 0.0625, "auto"), ops, V100Workspace()
    )
    result = executor.sequence(request)
    assert calls.count("should_split") == calls.count("should_bridge") == 1
    assert calls.index("should_split") < calls.index("should_bridge")
    if choice == "bhmd":
        assert result.skip_debug and result.output is None
        assert torch.all(output == 7)
    else:
        assert not result.skip_debug and torch.all(result.output == 7)
    if choice in {"fa2", "bridge"}:
        assert result.is_destination
        assert result.output.data_ptr() == output.data_ptr()
    if choice == "fa2_decline":
        assert (
            calls.index("gather")
            < calls.index("fa2")
            < calls.index("should_split")
            < calls.index("paged")
        )
    if choice == "bridge_decline":
        assert "splitkv" not in calls
        assert calls[-2:] == ["bridge", "paged"]
    if choice in {"bfla", "fa2"}:
        assert "should_contig" not in calls
    if choice == "mask_decline":
        assert calls.index("mask") < calls.index("should_contig") < calls.index("paged")


@pytest.mark.parametrize(
    "choice", ["noncausal", "tree", "small", "rows", "sequences", "terminal_none"]
)
def test_batch_candidates_with_independent_operators(monkeypatch, choice):
    calls = []
    query = torch.zeros((4, 6, 256), dtype=torch.float16)
    output = torch.zeros_like(query)
    cache = torch.zeros((2, 8, 1, 256), dtype=torch.float16)
    lengths = torch.tensor([2, 2])
    metadata = SimpleNamespace(block_table=torch.tensor([[0], [1]]), seq_lens=lengths)
    request = sequence.PrefillBatchRequest(
        SimpleNamespace(
            is_dflash_draft_attn=True, _k_scale_float=1.0, _v_scale_float=1.0
        ),
        query,
        None,
        None,
        cache,
        cache,
        metadata,
        output,
        output,
        torch.tensor([0, 2, 4]),
        lengths,
        lengths,
        2,
        2,
        1,
        256,
        8,
        choice != "noncausal",
        (-1, -1),
        None,
        False,
        False,
    )
    policy = cast(
        V100AttnConfig,
        SimpleNamespace(
            use_flash_v100_prefill_paged=True,
            use_flash_v100_decode=True,
            smallq_decode_max_query_len=4 if choice == "small" else 0,
            smallq_decode_max_model_len=0,
            use_decode_paged_prefill=False,
        ),
    )

    def complete(name):
        def execute(*args, **kwargs):
            calls.append(name)
            assert choice == name or (name == "tree" and choice == "terminal_none")
            if choice == "terminal_none":
                return None
            return output.fill_(9)

        return execute

    def paged(*args, **kwargs):
        calls.append("noncausal")
        assert choice == "noncausal"
        kwargs["out"].fill_(9)
        return kwargs["out"]

    def rows(*args):
        calls.append("rows")
        assert choice == "rows"
        output[:2].fill_(9)
        return {0}

    monkeypatch.setattr(
        sequence._masks,
        "_ddtree_parent_metadata_requires_branch",
        lambda *args: choice in {"tree", "terminal_none"},
    )
    monkeypatch.setattr(sequence._routing, "_record_route", lambda name: None)
    ops = sequence.PrefillOps(
        bridge=None,
        run_paged=lambda *, fn, **kwargs: fn(),
        should_bridge=None,
        should_bfla=None,
        should_contig=None,
        should_gather=None,
        should_split=None,
        bhmd=None,
        dense=None,
        paged=paged,
        bfla=None,
        splitkv=None,
        uniform=None,
        try_fa2=None,
        log_bfla=None,
        log_fa2=None,
        log_contiguous_bhmd=None,
        log_contiguous_dense=None,
        log_dense_fa2=None,
        log_fp8_bridge=None,
        log_splitkv=None,
        tree_prefill=complete("tree"),
        small_query=complete("small"),
        allow_rows=lambda **kwargs: choice == "rows",
        decode_rows=rows,
        log_noncausal=lambda *args: None,
        log_small_query=lambda *args: None,
    )
    result = sequence.PrefillExecutor(
        sequence.PrefillConfig(policy, 0.0625, "auto"), ops, V100Workspace()
    ).batch(request)
    if choice in {"noncausal", "tree", "small"}:
        assert result.complete and result.output is output
        assert torch.all(output == 9) and calls == [choice]
    elif choice == "terminal_none":
        assert result.complete and result.output is None and calls == ["tree"]
    else:
        assert not result.complete and result.output is None
        assert result.rows == ({0} if choice == "rows" else set())
        assert calls == (["rows"] if choice == "rows" else [])
        assert torch.all(output[2:] == 0)
