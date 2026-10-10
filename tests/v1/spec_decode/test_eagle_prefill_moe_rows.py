# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager MTP draft prefill runs the draft MoE only for the sampled rows.

The draft K/V (written before the MoE) must stay bitwise identical, the sampled
rows must match the full computation, and draft steps are skipped only when
every request in the batch is mid-prefill.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import vllm.v1.worker.gpu.spec_decode.eagle.speculator as speculator_module
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.eagle.prefill_moe_rows import (
    DraftMoERowSelector,
    PadRowOps,
)
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
    EagleSpeculator,
    _has_single_mtp_decoder_block,
    _is_context_only_prefill,
)

HIDDEN = 8


@pytest.mark.parametrize(
    ("num_mtp_layers", "layer_count", "expected"),
    [
        (1, 1, True),
        (2, 2, False),
        (None, 1, False),
        (1, 2, False),
    ],
)
def test_prefill_row_selection_requires_verified_single_mtp_block(
    num_mtp_layers, layer_count, expected
):
    predictor = SimpleNamespace(
        num_mtp_layers=num_mtp_layers,
        layers=[object() for _ in range(layer_count)],
    )
    model = SimpleNamespace(model=predictor)
    assert _has_single_mtp_decoder_block(model) is expected


class _Runner:
    """Row-wise stand-in for ``MoERunner._forward_impl``.

    Like the Qwen MoE runner that owns its gate, it receives hidden_states in
    place of router logits and returns ``(shared_output, fused_output)``.
    """

    def __init__(self, shared: bool = True) -> None:
        self.shared = shared
        self.calls: list[tuple] = []
        self.scale = torch.linspace(-1.0, 1.0, HIDDEN)

    def _forward_impl(
        self, layer, hidden_states, router_logits, shared_experts_input, input_ids=None
    ):
        self.calls.append(
            (layer, hidden_states, router_logits, shared_experts_input, input_ids)
        )
        # Element-wise, so a row's result does not depend on the row count.
        fused = torch.tanh(hidden_states * self.scale + 0.25)
        if not self.shared:
            return fused
        return shared_experts_input * 0.5, fused


def _moe_layer(shared: bool = True) -> SimpleNamespace:
    return SimpleNamespace(runner=_Runner(shared))


def _call(layer, hidden, router_logits=None, shared_input=None, input_ids=None):
    router_logits = hidden if router_logits is None else router_logits
    shared_input = hidden if shared_input is None else shared_input
    return layer.runner._forward_impl(
        layer, hidden, router_logits, shared_input, input_ids
    )


def test_inactive_selector_passes_inputs_through():
    layer = _moe_layer()
    DraftMoERowSelector([layer])
    hidden = torch.randn(5, HIDDEN)
    shared, fused = _call(layer, hidden)
    (_, h, r, s, i) = layer.runner.calls[-1]
    assert h is hidden and r is hidden and s is hidden and i is None
    assert torch.equal(fused, torch.tanh(hidden * layer.runner.scale + 0.25))
    assert torch.equal(shared, hidden * 0.5)


@pytest.mark.parametrize("shared", [True, False])
def test_selected_rows_are_scattered_and_other_rows_are_zero(shared):
    layer = _moe_layer(shared)
    selector = DraftMoERowSelector([layer])
    hidden = torch.randn(7, HIDDEN)
    full = _call(layer, hidden)
    rows = torch.tensor([6, 2])
    with selector.select(rows):
        partial = _call(layer, hidden)

    (_, h, r, s, _) = layer.runner.calls[-1]
    assert h.shape == (2, HIDDEN)
    assert torch.equal(h, hidden[rows])
    # The router alias and the shared-expert input follow the selected rows.
    assert r is h and s is h

    full = full if isinstance(full, tuple) else (full,)
    partial = partial if isinstance(partial, tuple) else (partial,)
    others = torch.tensor([0, 1, 3, 4, 5])
    for f, p in zip(full, partial, strict=True):
        assert p.shape == f.shape
        # Row-wise math: the selected rows equal the full computation.
        assert torch.equal(p[rows], f[rows])
        assert torch.count_nonzero(p[others]) == 0


def test_separate_per_token_inputs_are_selected():
    layer = _moe_layer()
    selector = DraftMoERowSelector([layer])
    hidden = torch.randn(6, HIDDEN)
    router_logits = torch.randn(6, 3)
    input_ids = torch.arange(6)
    rows = torch.tensor([1, 4])
    with selector.select(rows):
        _call(layer, hidden, router_logits=router_logits, input_ids=input_ids)
    (_, _, r, _, i) = layer.runner.calls[-1]
    assert torch.equal(r, router_logits[rows])
    assert torch.equal(i, input_ids[rows])


def test_select_restores_previous_rows_on_error():
    layer = _moe_layer()
    selector = DraftMoERowSelector([layer])
    hidden = torch.randn(4, HIDDEN)
    with pytest.raises(RuntimeError), selector.select(torch.tensor([3])):
        raise RuntimeError("boom")
    _call(layer, hidden)
    assert layer.runner.calls[-1][1] is hidden


def test_injected_pad_ops_preserve_real_routes_and_restore_nested_contexts():
    layer = _moe_layer()
    ids = torch.arange(12).reshape(6, 2)
    weights = torch.full((6, 2), 0.5)
    calls = []

    def route():
        calls.append("router")
        return weights, ids.clone()

    def mask(output_ids, num_valid):
        output_ids[int(num_valid[0]) :] = -1

    layer.runner.router = SimpleNamespace(select_experts=route)
    layer.runner._quant_method = SimpleNamespace(is_monolithic=False)
    selector = DraftMoERowSelector([layer], pad_ops=PadRowOps(mask, calls.append))
    assert selector.pad_rows_ready
    with selector.pad_rows(torch.tensor([4])):
        actual_weights, actual_ids = layer.runner.router.select_experts()
        assert actual_weights is weights
        assert torch.equal(actual_ids[:4], ids[:4])
        assert (actual_ids[4:] == -1).all()
        with pytest.raises(RuntimeError), selector.pad_rows(torch.tensor([2])):
            _, nested = layer.runner.router.select_experts()
            assert (nested[2:] == -1).all()
            raise RuntimeError("restore outer padding boundary")
        _, restored = layer.runner.router.select_experts()
        assert torch.equal(restored[:4], ids[:4])
        assert (restored[4:] == -1).all()
    _, inactive = layer.runner.router.select_experts()
    assert torch.equal(inactive, ids)
    assert calls == ["router"] * 4


def test_injected_pad_ops_report_unsupported_router():
    notes: list[str] = []
    selector = DraftMoERowSelector(
        [_moe_layer()], pad_ops=PadRowOps(lambda *args: None, notes.append)
    )
    assert not selector.pad_rows_ready
    assert notes == ["fallback:router"]


def test_without_pad_ops_does_not_wrap_router():
    layer = _moe_layer()
    original = lambda: (torch.ones(1), torch.zeros(1, dtype=torch.int64))
    layer.runner.router = SimpleNamespace(select_experts=original)
    layer.runner._quant_method = SimpleNamespace(is_monolithic=False)
    selector = DraftMoERowSelector([layer])
    assert selector.pad_ops is None and not selector.pad_rows_ready
    assert layer.runner.router.select_experts is original


def test_from_model_skips_target_layers(monkeypatch):
    import vllm.model_executor.layers.fused_moe.layer as layer_module

    class _FakeMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.runner = _Runner()

    monkeypatch.setattr(layer_module, "FusedMoE", _FakeMoE)
    shared_with_target = _FakeMoE()
    draft_only = _FakeMoE()
    draft = nn.ModuleList([shared_with_target, draft_only])
    target = nn.ModuleList([shared_with_target])

    selector = DraftMoERowSelector.from_model(draft, exclude=target.modules())
    assert selector is not None
    assert selector.layers == (draft_only,)
    assert DraftMoERowSelector.from_model(target, exclude=target.modules()) is None


class _ToyDraftLayer(nn.Module):
    """K/V written for every position before a MoE, like the MTP layer."""

    def __init__(self, num_slots: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.w_kv = torch.randn(HIDDEN, 2 * HIDDEN, generator=gen)
        self.kv_cache = torch.zeros(num_slots, 2 * HIDDEN)
        self.moe = _moe_layer()

    def forward(self, hidden, slot_mapping):
        self.kv_cache[slot_mapping] = hidden @ self.w_kv
        shared, fused = _call(self.moe, hidden)
        return hidden + shared + fused


@pytest.mark.parametrize(
    "query_lens",
    [
        [1632],  # prompt exactly one chunk
        [1633],  # one past the chunk boundary
        [5, 1632],  # decode request with a full prefill chunk
        [5, 416, 5],  # mixed batch, final chunk in the middle
        [1],
    ],
)
def test_kv_bitwise_and_sampled_rows_match_full_path(query_lens):
    num_tokens = sum(query_lens)
    layer = _ToyDraftLayer(num_slots=num_tokens + 11)
    selector = DraftMoERowSelector([layer.moe])
    gen = torch.Generator().manual_seed(1)
    hidden = torch.randn(num_tokens, HIDDEN, generator=gen)
    slot_mapping = torch.randperm(num_tokens + 11, generator=gen)[:num_tokens]
    last_token_indices = torch.tensor(np.cumsum(query_lens) - 1)

    full_out = layer(hidden, slot_mapping)
    full_kv = layer.kv_cache.clone()
    layer.kv_cache.zero_()
    with selector.select(last_token_indices):
        rows_out = layer(hidden, slot_mapping)

    assert torch.equal(layer.kv_cache, full_kv)
    assert torch.equal(rows_out[last_token_indices], full_out[last_token_indices])


@pytest.mark.parametrize(
    ("num_reqs", "incomplete", "expected"),
    [
        (2, None, False),
        (2, [True, True], True),
        (2, [True, False], False),
        (1, [False], False),
        # A mask that does not cover exactly the batch is not trusted.
        (2, [True], False),
        (2, [True, True, True], False),
    ],
)
def test_is_context_only_prefill(num_reqs, incomplete, expected):
    batch = _batch([1632] * num_reqs, incomplete)
    assert _is_context_only_prefill(batch) is expected


def _speculator(num_tokens: int, selector: DraftMoERowSelector | None):
    spec = object.__new__(EagleSpeculator)
    spec.method = "mtp"
    spec.num_speculative_steps = 4
    spec.max_num_reqs = 4
    spec.max_model_len = 4096
    spec.dp_size = 1
    spec.dp_rank = 0
    spec.share_mtp_topk_indices = False
    spec.prefill_cudagraph_manager = None
    spec.decode_cudagraph_manager = None
    spec.prefill_moe_rows = None
    spec._prefill_batch = None
    spec._skip_draft_decode = False
    spec.hidden_states = torch.zeros(num_tokens, HIDDEN)
    spec.temperature = torch.zeros(4)
    spec.seeds = torch.zeros(4, dtype=torch.int64)
    spec.idx_mapping = torch.zeros(4, dtype=torch.int32)
    spec.last_token_indices = torch.zeros(4, dtype=torch.int64)
    spec.draft_tokens = torch.full((4, 4), -7, dtype=torch.int64)
    spec.current_draft_step = torch.tensor(0)
    spec.draft_logits = None
    spec.input_buffers = SimpleNamespace(positions=torch.arange(num_tokens))
    if selector is not None:
        spec._enable_prefill_rows(selector)
    return spec


def _batch(query_lens, incomplete):
    if incomplete is not None:
        incomplete = np.array(incomplete)
    return SimpleNamespace(
        num_tokens_after_padding=sum(query_lens),
        num_tokens=sum(query_lens),
        num_reqs=len(query_lens),
        num_scheduled_tokens=np.array(query_lens),
        idx_mapping=torch.arange(len(query_lens), dtype=torch.int32),
        seq_lens=torch.tensor(query_lens),
        is_incomplete_prefilling_np=incomplete,
        is_prefilling_np=np.array(query_lens) > 5,
    )


@pytest.mark.parametrize(
    ("cg_mode", "dummy_run", "is_profile", "with_selector", "incomplete", "expect"),
    [
        # (sampled_rows_only, sample, decode steps run)
        (CUDAGraphMode.NONE, False, False, True, [True], (True, False, False)),
        (CUDAGraphMode.NONE, False, False, True, [True, True], (True, False, False)),
        (CUDAGraphMode.NONE, False, False, True, [False], (True, True, True)),
        (CUDAGraphMode.NONE, False, False, True, [True, False], (True, True, True)),
        (CUDAGraphMode.NONE, False, False, True, None, (True, True, True)),
        (CUDAGraphMode.PIECEWISE, False, False, True, [True], (False, True, True)),
        (CUDAGraphMode.NONE, True, False, True, [True], (False, True, True)),
        (CUDAGraphMode.NONE, False, True, True, [True], (False, True, True)),
        (CUDAGraphMode.NONE, False, False, False, [True], (False, True, True)),
    ],
)
def test_propose_restricts_rows_and_skips_steps_only_when_allowed(
    monkeypatch, cg_mode, dummy_run, is_profile, with_selector, incomplete, expect
):
    query_lens = [1632] if incomplete is None or len(incomplete) == 1 else [5, 1632]
    num_tokens = sum(query_lens)
    selector = DraftMoERowSelector([_moe_layer()]) if with_selector else None
    spec = _speculator(num_tokens, selector)
    spec.last_token_indices[: len(query_lens)] = torch.tensor(np.cumsum(query_lens) - 1)
    batch = _batch(query_lens, incomplete)

    desc = SimpleNamespace(cg_mode=cg_mode, num_tokens=num_tokens, num_reqs=None)
    monkeypatch.setattr(speculator_module, "prepare_eagle_inputs", lambda *a: None)
    monkeypatch.setattr(
        speculator_module, "get_uniform_decode_token_count", lambda *a, **k: None
    )
    monkeypatch.setattr(
        speculator_module, "dispatch_cg_and_sync_dp", lambda *a, **k: (desc, None)
    )
    monkeypatch.setattr(speculator_module, "prepare_eagle_decode", lambda *a, **k: None)
    seen = {}

    def run_model(n, *a, **k):
        seen["rows"] = None if selector is None else selector._rows
        h = torch.zeros(n, HIDDEN)
        return h, h

    sampled = []

    def sample_draft(hidden, *a):
        sampled.append(hidden.shape[0])
        return torch.zeros(hidden.shape[0], dtype=torch.int64)

    decode = []
    monkeypatch.setattr(spec, "run_model", run_model, raising=False)
    monkeypatch.setattr(spec, "_sample_draft", sample_draft, raising=False)
    monkeypatch.setattr(
        EagleSpeculator,
        "multi_step_decode",
        lambda self, *a, **k: decode.append(self._skip_draft_decode),
    )

    kwargs = dict(
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.zeros(num_tokens, HIDDEN),
        aux_hidden_states=None,
        num_sampled=torch.zeros(len(query_lens), dtype=torch.int32),
        num_rejected=torch.zeros(len(query_lens), dtype=torch.int32),
        last_sampled=torch.zeros(4, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(4, dtype=torch.int64),
        temperature=torch.zeros(4),
        seeds=torch.zeros(4, dtype=torch.int64),
    )
    if dummy_run or is_profile:
        # The model runner's dummy and profile runs pass everything by keyword.
        out = spec.propose(
            input_batch=batch, dummy_run=dummy_run, is_profile=is_profile, **kwargs
        )
    else:
        out = spec.propose(batch, **kwargs)

    rows_only, sample, steps = expect
    assert (seen["rows"] is not None) is rows_only
    if rows_only:
        assert torch.equal(seen["rows"], spec.last_token_indices[: len(query_lens)])
    assert bool(sampled) is sample
    # multi_step_decode runs its steps only when the skip flag is clear.
    assert decode == [not steps]
    assert spec._prefill_batch is None and spec._skip_draft_decode is False
    assert out.shape == (len(query_lens), 4)


def test_prefill_selects_last_rows_and_can_skip_sampling(monkeypatch):
    num_tokens = 12
    selector = DraftMoERowSelector([_moe_layer()])
    spec = _speculator(num_tokens, selector)
    spec.last_token_indices[:2] = torch.tensor([4, 11])
    seen_rows = []

    def run_model(*args, **kwargs):
        seen_rows.append(selector._rows)
        last = torch.arange(num_tokens * HIDDEN, dtype=torch.float32)
        last = last.view(num_tokens, HIDDEN)
        return last, last + 1

    monkeypatch.setattr(spec, "run_model", run_model, raising=False)
    monkeypatch.setattr(
        spec,
        "_sample_draft",
        lambda hidden, *a: hidden[:, 0].to(torch.int64),
        raising=False,
    )

    # Final chunks: sampled rows only, drafts sampled as before.
    spec._prefill_batch = _batch([5, 7], [False, False])
    spec.prefill(2, num_tokens, None, None, None)
    assert torch.equal(seen_rows[-1], torch.tensor([4, 11]))
    assert selector._rows is None
    assert spec._skip_draft_decode is False
    assert spec.draft_tokens[:2, 0].tolist() == [4 * HIDDEN, 11 * HIDDEN]
    assert torch.equal(spec.input_buffers.positions[:2], torch.tensor([4, 11]))

    # Every request mid-prefill: K/V only, no sampling, decode steps skipped.
    spec.draft_tokens.fill_(-7)
    hidden_before = spec.hidden_states.clone()
    spec.input_buffers.positions = torch.arange(num_tokens)
    spec._prefill_batch = _batch([5, 7], [True, True])
    spec.prefill(2, num_tokens, None, None, None)
    assert torch.equal(seen_rows[-1], torch.tensor([4, 11]))
    assert spec._skip_draft_decode is True
    assert (spec.draft_tokens == -7).all()
    assert torch.equal(spec.hidden_states, hidden_before)
    assert torch.equal(spec.input_buffers.positions, torch.arange(num_tokens))

    # No serving batch (dummy, profile or graph capture): the full path.
    spec._skip_draft_decode = False
    spec._prefill_batch = None
    spec.prefill(2, num_tokens, None, None, None)
    assert seen_rows[-1] is None

    # Piecewise graphs keep the full path.
    spec._prefill_batch = _batch([5, 7], [True, True])
    spec.prefill(2, num_tokens, None, None, None, CUDAGraphMode.PIECEWISE)
    assert seen_rows[-1] is None and spec._skip_draft_decode is False


def test_multi_step_decode_returns_early_when_skipped():
    spec = object.__new__(EagleSpeculator)
    spec._skip_draft_decode = True
    # Any attribute access beyond the flag would raise on this bare object.
    assert EagleSpeculator.multi_step_decode(spec, 2, False, None, None) is None


def test_recording_entry_point_marks_serving_batches_and_resets(monkeypatch):
    spec = _speculator(8, DraftMoERowSelector([_moe_layer()]))
    batch = _batch([8], [True])
    seen = []
    monkeypatch.setattr(
        EagleSpeculator,
        "propose",
        lambda self, *a, **k: seen.append(self._prefill_batch),
    )
    spec.propose(input_batch=batch, dummy_run=True)
    spec.propose(input_batch=batch, is_profile=True)
    spec.propose(batch)
    spec.propose(input_batch=batch)
    assert seen == [None, None, batch, batch]
    assert spec._prefill_batch is None

    # A call the entry point cannot interpret still reaches propose(), with no
    # recorded batch, so the draft keeps the full path.
    spec.propose()
    assert seen[-1] is None and len(seen) == 5

    def boom(self, *a, **k):
        self._skip_draft_decode = True
        raise RuntimeError("boom")

    monkeypatch.setattr(EagleSpeculator, "propose", boom)
    with pytest.raises(RuntimeError):
        spec.propose(batch)
    assert spec._prefill_batch is None and spec._skip_draft_decode is False
