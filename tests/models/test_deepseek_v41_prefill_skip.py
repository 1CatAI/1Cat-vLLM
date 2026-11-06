# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_PREFILL_SKIP_ROWS (prefill_skip.py), CPU only.

- the knob: 0/unset is off, values below MIN_ROWS and non-integers refuse;
- the step plan (skip_step) and the session-record guard;
- ``DeepseekV41Model._forward_fused`` / ``_forward_reference`` /
  ``_prefill_skipped`` / ``prefill_skip_bind``, taken from model.py by source
  and run over 40 stub layers (the harness of test_deepseek_v41_pp_forward):
  knob off and unbound give the same bits as before; a skipped step on the
  last PP2 stage runs layer 20's attention only, zeroes the rings of 21-39,
  and returns zero rows and zero aux taps; stage 1 is never skipped;
- the cone: a toy chunked prefill of 19 sliding-window layers with rings
  gives bitwise the same last row and the same decode-visible ring rows
  with skipped early chunks when K >= 20 windows, and differs when K is too
  small (so the check can fail).

    python tests/models/test_deepseek_v41_prefill_skip.py   # or pytest
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_deepseek_v41_pp_forward as harness  # noqa: E402

from vllm.models.deepseek_v41 import prefill_skip  # noqa: E402
from vllm.sequence import IntermediateTensors  # noqa: E402

LAYERS, HIDDEN, TOKENS = harness.LAYERS, harness.HIDDEN, harness.TOKENS
AUX = (37, 38, 39)
SKIP_AFTER = 20  # max(kv_source_layer_ids)


# ---- the knob and the plans ----
def _rows(value):
    old = os.environ.get(prefill_skip.KNOB)
    try:
        if value is None:
            os.environ.pop(prefill_skip.KNOB, None)
        else:
            os.environ[prefill_skip.KNOB] = value
        return prefill_skip.skip_rows()
    finally:
        if old is None:
            os.environ.pop(prefill_skip.KNOB, None)
        else:
            os.environ[prefill_skip.KNOB] = old


def test_knob_values():
    assert _rows(None) == 0
    assert _rows("0") == 0
    assert _rows(" 3072 ") == 3072
    assert _rows("2560") == prefill_skip.MIN_ROWS == 2560
    for bad in ("1", "2559", "-1", "abc", "3k"):
        try:
            _rows(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} accepted")


def test_skip_step():
    k = 3072
    p = 100_000
    # off, or no rows scheduled
    assert not prefill_skip.skip_step([0], [8192], [p], 0)
    assert not prefill_skip.skip_step([], [], [], k)
    assert not prefill_skip.skip_step([0], [0], [p], k)
    # early chunk; the boundary c + n == p - k still skips
    assert prefill_skip.skip_step([0], [8192], [p], k)
    assert prefill_skip.skip_step([p - k - 8192], [8192], [p], k)
    assert not prefill_skip.skip_step([p - k - 8191], [8192], [p], k)
    # last chunk, decode, short prompt
    assert not prefill_skip.skip_step([p - 100], [100], [p], k)
    assert not prefill_skip.skip_step([p + 5], [6], [p], k)
    assert not prefill_skip.skip_step([0], [2000], [2000], k)
    # every scheduled request must be early; unscheduled ones do not count
    assert not prefill_skip.skip_step([0, p - 10], [8192, 6], [p, p], k)
    assert prefill_skip.skip_step([0, 0], [8192, 0], [p, 10], k)


def test_session_record_ok():
    k = 3072
    p = 100_000
    assert prefill_skip.session_record_ok(10, p, 0)  # knob off: unchanged
    assert prefill_skip.session_record_ok(p, p, k)  # prompt done
    assert prefill_skip.session_record_ok(p + 40, p, k)  # decoded
    assert not prefill_skip.session_record_ok(8192, p, k)  # inside skipped
    assert not prefill_skip.session_record_ok(p - k, p, k)  # last skipped row
    assert not prefill_skip.session_record_ok(p - k + 2559, p, k)
    assert prefill_skip.session_record_ok(p - k + 2560, p, k)


# ---- the forward control flow (model.py by source) ----
NAMES = harness.NAMES + ("_prefill_skipped", "prefill_skip_bind")


class _Log:
    def info(self, *args):
        self.last = args


def _fns():
    namespace = harness._namespace(0)
    namespace.update(
        prefill_skip=prefill_skip,
        logger=_Log(),
        get_tensor_model_parallel_rank=lambda: 0,
    )
    fns = harness._bind(
        harness._methods(harness.NEW, "DeepseekV41Model", NAMES), namespace
    )
    block = harness._bind(
        harness._methods(harness.NEW, "V41Block", ("forward",)), namespace
    )["forward"]
    return fns, block


def _model(start, end, *, skip_layer=None, aux=AUX, bound=True):
    fns, block = _fns()
    model = harness._model(fns, block, start, end)
    model.aux_hidden_state_layers = tuple(i for i in aux if start <= i < end)
    calls = []
    for layer in model.layers:
        if isinstance(layer, harness.Missing):
            continue
        attn, ffn, i = layer.attn, layer.ffn, layer.i

        def traced_attn(x, p, t, attn=attn, i=i):
            calls.append(("attn", i))
            return attn(x, p, t)

        def traced_ffn(x, ffn=ffn, i=i):
            calls.append(("ffn", i))
            return ffn(x)

        traced_attn.ratio = attn.ratio
        traced_attn.window_ring = torch.full((4, 2), float(i))
        layer.attn, layer.ffn = traced_attn, traced_ffn
    if bound:
        model._v41_skip_layer = skip_layer
        model._v41_skip_after = SKIP_AFTER
        model._v41_skip_eligible = end == LAYERS and end > SKIP_AFTER
        model._v41_skip_steps = 0
    return model, calls


def _run(model, path="_forward_fused", hidden=None, pre=None):
    h, positions, p = harness._inputs()
    hidden = h if hidden is None else hidden
    pre = p if pre is None else pre
    aux = []
    out = getattr(model, path)(
        hidden.clone(), positions, pre.clone(), object(), None, aux
    )
    return out, aux


def test_knob_off_same_bits():
    for path in ("_forward_fused", "_forward_reference"):
        for stages in ([(0, LAYERS)], [(0, 20), (20, LAYERS)]):
            outs = []
            for bound in (False, True):
                hidden, pre = None, None
                out = aux = None
                for start, end in stages:
                    model, _ = _model(start, end, bound=bound)
                    out, aux = _run(model, path, hidden, pre)
                    if isinstance(out, IntermediateTensors):
                        hidden, pre = out["hidden"].clone(), out["pre"].clone()
                outs.append((out, aux))
            (a, a_aux), (b, b_aux) = outs
            assert torch.equal(a, b), path
            assert len(a_aux) == len(b_aux) == 3
            assert all(torch.equal(x, y) for x, y in zip(a_aux, b_aux))


def _stage2_input():
    model, _ = _model(0, 20)
    out, _ = _run(model)
    assert isinstance(out, IntermediateTensors)
    return out["hidden"].clone(), out["pre"].clone()


def test_skip_stage2_fused():
    hidden, pre = _stage2_input()
    model, calls = _model(20, LAYERS)
    model.prefill_skip_bind(True)
    assert model._v41_skip_layer == SKIP_AFTER and model._v41_skip_steps == 1
    out, aux = _run(model, "_forward_fused", hidden, pre)
    model.prefill_skip_bind(False)
    assert model._v41_skip_layer is None
    assert calls == [("attn", 20)]  # no FFN 20, nothing after
    assert out.shape == (TOKENS, HIDDEN) and out.dtype == torch.float16
    assert not out.any()
    assert len(aux) == 3 and all(a.dtype == torch.float16 and not a.any() for a in aux)
    rings = {layer.i: layer.attn.window_ring for layer in model.layers[20:]}
    assert torch.equal(rings[20], torch.full((4, 2), 20.0))  # layer 20 runs
    assert all(not rings[i].any() for i in range(21, LAYERS))
    # unbound again: the next (executed) step runs every layer
    calls.clear()
    out, aux = _run(model, "_forward_fused", hidden, pre)
    assert [c for c in calls if c[0] == "attn"] == [("attn", i) for i in range(20, 40)]
    assert len(aux) == 3 and out.any()


def test_skip_single_stage_and_reference():
    model, calls = _model(0, LAYERS)
    model.prefill_skip_bind(True)
    out, aux = _run(model, "_forward_fused")
    assert [c for c in calls if c[0] == "attn"] == [("attn", i) for i in range(21)]
    assert [c for c in calls if c[0] == "ffn"] == [("ffn", i) for i in range(20)]
    assert not out.any() and len(aux) == 3
    # reference path: whole blocks up to 20, nothing after
    model, calls = _model(0, LAYERS)
    model.prefill_skip_bind(True)
    out, aux = _run(model, "_forward_reference")
    assert [c for c in calls if c[0] == "attn"] == [("attn", i) for i in range(21)]
    assert not out.any() and len(aux) == 3


def test_bind_refuses_where_not_eligible():
    # stage 1 of PP2 (not last), and a bound step that is not a skip
    model, calls = _model(0, 20)
    assert not model._v41_skip_eligible
    model.prefill_skip_bind(True)
    assert model._v41_skip_layer is None
    out, _ = _run(model)
    assert isinstance(out, IntermediateTensors)
    model, _ = _model(20, LAYERS)
    model.prefill_skip_bind(False)
    assert model._v41_skip_layer is None and model._v41_skip_steps == 0


# ---- the cone (toy chunked prefill, numpy float64) ----
def _toy(n_tokens, chunk, skip_rows, window, n_layers=19, dim=3, seed=0):
    """Layers 21.. of a toy: in[l+1][p] = tanh(A_l @ sum(key_l[p-w+1..p]) +
    in[l][p]); key_l = tanh(B_l @ in[l]); rings of 2 * window rows keyed by
    position, as in attention_ops (keys before the chunk come from the ring).
    Returns the last row's output and each layer's ring rows."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_tokens, dim))  # layer 20's output: always exact
    a = rng.standard_normal((n_layers, dim, dim)) * 0.3
    b = rng.standard_normal((n_layers, dim, dim)) * 0.3
    size = 2 * window
    rings = np.zeros((n_layers, size, dim))
    last = None
    for c in range(0, n_tokens, chunk):
        n = min(chunk, n_tokens - c)
        if skip_rows and prefill_skip.skip_step([c], [n], [n_tokens], skip_rows):
            rings[:] = 0.0
            continue
        h = x[c : c + n]
        for layer in range(n_layers):
            key = np.tanh(h @ b[layer].T)
            out = np.empty_like(h)
            for r in range(n):
                p = c + r
                total = np.zeros(dim)
                for j in range(max(0, p - window + 1), p + 1):
                    total = total + (key[j - c] if j >= c else rings[layer, j % size])
                out[r] = np.tanh(total @ a[layer].T + h[r])
            for r in range(max(0, n - size), n):
                rings[layer, (c + r) % size] = key[r]
            h = out
        last = h[-1]
    return last, rings.copy()


def _cone_case(window, skip_rows, n_tokens=None, chunk=None):
    n_tokens = n_tokens or 26 * window
    chunk = chunk or 3 * window
    base_last, base_rings = _toy(n_tokens, chunk, 0, window)
    last, rings = _toy(n_tokens, chunk, skip_rows, window)
    # Rows a decode step (and a hit within window - 4 rows) can read.
    reach = window - 4
    visible = [(p % (2 * window)) for p in range(n_tokens - window - reach, n_tokens)]
    same_last = np.array_equal(last, base_last)
    same_rings = np.array_equal(rings[:, visible], base_rings[:, visible])
    return same_last, same_rings


def test_cone_bitwise_with_min_rows():
    # MIN_ROWS = 20 windows (2560 at window 128); the toy scales it.
    window = 8
    for chunk in (3 * window, 5 * window, 7 * window + 3):
        for n_tokens in (26 * window, 31 * window + 5):
            same_last, same_rings = _cone_case(
                window, 20 * window, n_tokens=n_tokens, chunk=chunk
            )
            assert same_last and same_rings, (chunk, n_tokens)


def test_cone_breaks_when_rows_too_small():
    window = 8
    same_last, same_rings = _cone_case(window, 4 * window)
    assert not (same_last and same_rings)


# ---- fail closed: outputs that need every prompt row ----
def test_block_reasons():
    ok = dict(pooling=False, spec_method="dspark", kv_transfer=False, ec_transfer=False)
    assert prefill_skip.static_block_reasons(**ok) == []
    assert prefill_skip.static_block_reasons(**{**ok, "spec_method": None}) == []
    for change in (
        {"pooling": True},
        {"spec_method": "eagle3"},
        {"spec_method": "extract_hidden_states"},
        {"spec_method": "mtp"},
        {"kv_transfer": True},
        {"ec_transfer": True},
    ):
        assert len(prefill_skip.static_block_reasons(**{**ok, **change})) == 1, change
    assert (
        prefill_skip.step_block_reasons(prompt_logprobs=False, routed_experts=False)
        == []
    )
    assert prefill_skip.step_block_reasons(prompt_logprobs=True, routed_experts=False)
    assert prefill_skip.step_block_reasons(prompt_logprobs=False, routed_experts=True)


def test_skip_counter():
    counter = prefill_skip.SkipCounter()
    assert counter.step(True, ["a"], [8192]) == []
    assert counter.step(True, ["a"], [8192]) == []
    assert counter.step(False, ["a"], [0]) == []  # empty PP step: no report
    lines = counter.step(False, ["a"], [8192])
    assert lines == [
        "request a: skipped layers 21-39 for 2 prefill chunks (16384 rows)"
    ]
    assert counter.step(False, ["a"], [6]) == []  # decode: nothing more
    assert counter.step(True, ["b"], [8192]) == []
    lines = counter.step(False, ["c"], [100])  # b aborted while skipping
    assert lines == [
        "request b: left while skipping after 1 prefill chunks (8192 rows)"
    ]
    assert counter.counts == {}


RUNNER = harness.ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py"


def _runner_plan(rows, spec_method="dspark", **flags):
    source = RUNNER.read_text()
    logged = []
    namespace = dict(
        v41_prefill_skip=prefill_skip,
        _V41_PREFILL_SKIP_ROWS=rows,
        get_pp_group=lambda: SimpleNamespace(is_last_rank=True),
        get_tp_group=lambda: SimpleNamespace(rank_in_group=0),
        has_kv_transfer_group=lambda: flags.get("kv_transfer", False),
        has_ec_transfer=lambda: False,
        logger=SimpleNamespace(info=lambda *a: logged.append(a[0] % a[1:])),
    )
    fn = harness._bind(
        harness._methods(source, "GPUModelRunner", ("_v41_prefill_skip_plan",)),
        namespace,
    )["_v41_prefill_skip_plan"]
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(method=spec_method) if spec_method else None,
        is_pooling_model=False,
        num_prompt_logprobs={},
        routed_experts_initialized=False,
        input_batch=SimpleNamespace(
            req_ids=["r"],
            num_computed_tokens_cpu=np.array([0]),
            num_prompt_tokens=np.array([100_000]),
        ),
    )
    runner.plan = fn.__get__(runner)
    return runner, logged


def test_runner_plan_fails_closed_and_logs_once():
    scheduled = np.array([8192], dtype=np.int32)
    runner, logged = _runner_plan(3072)
    assert runner.plan(1, scheduled)
    assert "skip layers 21-39" in logged[0]
    runner.num_prompt_logprobs = {"r": 5}
    assert not runner.plan(1, scheduled)
    assert not runner.plan(1, scheduled)
    assert sum("prompt logprobs" in line for line in logged) == 1  # once
    runner.num_prompt_logprobs = {}
    runner.input_batch.num_computed_tokens_cpu = np.array([100_000 - 4000])
    assert not runner.plan(1, np.array([4000], dtype=np.int32))
    assert any("skipped layers 21-39 for 1 prefill chunks" in x for x in logged)
    for method in ("eagle3", "extract_hidden_states"):
        runner, logged = _runner_plan(3072, spec_method=method)
        assert not runner.plan(1, scheduled)
        assert "disabled" in logged[0]
    runner, logged = _runner_plan(3072, kv_transfer=True)
    assert not runner.plan(1, scheduled)


# ---- session prefix: hits at P << N are never admitted ----
# KVCacheManager's session-hit path, taken from kv_cache_manager.py by source
# (importing the module needs vllm._C). The partial prefix cache is Mamba-only
# (_init_partial_prefix_cache), so it is stubbed out here.
MANAGER = harness.ROOT / "vllm" / "v1" / "core" / "kv_cache_manager.py"
REACH = 128  # _V41_RING_REACH in kv_cache_manager.py


def _manager_methods(rows):
    namespace = dict(
        _V41_SESSION_PREFIX=True,
        _V41_RING_REACH=REACH,
        _V41_PREFILL_SKIP_ROWS=rows,
        v41_prefill_skip=prefill_skip,
        Request=object,
        KVCacheBlocks=object,
    )
    return harness._bind(
        harness._methods(
            MANAGER.read_text(),
            "KVCacheManager",
            ("free", "_v41_session_hit", "get_computed_blocks"),
        ),
        namespace,
    )


def _manager(rows, hit):
    manager = SimpleNamespace(
        _v41_last=None,
        enable_caching=True,
        log_stats=False,
        empty_kv_cache_blocks="EMPTY",
        create_kv_cache_blocks=lambda blocks: "HIT",
        _partial_prefix_hit=lambda request, hit_length, max_length: None,
        coordinator=SimpleNamespace(
            find_longest_cache_hit=lambda hashes, length: ([[]], min(hit, length)),
            shared_prefix_boundary=0,
            free=lambda request_id: None,
        ),
    )
    for name, fn in _manager_methods(rows).items():
        setattr(manager, name, fn.__get__(manager))
    return manager


def _request(tokens, computed, prompt):
    return SimpleNamespace(
        request_id="q",
        all_token_ids=list(tokens),
        num_tokens=len(tokens),
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        block_hashes=[],
        skip_reading_prefix_cache=False,
        shared_prefix_boundary=0,
        partial_prefix_entry=None,
    )


def _hit(rows, prompt, freed_at, hit):
    """A request of `prompt` tokens is freed after `freed_at` computed rows; a
    second request continuing the same tokens finds a cache hit at `hit`."""
    tokens = list(range(prompt + 50))
    manager = _manager(rows, hit)
    manager.free(_request(tokens[:prompt], freed_at, prompt))
    blocks, computed = manager.get_computed_blocks(_request(tokens, 0, len(tokens)))
    return computed if blocks == "HIT" else 0


def test_session_hits_with_skip():
    n, k = 100_000, 3072
    # Finished prompt: a hit within the ring reach is admitted; any hit at a
    # shorter prefix P << N (inside or below the skipped region) is refused
    # (full recompute), with or without the knob.
    for rows in (0, k):
        assert _hit(rows, n, n, n - 100) == n - 100
        for p in (50_000, n - k, n - k - 8192, 8192, n - REACH - 2):
            assert _hit(rows, n, n, p) == 0, (rows, p)
    # Freed inside the skipped chunks (abort/preempt at 8192): the knob drops
    # the record; knob off keeps today's behaviour.
    assert _hit(0, n, 8192, 8192) == 8192
    assert _hit(k, n, 8192, 8192) == 0
    assert _hit(k, n, 8192, 8192 - 100) == 0
    # Just past the skip boundary (rings maybe outside the cone): refused;
    # MIN_ROWS past it: admitted.
    assert _hit(k, n, n - k + 100, n - k + 100) == 0
    edge = n - k + prefill_skip.MIN_ROWS
    assert _hit(k, n, edge, edge) == edge


# ---- the long-prompt greedy gate (benchmarks/.../prefill_skip_greedy.py) ----
def test_greedy_gate_verdict():
    import importlib.util

    path = harness.ROOT / "benchmarks/deepseek_v41/tools/prefill_skip_greedy.py"
    spec = importlib.util.spec_from_file_location("prefill_skip_greedy", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    def row(tokens, tops):
        steps = [{"token": t, "top": top} for t, top in zip(tokens, tops)]
        return {"text": "".join(tokens), "steps": steps}

    tie = [["x", -0.70], ["y", -0.75]]  # gap 0.05
    wide = [["x", -0.10], ["y", -2.0]]
    base = row(["a", "x", "b"], [[["a", 0.0], ["q", -3.0]], tie, [["b", 0.0]]])
    assert tool.verdict(base, base) == "IDENTICAL"
    near = row(
        ["a", "y", "c"], [[["a", 0.0], ["q", -3.0]], [["y", -0.7], ["x", -0.72]], []]
    )
    assert tool.verdict(base, near) == "NEAR"
    far = row(["a", "y"], [[["a", 0.0]], [["y", -0.1], ["x", -2.0]]])
    assert tool.verdict(base, far) == "DIFF"  # other's gap too wide
    base_wide = row(["a", "x"], [[["a", 0.0]], wide])
    assert tool.verdict(base_wide, near) == "DIFF"  # baseline's gap too wide
    other = row(["a", "z"], [[["a", 0.0]], [["z", -0.7], ["x", -0.72]]])
    assert tool.verdict(base, other) == "DIFF"  # not the baseline's #2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)} passed")
