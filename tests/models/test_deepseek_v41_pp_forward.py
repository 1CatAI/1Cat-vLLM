# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The PP2 forward split has the same control flow as one stage (CPU only).

``DeepseekV41Model._forward_fused`` / ``_forward_reference`` and their stage
helpers are taken from model.py by source, as is the pre-PP fused loop at
the #1277 merge (899e9ed). They run here over 40 stub layers with stub HC
kernels on CPU. The stub fused transition deliberately differs from
expand + mixes + collapse, so a wrongly taken branch changes the result.
Checks:

- one stage, no simulated boundary: the new fused loop gives the same bits as
  the pre-PP loop (TP4 unchanged);
- one stage with VLLM_V41_PP_BOUNDARY_SIM=20 gives the same bits as stage 1
  (layers 0-19) handing (hidden, pre) to stage 2 (layers 20-39), on both the
  fused and the reference path, and the boundary tap sees the hand-off;
- the simulated boundary does change the fused result (it is not a no-op).

Real kernels are not used, so this does not check that the boundary branch is
numerically close to the fused transition; M1 on GPUs does.

    python tests/models/test_deepseek_v41_pp_forward.py   # or pytest
"""

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.sequence import IntermediateTensors

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "vllm" / "models" / "deepseek_v41" / "model.py"
PRE_PP = "899e9ed"  # the #1277 merge
LAYERS, HIDDEN, TOKENS = 40, 16, 3
ENGRAM = (1, 14)


def _methods(source, cls, names):
    tree = ast.parse(source)
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    found = [n for n in body.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in found} == set(names), (cls, names)
    return found


# ---- stub HC kernels (CPU, deterministic) ----
def hc_mixes(residual, fn, scale, base):
    score = residual.mean(-1) * scale + base  # [T, 4]
    pre = torch.softmax(score + fn, -1)
    post = torch.sigmoid(score - fn)
    comb = torch.softmax(score[:, :, None] + score[:, None, :] * fn, -1)
    return pre, post, comb


def collapse_hc(hidden, pre, weight):
    return (hidden * pre[..., None]).sum(1) * weight


def expand_hc(y, hidden, post, comb, inplace=False):
    out = torch.einsum("tij,tjh->tih", comb, hidden) + post[..., None] * y[:, None]
    if inplace:
        hidden.copy_(out)
        return hidden
    return out


def hc_transition(hidden, y, post, comb, pre, weight, fn=None, scale=None, base=None):
    expand_hc(y, hidden, post, comb, inplace=True)
    inputs = collapse_hc(hidden, pre, weight)
    if fn is None:
        return inputs, None
    mix = hc_mixes(hidden, fn, scale, base)
    # Unlike the real kernel, not equal to the reference branch on purpose.
    return inputs + 1e-3, tuple(m + 1e-3 for m in mix)


class _CpuTorch:
    """torch for the extracted code, with no CUDA stream to ask about."""

    cuda = SimpleNamespace(is_current_stream_capturing=lambda: False)

    def __getattr__(self, name):
        return getattr(torch, name)


def _namespace(sim):
    return dict(
        torch=_CpuTorch(),
        hc_mixes=hc_mixes,
        collapse_hc=collapse_hc,
        expand_hc=expand_hc,
        hc_transition=hc_transition,
        IntermediateTensors=IntermediateTensors,
        _tmark=lambda name, tokens: None,
        _treport=lambda tokens: None,
        _ATT_T={"on": False},
        _PRED_STATS=False,
        _PREFETCH_HINTS=False,
        _PP_BOUNDARY_SIM=sim,
        # Side-stream mHC mixes (VLLM_V41_HC_SIDE) are off here.
        hc_join=lambda: None,
        hc_kick=lambda: None,
    )


def _bind(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(MODEL), "exec"), namespace)
    return {n.name: namespace[n.name] for n in nodes}


class Missing:
    """PPMissingLayer stand-in: any attribute access fails."""


class Layer:
    def __init__(self, i, block_forward):
        g = torch.Generator().manual_seed(1000 + i)

        def vec(*shape):
            return torch.randn(*shape, generator=g, dtype=torch.float32) * 0.5

        self.i = i
        self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base = vec(4), vec(4), vec(4)
        self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base = vec(4), vec(4), vec(4)
        self.attn_norm = SimpleNamespace(weight=1 + vec(HIDDEN))
        self.ffn_norm = SimpleNamespace(weight=1 + vec(HIDDEN))
        wa, wf, we = vec(HIDDEN, HIDDEN), vec(HIDDEN, HIDDEN), vec(HIDDEN)

        def attn(inputs, positions, table):
            return torch.tanh(inputs @ wa) + positions[:, None] * 1e-2

        attn.ratio = i % 2  # type: ignore[attr-defined]
        self.attn = attn
        self.ffn = lambda inputs: torch.tanh(inputs @ wf)
        self.engram = (
            (lambda hidden, io, valid: hidden + 0.1 * torch.tanh(hidden * we))
            if i in ENGRAM
            else None
        )
        self._forward = block_forward

    def __call__(self, x, positions, previous_pre, table):
        return self._forward(self, x, positions, previous_pre, table)


def _model(fns, block_forward, start, end, tap=None):
    model = SimpleNamespace(
        layers=[
            Layer(i, block_forward) if start <= i < end else Missing()
            for i in range(LAYERS)
        ],
        start_layer=start,
        end_layer=end,
        aux_hidden_state_layers=(),
        compressed_rope=None,
        window_rope=None,
        norm=SimpleNamespace(weight=torch.linspace(0.5, 1.5, HIDDEN)),
        engram_io=None,
        pp_boundary_tap=tap,
        # No KV mirror export or index top-k hand-off at these cuts.
        export_source=None,
        topk_export=None,
        topk_import=None,
    )
    for name, fn in fns.items():
        setattr(model, name, fn.__get__(model))
    return model


def _inputs():
    g = torch.Generator().manual_seed(7)
    hidden = torch.randn(TOKENS, HIDDEN, generator=g, dtype=torch.float32)
    hidden = hidden[:, None].expand(-1, 4, -1).contiguous()
    pre = torch.zeros(TOKENS, 4, dtype=torch.float32)
    pre[:, 0] = 1
    positions = torch.arange(TOKENS, dtype=torch.float32) + 40
    return hidden, positions, pre


NEW = MODEL.read_text()
NAMES = ("_forward_fused", "_forward_reference", "_stage_boundary", "_hand_off")


def _new(sim):
    namespace = _namespace(sim)
    fns = _bind(_methods(NEW, "DeepseekV41Model", NAMES), namespace)
    block = _bind(_methods(NEW, "V41Block", ("forward",)), namespace)["forward"]
    return fns, block


def _run(path, sim, stages, float32_handoff=False):
    """stages: [(start, end)]; returns the last stage's output and the taps."""
    fns, block = _new(sim)
    taps = []
    hidden, positions, pre = _inputs()
    out = None
    for start, end in stages:
        model = _model(
            fns, block, start, end, tap=lambda h, p: taps.append((h.clone(), p.clone()))
        )
        out = getattr(model, path)(hidden, positions, pre, object(), None, [])
        if isinstance(out, IntermediateTensors):
            # The PP hop delivers a copy into the next stage's buffer.
            hidden, pre = out["hidden"].clone(), out["pre"].clone()
    return out, taps


def test_tp4_fused_loop_unchanged():
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{PRE_PP}:vllm/models/deepseek_v41/model.py"],
        capture_output=True,
        text=True,
    )
    if shown.returncode != 0:
        pytest.skip(f"{PRE_PP} is not in this checkout (shallow clone)")
    base = shown.stdout
    old_ns = _namespace(0)
    old = _bind(_methods(base, "DeepseekV41Model", ("_forward_fused",)), old_ns)
    fns, block = _new(0)
    hidden, positions, pre = _inputs()
    reference = _model(old, block, 0, LAYERS)
    expected = reference._forward_fused(
        hidden.clone(), positions, pre, object(), None, []
    )
    got, taps = _run("_forward_fused", 0, [(0, LAYERS)])
    assert torch.equal(got, expected)
    assert taps == []


def _check_split(path):
    sim, sim_taps = _run(path, 20, [(0, LAYERS)])
    pp2, pp2_taps = _run(path, 0, [(0, 20), (20, LAYERS)])
    assert not isinstance(pp2, IntermediateTensors)
    assert torch.equal(sim, pp2), (sim - pp2).abs().max()
    assert len(sim_taps) == len(pp2_taps) == 1
    assert all(torch.equal(a, b) for a, b in zip(sim_taps[0], pp2_taps[0]))
    return sim


def test_simulated_boundary_equals_pp2_fused():
    sim = _check_split("_forward_fused")
    plain, _ = _run("_forward_fused", 0, [(0, LAYERS)])
    assert not torch.equal(sim, plain)  # the boundary branch was taken


def test_simulated_boundary_equals_pp2_reference():
    _check_split("_forward_reference")


def test_stage1_returns_the_hand_off():
    fns, block = _new(0)
    hidden, positions, pre = _inputs()
    out = _model(fns, block, 0, 20)._forward_fused(
        hidden, positions, pre, object(), None, []
    )
    assert isinstance(out, IntermediateTensors)
    assert set(out.tensors) == {"hidden", "pre"}
    assert out["hidden"].shape == (TOKENS, 4, HIDDEN)
    assert out["pre"].shape == (TOKENS, 4)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)} passed")
