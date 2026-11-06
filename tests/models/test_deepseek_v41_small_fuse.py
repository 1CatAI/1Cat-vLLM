# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 attention PREP / POST fusions against today's kernels (CPU, Triton
interpreter).

* VLLM_V41_FUSE_ATTN_PREP: ``_attn_prep`` must give ``_rotate`` (query) and
  ``_short_rms`` -> ``_rotate`` -> ``_qdq`` (window) bit for bit.
* VLLM_V41_FUSE_ATTN_POST: ``_attn_post`` must give ``_ring_insert`` ->
  ``_rotate`` (inverse) -> ``_qdq`` bit for bit, ring included.

Scope: the interpreter runs programs one after another and ``tl.sum`` as a
numpy reduction, so these tests prove the kernels' logic (masks, chunk
order, the deferred ring write), not the GPU's reduction trees. Reduction
layouts are compared on sm_70 TTGIR in test_deepseek_v41_small_fuse_sm70.py;
the GPU check is benchmarks/deepseek_v41/small_fuse_check.py.

    python tests/models/test_deepseek_v41_small_fuse.py   # or pytest
"""

import os
import sys

import pytest

if "triton" in sys.modules and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip(
        "needs TRITON_INTERPRET=1 in the environment: triton was imported "
        "before this file could turn the interpreter on",
        allow_module_level=True,
    )
os.environ["TRITON_INTERPRET"] = "1"  # before triton is imported

import importlib  # noqa: E402
import logging  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

_ENV = {
    "VLLM_V41_HC_WINDOW": "2048",
    "VLLM_V41_HC_COLS": "32",
    "VLLM_V41_HC_PW": "4",
    "VLLM_V41_HC_FW": "4",
    "VLLM_V41_HC_V3": "1",
    "VLLM_V41_HC_V3_GROUPS": "8",
}


class _Envs(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("VLLM_"):
            return _ENV.get(name, "0")
        raise AttributeError(name)


def _stubs():
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    if "vllm.triton_utils" not in sys.modules:
        tu = types.ModuleType("vllm.triton_utils")
        tu.tl, tu.triton = tl, triton
        sys.modules["vllm.triton_utils"] = tu
    if "vllm.logger" not in sys.modules:
        lg = types.ModuleType("vllm.logger")
        lg.init_logger = logging.getLogger
        sys.modules["vllm.logger"] = lg
    if "vllm.envs" not in sys.modules:
        envs = _Envs("vllm.envs")
        sys.modules["vllm.envs"] = envs
        sys.modules["vllm"].envs = envs


def _package(name, path=None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(ROOT / path)] if path else []
        sys.modules[name] = mod
    return sys.modules[name]


def _load_attention():
    _stubs()
    # The model package's __init__ pulls in the whole model; load just the
    # kernel modules, with the prefill BMM (unused here) stubbed out.
    _package("vllm.models", "vllm/models")
    _package("vllm.models.deepseek_v4", "vllm/models/deepseek_v4")
    _package("vllm.models.deepseek_v4.common", "vllm/models/deepseek_v4/common")
    _package("vllm.models.deepseek_v4.common.ops", "vllm/models/deepseek_v4/common/ops")
    bmm = types.ModuleType("vllm.models.deepseek_v4.common.ops.sparse_prefill_bmm")
    bmm.attend_gathered_keys = None
    sys.modules.setdefault(bmm.__name__, bmm)
    _package("vllm.models.deepseek_v41", "vllm/models/deepseek_v41")
    return tuple(
        importlib.import_module("vllm.models.deepseek_v41." + name)
        for name in ("attn_fuse", "quant_rope", "norm", "attention_ops")
    )


ATTN, ROPE, NORM, OPS = _load_attention()


@triton.jit
def _round_nearest_even_i32(x):
    # fp8_software's helper is one PTX cvt.rni, which the interpreter cannot
    # run; this is the same round-half-even, exact for |x| < 2**23. Old and
    # new kernels share it.
    f = tl.floor(x)
    d = x - f
    fi = f.to(tl.int32)
    return fi + ((d > 0.5) | ((d == 0.5) & ((fi & 1) == 1))).to(tl.int32)


sys.modules[
    "vllm.models.deepseek_v4.common.ops.fp8_software"
]._round_nearest_even_i32 = _round_nearest_even_i32


class _PoisonTorch(types.ModuleType):
    """``torch`` for the module under test: fresh buffers start as NaN / -1."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def _poison(t):
        return t.fill_(float("nan") if t.is_floating_point() else -1)

    def empty(self, *args, **kwargs):
        return self._poison(torch.empty(*args, **kwargs))

    def empty_like(self, x, **kwargs):
        return self._poison(torch.empty_like(x, **kwargs))


@pytest.fixture(autouse=True)
def _poisoned(monkeypatch):
    monkeypatch.setattr(ATTN, "torch", _PoisonTorch("torch"))
    yield


def _empty_like(x):
    return _PoisonTorch._poison(torch.empty_like(x))


def _bits(x):
    x = x.contiguous()
    return x.view({4: torch.int32, 2: torch.int16}[x.element_size()])


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


# ----------------------------------------------------------------------------
# Attention PREP / POST. The old wrappers refuse CPU tensors, so the reference
# below launches the old kernels exactly as apply_rope / short_rms_norm /
# attention_qdq / insert_window_ring do.


def _old_rope(x, positions, table, inverse=False):
    result = _empty_like(x)
    ROPE._rotate[(x.shape[0], x.shape[1])](
        x,
        positions,
        table,
        result,
        table.shape[0] - 1,
        HEADS=x.shape[1],
        D=x.shape[-1],
        INVERSE=inverse,
        BLOCK=triton.next_power_of_2(x.shape[-1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


def _old_rms(x, weight, eps):
    output = _empty_like(x)
    NORM._short_rms[(x.shape[0],)](
        x,
        weight,
        output,
        H=x.shape[1],
        EPS=eps,
        BLOCK=triton.next_power_of_2(x.shape[1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def _old_qdq(x):
    result = _empty_like(x)
    ROPE._qdq[(x.numel() // 128,)](
        x,
        result,
        GROUP=128,
        FP4=False,
        E4M3_SCALE=False,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


def _old_ring(ring, values, positions, valid):
    tokens = values.shape[0]
    first = max(0, tokens - OPS.WINDOW_RING)
    OPS._ring_insert[(tokens - first,)](
        values, positions, valid, ring, first, RING=OPS.WINDOW_RING
    )


def _rows(tokens, width, g, scale=1.0):
    x = torch.randn(tokens, width, generator=g) * scale
    x[0, :7] = 0.0
    x[0, 128:256] *= 1e-5  # a group under the QDQ amax floor
    if tokens > 1:
        x[1] = 0.0  # all-zero row: RMS eps and the QDQ amax floor
    if tokens > 2:
        x[2, ::3] *= 3000.0  # near the FP16 top
    return x.clamp(-60000, 60000).half()


def _positions(tokens, length, g):
    # Inside the table, the last row, beyond it (cos/sin on the fly), and 0.
    picks = [0, length - 1, length, length + 977, 5, 123456]
    pos = torch.tensor([picks[i % len(picks)] for i in range(tokens)])
    return (pos + torch.randint(0, 3, (tokens,), generator=g)).to(torch.int64)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 6])
def test_attn_prep_bitwise(tokens, compressed):
    g = torch.Generator().manual_seed(500 + tokens + 50 * compressed)
    length = 64
    table = ROPE.rope_table(length, compressed=compressed, device="cpu")
    q = _rows(tokens, 16 * 512, g).view(tokens, 16, 512)
    kv = _rows(tokens, 512, g, scale=0.3)
    weight = torch.rand(512, generator=g) + 0.5
    positions = _positions(tokens, length, g)
    query_old = _old_rope(q, positions, table)
    window_old = _old_qdq(
        _old_rope(_old_rms(kv, weight, 1e-20).unsqueeze(1), positions, table).view(
            tokens, 512
        )
    )
    query_new, window_new = ATTN._prep_launch(q, kv, weight, 1e-20, positions, table)
    assert _same(query_old, query_new), "query"
    assert _same(window_old, window_new), "window"
    assert not torch.isnan(window_new.float()).any()


@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize("tokens", [1, 2, 5, 6, 7, 8])
def test_attn_post_bitwise(tokens, ring):
    g = torch.Generator().manual_seed(900 + tokens + 50 * ring)
    length = 64
    table = ROPE.rope_table(length, device="cpu")
    output = _rows(tokens, 16 * 512, g).view(tokens, 16, 512)
    positions = _positions(tokens, length, g)
    if ring:
        # Consecutive decode positions (no ring slot is written twice).
        positions = torch.arange(tokens, dtype=torch.int64) + 300 + tokens
        valid = torch.ones(tokens, dtype=torch.bool)
        valid[-1] = tokens < 3  # an invalid token: not written
        values = _rows(tokens, 512, g)
        ring_old = torch.randn(OPS.WINDOW_RING, 512, generator=g).half()
        ring_new = ring_old.clone()
        _old_ring(ring_old, values, positions, valid)
    out_old = _old_qdq(
        _old_rope(output, positions, table, inverse=True).reshape(tokens, 2, 4096)
    )
    out_new = ATTN._post_launch(
        output,
        positions,
        table,
        2,
        ring_new if ring else None,
        values if ring else None,
        valid if ring else None,
    )
    assert _same(out_old, out_new), "output"
    if ring:
        assert _same(ring_old, ring_new), "ring"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
