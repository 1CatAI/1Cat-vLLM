# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU check for the V4.1 attention PREP / POST fusions (one GPU, synthetic inputs).

    VLLM_V41_FUSE_ATTN_PREP   _rotate(q), _short_rms(kv), _rotate(kv), _qdq
                              -> _attn_prep
    VLLM_V41_FUSE_ATTN_POST   _ring_insert, _rotate(inverse), _qdq
                              -> _attn_post

1. Bit equality, old vs new, every output into fresh NaN-poisoned buffers:
   PREP for M = 1..6 on the window and compressed RoPE tables; POST for
   M = 1..8 with and without a window ring. Positions inside the 16K table,
   at its end, beyond it (cos/sin on the fly) and 0; zero rows, sub-floor QDQ
   groups and near-FP16-max values; invalid ring tokens.
2. CUDA graphs: each fused op captured at several M and replayed with new
   inputs, against graphs of the old kernels.

Loads the kernel modules from this script's tree (no vllm install or build
needed). No timing. Exit status 1 on any difference.

    CUDA_VISIBLE_DEVICES=0 python benchmarks/deepseek_v41/small_fuse_check.py \\
        --json small_fuse.json
"""

import argparse
import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "small_fuse_sm70_check", _HERE / "small_fuse_sm70_check.py"
)
_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader)

ATTN, ROPE, NORM, OPS = _loader.load_modules()
HEADS, GROUPS = 16, 2  # per TP rank
TABLE_LENGTH = 16384
DEV = "cuda"


class _PoisonTorch(types.ModuleType):
    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def _poison(t):
        return t.fill_(float("nan") if t.is_floating_point() else -1)

    def empty(self, *args, **kwargs):
        return self._poison(torch.empty(*args, **kwargs))

    def empty_like(self, x, **kwargs):
        return self._poison(torch.empty_like(x, **kwargs))


_MODULES = (ATTN, ROPE, NORM, OPS)


@contextlib.contextmanager
def poisoned():
    saved = [m.torch for m in _MODULES]
    for m in _MODULES:
        m.torch = _PoisonTorch("torch")
    try:
        yield
    finally:
        for m, t in zip(_MODULES, saved):
            m.torch = t


def _bits(x):
    x = x.contiguous()
    return x.view({4: torch.int32, 2: torch.int16, 1: torch.int8}[x.element_size()])


def compare(failures, name, old, new):
    """Bitwise; records a failure (with a short diagnosis) on any difference."""
    ok = (
        old.shape == new.shape
        and old.dtype == new.dtype
        and torch.equal(_bits(old), _bits(new))
    )
    if ok and old.is_floating_point() and torch.isnan(new).any():
        ok = False  # finite inputs: a NaN is a buffer the new kernel never wrote
    if not ok:
        info = {"check": name, "shape": list(new.shape)}
        if old.shape == new.shape and old.is_floating_point():
            differ = _bits(old) != _bits(new)
            info["differ"] = int(differ.sum())
            info["nan_new"] = int(torch.isnan(new).sum())
            d = (old.float() - new.float()).abs()
            info["max_abs"] = float(d[torch.isfinite(d)].max()) if differ.any() else 0
        failures.append(info)
    return ok


# ---------------------------------------------------------------- inputs --


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def rows(tokens, width, seed, scale=1.0):
    g = _gen(seed)
    x = torch.randn(tokens, width, device=DEV, generator=g) * scale
    x[0, 128:256] *= 1e-5  # one QDQ group under the amax floor
    if tokens > 1:
        x[1] = 0  # all-zero row: RMS eps and the QDQ floor
    if tokens > 2:
        x[2, ::5] *= 2000.0  # near the FP16 top
    return x.clamp(-65000, 65000).half()


def position_sets(tokens):
    # Consecutive decode positions; table end and beyond (on-the-fly
    # cos/sin); zero and mixed.
    picks = [0, TABLE_LENGTH - 1, TABLE_LENGTH, TABLE_LENGTH + 4093, 7, 5000]
    yield "decode", torch.arange(tokens, device=DEV) + 4000
    yield (
        "edges",
        torch.tensor([picks[i % len(picks)] for i in range(tokens)], device=DEV),
    )
    yield "beyond", torch.arange(tokens, device=DEV) + TABLE_LENGTH - 2


TABLES = {}


def table(compressed):
    if compressed not in TABLES:
        TABLES[compressed] = ROPE.rope_table(
            TABLE_LENGTH, compressed=compressed, device=DEV
        )
    return TABLES[compressed]


def prep_old(q, kv, w, positions, tab):
    query = ROPE.apply_rope(q, positions, tab)
    window = ROPE.attention_qdq(
        ROPE.apply_rope(NORM.short_rms_norm(kv, w, 1e-20), positions, tab)
    )
    return query, window


def prep_new(q, kv, w, positions, tab):
    result = ATTN.attn_prep(q, kv, w, 1e-20, positions, tab)
    if result is None:
        raise RuntimeError("attn_prep declined a decode-shaped call")
    return result


def post_old(output, positions, tab, ring, values, valid):
    if ring is not None:
        OPS.insert_window_ring(ring, values, positions, valid)
    output = ROPE.apply_rope(output, positions, tab, inverse=True)
    return ROPE.attention_qdq(output.reshape(output.shape[0], GROUPS, 4096))


def post_new(output, positions, tab, ring, values, valid):
    result = ATTN.attn_post(output, positions, tab, GROUPS, ring, values, valid)
    if result is None:
        raise RuntimeError("attn_post declined a decode-shaped call")
    return result


def post_inputs(tokens, ring, seed):
    output = rows(tokens, HEADS * 512, seed).view(tokens, HEADS, 512)
    if not ring:
        return output, None, None, None
    g = _gen(seed + 1)
    values = rows(tokens, 512, seed + 2)
    valid = torch.ones(tokens, dtype=torch.bool, device=DEV)
    if tokens >= 3:
        valid[tokens // 2] = False  # a padded token: not written
    ring_buf = torch.randn(OPS.WINDOW_RING, 512, device=DEV, generator=g).half()
    return output, ring_buf, values, valid


# ------------------------------------------------------------ bit checks --


def check_prep(failures, report):
    n = 0
    for tokens in range(1, 7):
        for compressed in (False, True):
            tab = table(compressed)
            for pname, positions in position_sets(tokens):
                seed = 100 * tokens + 10 * compressed + len(pname)
                q = rows(tokens, HEADS * 512, seed).view(tokens, HEADS, 512)
                kv = rows(tokens, 512, seed + 1, scale=0.4)
                w = 1.0 + 0.2 * torch.randn(512, device=DEV, generator=_gen(seed))
                with poisoned():
                    old = prep_old(q, kv, w, positions, tab)
                    new = prep_new(q, kv, w, positions, tab)
                tag = f"prep M={tokens} compressed={compressed} {pname}"
                compare(failures, tag + " query", old[0], new[0])
                compare(failures, tag + " window", old[1], new[1])
                n += 1
    report["prep_cases"] = n


def check_post(failures, report):
    n = 0
    tab = table(False)
    for tokens in range(1, 9):
        for ring in (False, True):
            for pname, positions in position_sets(tokens):
                if ring and pname == "edges":
                    continue  # repeated ring slots race in the old kernel too
                seed = 300 * tokens + 10 * ring + len(pname)
                output, ring_buf, values, valid = post_inputs(tokens, ring, seed)
                ring_old = ring_buf.clone() if ring else None
                with poisoned():
                    old = post_old(output, positions, tab, ring_old, values, valid)
                    new = post_new(output, positions, tab, ring_buf, values, valid)
                tag = f"post M={tokens} ring={ring} {pname}"
                compare(failures, tag + " output", old, new)
                if ring:
                    compare(failures, tag + " ring", ring_old, ring_buf)
                n += 1
    report["post_cases"] = n


# ----------------------------------------------------------- CUDA graphs --


def _capture(fn):
    # Poisoned inside the graph too: every replay refills fresh buffers with
    # NaN before the kernels run, so an unwritten element shows.
    with poisoned():
        fn()  # eager warm-up (compiles)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn()
    return graph, result


def check_graphs(failures, report):
    n = 0
    # PREP / POST: replays with new inputs.
    tab = table(True)
    for tokens in (6, 2):
        positions = torch.arange(tokens, device=DEV) + 9000
        q = rows(tokens, HEADS * 512, 1).view(tokens, HEADS, 512)
        kv = rows(tokens, 512, 2, scale=0.4)
        w = 1.0 + 0.2 * torch.randn(512, device=DEV, generator=_gen(3))
        g_old, o_old = _capture(lambda: prep_old(q, kv, w, positions, tab))
        g_new, o_new = _capture(lambda: prep_new(q, kv, w, positions, tab))
        for step in range(3):
            q.copy_(rows(tokens, HEADS * 512, 40 + step).view(tokens, HEADS, 512))
            kv.copy_(rows(tokens, 512, 50 + step, scale=0.4))
            positions.add_(1)
            g_old.replay()
            g_new.replay()
            torch.accelerator.synchronize()
            tag = f"graph prep step={step} M={tokens}"
            compare(failures, tag + " query", o_old[0], o_new[0])
            compare(failures, tag + " window", o_old[1], o_new[1])
            n += 1
    for tokens in (8, 6):
        positions = torch.arange(tokens, device=DEV) + 9000
        output, ring_new, values, valid = post_inputs(tokens, True, 7)
        ring_old = ring_new.clone()
        g_old, o_old = _capture(
            lambda: post_old(output, positions, table(False), ring_old, values, valid)
        )
        g_new, o_new = _capture(
            lambda: post_new(output, positions, table(False), ring_new, values, valid)
        )
        for step in range(3):
            fresh = post_inputs(tokens, True, 60 + step)
            output.copy_(fresh[0])
            values.copy_(fresh[2])
            ring_old.copy_(fresh[1])
            ring_new.copy_(fresh[1])
            positions.add_(tokens)
            g_old.replay()
            g_new.replay()
            torch.accelerator.synchronize()
            tag = f"graph post step={step} M={tokens}"
            compare(failures, tag + " output", o_old, o_new)
            compare(failures, tag + " ring", ring_old, ring_new)
            n += 1
    report["graph_cases"] = n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    torch.accelerator.set_device_index(args.device)
    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "triton": _loader.triton.__version__,
    }
    failures = []
    check_prep(failures, report)
    check_post(failures, report)
    check_graphs(failures, report)
    report["mismatches"] = len(failures)
    report["failures"] = failures[:40]
    report["ok"] = not failures
    text = json.dumps(report, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
