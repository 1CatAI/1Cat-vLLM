# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_70 compile check for the V4.1 attention PREP / POST fusions (CPU only).

Compiles the old kernels and their fused replacements with
``triton.compile(..., target=GPUTarget("cuda", 70, 32))`` (no GPU, no driver)
and checks:

* every FP32 add-reduction (``tt.reduce`` + ``arith.addf``) in a fused kernel
  has an operand layout and axis that the replaced kernels use, so the
  in-thread / warp-shuffle / cross-warp summation tree is the same; max
  reductions are exact in any order and are only listed;
* ptxas (Triton's bundled CUDA 12.x one) reports no spills.

Prints one JSON report; exit 1 on any failure, 2 if this Triton cannot
target sm_70.

    CUDA_VISIBLE_DEVICES= python benchmarks/deepseek_v41/small_fuse_sm70_check.py
"""

import importlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path

os.environ.pop("TRITON_INTERPRET", None)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TARGET = GPUTarget("cuda", 70, 32)
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
        if name.startswith("VLLM_"):  # the process env first, as vllm.envs
            return os.environ.get(name, _ENV.get(name, "0"))
        raise AttributeError(name)


def _package(name, path=None):
    mod = types.ModuleType(name)
    mod.__path__ = [str(ROOT / path)] if path else []
    sys.modules[name] = mod
    return mod


def load_modules():
    # Load only the kernel modules (not vllm's or the model package's
    # __init__), with vllm.envs / logger / triton_utils stubbed.
    _package("vllm")
    tu = types.ModuleType("vllm.triton_utils")
    tu.tl, tu.triton = tl, triton
    sys.modules[tu.__name__] = tu
    lg = types.ModuleType("vllm.logger")
    lg.init_logger = logging.getLogger
    sys.modules[lg.__name__] = lg
    envs = _Envs("vllm.envs")
    sys.modules[envs.__name__] = envs
    sys.modules["vllm"].envs = envs
    _package("vllm.models", "vllm/models")
    _package("vllm.models.deepseek_v4", "vllm/models/deepseek_v4")
    _package("vllm.models.deepseek_v4.common", "vllm/models/deepseek_v4/common")
    _package("vllm.models.deepseek_v4.common.ops", "vllm/models/deepseek_v4/common/ops")
    bmm = types.ModuleType("vllm.models.deepseek_v4.common.ops.sparse_prefill_bmm")
    bmm.attend_gathered_keys = None
    sys.modules[bmm.__name__] = bmm
    _package("vllm.models.deepseek_v41", "vllm/models/deepseek_v41")
    attn, rope, norm, ops = (
        importlib.import_module("vllm.models.deepseek_v41." + name)
        for name in ("attn_fuse", "quant_rope", "norm", "attention_ops")
    )
    return attn, rope, norm, ops


def compile_sm70(fn, types_, constexprs, num_warps=4, maxnreg=None):
    sig, attrs = {}, {}
    for i, name in enumerate(fn.arg_names):
        if name in constexprs:
            sig[name] = "constexpr"
        else:
            sig[name] = types_[name]
            if types_[name].startswith("*"):
                attrs[(i,)] = [["tt.divisibility", 16]]
    options = {"num_warps": num_warps, "enable_fp_fusion": False}
    if maxnreg is not None:
        options["maxnreg"] = maxnreg
    kernel = triton.compile(
        ASTSource(fn=fn, signature=sig, constexprs=constexprs, attrs=attrs),
        target=TARGET,
        options=options,
    )
    ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
    with tempfile.TemporaryDirectory() as d:
        ptx = Path(d) / "k.ptx"
        ptx.write_text(kernel.asm["ptx"])
        run = subprocess.run(
            [str(ptxas), "-arch=sm_70", "-v", str(ptx), "-o", str(ptx) + ".o"],
            capture_output=True,
            text=True,
        )
    info = run.stderr + run.stdout
    regs = re.search(r"Used (\d+) registers", info)
    spills = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", info)
    if run.returncode or regs is None or spills is None:
        raise RuntimeError("ptxas sm_70 failed:\n" + info)
    return {
        "ttgir": kernel.asm["ttgir"],
        "ptx": kernel.asm["ptx"],
        "regs": int(regs.group(1)),
        "spill_bytes": int(spills.group(1)) + int(spills.group(2)),
    }


def reductions(ttgir):
    """Counter of (combine op, axis, operand type with layouts spelled out)."""
    aliases = {}
    for line in ttgir.splitlines():
        m = re.match(r"(#\w+) = (#ttg\.\w+<.*>)$", line.strip())
        if m:
            aliases[m.group(1)] = m.group(2)
    lines = ttgir.splitlines()
    found = Counter()
    for idx, line in enumerate(lines):
        m = re.search(r'"tt\.reduce"\(([^)]*)\) <\{axis = (\d+) : i32\}>', line)
        if not m:
            continue
        combine = typ = None
        for nxt in lines[idx + 1 : idx + 16]:
            if combine is None and (c := re.search(r"arith\.(\w+)", nxt)):
                combine = c.group(1)
            if t := re.search(r"\}\) : \((tensor<[^)]*>)\)", nxt):
                typ = t.group(1)
                break
        if typ is None:
            raise RuntimeError("cannot parse tt.reduce at TTGIR line %d" % idx)
        for alias, layout in aliases.items():
            typ = re.sub(re.escape(alias) + r"\b", layout, typ)
        found[(combine, int(m.group(2)), typ)] += 1
    return found


def _check_adds(name, old_kernels, new, report):
    old = Counter()
    for kernel in old_kernels:
        old += reductions(kernel["ttgir"])
    adds = Counter(
        {k: v for k, v in reductions(new["ttgir"]).items() if k[0] == "addf"}
    )
    known = {k for k in old if k[0] == "addf"}
    unmatched = [k for k in adds if k not in known]
    report[name] = {
        "fused_regs": new["regs"],
        "old_regs": [k["regs"] for k in old_kernels],
        "spill_bytes": new["spill_bytes"],
        "add_reductions": [
            f"{v} x axis {k[1]} {k[2]}" for k, v in sorted(adds.items())
        ],
        "max_reductions": sum(
            v for k, v in reductions(new["ttgir"]).items() if k[0] != "addf"
        ),
        "unmatched_add_layouts": [str(k) for k in unmatched],
    }
    return not unmatched and new["spill_bytes"] == 0


def main():
    attn, rope, norm, ops = load_modules()
    p32, p16, pi64, pb = "*fp32", "*fp16", "*i64", "*i1"
    report = {"triton": triton.__version__}
    ok = True

    # VLLM_V41_FUSE_ATTN_PREP: _rotate(q), _short_rms(kv), _rotate(kv), _qdq.
    rot_t = dict(x=p16, positions=pi64, table=p32, out=p16, LENGTH="i32")
    rms_t = dict(x=p16, weight=p32, out=p16)
    qdq_t = dict(x=p16, out=p16)
    old_rot_q = compile_sm70(
        rope._rotate, rot_t, dict(HEADS=16, D=512, INVERSE=False, BLOCK=512)
    )
    old_rms = compile_sm70(norm._short_rms, rms_t, dict(H=512, EPS=1e-20, BLOCK=512))
    old_rot_kv = compile_sm70(
        rope._rotate, rot_t, dict(HEADS=1, D=512, INVERSE=False, BLOCK=512)
    )
    old_qdq = compile_sm70(
        rope._qdq, qdq_t, dict(GROUP=128, FP4=False, E4M3_SCALE=False)
    )
    prep_t = dict(
        q=p16, kv=p16, kv_weight=p32, positions=pi64, table=p32, q_out=p16,
        normed=p16, roped=p16, window=p16, LENGTH="i32",
    )  # fmt: skip
    prep = compile_sm70(
        attn._attn_prep,
        prep_t,
        dict(HEADS=16, H=512, EPS=1e-20, BLOCK=512, GROUP=128),
    )
    ok &= _check_adds(
        "attn_prep", [old_rot_q, old_rms, old_rot_kv, old_qdq], prep, report
    )

    # VLLM_V41_FUSE_ATTN_POST: _ring_insert, _rotate(inverse), _qdq.
    old_rot_inv = compile_sm70(
        rope._rotate, rot_t, dict(HEADS=16, D=512, INVERSE=True, BLOCK=512)
    )
    old_ring = compile_sm70(
        ops._ring_insert,
        dict(values=p16, positions=pi64, valid_tokens=pb, ring=p16, FIRST="i32"),
        dict(RING=ops.WINDOW_RING),
    )
    post_t = dict(
        x=p16, positions=pi64, table=p32, roped=p16, out=p16, values=p16,
        valid_tokens=pb, ring=p16, LENGTH="i32",
    )  # fmt: skip
    for do_ring in (True, False):
        post = compile_sm70(
            attn._attn_post,
            post_t,
            dict(
                HEADS=16,
                H=512,
                BLOCK=512,
                GROUP=128,
                RING=ops.WINDOW_RING,
                DO_RING=do_ring,
            ),
        )
        ok &= _check_adds(
            f"attn_post ring={do_ring}", [old_rot_inv, old_qdq, old_ring], post, report
        )
    report["ok"] = bool(ok)
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


def _ptxas_targets_sm70():
    ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
    version = subprocess.run([str(ptxas), "--version"], capture_output=True, text=True)
    release = re.search(r"release (\d+)\.", version.stdout)
    return release is not None and int(release.group(1)) <= 12  # 13 dropped Volta


if __name__ == "__main__":
    if not _ptxas_targets_sm70():
        print(json.dumps({"ok": None, "skip": "Triton's ptxas cannot target sm_70"}))
        sys.exit(2)
    sys.exit(main())
