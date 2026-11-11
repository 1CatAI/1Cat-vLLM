# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Knob-off proof for the V4.1 attention PREP / POST fusions (source level, CPU).

With VLLM_V41_FUSE_ATTN_PREP / _ATTN_POST off, every fusion
branch is an ``if <knob>[ and ...]:`` whose test is False, so the code that
runs is what remains after replacing each such ``if`` by its ``else`` body.
This test does that replacement on the AST and requires the result to equal
the base commit's code (the parent of the commit that added attn_fuse.py,
or V41_FUSE_BASE):

* model.py: V41Attention.forward equal after pruning; every other method of
  the base V41Attention (project_output included) and every other top-level
  function/class unchanged;
* quant_rope.py, norm.py, attention_ops.py: unchanged;
* the two knobs default to "0" in vllm/envs.py.

Needs git and the base commit (skips otherwise).

    python tests/models/test_deepseek_v41_small_fuse_knob_off.py   # or pytest
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KNOBS = {"FUSE_ATTN_PREP", "FUSE_ATTN_POST"}


def _base():
    if os.environ.get("V41_FUSE_BASE"):
        return os.environ["V41_FUSE_BASE"]
    try:
        added = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "log",
                "--diff-filter=A",
                "--format=%H",
                "--",
                "vllm/models/deepseek_v41/attn_fuse.py",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        added = []
    if not added:
        pytest.skip("git history with the commit that added attn_fuse.py needed")
    return added[-1] + "^"


def _source(rel, rev=None):
    if rev is None:
        return (ROOT / rel).read_text()
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{rev}:{rel}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or base commit {rev} not available")


def _is_knob(test):
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        test = test.values[0]
    return isinstance(test, ast.Name) and test.id in KNOBS


class _KnobsOff(ast.NodeTransformer):
    pruned = 0

    def visit_If(self, node):
        self.generic_visit(node)
        if _is_knob(node.test):
            self.pruned += 1
            return node.orelse  # [] removes the statement
        return node


def _defs(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            out[node.name] = node
    return out


def _methods(cls):
    return {
        n.name: n
        for n in cls.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _pruned(node):
    pruner = _KnobsOff()
    node = pruner.visit(ast.parse(ast.unparse(node)).body[0])
    return node, pruner.pruned


def _module_statements(tree):
    return [
        _dump(n) for n in tree.body if not isinstance(n, ast.FunctionDef | ast.ClassDef)
    ]


def _is_subsequence(small, big):
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


def test_attention_knob_off():
    rel = "vllm/models/deepseek_v41/model.py"
    base, cur = ast.parse(_source(rel, _base())), ast.parse(_source(rel))
    base_defs, cur_defs = _defs(base), _defs(cur)
    for name, node in base_defs.items():
        if name != "V41Attention":
            assert _dump(cur_defs[name]) == _dump(node), name
    base_m = _methods(base_defs["V41Attention"])
    cur_m = _methods(cur_defs["V41Attention"])
    for name, node in base_m.items():
        if name == "forward":
            pruned, n = _pruned(cur_m[name])
            assert n == 4  # PREP x1; POST x3 (seed, deferred ring, return)
            assert _dump(pruned) == _dump(node), name
        else:
            assert _dump(cur_m[name]) == _dump(node), name
    assert set(cur_m) - set(base_m) == {
        "_prep_fused",
        "_post_fused",
        "_project_quantized",
    }
    assert _is_subsequence(_module_statements(base), _module_statements(cur))


@pytest.mark.parametrize(
    "rel",
    [
        "vllm/models/deepseek_v41/quant_rope.py",
        "vllm/models/deepseek_v41/norm.py",
        "vllm/models/deepseek_v41/attention_ops.py",
    ],
)
def test_old_kernels_untouched(rel):
    assert _source(rel) == _source(rel, _base())


def test_knobs_default_off():
    src = _source("vllm/envs.py")
    for knob in ("FUSE_ATTN_PREP", "FUSE_ATTN_POST"):
        name = "VLLM_V41_" + knob
        assert f'    {name}: str = "0"\n' in src
        assert f'os.getenv("{name}", "0")' in src
        assert f'        "{name}",\n' in src  # compile_factors ignore list


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
