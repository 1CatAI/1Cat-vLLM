# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real provider/caller branches without importing a CUDA extension."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_functions(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    if class_name is not None:
        tree = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == set(names)
    body: list[ast.stmt] = list(functions)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), namespace)


@pytest.mark.parametrize("method", ["forward_native", "forward_cuda"])
@pytest.mark.parametrize(
    "case,admitted",
    [
        ("admitted", True),
        ("old_bound_missing_library", True),
        ("disabled", False),
        ("batch_invariant", False),
        ("cpu", False),
        ("other_arch", False),
        ("grouped", False),
        ("bias", False),
        ("gate_before_norm", False),
        ("activation", False),
        ("unsupported_shape", False),
        ("missing_library", False),
    ],
)
def test_provider_preserves_caller_fallback(method, case, admitted, monkeypatch):
    calls = []

    def implementation(name):
        def call(*args, **kwargs):
            calls.append(name)
            return name

        return call

    torch = SimpleNamespace(
        Tensor=object,
        ops=SimpleNamespace(
            vllm=SimpleNamespace(sm70_rmsnorm_gated_exact=implementation("exact"))
        ),
    )
    provider = {
        "torch": torch,
        "_sm70_gated_norm_device_supported": lambda _: case != "other_arch",
        "_sm70_gated_norm_shape_supported": lambda *args: case != "unsupported_shape",
        "_PP7_EXACT_NORM_AVAILABLE": case
        not in ("missing_library", "old_bound_missing_library"),
    }
    load_functions(
        "vllm/model_executor/kernels/norm/sm70.py", ["maybe_gated_norm"], provider
    )
    caller: dict[str, Any] = {
        "torch": torch,
        "envs": SimpleNamespace(VLLM_BATCH_INVARIANT=case == "batch_invariant"),
        "maybe_gated_norm": provider["maybe_gated_norm"],
    }
    load_functions(
        "vllm/model_executor/layers/layernorm.py",
        ["forward_native", "forward_cuda"],
        caller,
        class_name="RMSNormGated",
    )
    guard = ModuleType("vllm.model_executor.layers.fla.ops.layernorm_guard")
    guard.__dict__["rmsnorm_fn"] = implementation("triton")
    monkeypatch.setitem(sys.modules, guard.__name__, guard)
    norm = SimpleNamespace(
        weight=object(),
        eps=1e-6,
        _sm70_rmsnorm_gated_exact=case != "disabled",
        group_size=128 if case == "grouped" else None,
        bias=object() if case == "bias" else None,
        norm_before_gate=case != "gate_before_norm",
        activation="unsupported" if case == "activation" else "silu",
        forward_static=implementation("reference"),
    )
    x = SimpleNamespace(
        is_cuda=case != "cpu",
        device=SimpleNamespace(index=0),
        shape=(192 if case == "old_bound_missing_library" else 240, 128),
        dtype=object(),
    )
    if method == "forward_native" and case in ("bias", "missing_library"):
        admitted = True
    fallback = "reference" if method == "forward_native" else "triton"
    expected = "exact" if admitted else fallback
    assert caller[method](norm, x, object()) == expected
    assert calls == [expected]
