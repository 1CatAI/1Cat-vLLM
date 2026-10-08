# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture mask/slot provenance checks; synthetic inputs do not select a format."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_tool(name):
    path = ROOT / "tools/kv_codec" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"kv_capture_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture = load_tool("capture")


def test_actual_qsa_selection_matches_sparse_attention_oracle():
    # Excluding an allowed earlier key must affect the oracle, even on first prefill.
    indices = torch.tensor([[0, 2, -1], [1, 4, 5]], dtype=torch.int32)
    positions = torch.tensor([2, 5], dtype=torch.int64)
    allowed = capture.qsa_allowed_from_indices(indices, positions, 6)
    generator = torch.Generator().manual_seed(20261008)
    q = torch.randn(2, 2, 4, generator=generator)
    k = torch.randn(6, 1, 4, generator=generator)
    v = torch.randn(6, 1, 4, generator=generator)
    # Load the evaluator without importing vLLM or initializing a CUDA driver.
    evaluator = load_tool("evaluate")
    actual = evaluator.attention(q, k, v, allowed, 0.5)
    expected = torch.stack(
        [
            (q[row] @ k[live, 0].T * 0.5).softmax(-1) @ v[live, 0]
            for row, selected in enumerate(indices)
            for live in [selected[selected >= 0].long()]
        ]
    )
    torch.testing.assert_close(actual, expected)
    assert not allowed[0, 1]  # A dense causal mask would incorrectly include it.


@pytest.mark.parametrize(
    "selected",
    [[-1, -1], [0, 0], [0, 3], [0, 6], [0, -2]],
)
def test_qsa_capture_rejects_unrepresentable_or_invalid_selection(selected):
    with pytest.raises(ValueError):
        capture.qsa_allowed_from_indices(
            torch.tensor([selected], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int64),
            6,
        )


def test_qsa_capture_copies_only_addressed_state_rows():
    cache = torch.arange(3 * 4 * 6).reshape(3, 4, 1, 6).to(torch.float16)
    cache = cache[:, :, :, ::2]  # Retain the actual strided state row.
    result = capture.capture_qsa_state(cache, torch.tensor([7, -1, 0, 7, 5]))
    assert result["slots"].tolist() == [0, 5, 7]
    assert torch.equal(
        result["rows"], torch.stack([cache[0, 0], cache[1, 1], cache[1, 3]])
    )
    with pytest.raises(ValueError):
        capture.capture_qsa_state(cache, torch.tensor([12]))


def test_qsa_launch_observer_preserves_arguments_and_counts_success_only():
    calls: list[tuple] = []
    routes: dict[str, int] = {}

    class Kernel:
        name = "original"

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                if kwargs.get("fail"):
                    raise RuntimeError("original launch failed")
                calls.append((grid, args, kwargs))
                return args[0]

            return launch

    observer = capture.ObservedQSAKernel(Kernel(), routes)
    assert observer.name == "original"
    value = torch.tensor([1])
    assert observer[(2, 3)](value, num_warps=4) is value
    assert calls == [((2, 3), (value,), {"num_warps": 4})]
    assert routes == {"qsa_sparse_triton_splitk": 1}
    with pytest.raises(RuntimeError):
        observer[(1,)](value, fail=True)
    assert routes == {"qsa_sparse_triton_splitk": 1}
