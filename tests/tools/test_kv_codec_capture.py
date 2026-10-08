# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture mask/slot provenance checks; synthetic inputs do not select a format."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    "packed,fail_second", [(False, False), (True, False), (True, True)]
)
def test_capture_loop_isolates_requests_and_retains_incomplete_marker(
    monkeypatch, tmp_path, packed, fail_second
):
    instances = []

    class Engine:
        def __init__(self, **config):
            instances.append(self)
            self.pending = None
            self.generated = 0
            assert config["max_num_batched_tokens"] == (3 if packed else 2)

        def collective_rpc(self, method, args=()):
            if method is capture.start_capture_on_worker:
                assert self.pending is None
                directory, ids, provenance, adapter = args
                self.pending = (ids, provenance)
                return []
            assert method is capture.finish_capture_on_worker
            assert self.pending is not None
            ids, provenance = self.pending
            self.pending = None
            return [
                {
                    "rank": 0,
                    "route_delta": {"test_only": 1},
                    "samples": [{**provenance, "tensor_path": "rank0-layer0.pt"}],
                }
            ]

        def generate(self, prompts, params, use_tqdm):
            assert self.pending is not None
            assert prompts == [{"prompt_token_ids": self.pending[0]}]
            self.generated += 1
            if fail_second and self.generated == 2:
                raise RuntimeError("synthetic request failure")
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7])])]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            __file__="/test/site-packages/vllm/__init__.py",
            LLM=Engine,
            SamplingParams=lambda **kwargs: kwargs,
            envs=SimpleNamespace(VLLM_ALLOW_INSECURE_SERIALIZATION=True),
        ),
    )
    monkeypatch.setattr(capture, "verify_runtime_wheel", lambda *args: {})
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "synthetic_control_flow_only"}))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            [
                {"id": "first", "prompt_token_ids": [1, 2]},
                {"id": "second", "prompt_token_ids": [3, 4, 5]},
            ]
            if packed
            else {"prompt_token_ids": [1, 2]}
        )
    )
    wheel = tmp_path / "test.whl"
    wheel.write_bytes(b"synthetic control flow; no installed/GPU evidence")
    out = tmp_path / "samples"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture",
            "--engine-config",
            str(config),
            "--request",
            str(request),
            "--wheel",
            str(wheel),
            "--source-sha",
            "test",
            "--out",
            str(out),
        ],
    )
    if fail_second:
        with pytest.raises(RuntimeError, match="synthetic request failure"):
            capture.main()
    else:
        capture.main()
        assert instances[0].pending is None
    assert len(instances) == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["performance_evidence"] is False
    if packed:
        assert manifest["complete"] is (not fail_second)
        assert [s["tensor_path"] for s in manifest["samples"]] == (
            ["first/rank0-layer0.pt"]
            if fail_second
            else ["first/rank0-layer0.pt", "second/rank0-layer0.pt"]
        )
        assert [s["request_id"] for s in manifest["samples"]] == (
            ["first"] if fail_second else ["first", "second"]
        )
    else:
        assert "complete" not in manifest and "request_id" not in manifest["samples"][0]
        assert manifest["output_token_ids"] == [7]


def test_request_corpus_preserves_order_and_single_request_compatibility():
    packed, requests = capture.capture_requests({"prompt_token_ids": [1, 2, 3]})
    assert not packed and requests == [("request", [1, 2, 3])]
    packed, requests = capture.capture_requests(
        [
            {"id": "zh_story", "prompt_token_ids": [1, 3]},
            {"id": "python_cache", "prompt_token_ids": [2]},
        ]
    )
    assert packed and requests == [("zh_story", [1, 3]), ("python_cache", [2])]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        {"prompt_token_ids": []},
        {"prompt_token_ids": [True]},
        {"prompt_token_ids": [-1]},
        {"prompt_token_ids": "123"},
        [{"id": "../escape", "prompt_token_ids": [1]}],
        [
            {"id": "same", "prompt_token_ids": [1]},
            {"id": "same", "prompt_token_ids": [2]},
        ],
    ],
)
def test_invalid_corpus_cannot_overwrite_or_misrepresent_samples(payload):
    with pytest.raises(ValueError):
        capture.capture_requests(payload)


def test_qsa_layers_cover_rare_compression_ratios_without_losing_depths():
    layers = [
        SimpleNamespace(indexer=SimpleNamespace(compress_ratio=ratio))
        for ratio in (4, 4, 128, 4, 4, 4, 4, 4, 128, 4, 4, 4)
    ]
    selected = capture.select_qsa_capture_layers(layers)
    assert selected == [layers[i] for i in (0, 2, 6, 11)]
    assert {layer.indexer.compress_ratio for layer in selected} == {4, 128}
    assert capture.select_qsa_capture_layers(layers[:1]) == layers[:1]


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
