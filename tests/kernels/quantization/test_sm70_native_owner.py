# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native policy/owner isolation with changed-input graph replay, no models."""

from types import SimpleNamespace

import pytest
import torch

from vllm._sm70.policy import NativeBindings
from vllm.config import KernelConfig, set_current_vllm_config
from vllm.runtime_resources import release_runtime_resources


@pytest.fixture(autouse=True)
def sm70():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401
    import vllm._moe_C  # noqa: F401


def engine(family, **overrides):
    cfg = SimpleNamespace(kernel_config=KernelConfig())
    policy = (
        cfg.kernel_config.layer_execution.native
        if family == "f16"
        else cfg.kernel_config.sm70_fp8.native
    )
    for field, value in overrides.items():
        setattr(policy, field, value)
    if family == "f16":
        # Compare the same deterministic selector, not independent autotuning
        # winners with different rounding (observed in the legacy baseline).
        policy.f16_dense_tune_max_m = 0
    policy.resolve(family)
    with set_current_vllm_config(cfg):
        binding = NativeBindings(policy.values)
    reference = NativeBindings(policy.values)
    return cfg, binding, reference


@pytest.mark.parametrize("rows", [8, 9, 16, 17, 31, 32])
def test_qpn8_policy_is_frozen_across_two_engines_and_replay(monkeypatch, rows):
    torch.manual_seed(951)
    k, n = 1536, 5120
    weight = (torch.randn(n, k, device="cuda") * 16).to(torch.float8_e4m3fn)
    scales = torch.rand(n, 1, device="cuda") * 0.005 + 0.002
    codes, packed_scales = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scales)
    records = []
    for enabled in (False, True):
        cfg, bound, reference = engine(
            "fp8",
            fp8_qpn8_m16=enabled,
            fp8_qpn8_m32_chunked=enabled,
            fp8_qpn8_m32_native=enabled,
        )
        x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.1
        out, expected = (
            torch.empty(rows, n, device="cuda", dtype=x.dtype),
            torch.empty(rows, n, device="cuda", dtype=x.dtype),
        )
        args = (x, codes, packed_scales, 12, 2, True, False)
        for _ in range(3):
            bound.fp8_qpn8_gemm_sm70_out(out, *args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bound.fp8_qpn8_gemm_sm70_out(out, *args)
        records.append((cfg, bound, reference, x, out, expected, args, graph))
    for alias in (
        "VLLM_SM70_FP8_QPN8_M16",
        "VLLM_SM70_FP8_QPN8_M32_CHUNKED",
        "VLLM_SM70_FP8_QPN8_M32_NATIVE",
    ):
        monkeypatch.setenv(alias, "invalid-after-init")
    for amplitude in (0.3, 0.01, 0):
        for cfg, bound, reference, x, out, expected, args, graph in records:
            x.normal_(0, amplitude)
            reference.fp8_qpn8_gemm_sm70_out(expected, *args)
            for replay in (False, True):
                out.fill_(float("nan"))
                if replay:
                    graph.replay()
                else:
                    bound.fp8_qpn8_gemm_sm70_out(out, *args)
                assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    for cfg, *_, graph in records:
        graph.reset()
        release_runtime_resources(cfg)


def test_gemm_resources_release_one_engine_without_invalidating_other_graph():
    torch.manual_seed(413)
    weight = torch.randn(512, 1024, device="cuda", dtype=torch.float16) * 0.01
    records = []
    for _ in range(2):
        cfg, bound, reference = engine("f16", f16_dense_max_m=64)
        x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
        for _ in range(3):
            bound.sm70_f16_gemm(x, weight)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = bound.sm70_f16_gemm(x, weight)
        assert sum(handle.resource_count() for handle in bound.owner.handles) > 0
        records.append((cfg, bound, reference, x, output, graph))
    assert records[0][1].owner is not records[1][1].owner
    records[0][-1].reset()
    release_runtime_resources(records[0][0])
    cfg, bound, reference, x, output, graph = records[1]
    for amplitude in (0.01, 0.3):
        x.normal_(0, amplitude)
        expected = reference.sm70_f16_gemm(x, weight)
        graph.replay()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    graph.reset()
    release_runtime_resources(cfg)
    assert all(handle.resource_count() == 0 for handle in bound.owner.handles)


def test_export_reload_uses_current_engine_slot(tmp_path):
    torch.manual_seed(823)
    first, bound, _ = engine("f16", f16_dense_max_m=64, tm_gemm_cache_summary=False)
    second, other, reference = engine(
        "f16", f16_dense_max_m=64, tm_gemm_cache_summary=True
    )
    assert first.kernel_config.compute_hash() == second.kernel_config.compute_hash()
    assert bound.arguments == other.arguments
    slot = bound.arguments[0]

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "weight",
                torch.randn(512, 1024, device="cuda", dtype=torch.float16) * 0.01,
            )

        def forward(self, x):
            return torch.ops._C.sm70_f16_gemm(x, self.weight, native_policy=slot)

    module = Projection()
    x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
    exported = torch.export.export(
        module, (x,), dynamic_shapes={"x": {0: torch.export.Dim("rows", min=1, max=32)}}
    )
    path = tmp_path / "projection.pt2"
    torch.export.save(exported, path)
    loaded = torch.export.load(path).module()
    release_runtime_resources(first)
    for rows in (1, 16, 32):
        x = torch.randn(rows, 1024, device="cuda", dtype=torch.float16)
        expected = reference.sm70_f16_gemm(x, module.weight)
        with other.owner.activate():
            actual = loaded(x)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    release_runtime_resources(second)
