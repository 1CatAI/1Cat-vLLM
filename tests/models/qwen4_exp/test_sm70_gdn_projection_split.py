# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as module

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def uncached_environment():
    envs.disable_envs_cache()


def metadata(rows, width, **overrides):
    values = dict(
        ndim=2,
        shape=(rows, width),
        dtype=torch.float16,
        is_cuda=True,
        device=torch.device("cuda:0"),
        stride=lambda: (width, 1),
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("rows", (0, 1, 2, 3, 4, 8, 16, 17, 32, 64, 127, 256, 4096))
def test_admission_has_no_max_batch_binding(monkeypatch, rows):
    monkeypatch.setenv("VLLM_SM70_GDN_BATCH_SPLIT_COPY", "1")
    monkeypatch.setattr(module.current_platform, "is_device_capability", lambda _: True)
    assert module._can_fuse_gdn_projection_split(
        metadata(rows, 4096), metadata(rows, 24)
    ) == (rows > 1)


@pytest.mark.parametrize(
    "change",
    (
        dict(dtype=torch.float32),
        dict(dtype=torch.bfloat16),
        dict(is_cuda=False),
        dict(ndim=3),
        dict(shape=(4, 4095)),
        dict(stride=lambda: (8192, 2)),
        dict(device=torch.device("cuda:1")),
    ),
)
def test_unsupported_input_falls_back(monkeypatch, change):
    monkeypatch.setenv("VLLM_SM70_GDN_BATCH_SPLIT_COPY", "1")
    monkeypatch.setattr(module.current_platform, "is_device_capability", lambda _: True)
    assert not module._can_fuse_gdn_projection_split(
        metadata(4, 4096, **change), metadata(4, 24)
    )


def test_off_and_non_sm70_fall_back(monkeypatch):
    q, ba = metadata(4, 4096), metadata(4, 24)
    monkeypatch.setenv("VLLM_SM70_GDN_BATCH_SPLIT_COPY", "0")
    assert not module._can_fuse_gdn_projection_split(q, ba)
    monkeypatch.setenv("VLLM_SM70_GDN_BATCH_SPLIT_COPY", "1")
    monkeypatch.setattr(
        module.current_platform, "is_device_capability", lambda _: False
    )
    assert not module._can_fuse_gdn_projection_split(q, ba)


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_GDN_BATCH_SPLIT_COPY", raising=False)
    assert not envs.environment_variables["VLLM_SM70_GDN_BATCH_SPLIT_COPY"]()


def test_fusion_keeps_both_linear_calls(monkeypatch):
    x = torch.empty(4, 2560)
    wq, wb = torch.empty(4096, 2560), torch.empty(24, 2560)
    q, ba = torch.empty(4, 4096), torch.empty(4, 24)
    calls = []

    def linear(value, weight):
        assert value is x
        calls.append(weight)
        return q if weight is wq else ba

    expected = tuple(torch.empty(4, width) for width in (2560, 1536, 12, 12))

    def fused(value, gate):
        assert value is q and gate is ba
        return expected

    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    monkeypatch.setattr(module, "_can_fuse_gdn_projection_split", lambda *args: True)
    monkeypatch.setattr(module, "_split_gdn_projection_outputs", fused)
    actual = module._qwen38_sm70_fp16_gdn_input(x, wq, wb)
    assert actual is expected
    assert len(calls) == 2 and calls[0] is wq and calls[1] is wb


@pytest.mark.parametrize("rows", (0, 1, 4, 17))
def test_cpu_original_fallback_is_unchanged(rows):
    # Small K is sufficient to exercise fallback slicing, including empty input.
    x, wq, wb = torch.randn(rows, 3), torch.randn(4096, 3), torch.randn(24, 3)
    actual = module._qwen38_sm70_fp16_gdn_input(x, wq, wb)
    q, ba = torch.nn.functional.linear(x, wq), torch.nn.functional.linear(x, wb)
    expected = q[:, :2560], q[:, 2560:], ba[:, :12], ba[:, 12:]
    assert all(
        torch.equal(a, b) and a.is_contiguous()
        for a, b in zip(actual, expected, strict=True)
    )


@pytest.mark.parametrize("rows", (1, 2, 4, 8, 16, 17, 32, 64))
def test_cuda_public_op_graph_replay_is_bitwise(monkeypatch, rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires an owned SM70 GPU")
    envs.disable_envs_cache()
    torch.manual_seed(51 + rows)
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    wq = torch.randn(4096, 2560, device="cuda", dtype=torch.float16) * 0.01
    wb = torch.randn(24, 2560, device="cuda", dtype=torch.float16) * 0.01
    op = torch.ops.vllm.qwen38_sm70_fp16_gdn_input
    route_hits = []
    fused = module._split_gdn_projection_outputs

    def tracked(q, ba):
        route_hits.append(q.shape[0])
        return fused(q, ba)

    monkeypatch.setattr(module, "_split_gdn_projection_outputs", tracked)
    monkeypatch.setenv("VLLM_SM70_GDN_BATCH_SPLIT_COPY", "1")
    op(x, wq, wb)
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outputs = op(x, wq, wb)
    assert route_hits == ([rows, rows] if rows > 1 else [])
    route_hits.clear()
    for i in range(4):
        x.normal_(0, 0.1 * (i + 1))
        wq.mul_(0.9)
        wb.mul_(0.9)
        for out in outputs:
            out.fill_(float("nan"))
        monkeypatch.setenv("VLLM_SM70_GDN_BATCH_SPLIT_COPY", "0")
        reference = op(x, wq, wb)
        g.replay()
        assert not route_hits
        assert all(
            a.is_contiguous() and torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(outputs, reference, strict=True)
        )
