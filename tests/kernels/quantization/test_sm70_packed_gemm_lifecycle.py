# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed dense/grouped views keep their strides and live replay inputs."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from vllm._sm70.policy import NativeBindings
from vllm.config import KernelConfig, set_current_vllm_config
from vllm.config.sm70_native import Sm70NativeConfig
from vllm.runtime_resources import release_runtime_resources

FORMATS = ("awq", "fp8", "mxfp4", "nvfp4")
K, N = 512, 256


def bind_format(fmt):
    config = SimpleNamespace(kernel_config=KernelConfig())
    policy = Sm70NativeConfig(**{f"{fmt}_tune_small_shapes": False})
    policy.resolve(fmt)
    with set_current_vllm_config(config):
        bindings = NativeBindings(policy.values)
    return config, bindings


def prepare_weight(bound, fmt, gated=False, compact=False):
    group = {"awq": 128, "fp8": 128, "mxfp4": 32, "nvfp4": 16}[fmt]
    scale = torch.rand(K // group, N, device="cuda", dtype=torch.float16) * 0.01
    if fmt == "awq":
        weight = torch.randint(
            -(2**31), 2**31 - 1, (K, N // 8), device="cuda", dtype=torch.int32
        )
        zeros = torch.randint(
            -(2**31), 2**31 - 1, (K // group, N // 8), device="cuda", dtype=torch.int32
        )
        prepare = bound.awq_sm70_prepare_compact if compact else bound.awq_sm70_prepare
        packed, scales, meta = prepare(weight, scale, zeros, group, gated)
    elif fmt == "fp8":
        weight = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.rand(N // 128, K // 128, device="cuda") * 0.01
        packed, scales, meta = bound.fp8_sm70_prepare(weight, scale, group, gated)
    else:
        weight = torch.randint(16, (K, N), device="cuda", dtype=torch.uint8)
        if fmt == "mxfp4":
            scale = torch.randint(
                118, 124, (K // group, N), device="cuda", dtype=torch.uint8
            )
        packed, scales, meta = getattr(bound, f"{fmt}_sm70_prepare")(
            weight, scale, group, gated
        )
    return packed, scales, group, int(meta[0]), int(meta[1])


@pytest.fixture(autouse=True)
def sm70():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401
    import vllm._moe_C  # noqa: F401


@pytest.mark.parametrize(
    "fmt,compact", [(fmt, False) for fmt in FORMATS] + [("awq", True)]
)
@pytest.mark.parametrize("gated", [False, True])
def test_dense_dynamic_rows_output_stride_and_replay(
    fmt, compact, gated, record_property
):
    torch.manual_seed(101)
    digest = hashlib.sha256()
    config, bound = bind_format(fmt)
    weight, scales, group, k_ld, q_ld = prepare_weight(bound, fmt, gated, compact)
    operation = getattr(bound, f"{fmt}_gemm_sm70_out")
    width = N // 2 if gated else N
    for rows in (1, 8, 32, 33, 64):
        # AWQ's native contract still uses LD=K. Other codecs accept row views.
        storage = torch.randn(
            rows, K if fmt == "awq" else K + 128, device="cuda", dtype=torch.float16
        )
        x = storage[:, :K]
        buffers = [
            torch.full((rows, width + 16), 42, device="cuda", dtype=x.dtype)
            for _ in range(2)
        ]
        expected, actual = [buffer[:, :width] for buffer in buffers]

        def run(out, x=x):
            operation(out, x, weight, scales, group, k_ld, q_ld, gated)

        run(actual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(actual)
        for amplitude in (0.01, 0.2):
            x.normal_(0, amplitude)
            run(expected)
            graph.replay()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert all(torch.all(buffer[:, width:] == 42) for buffer in buffers)
            digest.update(actual.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        graph.reset()
    with pytest.raises(RuntimeError, match="input must be float16"):
        operation(actual, x.float(), weight, scales, group, k_ld, q_ld, gated)
    record_property("output_sha256", digest.hexdigest())
    release_runtime_resources(config)


@pytest.mark.parametrize("fmt", FORMATS)
def test_grouped_replay_updates_experts_and_empty_segments(fmt, record_property):
    torch.manual_seed(107)
    digest = hashlib.sha256()
    config, bound = bind_format(fmt)
    prepared = [prepare_weight(bound, fmt) for _ in range(3)]
    weights = torch.stack([row[0] for row in prepared])
    scales = torch.stack([row[1] for row in prepared])
    _, _, group, k_ld, q_ld = prepared[0]
    ptrs = bound.awq_moe_build_strided_ptrs(weights, scales, k_ld, q_ld, 3)
    x = torch.randn(33, K + 128, device="cuda", dtype=torch.float16)[:, :K]
    offsets = torch.tensor([0, 19, 19, 33], device="cuda", dtype=torch.int32)
    experts = torch.tensor([2, 0, 2], device="cuda", dtype=torch.int32)
    expected, actual = [
        torch.empty(33, N, device="cuda", dtype=x.dtype) for _ in range(2)
    ]
    operation = getattr(bound, f"{fmt}_moe_dense_stage_sm70_out")

    def run(out):
        operation(out, x, offsets, experts, *ptrs, 3, K, N, group)

    run(actual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(actual)
    for ids, lengths in (([0, 2, 1], [0, 11, 11, 33]), ([1, 2, 0], [0, 0, 20, 33])):
        experts.copy_(torch.tensor(ids, device="cuda", dtype=experts.dtype))
        offsets.copy_(torch.tensor(lengths, device="cuda", dtype=offsets.dtype))
        x.normal_(0, 0.05)
        run(expected)
        graph.replay()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        digest.update(actual.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    graph.reset()
    record_property("output_sha256", digest.hexdigest())
    release_runtime_resources(config)
