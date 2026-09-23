# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a16_fp8,  # noqa: F401
)


def test_batched_layout_policy_is_shared_and_reversible(monkeypatch):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash", num_speculative_tokens=7),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
    )
    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)
    monkeypatch.setattr(sm70_tm.envs, "VLLM_SM70_BATCH_GEMM_LAYOUTS", True)

    assert sm70_tm.use_batched_gemm_layouts()
    config.scheduler_config.max_num_seqs = 4
    assert not sm70_tm.use_batched_gemm_layouts()
    config.scheduler_config.max_num_seqs = 16
    config.speculative_config.method = "mtp"
    assert not sm70_tm.use_batched_gemm_layouts()
    config.speculative_config.method = "dflash"
    monkeypatch.setattr(sm70_tm.envs, "VLLM_SM70_BATCH_GEMM_LAYOUTS", False)
    assert not sm70_tm.use_batched_gemm_layouts()


@pytest.mark.parametrize(
    ("m", "k", "n", "split_k", "gated_silu"),
    [
        (8, 1536, 5120, 12, False),
        (32, 5120, 3584, 16, False),
        (64, 1536, 5120, 12, False),
        (64, 5120, 8704, 8, True),
    ],
)
def test_batch_fp8_dispatch_matches_qpn8_and_replays(m, k, n, split_k, gated_silu):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA required")
    torch.manual_seed(20260924)
    weight = torch.randn((n, k), device="cuda").mul_(0.25).to(torch.float8_e4m3fn)
    scales = torch.full((n, 1), 0.125, device="cuda", dtype=torch.float32)
    codes, q_scales = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scales)
    tm_weight, tm_scales, meta = torch.ops._C.fp8_sm70_prepare(
        weight, scales, 128, gated_silu
    )
    x = torch.randn((m, k), device="cuda", dtype=torch.float16).mul_(0.1)
    expected = torch.empty(
        (m, n // 2 if gated_silu else n), device="cuda", dtype=torch.float16
    )
    actual = torch.empty_like(expected)
    workspace = torch.empty((k, n), device="cuda", dtype=torch.float16)

    def reference():
        torch.ops._C.fp8_qpn8_dispatch_sm70_out(
            expected,
            workspace.data_ptr(),
            x,
            codes,
            q_scales,
            split_k,
            2,
            False,
            gated_silu,
        )

    def candidate():
        torch.ops.vllm.sm70_ct_fp8_qpn8_batch_dispatch(
            actual,
            x,
            codes,
            q_scales,
            tm_weight,
            tm_scales,
            int(meta[0]),
            int(meta[1]),
            split_k,
            2,
            False,
            gated_silu,
        )

    candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    for _ in range(3):
        x.normal_(0, 0.1)
        reference()
        graph.replay()
        torch.accelerator.synchronize()
        if m <= 32:
            assert torch.equal(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
