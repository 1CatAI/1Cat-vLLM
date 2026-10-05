# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _sm70_ops as ops


@torch.inference_mode()
def test_fp32_head_matches_fp64_dense_and_live_graph_inputs():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")
    torch.manual_seed(20261004)
    n, k = 62080, 5120
    weight = (torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.125).to(
        torch.float8_e4m3fn
    )
    scale = torch.ones(n, 1, device="cuda", dtype=torch.float32)
    packed, packed_scale, meta = ops.fp8_sm70_prepare(weight, scale, 128, False)
    k_ld, q_ld = int(meta[0]), int(meta[1])
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A16Fp8,
    )

    scheme = object.__new__(CompressedTensorsW8A16Fp8)
    layer = SimpleNamespace(
        output_size_per_partition=n,
        sm70_fp8_turbomind=True,
        sm70_fp8_fp32_head=True,
        weight=packed,
        weight_scale_inv=packed_scale,
        sm70_fp8_k_ld=k_ld,
        sm70_fp8_q_ld=q_ld,
    )
    reference_weight = weight.double() * scale.double()
    for rows in (1, 8, 32):
        x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
        draft_output = torch.empty(rows, n, device="cuda", dtype=torch.float16)
        draft = scheme.apply_weights(layer, x, output=draft_output)
        layer.sm70_fp8_fp32_head = False
        original = scheme.apply_weights(layer, x)
        layer.sm70_fp8_fp32_head = True
        assert draft.data_ptr() == draft_output.data_ptr()
        assert torch.equal(draft.view(torch.int16), original.view(torch.int16))
        output = torch.empty(rows, n, device="cuda", dtype=torch.float32)

        def apply(values=x, target=output):
            torch.ops._C.fp8_gemm_sm70_fp32_head_out(
                target, values, packed, packed_scale, k_ld, q_ld
            )

        apply()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            apply()
        for amplitude in (0.0, 0.125, -0.25, 1.0):
            x.copy_(torch.randn_like(x) * amplitude)
            graph.replay()
            reference = x.double() @ reference_weight.T
            actual = output.double()
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=0.001)
            lp, lq = reference.log_softmax(-1), actual.log_softmax(-1)
            kl = (lp.exp() * (lp - lq)).sum(-1)
            assert kl.max().item() <= 0.001
            assert torch.equal(reference.argmax(-1), actual.argmax(-1))


def test_fp32_head_rejects_unsupported_verification_width():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")
    x = torch.empty(33, 5120, device="cuda", dtype=torch.float16)
    output = torch.empty(33, 62080, device="cuda", dtype=torch.float32)
    weight = torch.empty(5120, 62080, device="cuda", dtype=torch.uint8)
    scale = torch.empty(40, 62080, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="unsupported FP32 LM-head shape"):
        torch.ops._C.fp8_gemm_sm70_fp32_head_out(output, x, weight, scale, 0, 0)


def test_fp32_head_fake_tensor_registration():
    from torch._subclasses.fake_tensor import FakeTensorMode

    if not hasattr(torch.ops._C, "fp8_gemm_sm70_fp32_head_out"):
        pytest.skip("Native FP32 head operator required")
    with FakeTensorMode():
        x = torch.empty(8, 5120, device="cuda", dtype=torch.float16)
        output = torch.empty(8, 62080, device="cuda", dtype=torch.float32)
        weight = torch.empty(5120, 62080, device="cuda", dtype=torch.uint8)
        scale = torch.empty(40, 62080, device="cuda", dtype=torch.float16)
        result = torch.ops._C.fp8_gemm_sm70_fp32_head_out(
            output, x, weight, scale, 5120, 62080
        )
        assert result is None
        assert output.dtype == torch.float32
