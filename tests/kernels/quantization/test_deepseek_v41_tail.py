# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ragged verification uses bounded FP16 experts without CPU route readback."""

from types import SimpleNamespace
from typing import cast

import gguf
import pytest
import torch

from tests.models.gguf.test_v41_device_codecs import source
from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.quantization import gguf_v41_moe
from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
from vllm.model_executor.layers.quantization.gguf_native import native_available
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_device_capability(70),
    reason="SM70 qualification required",
)


@pytest.mark.parametrize(
    "rows,kind,down_kind,left",
    [
        (2, 29, 17, 0),
        (3, 16, 29, 64),
        (4, 17, 16, 128),
        (5, 29, 29, 192),
    ],
)
def test_tail_all_four_batches_tp_offsets_and_mixed_formats_are_fp16(
    monkeypatch, rows, kind, down_kind, left
):
    assert native_available()
    h, intermediate, experts = 5120, 576, 2
    packed = [
        source(kind, experts * intermediate, h, 0.00137, seed=41 + branch).reshape(
            experts, intermediate, -1
        )
        for branch in range(2)
    ]
    packed.append(source(down_kind, experts * h, 768, 0.00137).reshape(experts, h, -1))
    weights = [torch.from_numpy(p).to("cuda") for p in packed]
    dense = [
        torch.from_numpy(
            gguf.quants.dequantize(
                p.reshape(-1, p.shape[-1]), gguf.GGMLQuantizationType(k)
            )
        )
        .half()
        .reshape(experts, -1, 768 if branch == 2 else h)
        for branch, (p, k) in enumerate(zip(packed, (kind, kind, down_kind)))
    ]
    generator = torch.Generator().manual_seed(41)
    x = (torch.randn(rows, h, generator=generator) * 0.125).half()
    ids = torch.arange(rows * 6).reshape(rows, 6) % experts
    probabilities = torch.rand(rows, 6, generator=generator)
    probabilities /= probabilities.sum(-1, keepdim=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("FP16 tail used MMVQ or a CPU tensor readback")

    native_dense = gguf_v41_moe.native_dense
    boundaries = []

    def observed(values, packed, *args, **kwargs):
        result = native_dense(values, packed, *args, **kwargs)
        assert result is not None
        boundaries.append((values.clone(), result.clone()))
        return result

    state = SimpleNamespace(
        expert_trace_tokens=0,
        dp4a_admitted=True,
        is_v41=True,
        cache_layer=0,
        expert_cache=SimpleNamespace(resolve=lambda layer, ids: ids),
        weight_types={"w1": kind, "w3": kind, "w2": down_kind},
        input_padding={"w2": (left, 768)},
        swiglu_limit=10.0,
        fp16_workspace_bytes=4 * 1024**2,
    )
    method = cast(GGUFNativeMoEMethod, state)
    layer = SimpleNamespace(
        apply_router_weight_on_input=False,
        activation=MoEActivation.SILU,
        expert_map=None,
        gguf_w1=weights[0],
        gguf_w3=weights[1],
        gguf_w2=weights[2],
    )
    # Check real method dispatch too. Read back only test snapshots after the
    # hot call, with MMVQ and Tensor.cpu explicitly forbidden during dispatch.
    with monkeypatch.context() as patch:
        for name in ("ggml_dense_mmvq", "ggml_moe_mmvq", "ggml_moe_grouped_dense"):
            patch.setattr(torch.ops._C_gguf, name, forbidden)
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(gguf_v41_moe, "native_dense", observed)
        actual = GGUFNativeMoEMethod.apply(
            method,
            layer,
            x.to("cuda"),
            probabilities.to("cuda"),
            ids.to("cuda"),
            None,
            None,
        )
    assert len(boundaries) == rows * 6 * 3
    boundaries = [(values.cpu(), result.cpu()) for values, result in boundaries]
    expected = torch.empty_like(x)
    for token in range(rows):
        routes = []
        for branch in range(6):
            expert = ids[token, branch]
            route = (token * 6 + branch) * 3
            g_ideal = (x[token].float() @ dense[0][expert].float().T).half()
            u_ideal = (x[token].float() @ dense[1][expert].float().T).half()
            g, u = boundaries[route][1][0], boundaries[route + 1][1][0]
            # Compare each FP16 GEMM separately: CPU/TC FP32 reductions may
            # straddle a half rounding midpoint. Feeding the ideal half result
            # into another GEMM would compound that permitted boundary error.
            torch.testing.assert_close(g, g_ideal, rtol=2e-3, atol=2e-5)
            torch.testing.assert_close(u, u_ideal, rtol=2e-3, atol=2e-5)
            g, u = g.float(), u.float()
            hidden = (
                torch.nn.functional.silu(g.clamp(max=10))
                * u.clamp(-10, 10)
                * probabilities[token, branch]
            ).half()
            padded = boundaries[route + 2][0][0]
            torch.testing.assert_close(
                padded[left : left + 576], hidden, rtol=2e-3, atol=2e-5
            )
            assert not padded[:left].any() and not padded[left + 576 :].any()
            ideal_down = (padded.float() @ dense[2][expert].float().T).half()
            observed_down = boundaries[route + 2][1][0]
            torch.testing.assert_close(observed_down, ideal_down, rtol=2e-3, atol=2e-5)
            routes.append(observed_down)
        expected[token] = torch.stack(routes).float().sum(0).half()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
