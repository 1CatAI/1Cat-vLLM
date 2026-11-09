# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny QPN MoE (moe_backend="sm70_skinny") for NVFP4 and MXFP4."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    Mxfp4SkinnySm70Experts,
    Nvfp4SkinnySm70Experts,
    _ScaleCaches,
    grouped_splitk,
    nvfp4_skinny_scale_reason,
    qpn_prepack,
    rebase_e8m0_for_fp16,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _get_priority_backends,
    map_mxfp4_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    map_nvfp4_backend,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)
from vllm.platforms import current_platform


def _pow2(e8m0: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, e8m0.to(torch.float64) - 127)


# ---------------------------------------------------------------------------
# Host-side preparation
# ---------------------------------------------------------------------------


def test_rebase_reproduces_every_scale_exactly():
    generator = torch.Generator().manual_seed(0)
    low = torch.randint(95, 108, (4, 1, 1), generator=generator)
    offsets = torch.randint(0, 20, (4, 8, 6), generator=generator)
    scales = (low + offsets).to(torch.uint8)

    rebased = scales.clone()
    global_scale = rebase_e8m0_for_fp16(rebased)

    assert int(rebased.min()) >= 113 and int(rebased.max()) <= 142
    restored = _pow2(rebased) * global_scale.to(torch.float64).view(-1, 1, 1)
    assert torch.equal(restored, _pow2(scales))


def test_rebase_dry_run_preserves_checkpoint_bytes():
    scales = torch.tensor([[[100, 110, 119]]], dtype=torch.uint8)
    before = scales.clone()
    expected = rebase_e8m0_for_fp16(scales, dry_run=True)
    assert torch.equal(scales, before)
    assert torch.equal(rebase_e8m0_for_fp16(scales), expected)


def test_scale_conversion_checks_both_projections_before_mutating():
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        convert_weight_to_mxfp4_moe_kernel_format,
    )

    layer = nn.Module()
    s13 = torch.full((1, 32, 8), 100, dtype=torch.uint8)
    s2 = torch.full((1, 32, 8), 255, dtype=torch.uint8)
    before = s13.clone()
    with pytest.raises(ValueError, match="NaN"):
        convert_weight_to_mxfp4_moe_kernel_format(
            Mxfp4MoeBackend.SM70_SKINNY,
            layer,
            torch.zeros(1, 32, 128, dtype=torch.uint8),
            torch.zeros(1, 32, 128, dtype=torch.uint8),
            s13,
            s2,
        )
    assert torch.equal(s13, before)
    assert not list(layer.parameters())


def test_rebase_keeps_the_global_scale_inside_fp16():
    # All scales high: lifting the smallest to 113 would need a negative
    # shift whose global scale times 2^14 overflows fp16.
    scales = torch.full((1, 4, 4), 124, dtype=torch.uint8)
    scales[0, 0, 0] = 128

    rebased = scales.clone()
    global_scale = rebase_e8m0_for_fp16(rebased)

    assert float(global_scale) * 2**14 <= 2**15
    restored = _pow2(rebased) * float(global_scale)
    assert torch.equal(restored, _pow2(scales))


@pytest.mark.parametrize(
    ("low", "high", "message"),
    [
        (100, 255, "NaN"),
        (95, 125, "span"),
        (80, 90, "outside"),
    ],
)
def test_rebase_refuses_scales_the_kernels_cannot_represent(low, high, message):
    scales = torch.full((1, 2, 2), low, dtype=torch.uint8)
    scales[0, 1, 1] = high

    untouched = scales.clone()

    with pytest.raises(ValueError, match=message):
        rebase_e8m0_for_fp16(scales)
    assert torch.equal(scales, untouched)


def _experts_with_rasters(cls, w1_scales, w2_scales):
    experts = cls.__new__(cls)
    experts._w1_block_scales = w1_scales
    experts._w2_block_scales = w2_scales
    return experts


def test_nvfp4_rasters_keep_16_or_fold_to_32():
    # fp8-e4m3 bytes with zero mantissa are powers of two; duplicated pairs
    # are an MXFP4 raster in NVFP4 form.
    duplicated = torch.tensor([[[0x38, 0x38, 0x40, 0x40]]], dtype=torch.uint8)
    distinct = torch.tensor([[[0x38, 0x40, 0x40, 0x48]]], dtype=torch.uint8)
    layer = nn.Module()

    kept = _experts_with_rasters(Nvfp4SkinnySm70Experts, distinct, distinct.clone())
    folded = _experts_with_rasters(
        Nvfp4SkinnySm70Experts, duplicated, duplicated.clone()
    )

    assert kept._scale_rasters(layer)[2] == 16
    s13, _, group = folded._scale_rasters(layer)
    assert group == 32
    assert s13.tolist() == [[[0x38 // 8 + 120, 0x40 // 8 + 120]]]


def test_mxfp4_rasters_pass_through_at_32():
    scales = torch.full((2, 4, 2), 120, dtype=torch.uint8)
    experts = _experts_with_rasters(Mxfp4SkinnySm70Experts, scales, scales.clone())

    s13, s2, group = experts._scale_rasters(nn.Module())

    assert group == 32
    assert s13 is not None and torch.equal(s13, scales)
    assert torch.equal(s2, scales)


def test_zero_nvfp4_scales_are_not_folded_into_nonzero_mxfp4():
    scales = torch.tensor([[[0, 0, 0x38, 0x38]]], dtype=torch.uint8)
    experts = _experts_with_rasters(Nvfp4SkinnySm70Experts, scales, scales.clone())
    s13, s2, group = experts._scale_rasters(nn.Module())
    assert group == 16
    assert torch.equal(s13, scales) and torch.equal(s2, scales)


@pytest.mark.parametrize("value", [0.0, 0.001, 0.37, 3.99])
def test_nvfp4_global_scales_admit_finite_rebias(value):
    scales = torch.full((4,), value)
    assert nvfp4_skinny_scale_reason(scales, scales) is None


@pytest.mark.parametrize("value", [4.0, 100.0, float("inf"), float("nan"), -1.0])
def test_nvfp4_global_scales_reject_invalid_or_overflowing_rebias(value):
    scales = torch.full((4,), value)
    assert nvfp4_skinny_scale_reason(scales, scales) is not None


def test_nvfp4_gate_and_up_require_the_same_global_factor():
    g1 = torch.tensor([[0.25, 0.5], [0.5, 0.5]])
    assert "shared factor" in nvfp4_skinny_scale_reason(g1, torch.ones(2))


@pytest.mark.parametrize("scale_group", [16, 32])
def test_qpn_prepack_only_permutes_the_checkpoint_bytes(scale_group):
    generator = torch.Generator().manual_seed(2)
    codes = torch.randint(0, 256, (64, 128), generator=generator).to(torch.uint8)
    scales = torch.randint(0, 256, (64, 256 // scale_group), generator=generator).to(
        torch.uint8
    )

    qc, qs = qpn_prepack(codes, scales, scale_group)

    nibbles = torch.cat([codes & 0xF, codes >> 4]).flatten()
    packed_nibbles = torch.cat([qc & 0xF, qc >> 4])
    assert torch.equal(torch.sort(packed_nibbles).values, torch.sort(nibbles).values)
    assert torch.equal(torch.sort(qs).values, torch.sort(scales.flatten()).values)


def test_qpn_prepack_refuses_shapes_the_kernels_do_not_tile():
    with pytest.raises(ValueError, match="N % 32"):
        qpn_prepack(
            torch.zeros(48, 64, dtype=torch.uint8),
            torch.zeros(48, 8, dtype=torch.uint8),
        )


def test_backend_is_registered_for_automatic_selection():
    assert map_nvfp4_backend("sm70_skinny") == NvFp4MoeBackend.SM70_SKINNY
    assert map_mxfp4_backend("sm70_skinny") == [Mxfp4MoeBackend.SM70_SKINNY]
    assert Mxfp4MoeBackend.SM70_SKINNY in _get_priority_backends()


@pytest.mark.parametrize("k", [128, 256, 320, 512, 640, 1024, 4096, 5120])
def test_grouped_launch_uses_actual_alignment(k):
    split = grouped_splitk(k, 16)
    assert (k // 16) % split == 0


@pytest.mark.parametrize("k", [0, 64, 192, 319])
def test_grouped_launch_rejects_unsupported_alignment(k):
    with pytest.raises(ValueError):
        grouped_splitk(k, 16)


@pytest.mark.parametrize("scale_mode", [0, 1])
def test_large_batches_keep_the_scale_format_and_device_routing(scale_mode):
    experts = Nvfp4SkinnySm70Experts.__new__(Nvfp4SkinnySm70Experts)
    experts._scale_mode = scale_mode
    calls = []

    def grouped(output, hidden_states, *args):
        calls.append((hidden_states.shape[0], experts._scale_mode))
        output.copy_(hidden_states)

    experts._apply_grouped = grouped
    x = torch.arange(513 * 128, dtype=torch.float32).reshape(513, 128)
    out = torch.empty_like(x)
    experts.apply(
        out,
        x,
        torch.empty(4, 256, 64),
        torch.empty(4, 128, 64),
        torch.ones(513, 128),
        torch.zeros(513, 128, dtype=torch.int64),
        MoEActivation.SILU,
        4,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    assert torch.equal(out, x)
    assert len(calls) > 1
    assert sum(rows for rows, _ in calls) == 513
    assert all(rows * 128 <= 65535 and mode == scale_mode for rows, mode in calls)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({}, None),
        ({"hidden_dim": 320, "intermediate_size_per_partition": 640}, None),
        ({"hidden_dim": 192}, "hidden dimension"),
        ({"intermediate_size_per_partition": 64}, "intermediate dimension"),
        ({"in_dtype": torch.bfloat16}, "FP16"),
        ({"has_bias": True}, "expert bias"),
        ({"swiglu_limit": 7.0}, None),
        ({"experts_per_token": 0}, "routing width"),
        ({"ep_size": 2}, "expert-parallel"),
        ({"enable_eplb": True}, "load balancing"),
        ({"is_lora_enabled": True}, "LoRA"),
        ({"apply_router_weight_on_input": True}, "input-side"),
        ({"enabled": False}, "disabled"),
    ],
)
def test_kernel_admission_reports_real_constraints(monkeypatch, change, reason):
    import vllm.config
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.config.kernel import KernelConfig

    kernel = KernelConfig(sm70_skinny_moe=change.get("enabled", True))
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(kernel_config=kernel),
    )
    monkeypatch.setattr(
        Nvfp4SkinnySm70Experts, "_supports_current_device", lambda: True
    )
    monkeypatch.setattr(torch.ops._C, "skinny_moe_qpn_sm70", object(), raising=False)
    monkeypatch.setattr(
        mk.FusedMoEExperts, "is_supported_config", lambda *args: (True, None)
    )
    config = SimpleNamespace(
        hidden_dim=change.get("hidden_dim", 256),
        intermediate_size_per_partition=change.get(
            "intermediate_size_per_partition", 128
        ),
        in_dtype=change.get("in_dtype", torch.float16),
        has_bias=change.get("has_bias", False),
        is_lora_enabled=change.get("is_lora_enabled", False),
        apply_router_weight_on_input=change.get("apply_router_weight_on_input", False),
        swiglu_limit=change.get("swiglu_limit"),
        experts_per_token=change.get("experts_per_token", 3),
        moe_parallel_config=SimpleNamespace(
            ep_size=change.get("ep_size", 1),
            enable_eplb=change.get("enable_eplb", False),
        ),
    )
    supported, actual = Nvfp4SkinnySm70Experts.is_supported_config(
        Nvfp4SkinnySm70Experts, config, None, None, mk.FusedMoEActivationFormat.Standard
    )
    assert kernel.sm70_skinny_moe_applicable
    assert supported == (reason is None)
    if reason is not None:
        assert reason in actual


# ---------------------------------------------------------------------------
# Kernels, on SM70 and SM75
# ---------------------------------------------------------------------------


def _devices(capability: tuple[int, int]) -> list[int]:
    if not current_platform.is_cuda():
        return []
    return [
        i
        for i in range(torch.accelerator.device_count())
        if current_platform.get_device_capability(i) == capability
    ]


@pytest.fixture(params=[(7, 0), (7, 5)], ids=["sm70", "sm75"])
def device(request) -> torch.device:
    devices = _devices(request.param)
    if not devices:
        pytest.skip(f"no GPU with capability {request.param}")
    torch.accelerator.set_device_index(devices[0])
    return torch.device("cuda", devices[0])


def _nvfp4_expert(generator, rows, cols):
    codes = torch.randint(0, 256, (rows, cols // 2), generator=generator)
    scales = (torch.rand(rows, cols // 16, generator=generator) * 1.5 + 0.25).to(
        torch.float8_e4m3fn
    )
    return codes.to(torch.uint8), scales.view(torch.uint8)


def _nvfp4_weight(codes, scales, global_scale):
    values = break_fp4_bytes(codes, torch.float32)
    block = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return values * block * global_scale


@pytest.mark.parametrize("m", [1, 5, 8, 12, 16])
def test_qpn_gemm_matches_the_dequantized_weights(device, m):
    generator = torch.Generator().manual_seed(3)
    rows, cols, global_scale = 128, 512, 0.37
    codes, scales = _nvfp4_expert(generator, rows, cols)
    x = torch.randn(m, cols, generator=generator).to(torch.float16) * 0.1
    reference = x.float() @ _nvfp4_weight(codes, scales, global_scale).T

    qc, qs = qpn_prepack(codes.to(device), scales.to(device))
    out = torch.ops._C.skinny_qpn_gemm_sm70(x.to(device), qc, qs, global_scale, rows)

    torch.testing.assert_close(out.float().cpu(), reference, rtol=2e-2, atol=2e-3)


def test_moe_qpn_mxfp4_mode_matches_the_checkpoint_scales(device):
    generator = torch.Generator().manual_seed(1)
    experts, rows, cols, tokens = 2, 64, 256, 5
    codes = torch.randint(0, 256, (experts, rows, cols // 2), generator=generator)
    codes = codes.to(torch.uint8)
    scales = (
        torch.randint(0, 8, (experts, rows, cols // 32), generator=generator) + 112
    ).to(torch.uint8)
    hidden = torch.randn(tokens, cols, generator=generator).to(torch.float16)

    reference = []
    for e in range(experts):
        values = break_fp4_bytes(codes[e], torch.float32)
        weight = values * _pow2(scales[e]).repeat_interleave(32, dim=1).float()
        reference.append(hidden.float() @ weight.T)

    rebased = scales.clone()
    global_scale = rebase_e8m0_for_fp16(rebased)
    w = codes.to(device)
    s = rebased.to(device)
    for e in range(experts):
        qc, qs = qpn_prepack(w[e], s[e], 32)
        w[e].view(-1).copy_(qc)
        s[e].view(-1).copy_(qs)

    x = hidden.to(device)
    for e in range(experts):
        perm = torch.arange(tokens, dtype=torch.int32, device=device)
        gids = torch.full((tokens,), e, dtype=torch.int32, device=device)
        goff = torch.full((tokens + 1,), tokens, dtype=torch.int32, device=device)
        goff[0] = 0
        out = torch.empty((tokens, rows), dtype=torch.float16, device=device)
        torch.ops._C.skinny_moe_qpn_sm70(
            x,
            w,
            s,
            global_scale.to(device),
            perm,
            gids,
            goff,
            1,
            out,
            False,
            tokens,
            16,
            1,
            1,
        )
        torch.testing.assert_close(
            out.float().cpu(), reference[e], rtol=2e-2, atol=2e-2
        )


def _nvfp4_moe(generator, num_experts, hidden_size, intermediate):
    w13, s13, w2, s2 = [], [], [], []
    for _ in range(num_experts):
        c, s = _nvfp4_expert(generator, 2 * intermediate, hidden_size)
        w13.append(c)
        s13.append(s)
        c, s = _nvfp4_expert(generator, hidden_size, intermediate)
        w2.append(c)
        s2.append(s)
    g1 = torch.rand(num_experts, generator=generator) * 0.2 + 0.05
    g2 = torch.rand(num_experts, generator=generator) * 0.2 + 0.05
    return torch.stack(w13), torch.stack(s13), torch.stack(w2), torch.stack(s2), g1, g2


def _reference_moe(x, w13, s13, w2, s2, g1, g2, topk_weights, topk_ids):
    out = torch.zeros(x.size(0), w2.size(1))
    for t in range(x.size(0)):
        for j in range(topk_ids.size(1)):
            e = int(topk_ids[t, j])
            h = x[t].float() @ _nvfp4_weight(w13[e], s13[e], float(g1[e])).T
            gate, up = h.chunk(2)
            h = torch.nn.functional.silu(gate) * up
            y = h @ _nvfp4_weight(w2[e], s2[e], float(g2[e])).T
            out[t] += float(topk_weights[t, j]) * y
    return out


def _skinny_experts(device, w13, s13, w2, s2, g1, g2):
    experts = Nvfp4SkinnySm70Experts.__new__(Nvfp4SkinnySm70Experts)
    experts.quant_config = SimpleNamespace(gemm1_clamp_limit=None)
    experts._scale_mode = 0
    experts._w1_block_scales = s13
    experts._w2_block_scales = s2
    experts._scale_caches = _ScaleCaches(
        g1=g1.tolist(),
        g2=g2.tolist(),
        g1_t=g1.to(device),
        g2_t=g2.to(device),
        w1_scales_u8=s13,
        w2_scales_u8=s2,
    )
    return experts


@pytest.mark.parametrize("path", ["grouped", "dispatch"])
@pytest.mark.parametrize("num_tokens", [1, 6, 40])
def test_both_serving_paths_match_the_reference(device, path, num_tokens):
    generator = torch.Generator().manual_seed(4)
    num_experts, hidden_size, intermediate, top_k = 8, 256, 128, 3
    w13, s13, w2, s2, g1, g2 = _nvfp4_moe(
        generator, num_experts, hidden_size, intermediate
    )
    x = torch.randn(num_tokens, hidden_size, generator=generator).to(torch.float16)
    x = x * 0.1
    topk_ids = torch.stack(
        [torch.randperm(num_experts, generator=generator)[:top_k] for _ in x]
    )
    topk_weights = torch.rand(num_tokens, top_k, generator=generator)
    reference = _reference_moe(x, w13, s13, w2, s2, g1, g2, topk_weights, topk_ids)

    w13_d, s13_d, w2_d, s2_d = (t.to(device) for t in (w13, s13, w2, s2))
    for e in range(num_experts):
        for w, s in ((w13_d, s13_d), (w2_d, s2_d)):
            qc, qs = qpn_prepack(w[e], s[e])
            w[e].view(-1).copy_(qc)
            s[e].view(-1).copy_(qs)
    experts = _skinny_experts(device, w13_d, s13_d, w2_d, s2_d, g1, g2)
    output = torch.empty(num_tokens, hidden_size, dtype=torch.float16, device=device)
    args = dict(
        output=output,
        hidden_states=x.to(device),
        w1=w13_d,
        w2=w2_d,
        topk_weights=topk_weights.to(device),
        topk_ids=topk_ids.to(device),
        activation=MoEActivation.SILU,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    if path == "grouped":
        experts._apply_grouped(**args)
    else:
        experts.apply(
            **args,
            global_num_experts=num_experts,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
        )

    # fp16 is handed between the GEMMs and the activation; the largest error
    # measured against this fp32 reference is about 0.1% of the output
    # scale on both cards and both paths.
    scale = reference.abs().max().item()
    torch.testing.assert_close(
        output.float().cpu(), reference, rtol=0, atol=3e-3 * scale
    )


def test_large_mxfp4_dispatch_matches_reference(device):
    generator = torch.Generator().manual_seed(42)
    tokens, experts_count, hidden, intermediate, top_k = 513, 4, 256, 128, 2
    w13 = torch.randint(
        0,
        256,
        (experts_count, 2 * intermediate, hidden // 2),
        generator=generator,
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0,
        256,
        (experts_count, hidden, intermediate // 2),
        generator=generator,
        dtype=torch.uint8,
    )
    s13 = torch.randint(
        120,
        124,
        (experts_count, 2 * intermediate, hidden // 32),
        generator=generator,
        dtype=torch.uint8,
    )
    s2 = torch.randint(
        120,
        124,
        (experts_count, hidden, intermediate // 32),
        generator=generator,
        dtype=torch.uint8,
    )
    nv_s13 = (
        _pow2(s13).repeat_interleave(2, -1).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    nv_s2 = _pow2(s2).repeat_interleave(2, -1).to(torch.float8_e4m3fn).view(torch.uint8)
    global_scales = torch.ones(experts_count)
    x = (torch.randn(tokens, hidden, generator=generator) * 0.1).half()
    ids = torch.randint(experts_count, (tokens, top_k), generator=generator)
    weights = torch.rand(tokens, top_k, generator=generator)
    reference = _reference_moe(
        x, w13, nv_s13, w2, nv_s2, global_scales, global_scales, weights, ids
    )
    w13, s13, w2, s2 = (t.to(device) for t in (w13, s13, w2, s2))
    for e in range(experts_count):
        for w, s in ((w13, s13), (w2, s2)):
            qc, qs = qpn_prepack(w[e], s[e], 32)
            w[e].view(-1).copy_(qc)
            s[e].view(-1).copy_(qs)
    experts = _skinny_experts(device, w13, s13, w2, s2, global_scales, global_scales)
    experts._scale_mode = 1
    output = torch.empty_like(x, device=device)
    experts.apply(
        output,
        x.to(device),
        w13,
        w2,
        weights.to(device),
        ids.to(device),
        MoEActivation.SILU,
        experts_count,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    torch.testing.assert_close(
        output.float().cpu(),
        reference,
        rtol=0,
        atol=3e-3 * reference.abs().max().item(),
    )
