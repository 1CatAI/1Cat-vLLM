# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.models.minimax_h3.attention import (
    Attention,
    AttentionMetadata,
    attention_backend,
    chunked_attention_reference,
)
from vllm.model_executor.models.minimax_h3.comfy_checkpoint import (
    inspect_comfy_checkpoint,
)
from vllm.model_executor.models.minimax_h3.ops import (
    convrot_reference,
    dequantize_int8_reference,
)
from vllm.model_executor.models.minimax_h3.time_request import (
    MINIMAX_H3_SHAPE_PLANNER,
    minimax_h3_time_shift_sigmas,
)
from vllm.model_executor.models.minimax_h3.transformer import (
    MiniMaxH3DiTModel,
    _reorder_grouped_qkv_to_qkv,
)


def test_signed_int8_restores_each_rows_scale():
    weight = torch.tensor([[-128, -1, 0, 127], [127, 0, -1, -128]], dtype=torch.int8)
    scale = torch.tensor([0.125, 0.0001234567], dtype=torch.float32)
    actual = dequantize_int8_reference(weight, scale)
    torch.testing.assert_close(
        actual, (weight.double() * scale.double()[:, None]).half(), rtol=0, atol=0
    )


def test_convrot_matches_kronecker_and_inverse():
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float64,
    )
    h = h4
    for _ in range(3):
        h = torch.kron(h, h4)
    h /= 16
    x = torch.randn(3, 512)
    expected = (x.double().reshape(-1, 256) @ h).reshape(x.shape).float()
    torch.testing.assert_close(convrot_reference(x), expected, atol=5e-7, rtol=2e-5)
    torch.testing.assert_close(
        convrot_reference(convrot_reference(x)), x, atol=8e-7, rtol=3e-5
    )


def test_original_qkv_reorder_keeps_weights_and_row_scales_together():
    rows = torch.arange(3 * 4 * 8)
    expected = rows.reshape(4, 3, 8).permute(1, 0, 2).reshape(-1)
    for value in (rows, rows[:, None].repeat(1, 3)):
        actual = _reorder_grouped_qkv_to_qkv(
            value, num_query_groups=4, heads_per_group=1, head_dim=8
        )
        torch.testing.assert_close(actual, value[expected])


def test_pruned_adaln_interpolation_endpoints_and_midpoints():
    from types import SimpleNamespace

    table = torch.tensor([[0.0, 4.0], [2.0, 0.0], [8.0, 6.0]])
    model = SimpleNamespace(
        arch=SimpleNamespace(adaln_curve_grid=3), adaln_t_table=table
    )
    actual = MiniMaxH3DiTModel._embed_timesteps(
        model, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    )
    torch.testing.assert_close(
        actual,
        torch.tensor([[0.0, 4.0], [1.0, 2.0], [2.0, 0.0], [5.0, 3.0], [8.0, 6.0]]),
    )


def test_checkpoint_metadata_and_partition_validation(tmp_path):
    path = tmp_path / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    marker = json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    ).encode()
    save_file(
        {
            "blocks.0.attn.qkv_proj.weight": torch.ones(12, 256, dtype=torch.int8),
            "blocks.0.attn.qkv_proj.weight_scale": torch.ones(12, 1),
            "blocks.0.attn.qkv_proj.comfy_quant": torch.tensor(
                list(marker), dtype=torch.uint8
            ),
            "adaln_t_table": torch.zeros(1025, 8),
        },
        path,
    )
    config = inspect_comfy_checkpoint(path, expected_partition="fl2va")
    assert config.arch_overrides == {"adaln_curve_grid": 1025, "adaln_curve_dim": 8}
    with pytest.raises(ValueError, match="cannot serve"):
        inspect_comfy_checkpoint(path, expected_partition="ref2va")


def test_primary_workload_schedule_and_shape():
    planner = MINIMAX_H3_SHAPE_PLANNER
    assert planner.align_frame_count(240) == 243
    assert planner.video_latent_t(243) == 72
    assert planner.audio_latent_t(243 / 24) == 405
    sigmas = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=12)
    assert len(sigmas) - 1 == 49
    assert sigmas[0] == 1 and sigmas[-1] == 0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


@pytest.mark.parametrize(
    "frames,duration,expected",
    [
        (22, None, (22, 7, 37)),
        (39, None, (39, 12, 65)),
        (243, 1, (39, 12, 65)),
        (243, 2, (56, 17, 93)),
        (362, None, (362, 107, 603)),
    ],
)
def test_short_clip_shape_and_audio_alignment(frames, duration, expected):
    from vllm.model_executor.models.minimax_h3.config import H3SamplingParams
    from vllm.model_executor.models.minimax_h3.pipeline import MiniMaxH3Pipeline

    sampling = H3SamplingParams(
        num_frames=frames,
        extra_args={} if duration is None else {"duration_seconds": duration},
    )
    shape = MiniMaxH3Pipeline._resolve_shape(None, "t2va", sampling, None)
    assert shape == (768, 1344, *expected)


@pytest.mark.parametrize(
    "values", [{"num_frames": 21}, {"extra_args": {"duration_seconds": 0.5}}]
)
def test_short_clip_rejects_incomplete_vae_temporal_chunk(values):
    from vllm.model_executor.models.minimax_h3.config import H3SamplingParams

    with pytest.raises(ValueError, match="22"):
        H3SamplingParams(**values)


@pytest.mark.parametrize("height,width", [(768, 1344), (256, 256)])
def test_native_canvas_does_not_require_omni_aspect_ratio(height, width):
    from vllm.model_executor.models.minimax_h3.config import H3SamplingParams
    from vllm.model_executor.models.minimax_h3.pipeline import MiniMaxH3Pipeline

    shape = MiniMaxH3Pipeline._resolve_shape(
        None, "t2va", H3SamplingParams(height=height, width=width), None
    )
    assert shape == (height, width, 243, 72, 405)


@pytest.mark.parametrize("used,padded", [(31, 32), (129, 256)])
def test_attention_padding_excludes_poisoned_suffix(used, padded):
    token = attention_backend.set("TORCH_SDPA")
    try:
        attention = Attention(
            num_heads=2, num_kv_heads=2, head_size=128, softmax_scale=128**-0.5
        )
    finally:
        attention_backend.reset(token)
    q, k, v = [torch.randn(1, padded, 2, 128) for _ in range(3)]
    expected = chunked_attention_reference(
        q[:, :used], k[:, :used], v[:, :used], scale=128**-0.5
    )
    k[:, used:] = float("nan")
    v[:, used:] = float("nan")
    actual = attention(q, k, v, AttentionMetadata(extra={"valid_kv_length": used}))
    torch.testing.assert_close(actual[:, :used], expected)
    assert torch.count_nonzero(actual[:, used:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("length", [127, 128, 129, 191, 192, 193, 12323])
def test_flashinfer_online_softmax_across_tiles_and_batches(length):
    from vllm.model_executor.models.minimax_h3.cuda_ops import flashinfer_extension

    torch.manual_seed(42)
    q, k, v = [
        torch.randn(2, length, 2, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    # Later key tiles raise the softmax maximum and exercise accumulator
    # rescaling. Check spread-out query rows without a full square matrix.
    k[:, length // 2 :] *= 4
    rows = torch.linspace(0, length - 1, min(length, 65), device="cuda").long()
    expected = chunked_attention_reference(q[:, rows], k, v, scale=128**-0.5)
    actual = flashinfer_extension().forward(q, k, v, 128**-0.5)
    torch.testing.assert_close(actual[:, rows], expected, atol=0.002, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "length", [31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 385]
)
def test_flashinfer_prefetch_tail_and_unaligned_storage(length):
    from vllm.model_executor.models.minimax_h3.cuda_ops import flashinfer_extension

    torch.manual_seed(42)
    shape = (2, length, 2, 128)
    count = 2 * length * 2 * 128
    q, k, v = [
        torch.randn(count + offset, device="cuda", dtype=torch.float16)[
            offset:
        ].reshape(shape)
        for offset in (1, 3, 5)
    ]
    # The next K/V tile can be absent, partial or complete. Storage offsets
    # also exercise the scalar load path without changing contiguous layout.
    expected = chunked_attention_reference(q, k, v, scale=128**-0.5)
    actual = flashinfer_extension().forward(q, k, v, 128**-0.5)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("length", [129, 193, 257, 385])
def test_flashinfer_query_groups_have_independent_softmax_state(length):
    from vllm.model_executor.models.minimax_h3.cuda_ops import flashinfer_extension

    torch.manual_seed(2026)
    q, k, v = [
        torch.randn(2, length, 3, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    # Adjacent 16-query groups have different maxima and rescaling histories.
    # The last query tile is partial; its inactive rows still join barriers.
    q[:, 16::32] *= 4
    k[:, 64:128] *= 4
    k[:, 128:] *= 2
    expected = chunked_attention_reference(q, k, v, scale=128**-0.5)
    actual = flashinfer_extension().forward(q, k, v, 128**-0.5)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_flashinfer_graph_replay_uses_current_inputs():
    from vllm.model_executor.models.minimax_h3.cuda_ops import flashinfer_extension

    torch.manual_seed(123)
    q, k, v = [
        torch.randn(1, 193, 2, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    op = flashinfer_extension()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.forward(q, k, v, 128**-0.5)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op.forward(q, k, v, 128**-0.5)
    for multiplier in (1.0, 3.0):
        q.mul_(multiplier)
        k[:, 64:].neg_()
        graph.replay()
        expected = chunked_attention_reference(q, k, v, scale=128**-0.5)
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("backend", ["FLASH_ATTN_V100", "FLASHINFER_SM70"])
def test_noncausal_backend_matches_fp32_reference(backend):
    token = attention_backend.set(backend)
    try:
        attention = Attention(
            num_heads=14, num_kv_heads=14, head_size=128, softmax_scale=128**-0.5
        )
    finally:
        attention_backend.reset(token)
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(1, 257, 14, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    expected = chunked_attention_reference(
        q[:, :243], k[:, :243], v[:, :243], scale=128**-0.5
    )
    actual = attention(q, k, v, AttentionMetadata(extra={"valid_kv_length": 243}))
    torch.testing.assert_close(actual[:, :243], expected, atol=8e-4, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_sm70_w8a16_uses_signed_scales_and_fp32_gemm_reduction():
    from vllm.model_executor.models.minimax_h3.cuda_ops import w8a16_extension

    ops = w8a16_extension()
    torch.manual_seed(42)
    x = torch.randn(65, 512, device="cuda", dtype=torch.float16)
    weight = torch.randint(-128, 128, (257, 512), device="cuda", dtype=torch.int8)
    scale = torch.rand(257, device="cuda") * 0.001 + 0.0001
    rotated = ops.rotate(x)
    decoded = ops.dequantize(weight, scale)
    torch.testing.assert_close(rotated, convrot_reference(x), atol=0, rtol=0)
    torch.testing.assert_close(
        decoded, dequantize_int8_reference(weight, scale), atol=0, rtol=0
    )
    result = ops.gemm(rotated, decoded)
    reference = (rotated.float() @ decoded.float().T).half()
    torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("groups", [0, 1, 3, 4, 5, 262141])
def test_sm70_convrot_warp_tails_and_grid_stride(groups):
    from vllm.model_executor.models.minimax_h3.cuda_ops import w8a16_extension

    torch.manual_seed(42)
    # Offset storage also verifies scalar FP16 loads do not require vector
    # alignment. The largest case crosses the grid's four-warps/block cap.
    storage = torch.randn(groups * 256 + 1, device="cuda", dtype=torch.float16)
    x = storage[1:].view(groups, 256)
    actual = w8a16_extension().rotate(x)
    torch.testing.assert_close(actual, convrot_reference(x), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("shape", [(0, 256), (1, 1), (17, 31), (19, 7168), (65536, 1)])
def test_fp16_preparation_preserves_power_of_two_scales(shape):
    from vllm.model_executor.models.minimax_h3.quantization import fp16_gemm_input

    torch.manual_seed(42)
    x = torch.randn(*shape, device="cuda")
    exponents = torch.arange(shape[0], device="cuda") % 140 - 20
    x = torch.ldexp(x, exponents[:, None])
    if shape[0] and shape[1] > 1:
        x[0] = 0
        # Exact threshold values and their immediate neighbors exercise the
        # frexp exponent transition used to leave room for ConvRot.
        if shape[0] > 4:
            x[1] = 2048
            x[2] = torch.nextafter(x[1], torch.zeros_like(x[1]))
            x[3] = torch.nextafter(x[1], torch.full_like(x[1], float("inf")))
            x[4] = torch.finfo(torch.float32).max
    maximum = x.abs().amax(-1, keepdim=True)
    _, exponent = torch.frexp(maximum)
    expected_scale = torch.ldexp(torch.ones_like(maximum), (exponent - 11).clamp_min(0))
    expected = (x / expected_scale).half()
    actual, scale = fp16_gemm_input(x)
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fp16_preparation_keeps_nonfinite_values_visible():
    from vllm.model_executor.models.minimax_h3.quantization import fp16_gemm_input

    x = torch.tensor(
        [[float("nan"), 1e10], [float("inf"), 4096], [-float("inf"), -0.0]],
        device="cuda",
    )
    _, exponent = torch.frexp(x.abs().amax(-1, keepdim=True))
    expected_scale = torch.ldexp(
        torch.ones_like(exponent, dtype=torch.float32), (exponent - 11).clamp_min(0)
    )
    actual, scale = fp16_gemm_input(x)
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
    torch.testing.assert_close(
        actual, (x / expected_scale).half(), atol=0, rtol=0, equal_nan=True
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_encoder_causal_gqa_matches_fp32():
    from vllm.model_executor.models.minimax_h3.encoder import (
        _scaled_dot_product_attention,
    )

    q = torch.randn(1, 16, 127, 128, device="cuda", dtype=torch.float16)
    k, v = [
        torch.randn(1, 2, 127, 128, device="cuda", dtype=torch.float16)
        for _ in range(2)
    ]
    actual = _scaled_dot_product_attention(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(8, 1),
        v.float().repeat_interleave(8, 1),
        is_causal=True,
    ).half()
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)


def test_encoder_residual_rmsnorm_preserves_residual_rounding():
    from vllm.model_executor.models.minimax_h3.ops import RMSNorm

    norm = RMSNorm(128, eps=1e-6, dtype=torch.float16)
    x, residual = [torch.randn(3, 128, dtype=torch.float16) for _ in range(2)]
    expected_residual = residual + x
    value = expected_residual.float()
    expected = (
        value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
    ).half()
    output, updated = norm(x, residual)
    torch.testing.assert_close(updated, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(output, expected)


def test_encoder_uses_functional_all_reduce_return():
    from types import SimpleNamespace

    from torch import nn

    from vllm.model_executor.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLRowParallelLinear,
        MiniMaxH3Qwen3VLVocabParallelEmbedding,
    )

    # Native GroupCoordinator may return a new tensor without modifying input.
    group = SimpleNamespace(
        rank_in_group=0, world_size=2, all_reduce=lambda value: value + 3
    )
    embedding = MiniMaxH3Qwen3VLVocabParallelEmbedding(group, 8, 4, torch.float32)
    embedding.weight.data.fill_(1)
    actual = embedding(torch.tensor([0, 6]))
    torch.testing.assert_close(
        actual, torch.tensor([[4.0, 4.0, 4.0, 4.0], [3.0, 3.0, 3.0, 3.0]])
    )
    projection = MiniMaxH3Qwen3VLRowParallelLinear.__new__(
        MiniMaxH3Qwen3VLRowParallelLinear
    )
    nn.Module.__init__(projection)
    projection.input_is_parallel = True
    projection._tp_size = 2
    projection.group = group
    projection.output_dtype = torch.float16
    projection.quant_method = SimpleNamespace(apply=lambda layer, value: value * 2)
    value = torch.ones(2, 4, dtype=torch.float16)
    torch.testing.assert_close(projection(value), value * 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("pin_memory", [True, False])
def test_staging_preserves_aliased_weights_across_repeated_transfers(pin_memory):
    from torch import nn

    from vllm.model_executor.models.minimax_h3.residency import PinnedModuleStager

    module = nn.Module()
    backing = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    module.weight = nn.Parameter(backing)
    module.register_buffer("view", backing[3:7, 1:5])
    expected = module.view.clone()
    stager = PinnedModuleStager(module, torch.device("cuda"), pin_memory=pin_memory)
    for _ in range(2):
        stager.load()
        assert module.weight.is_cuda and module.view.is_cuda
        assert (
            module.weight.untyped_storage().data_ptr()
            == module.view.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(module.view.cpu(), expected)
        stager.offload()
        assert module.weight.is_pinned() is pin_memory
        assert module.view.is_pinned() is pin_memory
        torch.testing.assert_close(module.view, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fp32_residual_is_not_rounded_through_fp16_before_normalization():
    from vllm.model_executor.models.minimax_h3.modulation import (
        indexed_gate_rms_norm_scale_shift,
    )

    torch.manual_seed(42)
    residual = torch.randn(17, 256, device="cuda") * 100000
    branch = torch.randn_like(residual) * 10000
    gate = torch.randn(3, 256, device="cuda")
    weight = torch.ones(256, device="cuda", dtype=torch.float16)
    shift, scale = [torch.randn_like(gate) * 0.1 for _ in range(2)]
    indices = torch.arange(17, device="cuda") % 3
    expected = residual + gate[indices] * branch
    normalized = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-5)
    modulated = (normalized * (1 + scale[indices]) + shift[indices]).half()
    actual, actual_modulated = indexed_gate_rms_norm_scale_shift(
        residual,
        gate,
        branch,
        weight,
        shift,
        scale,
        indices,
        1e-5,
        output_dtype=torch.float16,
    )
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(actual_modulated).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=0.02)
    torch.testing.assert_close(actual_modulated, modulated, rtol=0.002, atol=0.002)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_tensor_core_projection_can_return_values_above_fp16_range():
    from vllm.model_executor.models.minimax_h3.cuda_ops import w8a16_extension

    x = torch.full((17, 512), 300.0, device="cuda", dtype=torch.float16)
    weight = torch.ones(256, 512, device="cuda", dtype=torch.float16)
    output = w8a16_extension().gemm(x, weight, True)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, x.float() @ weight.float().T, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_convrot_projection_restores_wide_activation_row_scales():
    from types import SimpleNamespace

    from vllm.model_executor.models.minimax_h3.quantization import (
        DiffusionInt8ConvRotConfig,
        Int8ConvRotLayerConfig,
        Int8ConvRotLinearMethod,
    )

    torch.manual_seed(42)
    layer = SimpleNamespace(
        weight=torch.randint(-3, 4, (256, 512), device="cuda", dtype=torch.int8),
        weight_scale=torch.full((256,), 1 / 128, device="cuda"),
        h3_output_fp32=True,
    )
    x = torch.randint(-16, 17, (17, 512), device="cuda").float() * 8192
    method = Int8ConvRotLinearMethod(
        DiffusionInt8ConvRotConfig(),
        Int8ConvRotLayerConfig(convrot=True),
        prefix="blocks.0.mlp.fc2",
    )
    output = method.apply(layer, x)
    weight = dequantize_int8_reference(layer.weight, layer.weight_scale)
    expected = convrot_reference(x) @ weight.float().T
    assert output.dtype == torch.float32 and torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
