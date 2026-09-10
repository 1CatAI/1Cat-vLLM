# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
import textwrap

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.mtp_fp8_checkpoint import (
    pad_mtp_fp8_checkpoint_matrix,
    padded_mtp_fp8_width,
)


@pytest.fixture
def should_do_global_cleanup_after_test():
    return False


def reconstruct(weight, scale):
    scales = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight.float() * scales[: weight.shape[0], : weight.shape[1]]


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("down", [False, True])
def test_checkpoint_shards_preserve_fp8_bytes_and_original_block_scales(tp_size, down):
    torch.manual_seed(37)
    shape = (256, 640) if down else (640, 256)
    weight = torch.randint(-16, 16, shape).to(torch.float8_e4m3fn)
    scale_shape = tuple(dim // 128 for dim in shape)
    # Distinct blocks make incorrect TP offsets visible in the reconstruction.
    scale = torch.arange(1, 11, dtype=torch.float32).reshape(scale_shape) / 16
    original = weight.view(torch.uint8).clone()
    original_scale = scale.clone()
    padded_weight, padded_scale = pad_mtp_fp8_checkpoint_matrix(
        weight, scale, down_projection=down, tp_size=tp_size
    )
    axis = 1 if down else 0
    width = padded_mtp_fp8_width(640 // tp_size, tp_size)
    reference = reconstruct(weight, scale)
    for rank in range(tp_size):
        start = rank * (640 // tp_size)
        offset = start % 128
        local_w = padded_weight.narrow(axis, rank * width, width)
        local_s = padded_scale.narrow(axis, rank * width // 128, width // 128)
        actual = reconstruct(local_w, local_s)
        valid = actual.narrow(axis, offset, 640 // tp_size)
        torch.testing.assert_close(
            valid, reference.narrow(axis, start, 640 // tp_size), rtol=0, atol=0
        )
        mask = actual.clone()
        mask.narrow(axis, offset, 640 // tp_size).zero_()
        assert torch.count_nonzero(mask) == 0
    assert torch.equal(weight.view(torch.uint8), original)
    assert torch.equal(scale, original_scale)


def test_tp4_width_retains_block_offsets_without_extra_quantization():
    assert padded_mtp_fp8_width(160, 4) == 256
    assert [(rank * 160) % 128 for rank in range(4)] == [0, 32, 64, 96]


@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), -1.0])
def test_bad_checkpoint_scales_are_rejected(bad_scale):
    weight = torch.zeros((640, 256), dtype=torch.float8_e4m3fn)
    scale = torch.ones((5, 2))
    scale[1, 0] = bad_scale
    with pytest.raises(ValueError, match="scales must be finite and nonnegative"):
        pad_mtp_fp8_checkpoint_matrix(weight, scale, down_projection=False, tp_size=4)


@pytest.mark.parametrize("bad_scale", [1e10, 1e-15])
def test_scales_that_overflow_or_underflow_in_the_sm70_kernel_are_rejected(bad_scale):
    weight = torch.ones((640, 256), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.full((5, 2), bad_scale)
    with pytest.raises(ValueError, match="outside the SM70 FP16 scale range"):
        pad_mtp_fp8_checkpoint_matrix(weight, scale, down_projection=False, tp_size=4)


@pytest.mark.parametrize("algo", ["FP8_PB_WO", "FP8_BLOCK_SCALES"])
def test_upstream_modelopt_dispatch_and_sm70_checkpoint_selection(algo):
    from unittest.mock import MagicMock, patch

    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )
    from vllm.models.qwen4_exp.nvidia.mtp_fp8_experts import checkpoint_fp8_prefixes

    prefix = "mtp.layers.48.mlp.experts"
    config = ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {prefix: {"quant_algo": algo, "group_size": 128}},
            }
        }
    )
    layer = MagicMock(spec=RoutedExperts)
    with patch(
        "vllm.model_executor.layers.quantization.modelopt.Fp8MoEMethod"
    ) as method:
        assert config.get_quant_method(layer, prefix) is method.return_value
        assert method.call_args.args[0] is config.fp8_block_config
    assert config.fp8_block_config.weight_block_size == [128, 128]
    assert config.has_blocked_weights()
    assert checkpoint_fp8_prefixes(config, {prefix}) == {prefix}
    config.exclude_modules = [prefix]
    assert config.get_quant_method(layer, prefix) is None
    assert checkpoint_fp8_prefixes(config, {prefix}) == set()


@pytest.mark.parametrize("backend", ["amd", "nvidia"])
def test_upstream_mtp_metadata_remapping_preserves_main_layers(backend):
    # Both backends register the same custom-op names. Import them in separate
    # interpreters, as in the upstream test, without importing API-server helpers.
    script = textwrap.dedent("""
        import sys
        from importlib import import_module
        module = import_module(f"vllm.models.qwen4_exp.{sys.argv[1]}.mtp")
        main = {"quant_algo": "NVFP4"}
        draft = {"quant_algo": "FP8_BLOCK_SCALES", "group_size": 128}
        source = {"model.layers.0.mlp.experts": main,
                  "mtp.layers.0.mlp.experts": draft}
        actual = module._remap_quantized_layers(source, 48)
        assert actual == {"model.layers.0.mlp.experts": main,
                          "mtp.layers.48.mlp.experts": draft}
        assert "mtp.layers.0.mlp.experts" in source
    """)
    subprocess.run([sys.executable, "-c", script, backend], check=True)


def checkpoint_pairs():
    prefix = "model.layers.0.mlp.experts"
    rows = []
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = (256, 640) if projection == "down_proj" else (640, 256)
            weight = torch.ones(shape, dtype=torch.float32).to(torch.float8_e4m3fn)
            scale = torch.full(tuple(dim // 128 for dim in shape), expert + 0.5)
            base = f"{prefix}.{expert}.{projection}"
            rows.extend(
                [(base + ".weight", weight), (base + ".weight_scale_inv", scale)]
            )
    return prefix, rows


def test_checkpoint_stream_orders_preserve_pairs_and_unrelated_tensors():
    from vllm.models.qwen4_exp.nvidia.mtp_fp8_checkpoint import (
        prepare_mtp_fp8_checkpoint,
    )

    prefix, rows = checkpoint_pairs()
    head = torch.ones(2, 2)
    for weights in (rows, list(reversed(rows))):
        actual = dict(
            prepare_mtp_fp8_checkpoint(
                [("lm_head.weight", head), *weights], {prefix}, tp_size=4, num_experts=2
            )
        )
        assert actual["lm_head.weight"] is head
        assert len(actual) == 13
        assert actual[prefix + ".0.gate_proj.weight"].dtype == torch.float8_e4m3fn
        assert actual[prefix + ".0.gate_proj.weight"].shape == (1024, 256)


@pytest.mark.parametrize("failure", ["missing_pair", "duplicate", "extra_expert"])
def test_checkpoint_stream_rejects_incomplete_or_duplicate_experts(failure):
    from vllm.models.qwen4_exp.nvidia.mtp_fp8_checkpoint import (
        prepare_mtp_fp8_checkpoint,
    )

    prefix, rows = checkpoint_pairs()
    if failure == "missing_pair":
        rows = rows[2:]
    elif failure == "duplicate":
        rows += rows[:2]
    else:
        rows += [(prefix + ".2.gate_proj.weight", rows[0][1])]
    with pytest.raises(ValueError):
        list(prepare_mtp_fp8_checkpoint(rows, {prefix}, tp_size=4, num_experts=2))


def test_checkpoint_selection_respects_unquantized_expert_exclusions():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.models.qwen4_exp.nvidia.mtp_fp8_experts import checkpoint_fp8_prefixes

    prefix = "mtp.layers.48.mlp.experts"
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
        ignored_layers=[prefix],
    )
    assert checkpoint_fp8_prefixes(config, {prefix}) == set()
    assert checkpoint_fp8_prefixes(config, {prefix, "mtp.layers.49.mlp.experts"}) == {
        "mtp.layers.49.mlp.experts"
    }


@pytest.mark.parametrize("quant_name", ["awq", "modelopt_fp4", "modelopt_mixed"])
@pytest.mark.parametrize("rank", range(4))
def test_native_checkpoint_uses_normal_tp_loader_with_each_target_format(
    quant_name, rank
):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.models.qwen4_exp.nvidia.mtp_fp8_experts import (
        MTPCheckpointFp8SM70MoEMethod,
    )

    # Exercise the real parameter allocator and weight_loader without a GPU
    # process group or a runner. The target's quantization name must not cause
    # the MTP FP8 weight/scale tensors to take a ModelOpt FP4 loading branch.
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    parallel = SimpleNamespace(tp_size=4, tp_rank=rank)
    layer.moe_parallel_config = parallel
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=parallel, has_bias=False, is_act_and_mul=True
    )
    layer.quant_config = SimpleNamespace(get_name=lambda: quant_name)
    layer.expert_map_manager = SimpleNamespace(
        map_global_to_local=lambda expert: expert
    )
    method = MTPCheckpointFp8SM70MoEMethod(layer)
    layer.quant_method = method
    h, i = method.maybe_roundup_sizes(256, 160, torch.float16, parallel)
    method.create_weights(layer, 2, h, i, torch.float16)
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.float8_e4m3fn
    prefix, rows = checkpoint_pairs()
    raw = dict(rows)
    for name, tensor in rows:
        if not name.endswith(".weight"):
            continue
        base = name.removesuffix(".weight")
        expert = int(base.split(".")[-2])
        projection = base.split(".")[-1]
        down = projection == "down_proj"
        shard = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[projection]
        weight, scale = pad_mtp_fp8_checkpoint_matrix(
            tensor, raw[base + ".weight_scale_inv"], down_projection=down, tp_size=4
        )
        stem = "w2" if down else "w13"
        for suffix, value in [("weight", weight), ("weight_scale_inv", scale)]:
            param_name = stem + "_" + suffix
            assert layer.weight_loader(
                getattr(layer, param_name), value, param_name, shard, expert, True
            )
        actual = reconstruct(
            getattr(layer, stem + "_weight")[expert],
            getattr(layer, stem + "_weight_scale_inv")[expert],
        )
        if not down:
            actual = actual[:i] if shard == "w1" else actual[i:]
        axis = 1 if down else 0
        start = rank * 160
        actual = actual.narrow(axis, start % 128, 160)
        expected = reconstruct(tensor, raw[base + ".weight_scale_inv"]).narrow(
            axis, start, 160
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
