# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from benchmarks.benchmark_sm70_turbomind_exactness import (
    _prepare_awq_moe_checkpoint_for_tp,
)
from vllm import envs
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _QWEN38_INDEXED_PREFILL_MIN_TOKENS,
    AWQSM70MoEMethod,
    _use_qwen38_indexed_prefill,
)

pytestmark = pytest.mark.skip_global_cleanup


def _qwen38_layer() -> SimpleNamespace:
    return SimpleNamespace(
        moe_config=SimpleNamespace(tp_size=4),
        sm70_awq_qwen38_indexed_prefill=True,
        sm70_awq_moe_batched_gemm=True,
        sm70_awq_checkpoint_group_size=32,
        sm70_awq_group_size=32,
        sm70_num_experts=512,
        sm70_hidden_logical_size=2560,
        sm70_intermediate_size=160,
        sm70_w13_k_dim=2560,
        sm70_w13_n_dim=320,
    )


def test_qwen38_awq_indexed_prefill_env_defaults_on(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_AWQ_QWEN38_MOE_INDEXED_PREFILL", raising=False)
    assert envs.VLLM_SM70_AWQ_QWEN38_MOE_INDEXED_PREFILL

    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_MOE_INDEXED_PREFILL", "0")
    assert not envs.VLLM_SM70_AWQ_QWEN38_MOE_INDEXED_PREFILL


def test_qwen38_awq_indexed_prefill_gate_is_exact():
    layer = _qwen38_layer()
    x = torch.empty(_QWEN38_INDEXED_PREFILL_MIN_TOKENS, 2560, dtype=torch.float16)
    topk_ids = torch.empty(_QWEN38_INDEXED_PREFILL_MIN_TOKENS, 10, dtype=torch.int32)

    assert _use_qwen38_indexed_prefill(layer, x, topk_ids)
    assert not _use_qwen38_indexed_prefill(layer, x[:127], topk_ids[:127])
    assert not _use_qwen38_indexed_prefill(layer, x, topk_ids[:, :9])

    layer.moe_config.tp_size = 2
    assert not _use_qwen38_indexed_prefill(layer, x, topk_ids)
    layer.moe_config.tp_size = 4

    layer.sm70_awq_checkpoint_group_size = 128
    assert not _use_qwen38_indexed_prefill(layer, x, topk_ids)
    layer.sm70_awq_checkpoint_group_size = 32

    layer.sm70_awq_group_size = 128
    assert not _use_qwen38_indexed_prefill(layer, x, topk_ids)
    layer.sm70_awq_group_size = 32

    layer.sm70_awq_qwen38_indexed_prefill = False
    assert not _use_qwen38_indexed_prefill(layer, x, topk_ids)


def test_qwen38_awq_indexed_prefill_rejects_noncontiguous_input():
    layer = _qwen38_layer()
    x = torch.empty(2560, _QWEN38_INDEXED_PREFILL_MIN_TOKENS).t()
    topk_ids = torch.empty(_QWEN38_INDEXED_PREFILL_MIN_TOKENS, 10, dtype=torch.int32)

    assert x.shape == (_QWEN38_INDEXED_PREFILL_MIN_TOKENS, 2560)
    assert not x.is_contiguous()
    assert not _use_qwen38_indexed_prefill(layer, x, topk_ids)


def _synthetic_awq_checkpoint(group_size: int) -> tuple[torch.Tensor, ...]:
    groups_w13 = 2560 // group_size
    groups_w2 = 640 // group_size
    w13_qweight = (
        torch.arange(160, dtype=torch.int32).view(1, 1, 160).expand(1, 2560, 160)
    )
    w13_scales = (
        torch.arange(1280, dtype=torch.float16)
        .view(1, 1, 1280)
        .expand(1, groups_w13, 1280)
    )
    w13_qzeros = (
        torch.arange(groups_w13, dtype=torch.int32)
        .view(1, groups_w13, 1)
        .expand(1, groups_w13, 160)
    )
    w2_qweight = (
        torch.arange(640, dtype=torch.int32).view(1, 640, 1).expand(1, 640, 320)
    )
    w2_scales = (
        torch.arange(groups_w2, dtype=torch.float16)
        .view(1, groups_w2, 1)
        .expand(1, groups_w2, 2560)
    )
    w2_qzeros = (
        torch.arange(groups_w2, dtype=torch.int32)
        .view(1, groups_w2, 1)
        .expand(1, groups_w2, 320)
    )
    return (
        w13_qweight,
        w13_scales,
        w13_qzeros,
        w2_qweight,
        w2_scales,
        w2_qzeros,
    )


def test_awq_checkpoint_tp4_layout_matches_loader_axes():
    tensors = _prepare_awq_moe_checkpoint_for_tp(
        _synthetic_awq_checkpoint(32), 4, 3, 32
    )
    w13_qweight, w13_scales, _, w2_qweight, w2_scales, _, group_size = tensors

    assert group_size == 32
    assert w13_qweight.shape == (1, 2560, 40)
    assert w13_scales.shape == (1, 80, 320)
    assert w2_qweight.shape == (1, 160, 320)
    assert w2_scales.shape == (1, 5, 2560)
    assert torch.equal(
        w13_qweight[0, 0],
        torch.cat((torch.arange(60, 80), torch.arange(140, 160))).to(torch.int32),
    )
    assert torch.equal(w2_qweight[0, :, 0], torch.arange(480, 640).to(torch.int32))
    assert torch.equal(w2_scales[0, :, 0], torch.arange(15, 20).to(torch.float16))


def test_g128_checkpoint_tp4_is_rejected_by_service_loader_geometry():
    with pytest.raises(ValueError, match="divisible by the checkpoint group"):
        _prepare_awq_moe_checkpoint_for_tp(_synthetic_awq_checkpoint(128), 4, 0, 128)


class _FakeRoutedExpertsLoader:
    weight_loader = FusedMoE.weight_loader
    _map_global_expert_id_to_local_expert_id = (
        FusedMoE._map_global_expert_id_to_local_expert_id
    )
    _get_hidden_dim = staticmethod(FusedMoE._get_hidden_dim)
    _narrow_expert_data_for_padding = staticmethod(
        FusedMoE._narrow_expert_data_for_padding
    )
    _load_w13 = FusedMoE._load_w13
    _load_w2 = FusedMoE._load_w2
    _load_model_weight_or_group_weight_scale = (
        FusedMoE._load_model_weight_or_group_weight_scale
    )

    def __init__(self, tp_rank: int) -> None:
        self.quant_config = None
        self.quant_method = object.__new__(AWQSM70MoEMethod)
        self.quant_method.group_size_div_factor = 1
        self.expert_map_manager = SimpleNamespace(
            map_global_to_local=lambda expert_id: expert_id
        )
        self.moe_config = SimpleNamespace(
            is_act_and_mul=True, moe_parallel_config=SimpleNamespace(tp_size=4)
        )
        self.tp_rank = tp_rank


def _service_loaded_awq_tensors(
    checkpoint: tuple[torch.Tensor, ...], tp_rank: int
) -> tuple[torch.Tensor, ...]:
    w13_qweight, w13_scales, w13_qzeros, w2_qweight, w2_scales, w2_qzeros = checkpoint

    def _parameter(tensor: torch.Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(tensor, requires_grad=False)

    destinations = (
        _parameter(torch.zeros(1, 2560, 40, dtype=torch.int32)),
        _parameter(torch.zeros(1, 80, 320, dtype=torch.float16)),
        _parameter(torch.zeros(1, 80, 40, dtype=torch.int32)),
        _parameter(torch.zeros(1, 160, 320, dtype=torch.int32)),
        _parameter(torch.zeros(1, 5, 2560, dtype=torch.float16)),
        _parameter(torch.zeros(1, 5, 320, dtype=torch.int32)),
    )
    for param in destinations:
        param.is_transposed = True
        param.quant_method = FusedMoeWeightScaleSupported.GROUP.value

    loader = _FakeRoutedExpertsLoader(tp_rank)
    checkpoint_shards = (
        (w13_qweight[..., :80], "qweight", "w1", destinations[0]),
        (w13_qweight[..., 80:], "qweight", "w3", destinations[0]),
        (w13_scales[..., :640], "scales", "w1", destinations[1]),
        (w13_scales[..., 640:], "scales", "w3", destinations[1]),
        (w13_qzeros[..., :80], "qzeros", "w1", destinations[2]),
        (w13_qzeros[..., 80:], "qzeros", "w3", destinations[2]),
        (w2_qweight, "qweight", "w2", destinations[3]),
        (w2_scales, "scales", "w2", destinations[4]),
        (w2_qzeros, "qzeros", "w2", destinations[5]),
    )
    for loaded, name, shard_id, param in checkpoint_shards:
        loader.weight_loader(param, loaded[0], name, shard_id, 0)
    return tuple(param.detach() for param in destinations)


@pytest.mark.parametrize("tp_rank", [0, 3])
def test_benchmark_tp4_preparation_matches_fused_moe_weight_loader(tp_rank: int):
    checkpoint = _synthetic_awq_checkpoint(32)
    checkpoint = (
        checkpoint[0],
        checkpoint[1].to(torch.bfloat16),
        checkpoint[2],
        checkpoint[3],
        checkpoint[4].to(torch.bfloat16),
        checkpoint[5],
    )
    benchmark_input = (
        checkpoint[0],
        checkpoint[1].half(),
        checkpoint[2],
        checkpoint[3],
        checkpoint[4].half(),
        checkpoint[5],
    )
    prepared = _prepare_awq_moe_checkpoint_for_tp(benchmark_input, 4, tp_rank, 32)
    service_loaded = _service_loaded_awq_tensors(checkpoint, tp_rank)

    assert prepared[-1] == 32
    for benchmark_tensor, service_tensor in zip(prepared[:-1], service_loaded):
        assert torch.equal(benchmark_tensor, service_tensor)
