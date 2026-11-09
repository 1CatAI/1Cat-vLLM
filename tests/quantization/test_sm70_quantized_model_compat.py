# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.attention import attention as attention_module
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsKVCacheMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models import qwen3_dflash2 as dflash2_module


def _compressed_config(strategy, method_cls=CompressedTensorsKVCacheMethod):
    scheme = None
    if strategy is not None:
        scheme = dict(type="float", num_bits=8, strategy=strategy, symmetric=True)
    config = SimpleNamespace(kv_cache_scheme=scheme)
    config.get_quant_method = lambda layer, prefix: method_cls(config)
    return config


def _set_platform(monkeypatch, capability=70, enabled=True, cuda=True):
    monkeypatch.setattr(
        attention_module,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: cuda,
            has_device_capability=lambda required: capability >= required,
        ),
    )
    monkeypatch.setattr(attention_module.envs, "VLLM_SM70_FLASH_ATTN_V100", enabled)


@pytest.mark.parametrize("strategy", [None, "tensor", "attn_head"])
@pytest.mark.parametrize("kv_dtype", ["fp8_e5m2", "fp8_e4m3"])
def test_compressed_checkpoint_scale_processing(monkeypatch, strategy, kv_dtype):
    _set_platform(monkeypatch)
    layer = torch.nn.Module()
    layer.kv_cache_dtype = kv_dtype
    layer.num_kv_heads = 4
    attention_module._init_kv_cache_quant(layer, _compressed_config(strategy), "attn")
    for name, value in (("k", 7.0), ("v", 11.0), ("q", 13.0)):
        getattr(layer, f"{name}_scale").data.fill_(value)
    layer.quant_method.process_weights_after_loading(layer)
    for name, loaded in (("k", 7.0), ("v", 11.0), ("q", 13.0)):
        expected = 1.0 if kv_dtype == "fp8_e5m2" else loaded
        actual = getattr(layer, f"_{name}_scale")
        torch.testing.assert_close(actual, torch.full_like(actual, expected))
        assert getattr(layer, f"_{name}_scale_float") == expected
        assert not hasattr(layer, f"{name}_scale")
        assert not hasattr(layer, f"{name}_zero_point")
    assert layer._prob_scale.item() == 1.0


@pytest.mark.parametrize(
    "capability,enabled,cuda",
    [(75, True, True), (80, True, True), (70, False, True), (70, True, False)],
)
def test_compressed_e5m2_rejects_unsupported_route(
    monkeypatch, capability, enabled, cuda
):
    _set_platform(monkeypatch, capability, enabled, cuda)
    layer = torch.nn.Module()
    layer.kv_cache_dtype = "fp8_e5m2"
    with pytest.raises(ValueError, match="unit-scale override"):
        attention_module._init_kv_cache_quant(
            layer, _compressed_config("tensor"), "attn"
        )
    assert not hasattr(layer, "_force_unit_fp8_e5m2_kv_scales")


def test_compressed_subclass_cannot_bypass_scale_processing_guard(monkeypatch):
    class DifferentScaleMethod(CompressedTensorsKVCacheMethod):
        def process_weights_after_loading(self, layer):
            pass

    _set_platform(monkeypatch)
    layer = torch.nn.Module()
    layer.kv_cache_dtype = "fp8_e5m2"
    with pytest.raises(ValueError, match="unit-scale override"):
        attention_module._init_kv_cache_quant(
            layer, _compressed_config("tensor", DifferentScaleMethod), "attn"
        )


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("softcap", [None, 3.0])
@pytest.mark.parametrize("quantized", [False, True])
def test_dflash2_dense_candidates_padding_tp_and_scaling(
    monkeypatch, world_size, softcap, quantized
):
    monkeypatch.setattr(dflash2_module.envs, "VLLM_SM70_DFLASH2_QUANT_LM_HEAD", True)
    monkeypatch.setattr(
        dflash2_module, "get_tensor_model_parallel_world_size", lambda: world_size
    )
    # Padded entries would win both rows if masking were omitted.
    local_logits = torch.tensor([[1, 4, 2, 99, 100], [5, 1, 3, 98, 99.0]])
    apply = Mock(return_value=local_logits.clone())
    method = SimpleNamespace(apply=apply) if quantized else UnquantizedEmbeddingMethod()
    method.apply = apply
    fast_path = Mock(return_value=None)
    if quantized:
        fast_path.side_effect = AssertionError("quantized heads must use dense apply")
    lm_head = SimpleNamespace(
        quant_method=method,
        maybe_get_sm70_dflash2_top20=fast_path,
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=2, org_vocab_start_index=10
        ),
    )
    outer = SimpleNamespace(
        lm_head=lm_head,
        model=SimpleNamespace(candidate_selector=SimpleNamespace(top_k=2)),
        output_multiplier=0.5,
        final_logit_softcapping=softcap,
    )
    local_values, local_ids = torch.topk(local_logits[:, :3], 2, dim=-1)
    shard_scores = [local_logits[:, :3]]
    shard_ids = [torch.tensor([[10, 11, 12], [10, 11, 12]])]
    for rank in range(1, world_size):
        shard_scores.append(torch.tensor([[2.5, 6.0, 0.5], [4.0, 0.5, 7.0]]) + rank)
        shard_ids.append(torch.tensor([[20, 21, 22], [20, 21, 22]]) + rank * 10)

    def all_gather(tensor, dim):
        assert dim == -1
        if tensor.dtype == torch.int64:
            torch.testing.assert_close(tensor, local_ids + 10)
        else:
            torch.testing.assert_close(tensor, local_values)
        parts = [tensor]
        for scores, ids in zip(shard_scores[1:], shard_ids[1:]):
            values, indices = torch.topk(scores, 2, dim=-1)
            parts.append(
                ids.gather(-1, indices) if tensor.dtype == torch.int64 else values
            )
        return torch.cat(parts, dim=-1)

    gather = Mock(side_effect=all_gather)
    monkeypatch.setattr(dflash2_module, "tensor_model_parallel_all_gather", gather)
    hidden = torch.zeros((2, 8))
    ids, values = dflash2_module.DFlash2Qwen3ForCausalLM.compute_candidates(
        outer, hidden
    )
    expected, indices = torch.topk(torch.cat(shard_scores, dim=-1), 2, dim=-1)
    expected_ids = torch.cat(shard_ids, dim=-1).gather(-1, indices)
    expected *= 0.5
    if softcap is not None:
        expected = torch.tanh(expected / softcap) * softcap
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(values, expected)
    apply.assert_called_once_with(lm_head, hidden, bias=None)
    assert gather.call_count == (2 if world_size > 1 else 0)
    if quantized:
        fast_path.assert_not_called()
    else:
        fast_path.assert_called_once_with(hidden, 2)


def test_quantized_dflash2_head_keeps_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(dflash2_module.envs, "VLLM_SM70_DFLASH2_QUANT_LM_HEAD", False)
    apply = Mock()
    outer = SimpleNamespace(
        lm_head=SimpleNamespace(quant_method=SimpleNamespace(apply=apply))
    )
    with pytest.raises(ValueError, match="VLLM_SM70_DFLASH2_QUANT_LM_HEAD=1"):
        dflash2_module.DFlash2Qwen3ForCausalLM.compute_candidates(
            outer, torch.zeros((1, 8))
        )
    apply.assert_not_called()
