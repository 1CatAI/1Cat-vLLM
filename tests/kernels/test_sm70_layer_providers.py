# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.model_executor.kernels.lm_head.sm70 import VocabShard
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    _maybe_sm70_dflash2_qpn8_rerank,
    _maybe_sm70_lm_head_forward,
    _maybe_sm70_lm_head_top1,
    maybe_prepare_sm70_lm_head_top1,
)

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
        reason="requires SM70 GPU",
    ),
]


def make_head(monkeypatch, mode):
    for name, enabled in {
        "VLLM_SM70_ENABLE_LM_HEAD_FASTPATH": mode == "dense",
        "VLLM_SM70_LM_HEAD_TOP1": mode == "raw_top1",
        "VLLM_SM70_LM_HEAD_TOP1_TC": mode == "packed_top1",
        "VLLM_SM70_DFLASH2_FP32_LOGITS": mode in ("fp32", "rerank"),
        "VLLM_SM70_DFLASH2_QPN8_RERANK": mode == "rerank",
        "VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW": False,
    }.items():
        monkeypatch.setenv(name, str(int(enabled)))
    from vllm import envs

    envs.disable_envs_cache()
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    dflash = Sm70DFlash2Config()
    dflash.resolve(qualified=False)
    cfg.speculative_config = SimpleNamespace(sm70_dflash2=dflash)
    with set_current_vllm_config(cfg):
        layer = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        torch.nn.Module.__init__(layer)
        layer.prefix = "target.lm_head"
        layer.shard_indices = VocabShard()
        layer.weight = torch.nn.Parameter(
            torch.randn((128, 128), dtype=torch.float16, device="cuda") * 0.02,
            requires_grad=False,
        )
        assert maybe_prepare_sm70_lm_head_top1(layer)
    return layer


def bits(result):
    values = result if isinstance(result, tuple) else (result,)
    assert all(isinstance(x, torch.Tensor) for x in values)
    return tuple(x.detach().clone() for x in values)


@pytest.mark.parametrize(
    "mode,rows",
    [
        ("raw_top1", 1),
        ("packed_top1", 1),
        ("packed_top1", 8),
        ("dense", 1),
        ("dense", 8),
        ("fp32", 1),
        ("fp32", 8),
        ("rerank", 1),
        ("rerank", 8),
    ],
)
@torch.inference_mode()
def test_prepared_head_graph_updates_inputs_and_keeps_buffer_addresses(
    monkeypatch, mode, rows
):
    torch.manual_seed(710)
    layer = make_head(monkeypatch, mode)
    x = torch.randn((rows, 128), dtype=torch.float16, device="cuda")

    def run():
        if mode.endswith("top1"):
            return _maybe_sm70_lm_head_top1(layer, x)
        if mode == "rerank":
            return _maybe_sm70_dflash2_qpn8_rerank(layer, x, 16)
        return _maybe_sm70_lm_head_forward(layer, x)

    run()
    addresses = [b.data_ptr() for b in layer.buffers()]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for scale in (0.1, 0.0, 2.0):
        x.normal_(0, scale)
        graph.replay()
        actual = bits(result)
        expected = bits(run())
        for a, e in zip(actual, expected):
            assert torch.equal(a.view(torch.uint8), e.view(torch.uint8))
        assert [b.data_ptr() for b in layer.buffers()] == addresses
    if mode == "fp32":
        assert result.dtype == torch.float32
        assert torch.equal(
            result, torch.mm(x, layer.weight.t(), out_dtype=torch.float32)
        )
    empty = x[:0]
    if mode.endswith("top1"):
        assert _maybe_sm70_lm_head_top1(layer, empty) is None
    elif mode == "rerank":
        assert _maybe_sm70_dflash2_qpn8_rerank(layer, empty, 16) is None
    else:
        assert _maybe_sm70_lm_head_forward(layer, empty).shape == (0, 128)


@torch.inference_mode()
def test_sparse_rerank_keeps_dense_vocab_tie_order(monkeypatch):
    layer = make_head(monkeypatch, "rerank")
    x = torch.zeros((1, 128), dtype=torch.float16, device="cuda")
    values, ids = _maybe_sm70_dflash2_qpn8_rerank(layer, x, 16)
    state = layer._sm70_lm_head_state
    expected_values, expected_ids = torch.topk(
        state._sm70_dflash2_rerank_dense_logits[:1], 16, sorted=True
    )
    assert torch.equal(values, expected_values)
    assert torch.equal(ids, expected_ids)


@torch.inference_mode()
def test_dense_projection_loading_and_fullgraph_views(monkeypatch):
    import vllm.model_executor.parameter as parameters
    from vllm.model_executor.layers.linear import ReplicatedLinear

    monkeypatch.setattr(parameters, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameters, "get_tensor_model_parallel_world_size", lambda: 1)

    monkeypatch.setenv("VLLM_SM70_F16_DENSE_TUNE_MAX_M", "0")
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(cfg):
        layer = ReplicatedLinear(
            128,
            128,
            bias=False,
            params_dtype=torch.float16,
            prefix="model.down_proj",
            disable_tp=True,
        ).cuda()
        layer.weight.normal_(0, 0.02)
        layer._sm70_f16_force_enable = True
        layer.quant_method.process_weights_after_loading(layer)
        assert layer._sm70_dense_state.weight is layer.weight
        assert layer._sm70_f16_prepared
        compiled = torch.compile(layer, backend="eager", fullgraph=True)
        for rows in (1, 8, 1):
            x = torch.randn((rows, 128), dtype=torch.float16, device="cuda")
            expected, _ = layer(x)
            # Compiled engine calls borrow the runner's runtime scope. Eager
            # layer calls above activate their own bound owner automatically.
            from vllm._sm70.runtime import bind_native_runtime

            with bind_native_runtime().activate():
                actual, _ = compiled(x)
                flat, _ = compiled(x.view(1, rows, 128))
            assert torch.equal(actual, expected)
            assert torch.equal(flat.view_as(expected), expected)
