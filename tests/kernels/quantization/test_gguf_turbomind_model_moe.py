# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size


def packed_weight(source_type, rows, k, seed):
    block, size = quant_size(source_type)
    data = np.random.default_rng(seed).integers(
        0, 256, (rows, k // block, size), dtype=np.uint8
    )
    data[:, :, :2] = (
        np.full((rows, k // block), 0.03125, np.float16)
        .view(np.uint8)
        .reshape(rows, k // block, 2)
    )
    return data.reshape(rows, -1)


def test_canonical_adapter_retains_q2_source_without_expansion():
    from vllm.model_executor.model_loader.gguf_adapters.qwen4exp import Qwen4ExpAdapter

    config = SimpleNamespace(
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
        num_experts=2,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
    )
    adapter = Qwen4ExpAdapter(config, tp_size=4)
    adapter.canonical_expert_storage = True
    data = np.stack([packed_weight(42, 64, 640, i) for i in range(2)])
    raw = "blk.0.ffn_down_exps.weight"
    tensor = SimpleNamespace(shape=np.array([640, 64, 2]), tensor_type=42, data=data)
    result = list(
        adapter.weights(
            {raw: tensor}, adapter.build_name_map({raw: tensor}), torch.float16
        )
    )
    types = [w.item() for n, w in result if n.endswith("qweight_type")]
    weights = [w for n, w in result if n.endswith("qweight")]
    assert types == [42, 42]
    assert [tuple(w.shape) for w in weights] == [(64, 180), (64, 180)]
    assert weights[0].data_ptr() == torch.from_numpy(data[0]).data_ptr()


@pytest.mark.parametrize("storage", ["canonical", "original"])
def test_load_model_retains_prepared_adapter(monkeypatch, storage):
    from vllm.model_executor.kernels.ple import gguf_pinned
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.model_loader import gguf_loader as module
    from vllm.model_executor.model_loader.gguf_adapters.qwen4exp import Qwen4ExpAdapter

    config = SimpleNamespace(
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
        num_experts=2,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
    )
    data = np.stack([packed_weight(42, 64, 640, i) for i in range(2)])
    raw = "blk.0.ffn_down_exps.weight"
    tensor = SimpleNamespace(shape=np.array([640, 64, 2]), tensor_type=42, data=data)
    loader = object.__new__(module.GGUFModelLoader)
    calls = []

    def mapping(model_config):
        calls.append(model_config)
        loader._native_adapter = Qwen4ExpAdapter(config, tp_size=4)
        loader._native_tensors = {raw: tensor}
        return loader._native_adapter.build_name_map(loader._native_tensors)

    class CapturedModel(torch.nn.Module):
        def load_weights(self, weights):
            self.entries = list(weights)

    model = CapturedModel()
    monkeypatch.setattr(loader, "_prepare_weights", lambda _: "unused.gguf")
    monkeypatch.setattr(loader, "_get_gguf_weights_map", mapping)
    monkeypatch.setattr(loader, "_get_all_gguf_files", lambda _: [])
    monkeypatch.setattr(loader, "_get_gguf_weight_type", lambda *args: {raw: "Q2_0"})
    monkeypatch.setattr(module, "initialize_model", lambda **kwargs: model)
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: None)
    monkeypatch.setattr(gguf_pinned, "prepare_pinned_gguf_ple", lambda *args: None)
    monkeypatch.setattr(
        module,
        "current_platform",
        SimpleNamespace(get_device_capability=lambda: (7, 0)),
    )
    monkeypatch.setattr(
        torch.ops._C,
        "gguf_affine_grouped_gemm_sm70_out",
        lambda *args: None,
        raising=False,
    )
    cfg = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        kernel_config=SimpleNamespace(
            sm70_gguf=SimpleNamespace(enabled=True, expert_storage=storage)
        ),
        quant_config=GGUFConfig(),
    )
    model_config = SimpleNamespace(dtype=torch.float16, hf_config=SimpleNamespace())
    assert loader.load_model(cfg, model_config) is model
    assert cfg.quant_config.canonical_expert_storage == (storage == "canonical")
    types = [w.item() for n, w in model.entries if n.endswith("qweight_type")]
    weights = [w for n, w in model.entries if n.endswith("qweight")]
    assert types == ([42, 42] if storage == "canonical" else [3, 3])
    assert [tuple(w.shape) for w in weights] == (
        [(64, 180), (64, 180)] if storage == "canonical" else [(64, 400), (64, 400)]
    )
    if storage == "canonical":
        assert weights[0].data_ptr() == torch.from_numpy(data[0]).data_ptr()
    else:
        # The TP4 K160 boundary cuts a Q2_0 K64 block. The existing Q4_1
        # conversion keeps its FP16 coefficient and every integer code.
        for expert, weight in enumerate(weights):
            np.testing.assert_array_equal(
                dequantize(weight.numpy(), 3), dequantize(data[expert], 42)
            )
    assert len(calls) == 1
    assert (
        list(loader.get_all_weights(model_config, model, skip_weight=lambda _: True))
        == []
    )
    retained = list(loader.get_all_weights(model_config, model))
    assert [w.item() for n, w in retained if n.endswith("qweight_type")] == types
    assert len(calls) == 1


def reference_part(x, ids, scores, decoded, rank, size):
    tokens, topk = ids.shape
    intermediate = decoded["w1"].shape[1] // size
    selection = slice(rank * intermediate, (rank + 1) * intermediate)
    output = torch.empty((tokens, topk, x.shape[1]), dtype=x.dtype, device=x.device)
    for expert in range(decoded["w1"].shape[0]):
        positions = (ids == expert).nonzero()
        if not positions.numel():
            continue
        rows = x[positions[:, 0]].float()
        gate = (rows @ decoded["w1"][expert, selection].float().T).half()
        up = (rows @ decoded["w3"][expert, selection].float().T).half()
        hidden = torch.nn.functional.silu(gate) * up
        down = (hidden.float() @ decoded["w2"][expert, :, selection].float().T).half()
        output[positions[:, 0], positions[:, 1]] = down
    return (output.float() * scores[..., None].float()).sum(1).half()


@pytest.mark.parametrize("m", [1, 8, 32, 512])
@torch.inference_mode()
def test_mixed_canonical_experts_tp4_and_graph(m):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.model_executor.layers.fused_moe import MoEActivation
    from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
        GGUFExpertBank,
        GGUFTurboMindMoEMethod,
    )

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    experts, hidden, intermediate = 4, 256, 640
    descriptors = {
        "w1": (18, intermediate, hidden),
        "w3": (20, intermediate, hidden),
        "w2": (42, hidden, intermediate),
    }
    sources, decoded = {}, {}
    for shard, (source_type, n, k) in descriptors.items():
        source = np.stack(
            [packed_weight(source_type, n, k, 60 + e) for e in range(experts)]
        )
        sources[shard] = source
        decoded[shard] = (
            torch.from_numpy(np.stack([dequantize(w, source_type) for w in source]))
            .half()
            .cuda()
        )
    torch.manual_seed(951 + m)
    x = (torch.randn(m, hidden, device="cuda") * 0.03125).half()
    ids = torch.randint(experts, (m, 2), device="cuda")
    scores = torch.randn(m, 2, device="cuda").softmax(-1)
    partials, references = [], []
    for rank in range(4):
        banks = torch.nn.ModuleDict()
        for shard, (source_type, _, _) in descriptors.items():
            bank = GGUFExpertBank(source_type, experts, x.device, x.dtype)
            for expert in range(experts):
                bank.add(
                    expert,
                    torch.from_numpy(sources[shard][expert]),
                    rank,
                    4,
                    1 if shard == "w2" else 0,
                )
            bank.finalize()
            assert not bank.pending
            if shard == "w2":
                assert (bank.decoder, bank.group, bank.k) == (2, 32, 160)
            banks[shard] = bank
        method = object.__new__(GGUFTurboMindMoEMethod)
        method.num_experts, method.hidden_size = experts, hidden
        method.small_routing = False
        method.raw_gate_up = False
        layer = SimpleNamespace(
            gguf_expert_banks=banks,
            expert_map=None,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
        )
        expected = reference_part(x, ids, scores, decoded, rank, 4)
        actual = method.apply(layer, x, scores, ids, None, None)
        torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.002)
        partials.append(actual)
        references.append(expected)
        if m == 8 and rank == 0:
            for _ in range(3):
                method.apply(layer, x, scores, ids, None, None)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay = method.apply(layer, x, scores, ids, None, None)
            x.mul_(0.5)
            ids.copy_(torch.flip(ids, (0,)))
            expected = reference_part(x, ids, scores, decoded, rank, 4)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(replay, expected, rtol=0.025, atol=0.002)
            x.mul_(2)
            ids.copy_(torch.flip(ids, (0,)))
    torch.testing.assert_close(
        torch.stack(partials).float().sum(0),
        torch.stack(references).float().sum(0),
        rtol=0.025,
        atol=0.004,
    )
