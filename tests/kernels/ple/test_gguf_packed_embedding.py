# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch


def packed_table(tmp_path):
    rng = np.random.default_rng(950)
    data = rng.integers(0, 256, (128, 5, 18), dtype=np.uint8)
    data[:, :, :2] = (
        np.full((128, 5), 0.03125, dtype=np.float16).view(np.uint8).reshape(128, 5, 2)
    )
    data = data.reshape(128, 90)
    path = tmp_path / "rows.bin"
    data.tofile(path)
    mapped = np.memmap(path, dtype=np.uint8, mode="c", shape=data.shape)
    return torch.from_numpy(mapped), data


def cpu_embedding(monkeypatch, tmp_path):
    from vllm.model_executor.layers import vocab_parallel_embedding as vocab
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.models.qwen4_exp.nvidia import gguf_embedding as packed

    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(packed, "is_offload_process", lambda: True)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={})
    )
    monkeypatch.setattr(packed, "get_current_vllm_config", lambda: config)
    method = packed.Qwen4ExpPLEGGUFEmbeddingMethod(GGUFConfig())
    module = packed.Qwen4ExpPackedGGUFEmbedding(
        64, 160, torch.float16, 128, "ple.rows", method
    )
    source, original = packed_table(tmp_path)
    module.weight_loader(module.qweight_type, torch.tensor(20, dtype=torch.uint8))
    module.weight_loader(module.qweight, source)
    method.process_weights_after_loading(module)
    return module, source, original


def test_cpu_embedding_retains_mmap_and_decodes_requested_rows(monkeypatch, tmp_path):
    module, source, original = cpu_embedding(monkeypatch, tmp_path)
    assert module.qweight.data_ptr() == source.data_ptr()
    assert module._storage_dim == 90
    assert module.ple_device_table is None and module.ple_host_storage is None
    ids = torch.tensor([[7, 9, 7], [2, 0, 9]])
    expected = torch.from_numpy(
        gguf.quants.dequantize(original[ids.numpy()], gguf.GGMLQuantizationType.IQ4_NL)
    ).half()
    torch.testing.assert_close(module.embedding_lookup(ids), expected, rtol=0, atol=0)


@pytest.mark.parametrize("cascade", [False, True])
def test_cpu_transport_preserves_row_order_and_output_dtype(
    monkeypatch, tmp_path, cascade
):
    from vllm.models.qwen4_exp.common.ple import PLEDiskSegment
    from vllm.models.qwen4_exp.nvidia import ple_layer

    embedding, _, original = cpu_embedding(monkeypatch, tmp_path)
    monkeypatch.setattr(ple_layer, "is_offload_process", lambda: True)
    layer = object.__new__(ple_layer.Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(layer)
    layer.ngram_embedding = embedding
    layer._packed_gguf = True
    layer._cascade = cascade
    layer._disk_segments = [PLEDiskSegment(7, 10)]
    layer.head_dim = 160
    layer.embedding_dim = 320
    ids = torch.tensor([[7, 9], [9, 1]])
    layer.compute_ngram_ids = lambda *_: ids
    output = torch.full((3, 320), -17, dtype=torch.float16)
    actual = layer.forward_impl(
        torch.empty(2, 4),
        torch.tensor([1, 2]),
        torch.tensor([0, 2]),
        torch.zeros(1, 2, dtype=torch.int64),
        output_buffer=output,
    )
    expected = (
        torch.from_numpy(
            gguf.quants.dequantize(
                original[ids.numpy()], gguf.GGMLQuantizationType.IQ4_NL
            )
        )
        .half()
        .reshape(2, 320)
    )
    if cascade:
        expected[1, 160:] = -17
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all(output[2] == -17)
    assert layer.get_offload_output_dtype(torch.float32) == torch.float16


def test_loader_filters_before_payload_iteration(monkeypatch):
    from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader

    loader = object.__new__(GGUFModelLoader)
    names = {
        "regular": "model.layers.0.mlp.up.weight",
        "table": "model.layers.1.ple.embedding.weight",
    }
    monkeypatch.setattr(loader, "_prepare_weights", lambda _: "model.gguf")
    monkeypatch.setattr(loader, "_get_gguf_weights_map", lambda _: names)
    touched = []

    def iterator(config, path, filtered):
        for raw, name in filtered.items():
            assert raw != "regular", "Unrelated payload was touched"
            touched.append(raw)
            yield name, torch.tensor(1)

    monkeypatch.setattr(loader, "_get_weights_iterator", iterator)
    result = list(
        loader.get_all_weights(None, None, skip_weight=lambda name: ".ple." not in name)
    )
    assert touched == ["table"]
    assert result[0][0] == names["table"]
    assert len(names) == 2


def test_disk_only_placement_never_gathers_missing_resident_rows(monkeypatch):
    from vllm.models.qwen4_exp.nvidia import gguf_embedding as packed

    module = object.__new__(packed.Qwen4ExpPackedGGUFEmbedding)
    torch.nn.Module.__init__(module)
    module._device_rows = module._host_rows = 0
    module._disk_rows = 128
    module._output_dtype = torch.float16
    module.embedding_dim = 4
    module._storage_dim = 8
    module.layer_name = "ple.missing.resident"

    def unexpected_gather(*_args):
        raise AssertionError("Disk-only placement has no resident pointer")

    monkeypatch.setattr(
        torch.ops.vllm, "qwen4_exp_ple_packed_gather", unexpected_gather
    )
    empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *args, **kwargs: empty(*args, **{**kwargs, "device": "cpu"})
    )
    ids = torch.tensor([[7, 9], [9, 1]])
    # Exercise the GPU dispatch decision using CPU buffers. Any resident gather
    # would need missing pointers and must be skipped before allocation.
    request = SimpleNamespace(
        device=torch.device("cuda"), reshape=ids.reshape, shape=ids.shape
    )
    rows = torch.arange(16, dtype=torch.float16).reshape(4, 4)
    actual = module.embedding_lookup(request, rows)
    torch.testing.assert_close(actual, rows.reshape(2, 2, 4), rtol=0, atol=0)
    assert actual.data_ptr() != rows.data_ptr()
    for invalid in (rows.float(), rows[:, :3]):
        with pytest.raises(ValueError, match="invalid dtype or width"):
            module.embedding_lookup(request, invalid)
    with pytest.raises(ValueError, match="require offloader output"):
        module.embedding_lookup(request)


@pytest.mark.parametrize("placement", ["device", "host", "split_disk", "disk"])
def test_gpu_packets_match_cpu_and_changed_graph_ids(monkeypatch, tmp_path, placement):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.model_executor.layers import vocab_parallel_embedding as vocab
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.models.qwen4_exp.common.ple import plan_ple_placement
    from vllm.models.qwen4_exp.nvidia import gguf_embedding as packed

    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(packed, "is_offload_process", lambda: False)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={})
    )
    monkeypatch.setattr(packed, "get_current_vllm_config", lambda: config)
    method = packed.Qwen4ExpPLEGGUFEmbeddingMethod(GGUFConfig())
    module = packed.Qwen4ExpPackedGGUFEmbedding(
        128, 160, torch.float16, 128, "ple.gpu.rows", method
    )
    host_rows = {"device": 0, "host": 128, "split_disk": 64, "disk": 0}[placement]
    vram_budget = {"split_disk": 32 * 90, "disk": 0}.get(placement)
    plan = plan_ple_placement(
        total_rows=128,
        row_bytes=90,
        host_budget_bytes=host_rows * 90,
        vram_budget_bytes=vram_budget,
        disk_allowed=placement in ("split_disk", "disk"),
    )
    monkeypatch.setattr(module, "_plan_placement", lambda _: plan)
    monkeypatch.setattr(
        packed,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple.gpu.rows": module}),
    )
    source, original = packed_table(tmp_path)
    module.weight_loader(module.qweight_type, torch.tensor(20, dtype=torch.uint8))
    module.weight_loader(module.qweight, source)
    method.process_weights_after_loading(module)
    reference = (
        torch.from_numpy(
            gguf.quants.dequantize(original, gguf.GGMLQuantizationType.IQ4_NL)
        )
        .half()
        .cuda()
    )
    ids = torch.tensor([0, 31, 63, 127], device="cuda")
    remote = reference[ids].clone()
    for _ in range(3):
        result = module.embedding_lookup(ids, remote)
    torch.testing.assert_close(result, reference[ids], rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = module.embedding_lookup(ids, remote)
    ids.copy_(torch.tensor([127, 0, 95, 32], device="cuda"))
    remote.copy_(reference[ids])
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(result, reference[ids], rtol=0, atol=0)
