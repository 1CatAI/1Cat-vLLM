# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 mmap ownership and result ABI; these are CPU tests, not GPU admission."""

from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import vllm.model_executor.layers.ple_offload_layer as offload
import vllm.model_executor.layers.vocab_parallel_embedding as vocab
import vllm.model_executor.parameter as parameter
import vllm.models.qwen4_exp.nvidia.ple_layer as ple
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.ple_offload.protocol import PleOffloadRegistration
from vllm.v1.ple_offload.worker import PleOffloadRunner


@pytest.fixture
def runtime(monkeypatch):
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.offload_config.ple.cpu = True
    config.offload_config.ple.disk = True
    config.offload_config.ple.hybrid = False
    config.kernel_config.ple_disk_cascade_active = False
    config.kernel_config.ple_row_cache_mib = 0
    config.kernel_config.ple_disk_prefetch_window_kib = 0
    monkeypatch.setattr(offload, "_offload_worker_flag", True)
    monkeypatch.setattr(ple.envs, "VLLM_PLE_DISK_OFFLOAD_NUM_THREADS", 2)
    for module in (vocab, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    with set_current_vllm_config(config):
        yield config


def make_layer(*, dtype=torch.float16):
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        eos_token_id=2,
        vocab_size=64,
        split_ngram_parts=3,
        seed=1234,
        ngram_vocab_size_base=7,
        make_ngram_vocab_size_divisible_by=1,
    )
    return ple.Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=2560,
        ple_dense_layer_id=0,
        max_total_tokens=256,
        max_num_reqs=4,
        prefix="model.layers.1.ple.ple_embedding",
        layer_name="model.layers.1.ple",
        params_dtype=dtype,
    )


def checkpoint(tmp_path, layer, dtype=torch.bfloat16):
    count, dim = layer.ngram_embedding.weight.shape
    values = (torch.arange(count * dim).reshape(count, dim) % 997 - 500) / 1000
    table = values.to(dtype)
    tensors = {
        f"ngram_embedding.shard_{i}.weight": part.contiguous()
        for i, part in enumerate(table.split(layer._disk_shard_size))
    }
    path = str(tmp_path / "ple.safetensors")
    save_file(tensors, path)
    with safe_open(path, framework="pt") as source:
        names = source.keys()
        weights = [(name, source.get_tensor(name)) for name in names]
    return table, weights


def test_meta_storage_and_checkpoint_bytes(runtime, tmp_path):
    layer = make_layer()
    try:
        assert layer.ngram_embedding.weight.is_meta
        assert layer.ngram_embedding.weight.dtype == torch.bfloat16
        table, weights = checkpoint(tmp_path, layer)
        assert layer.load_weights(weights) == {"ngram_embedding.weight"}
        assert layer.ngram_embedding.weight.is_meta
        assert layer.get_offload_output_dtype(torch.float16) == torch.uint8
        assert layer.get_offload_output_dim(2560) == 5120
        assert layer._disk_row_kernel.row_bytes == 320
        for _, tensor in weights:
            assert any(tensor is shard for shard in layer._disk_shards)
        assert sum(t.numel() for t in layer._disk_shards) == table.numel()
        pointers, _, row_bytes, rows = PleOffloadRunner._fast_lane_fp8_shards(layer)
        assert row_bytes == 320 and rows == table.shape[0]
        assert pointers == [t.data_ptr() for t in layer._disk_shards]
    finally:
        layer._disk_executor.shutdown()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("rows", [0, 1, 16, 128, 129, 257])
def test_lookup_boundaries_order_reuse_and_output_guards(
    runtime, tmp_path, native, rows
):
    runtime.kernel_config.ple_disk_row_gather = native
    layer = make_layer()
    try:
        table, weights = checkpoint(tmp_path, layer)
        layer.load_weights(weights)
        assert (layer._disk_row_kernel is not None) == native
        count = table.shape[0]
        ids = (torch.arange(rows) * 97 % count).reshape(-1, 1)
        if rows >= 4:
            ids[:4, 0] = torch.tensor(
                [0, layer._disk_shard_size - 1, layer._disk_shard_size, count - 1]
            )
        output = torch.full((rows + 2, 320), 123, dtype=torch.uint8)
        for current in (ids, ids.flip(0), ids * 0 + count - 1):
            layer._disk_embedding_lookup(current, output[:rows])
            expected = table[current.reshape(-1)].view(torch.uint8)
            assert torch.equal(output[:rows], expected)
            assert output[rows:].eq(123).all()
        for invalid in (-1, count):
            sentinel = torch.full((2, 320), 123, dtype=torch.uint8)
            with pytest.raises(IndexError):
                layer._disk_embedding_lookup(torch.tensor([[0], [invalid]]), sentinel)
            assert sentinel.eq(123).all()
        with pytest.raises(ValueError, match="geometry"):
            layer._disk_embedding_lookup(
                torch.tensor([[0]]), torch.empty((1, 160), dtype=torch.uint8)
            )
    finally:
        layer._disk_executor.shutdown()


def test_forward_keeps_bf16_bytes_and_padding(runtime, tmp_path):
    layer = make_layer()
    try:
        table, weights = checkpoint(tmp_path, layer)
        layer.load_weights(weights)
        tokens = torch.tensor([4, 7, 8, 6], dtype=torch.int32)
        starts = torch.tensor([0, 1, 4], dtype=torch.int32)
        context = torch.tensor([[2, 2], [1, 2]], dtype=torch.int32)
        ids = layer.compute_ngram_ids(tokens, starts, context)
        target = torch.full((7, 5120), 123, dtype=torch.uint8)
        output = layer.forward_impl(torch.empty(4, 0), tokens, starts, context, target)
        expected = table[ids.reshape(-1)].view(torch.uint8).reshape(4, 5120)
        assert torch.equal(output, expected)
        assert target[4:].eq(123).all()
    finally:
        layer._disk_executor.shutdown()


@pytest.mark.parametrize("bad_dtype", [torch.float16, torch.float32])
def test_storage_mismatch_rejected(runtime, tmp_path, bad_dtype):
    layer = make_layer()
    try:
        _, weights = checkpoint(tmp_path, layer, bad_dtype)
        with pytest.raises(ValueError, match="dtype mismatch"):
            layer.load_weights(weights)
    finally:
        layer._disk_executor.shutdown()


def test_incomplete_or_scaled_checkpoint_rejected(runtime, tmp_path):
    layer = make_layer()
    try:
        _, weights = checkpoint(tmp_path, layer)
        with pytest.raises(RuntimeError, match="did not load shards"):
            layer.load_weights(weights[:-1])
        with pytest.raises(ValueError, match="FP8 scale"):
            layer.load_weights([("ngram_embedding.weight_scale", torch.tensor([1.0]))])
    finally:
        layer._disk_executor.shutdown()


def test_result_layout_matches_cpu_storage_and_all_consumers(runtime, tmp_path):
    layer = make_layer()
    try:
        _, weights = checkpoint(tmp_path, layer)
        layer.load_weights(weights)
        runner = PleOffloadRunner.__new__(PleOffloadRunner)
        runner._layers = {"ple": layer}
        runner.vllm_config = runtime
        runtime.kernel_config.ple_packed_gguf_results = False
        layout = layer.offload_result_layout()
        registrations = [
            SimpleNamespace(result_layouts={"ple": layout}) for _ in range(4)
        ]
        runner._bind_result_layouts(registrations)
        for change in ({"embedding_dim": 1280}, {"output_dtype": "bfloat16"}):
            with pytest.raises(ValueError, match="disagrees"):
                layer.bind_offload_result_layout(layout | change)
        with pytest.raises(ValueError, match="matching"):
            layer.bind_offload_result_layout(None)
        registrations[-1].result_layouts = {"ple": layout | {"embedding_dim": 1280}}
        with pytest.raises(ValueError, match="differ"):
            runner._bind_result_layouts(registrations)
    finally:
        layer._disk_executor.shutdown()


def test_gpu_placeholder_learns_storage_without_copying_table(runtime, monkeypatch):
    monkeypatch.setattr(offload, "_offload_worker_flag", False)
    layer = make_layer()
    assert not hasattr(layer, "ngram_embedding")
    # The enclosing PLE layer also sets these on the skipped GPU constructor.
    layer._offload_model_dtype = torch.float16
    layer._offload_embedding_dim = 2560
    weight = torch.empty((31, 160), dtype=torch.bfloat16, device="meta")
    assert layer.load_weights([("ngram_embedding.shard_0.weight", weight)]) == set()
    assert list(layer.parameters()) == []
    assert layer.get_offload_output_dim(2560) == 5120
    assert layer.get_offload_output_dtype(torch.float16) == torch.uint8
    assert layer.offload_result_layout()["format"] == "bf16"
    with pytest.raises(ValueError, match="mixed dtypes"):
        layer.load_weights(
            [("ngram_embedding.shard_1.weight", weight.to(torch.float16))]
        )


def test_bf16_four_consumer_registration(runtime, tmp_path):
    layer = make_layer()
    try:
        _, weights = checkpoint(tmp_path, layer)
        layer.load_weights(weights)
        runner = PleOffloadRunner.__new__(PleOffloadRunner)
        runner._layers = {"ple": layer}
        runner._worker_targets, runner._input_bufs, runner._pinned_bufs = {}, {}, {}
        runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1, tensor_parallel_size=4
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
            model_config=SimpleNamespace(
                dtype=torch.float16,
                hf_text_config=SimpleNamespace(ple_embed_dim=2560),
            ),
            kernel_config=SimpleNamespace(
                ple_packed_gguf_results=False, ple_fast_lane=False
            ),
        )
        registrations = []
        for rank in range(4):
            registrations.append(
                PleOffloadRegistration(
                    worker_id=rank,
                    tp_rank=rank,
                    dp_rank=0,
                    gpu_output_buffers={},
                    cpu_output_buffers={
                        "ple": torch.empty(8, 5120, dtype=torch.uint8).share_memory_()
                    },
                    sem_flag_tensors={
                        "ple": torch.zeros(16, dtype=torch.int32).share_memory_()
                    },
                    input_ids_buf=torch.zeros(8, dtype=torch.int32).share_memory_(),
                    query_start_loc_buf=torch.zeros(
                        5, dtype=torch.int32
                    ).share_memory_(),
                    ngram_context_buf=torch.zeros(
                        4, 2, dtype=torch.int32
                    ).share_memory_(),
                    result_layouts={"ple": layer.offload_result_layout()},
                )
            )
        payloads = iter(bytes(ForkingPickler.dumps(item)) for item in registrations)
        runner.accept_registrations(SimpleNamespace(recv=lambda: next(payloads)), 4)
        assert runner._pinned_bufs[0]["ple"].shape == (8, 5120)
        assert len(runner._worker_targets[0]["ple"]) == 4
    finally:
        layer._disk_executor.shutdown()


def test_bf16_file_storage_rejects_unqualified_placement(runtime):
    with pytest.raises(NotImplementedError, match="FP16"):
        make_layer(dtype=torch.bfloat16)
    runtime.kernel_config.ple_disk_cascade_active = True
    with pytest.raises(NotImplementedError, match="whole-table"):
        make_layer()
