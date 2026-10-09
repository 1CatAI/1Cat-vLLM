# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.kernels.ple.host_result import (
    HostResultRegion,
    publish_host_flag,
    wait_host_resets,
)
from vllm.model_executor.kernels.ple.packed_result import (
    packed_result_capability,
    prepare_packed_gguf_results,
)
from vllm.model_executor.layers.ple_offload_layer import CpuGpuSemaphore
from vllm.models.qwen4_exp.nvidia import ple_layer
from vllm.transformers_utils.gguf_rows import PackedGGUFRowReader


def rows(count=41, k=160):
    rng = np.random.default_rng(1710)
    data = rng.integers(0, 256, (count, k // 32, 18), dtype=np.uint8)
    scales = rng.uniform(-0.1, 0.1, (count, k // 32)).astype("<f2")
    data[:, :, :2] = scales.view(np.uint8).reshape(count, k // 32, 2)
    return data.reshape(count, -1)


def layout(**changes):
    options = dict(
        enabled=True,
        source_type=20,
        row_width=160,
        heads=16,
        fp16=True,
        sm70=True,
        offloaded=True,
        local_tables=False,
    )
    options.update(changes)
    return packed_result_capability(**options)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"enabled": False}, "disabled_by_kernel_config"),
        ({"source_type": 23}, "requires_iq4nl_source"),
        ({"row_width": 159}, "requires_complete_iq4nl_rows"),
        ({"heads": 0}, "requires_complete_iq4nl_rows"),
        ({"fp16": False}, "requires_sm70_fp16_output"),
        ({"sm70": False}, "requires_sm70_fp16_output"),
        ({"offloaded": False}, "cpu_result_transport_not_active"),
        ({"local_tables": True}, "local_table_decode_takes_precedence"),
    ],
)
def test_capability_preserves_other_result_paths(changes, reason):
    status = layout(**changes)
    assert not status["enabled"] and status["reason"] == reason


def test_metadata_plan_names_the_registered_offload_layer(monkeypatch):
    import vllm.envs as envs
    from vllm.model_executor.layers import ple_offload_layer
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_device_capability", lambda value: True)
    monkeypatch.setattr(ple_offload_layer, "ple_offload_enabled", lambda: True)
    monkeypatch.setenv("VLLM_SM70_QWEN38_HYBRID_PLE", "0")
    envs.disable_envs_cache()
    name = "model.layers.1.ple.ple_embedding"
    policy = SimpleNamespace(
        ple_packed_gguf_results=True,
        ple_packed_result_decoders={},
        ple_pinned_decode_active=False,
        ple_disk_cascade_active=False,
    )
    config = SimpleNamespace(
        kernel_config=policy,
        model_config=SimpleNamespace(
            dtype=torch.float16, hf_text_config=SimpleNamespace(ple_embed_dim=2560)
        ),
    )
    tensor = SimpleNamespace(shape=[160, 320001536], tensor_type=20)
    prepare_packed_gguf_results(
        config, {"table": tensor}, {"table": name + ".ngram_embedding.weight"}
    )
    assert policy.ple_packed_result_decoders == {name: layout()}
    policy.ple_pinned_decode_active = True
    prepare_packed_gguf_results(
        config, {"table": tensor}, {"table": name + ".ngram_embedding.weight"}
    )
    assert (
        policy.ple_packed_result_decoders[name]["reason"]
        == "local_table_decode_takes_precedence"
    )


@pytest.mark.parametrize("m", [1, 5, 20])
def test_cpu_producer_keeps_raw_rows_order_and_padding(monkeypatch, m):
    data = rows()
    reader = PackedGGUFRowReader(data, 20, 160)
    module = ple_layer.Qwen4ExpNGramEmbedding.__new__(ple_layer.Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.layer_name = "ple"
    module._packed_result_layout = layout()
    module.ngram_embedding = SimpleNamespace(_cpu_reader=reader)
    indices = torch.arange(m * 16).reshape(m, 16).remainder(41)
    module.compute_ngram_ids = lambda *args: indices
    monkeypatch.setattr(ple_layer, "is_offload_process", lambda: True)
    output = torch.full((m + 3, 1440), 0xEE, dtype=torch.uint8)
    ids = torch.zeros(m, dtype=torch.int32)
    result = module.forward_impl(
        ids, ids, torch.tensor([0, m]), torch.zeros(1, 2), output
    )
    np.testing.assert_array_equal(result.numpy(), data[indices.numpy()].reshape(m, -1))
    assert output[m:].eq(0xEE).all()
    assert module.get_offload_output_dim(2560) == 1440
    assert module.get_offload_output_dtype(torch.float16) == torch.uint8


def test_packed_rows_preserve_bounds_empty_and_value_checks():
    data = rows(3)
    reader = PackedGGUFRowReader(data, 20, 160, logical_rows=2)
    assert reader.lookup_packed_iq4nl(np.empty((0, 4), np.int64)).shape == (0, 4, 90)
    for ids in (np.array([-1]), np.array([2]), np.array([2**63], np.uint64)):
        with pytest.raises(IndexError):
            reader.lookup_packed_iq4nl(ids)
    with pytest.raises(TypeError):
        reader.lookup_packed_iq4nl(np.array([1.5]))
    blocks = data.reshape(3, 5, 18)
    for scale, code, raises in [
        (np.inf, 0, True),
        (np.nan, 0, True),
        (1024, 0xFF, True),
        (1024, 0x88, False),
    ]:
        blocks[0, :, :2] = np.array([scale], dtype="<f2").view(np.uint8)
        blocks[0, :, 2:] = code
        if raises:
            with pytest.raises(ValueError):
                reader.lookup_packed_iq4nl(np.array([0]))
            with pytest.raises(ValueError):
                reader.lookup(np.array([0]))
        else:
            packed = reader.lookup_packed_iq4nl(np.array([0]))
            expected = reader.lookup(np.array([0]))
            actual = gguf.quants.dequantize(
                packed, gguf.GGMLQuantizationType.IQ4_NL
            ).astype(np.float16)
            np.testing.assert_array_equal(actual, expected)


def cpu_owner_without_postload_configuration():
    module = ple_layer.Qwen4ExpNGramEmbedding.__new__(ple_layer.Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module._packed_result_layout = None
    module.head_dim, module.ngram_heads = 160, 16
    module.ngram_embedding = SimpleNamespace(
        _cpu_reader=PackedGGUFRowReader(rows(), 20, 160),
        _output_dtype=torch.float16,
    )
    return module


def test_cpu_spawn_configuration_binds_gpu_result_geometry_later():
    from vllm.v1.ple_offload.worker import PleOffloadRunner

    module = cpu_owner_without_postload_configuration()
    runner = PleOffloadRunner.__new__(PleOffloadRunner)
    runner._layers = {"ple": module}
    runner.vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(
            ple_packed_gguf_results=True, ple_packed_result_decoders={}
        )
    )
    assert module.get_offload_output_dtype(torch.float16) == torch.float16
    registrations = [
        SimpleNamespace(result_layouts={"ple": layout()}) for _ in range(4)
    ]
    runner._bind_result_layouts(registrations)
    assert module.get_offload_output_dtype(torch.float16) == torch.uint8
    assert module.get_offload_output_dim(2560) == 1440
    assert module.offload_result_layout() == layout()


def test_mapped_registration_binds_geometry_before_buffer_validation():
    from multiprocessing.reduction import ForkingPickler

    from vllm.v1.ple_offload.protocol import PleOffloadRegistration
    from vllm.v1.ple_offload.worker import PleOffloadRunner

    module = cpu_owner_without_postload_configuration()
    runner = PleOffloadRunner.__new__(PleOffloadRunner)
    runner._layers = {"ple": module}
    runner._worker_targets, runner._input_bufs, runner._pinned_bufs = {}, {}, {}
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=4),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            dtype=torch.float16, hf_text_config=SimpleNamespace(ple_embed_dim=2560)
        ),
        kernel_config=SimpleNamespace(ple_packed_gguf_results=True),
    )
    registrations = []
    for rank in range(4):
        result = torch.empty(8, 1440, dtype=torch.uint8).share_memory_()
        flag = torch.zeros(16, dtype=torch.int32).share_memory_()
        registrations.append(
            PleOffloadRegistration(
                worker_id=rank,
                tp_rank=rank,
                dp_rank=0,
                gpu_output_buffers={},
                cpu_output_buffers={"ple": result},
                sem_flag_tensors={"ple": flag},
                input_ids_buf=torch.zeros(8, dtype=torch.int32).share_memory_(),
                query_start_loc_buf=torch.zeros(5, dtype=torch.int32).share_memory_(),
                ngram_context_buf=torch.zeros(4, 2, dtype=torch.int32).share_memory_(),
                result_layouts={"ple": layout()},
            )
        )
    payloads = iter(bytes(ForkingPickler.dumps(item)) for item in registrations)
    runner.accept_registrations(SimpleNamespace(recv=lambda: next(payloads)), 4)
    assert module.offload_result_layout() == layout()
    assert runner._pinned_bufs[0]["ple"].shape == (8, 1440)
    assert runner._pinned_bufs[0]["ple"].dtype == torch.uint8
    assert len(runner._worker_targets[0]["ple"]) == 4


@pytest.mark.parametrize(
    "failure", ["mixed_ranks", "bad_source", "disabled", "unknown_layer"]
)
def test_result_negotiation_rejects_inconsistent_consumers_or_cpu_rows(failure):
    from vllm.v1.ple_offload.worker import PleOffloadRunner

    module = cpu_owner_without_postload_configuration()
    runner = PleOffloadRunner.__new__(PleOffloadRunner)
    runner._layers = {"ple": module}
    runner.vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(ple_packed_gguf_results=failure != "disabled")
    )
    status = layout()
    name = "missing" if failure == "unknown_layer" else "ple"
    if failure == "bad_source":
        status["source_type"] = 23
    registrations = [SimpleNamespace(result_layouts={name: status}) for _ in range(4)]
    if failure == "mixed_ranks":
        registrations[-1].result_layouts = {}
    with pytest.raises(ValueError):
        runner._bind_result_layouts(registrations)
    assert module.offload_result_layout() is None


@pytest.mark.parametrize("m", [1, 5, 20, 512])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_decoder_matches_official_and_changed_graph_replays(m):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 qualification")
    data = rows()
    indices = np.arange(m * 16).reshape(m, 16) % 41
    packed = torch.from_numpy(data[indices].reshape(m, -1).copy()).cuda()
    book = torch.tensor(gguf.quants.IQ4_NL.kvalues, dtype=torch.float32, device="cuda")
    decode = lambda: torch.ops.vllm.ple_decode_iq4nl_result(packed, book, 160)
    expected = torch.from_numpy(
        gguf.quants.dequantize(data, gguf.GGMLQuantizationType.IQ4_NL)[indices]
        .astype(np.float16)
        .reshape(m, -1)
    ).cuda()
    torch.testing.assert_close(decode(), expected, rtol=0, atol=0)
    warmup = torch.cuda.Stream()
    with torch.cuda.stream(warmup):
        for _ in range(3):
            decode()
    warmup.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = decode()
    for step in range(3):
        changed = rows(count=47)
        ids = (indices + 7 * step) % 47
        packed.copy_(torch.from_numpy(changed[ids].reshape(m, -1)))
        expected = torch.from_numpy(
            gguf.quants.dequantize(changed, gguf.GGMLQuantizationType.IQ4_NL)[ids]
            .astype(np.float16)
            .reshape(m, -1)
        ).cuda()
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [5, 20])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_ple_consumer_preserves_fp16_result_boundary(m):
    """Compare decoder storage policies through the PLE projection/gate chain."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 qualification")
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLEGroupedNorm

    torch.manual_seed(1094)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    packet = torch.empty((m, 1440), dtype=torch.uint8, device=device)
    reference = torch.empty((m, 2560), dtype=torch.float16, device=device)
    decoded = torch.empty_like(reference)
    book = torch.tensor(gguf.quants.IQ4_NL.kvalues, dtype=torch.float32, device=device)
    hidden = torch.randn(m, 10240, dtype=torch.float16, device=device)
    key = torch.nn.Linear(2560, 10240, bias=False, dtype=torch.float16, device=device)
    value = torch.nn.Linear(2560, 2560, bias=False, dtype=torch.float16, device=device)
    norms = [
        Qwen4ExpPLEGroupedNorm(10240, 1e-6, 2560, torch.float16).to(device)
        for _ in range(3)
    ]

    def consume(embeddings):
        projected_key = key(embeddings).reshape(m, 4, 2560)
        projected_value = value(embeddings)
        normalized_key = norms[0](projected_key.flatten(-2)).reshape(m, 4, 2560)
        normalized_query = norms[1](hidden).reshape(m, 4, 2560)
        gate = (normalized_key * normalized_query).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(2560)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated = gate * projected_value.unsqueeze(-2)
        normalized = norms[2](gated.flatten(-2))
        return (
            embeddings,
            projected_key,
            projected_value,
            normalized_key,
            gate,
            normalized,
        )

    def allocate():
        return consume(torch.ops.vllm.ple_decode_iq4nl_result(packet, book, 160))

    def fixed():
        torch.ops.vllm.ple_decode_iq4nl_result_out(packet, book, 160, decoded)
        return consume(decoded)

    compiled = [
        torch.compile(fn, fullgraph=True)
        for fn in (lambda: consume(reference), allocate, fixed)
    ]
    with torch.inference_mode():
        for step in range(3):
            data = rows(count=47)
            ids = (np.arange(m * 16).reshape(m, 16) + step * 7) % 47
            packet.copy_(torch.from_numpy(data[ids].reshape(m, -1)))
            official = gguf.quants.dequantize(data, gguf.GGMLQuantizationType.IQ4_NL)
            reference.copy_(
                torch.from_numpy(official[ids].astype(np.float16).reshape(m, -1))
            )
            control = compiled[0]()
            for fn in compiled[1:]:
                for actual, expected in zip(fn(), control):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decoder_out_keeps_workspace_tail_and_rejects_bad_storage():
    packet = torch.from_numpy(rows()[np.arange(80) % 41].reshape(5, -1)).cuda()
    book = torch.tensor(gguf.quants.IQ4_NL.kvalues, dtype=torch.float32, device="cuda")
    output = torch.full((8, 2560), 1094, dtype=torch.float16, device="cuda")
    torch.ops.vllm.ple_decode_iq4nl_result_out(packet, book, 160, output[:5])
    torch.testing.assert_close(
        output[:5],
        torch.ops.vllm.ple_decode_iq4nl_result(packet, book, 160),
        rtol=0,
        atol=0,
    )
    assert output[5:].eq(1094).all()
    for bad in (output, output[:5].float(), output[:5, ::2]):
        with pytest.raises(ValueError, match="output"):
            torch.ops.vllm.ple_decode_iq4nl_result_out(packet, book, 160, bad)


@pytest.mark.parametrize("m", [5, 20])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mapped_packet_consumer_decodes_before_releasing_result(m):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 qualification")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    packet = torch.empty((m, 1440), dtype=torch.uint8, device=device)
    region = HostResultRegion.create(packet)
    try:
        module = ple_layer.Qwen4ExpNGramEmbedding.__new__(
            ple_layer.Qwen4ExpNGramEmbedding
        )
        torch.nn.Module.__init__(module)
        module._packed_result_layout = layout()
        module._packed_result_codebook = torch.tensor(
            gguf.quants.IQ4_NL.kvalues, dtype=torch.float32, device=device
        )
        module._packed_result_output = torch.empty(
            (m + 3, 2560), dtype=torch.float16, device=device
        )
        module._cpu_output_buffer = region.result
        module.setup_cross_process_offload(
            packet, CpuGpuSemaphore(device, host_region=region)
        )
        hidden = torch.empty((m, 2560), dtype=torch.float16, device=device)
        # Compile the decoder before capture. The semaphore itself is captured
        # without executing a wait on the zero-valued producer flag.
        torch.ops.vllm.ple_decode_iq4nl_result(
            packet, module._packed_result_codebook, 160
        )
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module.wait_offloaded_output(hidden, m)
            module.release_offloaded_output()
        assert output.data_ptr() == module._packed_result_output.data_ptr()
        for step in range(4):
            wait_host_resets([region.flag], timeout_s=1)
            data = rows()
            ids = (np.arange(m * 16).reshape(m, 16) + step * 7) % 41
            region.result.copy_(torch.from_numpy(data[ids].reshape(m, -1)))
            publish_host_flag(region.flag)
            graph.replay()
            torch.accelerator.synchronize()
            expected = (
                gguf.quants.dequantize(data, gguf.GGMLQuantizationType.IQ4_NL)[ids]
                .astype(np.float16)
                .reshape(m, -1)
            )
            np.testing.assert_array_equal(output.cpu().numpy(), expected)
            wait_host_resets([region.flag], timeout_s=1)
    finally:
        region.close()
