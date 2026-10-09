# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configured native dispatch and independent scratch/capture owners."""

import pytest
import torch

from vllm.config.execution_policy import FlashV100Policy, GraphPolicy
from vllm.config.flash_v100 import NATIVE_FIELDS, FlashV100Diagnostics, native_value

interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
runtime_module = pytest.importorskip("flash_attn_v100.runtime")


def runtime():
    policy = FlashV100Policy()
    policy.resolve()
    policy.options.finalize(GraphPolicy(), FlashV100Diagnostics())
    return runtime_module.AttentionRuntime(
        runtime_module.PythonPolicy(**policy.options.python_policy),
        policy.options.native_inputs,
    )


@pytest.mark.parametrize(
    "raw", [None, "", "0", "1", "01", "true", " 6tail", "-8", "32"]
)
def test_native_parser_projection_matches_python(raw):
    extension = interface.flash_attn_v100_cuda
    values = [raw] * len(NATIVE_FIELDS)
    prepared = extension.PreparedPolicy(values)
    expected = [
        int(native_value(rule, raw, default)) for _, _, rule, default in NATIVE_FIELDS
    ]
    assert list(prepared.values) == expected
    assert not any(prepared.observations)


def test_native_field_positions_and_scalar_alias_priority():
    extension = interface.flash_attn_v100_cuda
    for changed in range(len(NATIVE_FIELDS)):
        values: list[str | None] = [None] * len(NATIVE_FIELDS)
        values[changed] = "0"
        expected = []
        for index, (_, _, rule, default) in enumerate(NATIVE_FIELDS):
            raw = values[index]
            if rule == "scalar_alias" and raw is None:
                raw = values[-1]
            expected.append(int(native_value(rule, raw, default)))
        assert list(extension.PreparedPolicy(values).values) == expected


def test_old_native_binary_rejects_engine_policy_before_execution(monkeypatch):
    from types import SimpleNamespace

    owner = runtime()
    monkeypatch.setattr(interface, "flash_attn_v100_cuda", SimpleNamespace())
    with pytest.raises(RuntimeError, match="rebuilt extension"):
        runtime_module.AttentionRuntime(owner.policy, [None] * len(NATIVE_FIELDS))


def test_runtime_release_does_not_clear_other_owner():
    first, second = runtime(), runtime()
    first.decode_plan_cache["plan"] = object()
    second.decode_plan_cache["plan"] = object()
    first.close()
    assert first.native_policy is None and not first.decode_plan_cache
    assert second.native_policy is not None and second.decode_plan_cache


def test_backend_workspace_capture_growth_and_engine_shutdown(device):
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.runtime_resources import release_runtime_resources
    from vllm.v1.attention.backends.flash_v100.dense_prefill import (
        get_fp8_prefill_bridge_tail_workspace,
    )

    engines = [SimpleNamespace(), SimpleNamespace()]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    q = torch.ones(1, 4, 6, 256, device=device, dtype=torch.float16)
    outputs = [torch.empty_like(q), torch.empty_like(q)]
    for engine, stream, graph, output in zip(engines, streams, graphs, outputs):
        with set_current_vllm_config(engine), torch.cuda.stream(stream):
            get_fp8_prefill_bridge_tail_workspace(q, 4)
        torch.accelerator.synchronize()
        with set_current_vllm_config(engine), torch.cuda.graph(graph, stream=stream):
            workspace, _ = get_fp8_prefill_bridge_tail_workspace(q, 4)
            workspace.copy_(q)
            output.copy_(workspace * 2)
    with set_current_vllm_config(engines[0]), torch.cuda.stream(streams[0]):
        get_fp8_prefill_bridge_tail_workspace(q, 16)
    q.fill_(3)
    for graph in graphs:
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(outputs[0], q * 2)
    assert torch.equal(outputs[1], outputs[0])
    graphs[0].reset()
    release_runtime_resources(engines[0])
    q.fill_(5)
    graphs[1].replay()
    torch.accelerator.synchronize()
    assert torch.equal(outputs[1], q * 2)


@pytest.fixture
def device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    return torch.device("cuda")


@pytest.mark.parametrize("storage", ["fp16", "fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("partition", [256, 1024])
def test_configured_decode_matches_legacy_and_survives_environment_change(
    monkeypatch, device, storage, partition
):
    monkeypatch.setenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", str(partition))
    owner = runtime()
    torch.manual_seed(211)
    q = torch.randn(1, 6, 256, device=device, dtype=torch.float16)
    k = torch.randn(8, 256, 1, 256, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    if storage != "fp16":
        dtype = torch.float8_e4m3fn if storage == "fp8_e4m3" else torch.float8_e5m2
        k, v = k.to(dtype).view(torch.uint8), v.to(dtype).view(torch.uint8)
    table = torch.arange(8, device=device, dtype=torch.int32).view(1, -1)
    lengths = torch.tensor([1537], device=device, dtype=torch.int32)
    kwargs = dict(
        kv_cache_dtype="auto" if storage == "fp16" else storage, max_seq_len_hint=1537
    )
    reference = interface.flash_attn_decode_paged_xqa(
        q, k, v, table, lengths, **kwargs
    ).clone()
    monkeypatch.setenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", "invalid-after-init")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE", "0")
    result = owner.bind(interface.flash_attn_decode_paged_xqa)(
        q, k, v, table, lengths, **kwargs
    )
    assert torch.equal(result, reference)


def test_distinct_owners_capture_growth_and_updated_input(monkeypatch, device):
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    first, second = runtime(), runtime()
    q = torch.ones(1, 6, 256, device=device, dtype=torch.float16)

    def scratch(owner, partitions):
        return interface._get_xqa_staged_rescale_workspace(
            q,
            batch_capacity=1,
            num_heads=6,
            plan=interface._DecodePlan(256, partitions, partitions, partitions),
            _runtime=owner,
        )

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = [torch.empty((), device=device) for _ in streams]
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for owner, stream, output, graph in zip((first, second), streams, outputs, graphs):
        with torch.cuda.stream(stream):
            scratch(owner, 4)
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            workspace = scratch(owner, 4)
            workspace.fill_(3)
            output.copy_(workspace.sum() + q.sum())
    assert (
        first.xqa_staged_rescale_workspace_cache
        is not second.xqa_staged_rescale_workspace_cache
    )
    with torch.cuda.stream(streams[0]):
        scratch(first, 4)
        old = next(iter(first.xqa_staged_rescale_workspace_cache.values()))
        scratch(first, 64)
        grown = next(iter(first.xqa_staged_rescale_workspace_cache.values()))
        assert grown.previous is old
    q.fill_(2)
    for graph in graphs:
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(outputs[0], outputs[1])
    assert outputs[0].item() == old.buffer.numel() * 3 + q.sum().item()


def test_bound_decode_capture_reads_new_query_and_lengths(device):
    owner = runtime()
    q = torch.ones(1, 6, 256, device=device, dtype=torch.float16)
    k = torch.randn(8, 256, 1, 256, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    table = torch.arange(8, device=device, dtype=torch.int32).view(1, -1)
    lengths = torch.tensor([1800], device=device, dtype=torch.int32)
    output = torch.empty_like(q)
    operation = owner.bind(interface.flash_attn_decode_paged_xqa)
    kwargs = dict(out=output, max_seq_len_hint=2048, workspace_seq_capacity_hint=2048)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        operation(q, k, v, table, lengths, **kwargs)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        operation(q, k, v, table, lengths, **kwargs)
    q.fill_(0.25)
    lengths.fill_(513)
    graph.replay()
    captured = output.clone()
    operation(q, k, v, table, lengths, **kwargs)
    torch.accelerator.synchronize()
    assert torch.equal(captured, output)


def test_dense_prefill_binding_preserves_output(device):
    owner = runtime()
    q = torch.randn(1, 5, 6, 256, device=device, dtype=torch.float16)
    k = torch.randn(1, 17, 1, 256, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    legacy = interface.flash_attn_func(q, k, v, causal=True)
    configured = owner.bind(interface.flash_attn_func)(q, k, v, causal=True)
    assert torch.equal(legacy, configured)
