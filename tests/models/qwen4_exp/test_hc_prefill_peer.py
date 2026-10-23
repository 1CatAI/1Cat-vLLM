# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import sm70_hc_prefill_peer as peer
from vllm.runtime_resources import release_runtime_resources, runtime_resources_for


@pytest.mark.parametrize("tokens", [0, 1, 3, 4, 5, 7, 8, 9])
@pytest.mark.parametrize("exact_capacity", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_gather_rows_into_preserves_owner_tail(
    monkeypatch, tokens, exact_capacity, disabled
):
    from vllm import distributed

    rows = (tokens + 3) // 4
    local = torch.arange(rows * 8).reshape(rows, 8).half()
    before = local.clone()
    gathered = torch.cat([local + rank * 100 for rank in range(4)])
    calls = []

    def gather_into(output, source):
        assert source is local
        calls.append("into")
        output.copy_(gathered)

    def gather(source, dim):
        assert source is local and dim == 0
        calls.append("fallback")
        return gathered

    comm = NS(disabled=disabled, all_gather=gather_into)
    group = NS(
        world_size=4, device_communicator=NS(pynccl_comm=comm), all_gather=gather
    )
    monkeypatch.setattr(distributed, "get_tp_group", lambda: group)
    capacity = tokens if exact_capacity else 4 * rows + 2
    output = torch.full((capacity, 8), -7, dtype=torch.half)
    peer._gather_rows_into(output, local, tokens)
    assert torch.equal(output[:tokens], gathered[:tokens])
    assert torch.all(output[tokens:] == -7)
    assert torch.equal(local, before)
    expected = "fallback" if disabled or capacity < 4 * rows else "into"
    assert calls == ([expected] if tokens else [])


def test_gather_rows_into_rejects_invalid_capacity(monkeypatch):
    from vllm import distributed

    monkeypatch.setattr(distributed, "get_tp_group", lambda: NS(world_size=4))
    with pytest.raises(ValueError, match="capacity"):
        peer._gather_rows_into(torch.empty(6, 8), torch.empty(2, 8), 7)


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize(
    "rows,accepted",
    [(4095, False), (4096, True), (4097, True), (16384, True), (16385, False)],
)
def test_prefill_admission_preserves_mtp_precision_policy(
    monkeypatch, reduced, rows, accepted
):
    from vllm.config import CUDAGraphMode
    from vllm.models.qwen4_exp.nvidia import model

    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", reduced
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    monkeypatch.setattr(model, "is_forward_context_available", lambda: True)
    context = NS(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(model, "get_forward_context", lambda: context)
    owner = NS(_hc_prefill_sequence_qualified=True)
    inputs = torch.empty(rows, dtype=torch.long)
    selected = model.Qwen4ExpForCausalLM._use_hc_prefill_sequence_rows(
        owner, inputs, None, None, {}
    )
    assert selected is accepted
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is reduced
    context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    assert not model.Qwen4ExpForCausalLM._use_hc_prefill_sequence_rows(
        owner, inputs, None, None, {}
    )
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", True)
    assert not model.Qwen4ExpForCausalLM._use_hc_prefill_sequence_rows(
        owner, inputs, None, None, {}
    )


def test_single_attention_partial_uses_fp32_transport(monkeypatch):
    from vllm import distributed
    from vllm.models.qwen4_exp.nvidia.ops import hc

    seen = []

    def reduce_scatter(value, dim):
        seen.append(value)
        assert value.dtype == torch.float32 and dim == 0
        return value[:1024]

    monkeypatch.setattr(
        distributed, "get_tp_group", lambda: NS(reduce_scatter=reduce_scatter)
    )
    monkeypatch.setattr(peer, "current_runtime_resources", lambda: {})
    monkeypatch.setattr(
        hc, "hc_combine_norm", lambda hidden, payload, *args: (payload, hidden)
    )
    payload = torch.arange(4096 * 4).reshape(4096, 4).half()
    hidden = torch.zeros(1024, 16, dtype=torch.half)
    actual, _ = peer._combine_norm(
        hidden, payload, torch.zeros(1024, 4), torch.zeros(16), 1e-6, False, True, True
    )
    assert len(seen) == 1 and torch.equal(seen[0], payload.float())
    assert actual.dtype == payload.dtype and torch.equal(actual, payload[:1024])


@pytest.mark.parametrize("separate", [False, True])
def test_peer_fallback_keeps_unrounded_moe_sum(monkeypatch, separate):
    from vllm import distributed
    from vllm.models.qwen4_exp.nvidia.ops import hc

    hidden = torch.zeros(1024, 10240, dtype=torch.half)
    payload = torch.full((8192, 2560), 40000, dtype=torch.half)
    injection = torch.zeros(1024, 4, dtype=torch.half)
    weight = torch.zeros(10240, dtype=torch.half)
    seen = []

    def reduce_scatter(source, dim):
        assert dim == 0 and source.dtype == torch.float32
        assert torch.all(source == 80000)
        seen.append(source)
        return torch.zeros(1024, 2560)

    group = NS(reduce_scatter=reduce_scatter)
    monkeypatch.setattr(distributed, "get_tp_group", lambda: group)
    monkeypatch.setattr(peer, "current_runtime_resources", lambda: {})
    monkeypatch.setattr(hc, "hc_combine_norm", lambda h, *args: (h, h))
    actual, normalized = peer._combine_norm(
        hidden,
        payload[:4096] if separate else payload,
        injection,
        weight,
        1e-6,
        not separate,
        True,
        True,
        payload[4096:] if separate else None,
    )
    assert actual is hidden and normalized is hidden and len(seen) == 1


@pytest.mark.parametrize("reduced,zero", [(False, False), (True, False), (False, True)])
def test_runner_partials_preserve_producer_storage_and_reject_reduced(reduced, zero):
    from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
        ZeroExpertRouter,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    routed, shared = torch.randn(5, 8), torch.randn(5, 6)
    calls = []

    def outputs(*args):
        calls.append(args)
        return shared, routed, 4

    runner = NS(
        _fused_output_is_reduced=reduced,
        router=ZeroExpertRouter.__new__(ZeroExpertRouter) if zero else NS(),
        _forward_outputs=outputs,
    )
    if reduced or zero:
        with pytest.raises(NotImplementedError, match="unreduced"):
            MoERunner.forward_partials(runner, routed, shared)
        assert not calls
    else:
        first, second = MoERunner.forward_partials(runner, routed, shared)
        assert second is not None
        assert first.data_ptr() == routed.data_ptr()
        assert second.data_ptr() == shared.data_ptr()
        assert first.shape == second.shape == (5, 4)
        assert len(calls) == 1


def test_compiled_separate_partials_keep_both_producers_live(monkeypatch):
    from vllm import distributed

    group = NS()
    monkeypatch.setattr(distributed, "get_tp_group", lambda: group)

    def combine(hidden, first, second, *args):
        block = (first.float() + second.float()).reshape(4, -1, 4).sum(0).half()
        output = hidden + block.repeat(1, 4)
        return output, output.clone()

    owner = NS(group=group, can_use=lambda *args: True, combine_norm=combine)
    monkeypatch.setattr(
        peer, "current_runtime_resources", lambda: {"hc_prefill_peer": owner}
    )
    library = torch.library.Library("vllm", "IMPL", "CPU")
    # An UnspecifiedPlatform harness already registers the source CPU path.
    # NVIDIA collection instead needs this test-local CPU implementation.
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "vllm::qwen38_hc_sequence_combine_norm", "CPU"
    ):
        library.impl("qwen38_hc_sequence_combine_norm", peer._combine_norm)

    @torch.compile(backend="inductor", dynamic=True, fullgraph=True)
    def compiled(x, hidden, injection, weight):
        first, second = x * 3, x * 4
        scratch = x * 100
        output, _ = peer.hc_sequence_combine_norm(
            hidden, first, injection, weight, 1e-6, False, True, True, second
        )
        return output, scratch

    for rows in (16, 17, 19, 20, 16):
        x = torch.arange(rows * 4).reshape(rows, 4).half() * 0.125
        local = (rows + 3) // 4
        hidden = torch.ones(local, 16, dtype=torch.half)
        injection = torch.ones(local, 4, dtype=torch.half)
        weight = torch.ones(16, dtype=torch.half)
        output, scratch = compiled(x, hidden, injection, weight)
        expected = peer._combine_norm(
            hidden, x * 3, injection, weight, 1e-6, False, True, True, x * 4
        )[0]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(scratch, x * 100, rtol=0, atol=0)
    del library


def test_peer_borrows_only_the_current_engines_owner(monkeypatch):
    from vllm import distributed

    group = NS()
    monkeypatch.setattr(distributed, "get_tp_group", lambda: group)
    calls = []

    def owner(number):
        def execute(hidden, *args):
            calls.append(number)
            return hidden + number, hidden + number

        return NS(group=group, can_use=lambda *args: True, combine_norm=execute)

    engines = [NS(), NS()]
    for number, cfg in enumerate(engines, 1):
        runtime_resources_for(cfg)["hc_prefill_peer"] = owner(number)
    hidden = torch.zeros(1, 10240, dtype=torch.half)
    args = (
        hidden,
        torch.zeros(4, 2560, dtype=torch.half),
        torch.zeros(1, 4, dtype=torch.half),
        torch.zeros(10240, dtype=torch.half),
        1e-6,
        False,
        True,
        True,
    )
    for number in (1, 2, 1):
        monkeypatch.setattr(
            peer,
            "current_runtime_resources",
            lambda number=number: runtime_resources_for(engines[number - 1]),
        )
        assert torch.all(peer._combine_norm(*args)[0] == number)
    assert calls == [1, 2, 1]


def test_peer_release_closes_only_owned_ipc_mappings_once(monkeypatch):
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    closed: list[int] = []
    freed: list[int] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.ops._C, "sm70_ring_close_mem_handle", closed.append)
    monkeypatch.setattr(
        CustomAllreduce,
        "free_shared_buffer",
        lambda pointers, rank: freed.append(pointers[rank]),
    )
    engines = [NS(), NS()]
    owners = []
    for base, cfg in zip((10, 20), engines):
        owner = peer.HcPrefillPeerRuntime.__new__(peer.HcPrefillPeerRuntime)
        owner.pointers = [base + i for i in range(4)]
        owner.rank, owner.device = 2, torch.device("cuda", 0)
        runtime_resources_for(cfg)["hc_prefill_peer"] = owner
        owners.append(owner)
    release_runtime_resources(engines[0])
    owners[0].close()
    assert freed == [12] and closed == [10, 11, 13]
    assert owners[1].pointers == [20, 21, 22, 23]
    release_runtime_resources(engines[1])
    assert freed == [12, 22] and closed == [10, 11, 13, 20, 21, 23]


def test_ragged_rows_pad_only_the_carrier_and_do_not_alias_input(monkeypatch):
    from vllm import distributed

    source = torch.arange(10).reshape(5, 2)
    outputs = []
    for rank in range(4):
        monkeypatch.setattr(
            distributed, "get_tp_group", lambda rank=rank: NS(rank_in_group=rank)
        )
        outputs.append(peer._shard_rows(source))
    expected = torch.cat((source.clone(), torch.zeros(3, 2, dtype=source.dtype)))
    source.zero_()
    assert torch.equal(torch.cat(outputs), expected)


def test_peer_native_allocation_follows_loading_cache_release(monkeypatch):
    from torch import nn

    from vllm.model_executor.layers.quantization import sm70_dmv13_projection as dense
    from vllm.models.qwen4_exp.nvidia import model, sm70_fp16_hc

    owner = model.Qwen4ExpForCausalLM.__new__(model.Qwen4ExpForCausalLM)
    nn.Module.__init__(owner)
    owner.model_config = NS(dtype=torch.half)
    owner.vllm_config = NS(kernel_config=NS(hc_weight_storage="sharded"))
    owner._hc_prefill_sequence_qualified = True
    owner.model = nn.Module()
    owner.model.start_layer = 0
    owner.model.layers = nn.ModuleList([nn.Module()])
    layer = owner.model.layers[0]
    layer.attn_hyper_connection = nn.Module()
    layer.attn_hyper_connection.hc_norm = nn.Linear(1, 1)
    live_loading_pages = [True]
    monkeypatch.setattr(model, "enable_qwen38_sm70_fp16_gemv", lambda *args: None)
    monkeypatch.setattr(model, "enable_qwen38_sm70_fp16_fused_hc", lambda *args: None)
    monkeypatch.setattr(dense, "share_prepared_banks", lambda module: 0)
    monkeypatch.setattr(sm70_fp16_hc, "prepare_sharded_hc_storage", lambda *args: None)
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: live_loading_pages.clear()
    )

    def allocate(*args):
        assert not live_loading_pages, "native IPC cannot reclaim Torch's cache"

    monkeypatch.setattr(peer, "prepare_hc_prefill_peer_runtime", allocate)
    owner.prepare_loaded_linear_weights()
