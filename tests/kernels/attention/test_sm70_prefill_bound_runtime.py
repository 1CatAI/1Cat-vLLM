# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA2 engine policy and ownership without loading a model."""

import pytest
import torch

from tests.config.test_sm70_prefill_runtime import engine
from vllm.config import set_current_vllm_config
from vllm.runtime_resources import release_runtime_resources, runtime_resources_for
from vllm.v1.attention.backends.flash_v100.runtime import (
    bind_prefill_operation,
    prepare_prefill_runtime,
)


@pytest.fixture
def native():
    if not torch.accelerator.is_available() or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("requires SM70")
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    load_fa2_library(torch.device("cuda"))
    assert torch.ops._vllm_fa2_C.sm70_prefill_policy_abi() == 1
    return torch.ops._vllm_fa2_C


def bind(config, native, length):
    name = (
        "sm70_d256_gqa_architecture_fwd"
        if length == 8000
        else "sm70_d256_gqa_architecture_q8192_fwd"
    )
    with set_current_vllm_config(config):
        prepare_prefill_runtime(native)
        return bind_prefill_operation(getattr(native, name), length)


def inputs(length, kv_len):
    q = torch.randn(1, length, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    return q, k, torch.randn_like(k), torch.empty_like(q)


@pytest.mark.parametrize("length", [8000, 8192])
@torch.inference_mode()
def test_bound_matches_legacy_and_rejects_original_shape_errors(native, length):
    config = engine()
    op = bind(config, native, length)
    legacy = getattr(
        native,
        "sm70_d256_gqa_architecture_fwd"
        if length == 8000
        else "sm70_d256_gqa_architecture_q8192_fwd",
    )
    torch.manual_seed(1308)
    for kv_len in (length, length + 8192):
        q, k, v, out = inputs(length, kv_len)
        reference = torch.empty_like(out)
        legacy(q, k, v, reference, 0.0625, True)
        op(q, k, v, out, 0.0625, True)
        assert torch.equal(out, reference)
        for execute in (legacy, op):
            with pytest.raises(RuntimeError, match="32-token alignment"):
                execute(q, k[:, :-1], v[:, :-1], out, 0.0625, True)
    observations = runtime_resources_for(config)["sm70_prefill"].explain()
    assert observations[f"q{length}"] == 2
    release_runtime_resources(config)
    with pytest.raises(RuntimeError, match="has been closed"):
        op(q, k, v, out, 0.0625, True)


@torch.inference_mode()
def test_two_policies_keep_capture_buffers_and_device_global_order(native, monkeypatch):
    torch.manual_seed(1309)
    configs = [engine(prefill_score_block_tokens=size) for size in (8192, 16384)]
    cases = []
    for cfg in configs:
        op = bind(cfg, native, 8192)
        q, k, v, out = inputs(8192, 16384)
        op(q, k, v, out, 0.0625, True)
        expected = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(q, k, v, out, 0.0625, True)
        cases.append((op, q, k, v, out, graph, expected))
    # Changes after binding must not alter either engine, including malformed
    # workspace input and return-affecting presence flags.
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS", "bad")
    monkeypatch.setenv("PREFIX_TORCH_EXACT_TAIL", "")
    monkeypatch.setenv("PREFIX_QK_CUBLAS_ALGO_RUNTIME", "-999")
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.accelerator.synchronize()
    for _ in range(3):
        for index in (1, 0):
            op, q, k, v, out, graph, _ = cases[index]
            with torch.cuda.stream(streams[index]):
                graph.replay()
        torch.accelerator.synchronize()
        for _, _, _, _, out, _, expected in cases:
            assert torch.equal(out, expected)
    # A later allocation for the other query family must keep both graphs'
    # score and host-metadata addresses alive.
    for cfg in configs:
        other = bind(cfg, native, 8000)
        other(*inputs(8000, 16000), 0.0625, True)
    for op, q, k, v, out, graph, expected in cases:
        q.mul_(0.5)
        op(q, k, v, out, 0.0625, True)
        expected.copy_(out)
        graph.replay()
        assert torch.equal(out, expected)
    # Destroy graph first, then its owner. The other owner remains usable.
    cases[0][5].reset()
    release_runtime_resources(configs[0])
    cases[1][5].replay()
    assert torch.equal(cases[1][4], cases[1][6])
    cases[1][5].reset()
    release_runtime_resources(configs[1])


@torch.inference_mode()
def test_invalid_block_size_stays_deferred_to_qualified_workspace(native, monkeypatch):
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS", "8192 ")
    config = engine()
    op = bind(config, native, 8192)  # Invalid legacy size does not fail loading.
    with pytest.raises(RuntimeError, match="multiple of 8192 between 8192 and 131072"):
        op(*inputs(8192, 8192), 0.0625, True)
    release_runtime_resources(config)
