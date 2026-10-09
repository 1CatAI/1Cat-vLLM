# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from multiprocessing import Pipe
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.config.execution_policy import FlashV100Policy, GraphPolicy
from vllm.distributed.device_communicators.collective_provider import (
    CollectiveCapabilities,
    gemma_fusion_modes,
)
from vllm.platforms.sm70 import graph_policy as module


def _transfer(value):
    sender, receiver = Pipe()
    try:
        sender.send(value)
        return receiver.recv()
    finally:
        sender.close()
        receiver.close()


def _config(**overrides):
    graph = GraphPolicy(**overrides)
    graph.resolve()
    flash = FlashV100Policy(enabled=True)
    flash.resolve()
    return NS(
        compilation_config=NS(runtime=graph),
        attention_config=NS(flash_v100=flash, backend=None),
        speculative_config=None,
        model_config=NS(
            max_model_len=65536,
            hf_config=NS(),
            hf_text_config=NS(
                num_attention_heads=48, num_key_value_heads=8, head_dim=256
            ),
        ),
        cache_config=NS(cache_dtype="fp8_e5m2"),
    )


@pytest.fixture
def sm70(monkeypatch):
    monkeypatch.setattr(module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        module.current_platform, "is_device_capability", lambda *_: True
    )
    monkeypatch.setattr(
        module.current_platform, "is_device_capability_family", lambda *_: True
    )


def test_graph_plan_explicit_empty_and_worker_isolation(sm70, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FP8_KV_DECODE_CONTEXT_BUCKETS", "4096")
    first = _config(fp8_context_buckets="")
    second = _config(fp8_context_buckets=(8192, 4096, 8192))
    plans = [module.resolve_graph_plan(c) for c in (first, second)]
    monkeypatch.setenv("VLLM_SM70_FP8_KV_DECODE_CONTEXT_BUCKETS", "123")
    assert plans[0].fp8_buckets == ()
    assert plans[1].fp8_buckets == (4096, 8192)
    assert module.resolve_graph_plan(_transfer(second)) == plans[1]


@pytest.mark.parametrize("maximum,expected", [(8192, ()), (8193, (8192,))])
def test_auto_bucket_boundary(sm70, monkeypatch, maximum, expected):
    monkeypatch.delenv("VLLM_SM70_FP8_KV_DECODE_CONTEXT_BUCKETS", raising=False)
    cfg = _config()
    cfg.model_config.max_model_len = maximum
    assert module.resolve_graph_plan(cfg).fp8_buckets == expected


def test_non_sm70_and_explicit_partition_fallback(sm70, monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FP8_KV_DECODE_CONTEXT_BUCKETS", raising=False)
    cfg = _config(batch_context_routing=True, decode_partition_size="256")
    assert not module.resolve_graph_plan(cfg).batch_context_routing
    monkeypatch.setattr(
        module.current_platform, "is_device_capability", lambda *_: False
    )
    assert module.resolve_graph_plan(cfg).fp8_buckets == ()


@pytest.mark.parametrize("value", ["0,8", "8,-1", "not-a-number"])
def test_bad_bucket_keeps_initialization_error(value):
    with pytest.raises(ValueError, match="VLLM_SM70_MTP_CONTEXT_BUCKETS"):
        module.parse_context_buckets(value, "VLLM_SM70_MTP_CONTEXT_BUCKETS")


@pytest.mark.parametrize("topology", [2, 4, 8])
@pytest.mark.parametrize("long_requested", [False, True])
def test_fusion_modes_consume_registered_capabilities(topology, long_requested):
    common = dict(
        hidden_size=5120,
        dtype=torch.float16,
        sm70=True,
        long_requested=long_requested,
        tp_size=topology,
        pp_size=1,
        speculative=False,
    )
    caps = CollectiveCapabilities(topology, True, False, 1024, 1024, True, True)
    tp2, long, push = gemma_fusion_modes(capabilities=caps, **common)
    assert tp2 == (topology == 2)
    assert long == (topology == 4 and long_requested)
    assert push == (topology == 4 and not long_requested)
    assert not gemma_fusion_modes(capabilities=None, **common)[2]
    common["sm70"] = False
    assert gemma_fusion_modes(capabilities=caps, **common) == (False, False, False)
