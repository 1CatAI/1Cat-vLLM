# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only validation of the offline qualification transport."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.benchmark_qwen38_dcp_quality import (
    validate_manifest_transport,
    worker_manifest,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test


def test_manifest_transport_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    with pytest.raises(TypeError, match="VLLM_ALLOW_INSECURE_SERIALIZATION"):
        validate_manifest_transport()


def test_manifest_callable_and_result_roundtrip(monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    validate_manifest_transport()
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(torch.accelerator, name, lambda: 1234)
    worker = SimpleNamespace(
        rank=0,
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                cache_dtype="fp8_e4m3",
                mamba_ssm_cache_dtype="float32",
                enable_prefix_caching=False,
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=2, dcp_comm_backend="a2a"
            ),
            speculative_config=None,
            compilation_config=SimpleNamespace(cudagraph_mode="FULL_AND_PIECEWISE"),
        ),
        model_runner=SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                num_blocks=32,
                kv_cache_tensors=[SimpleNamespace(size=4096)],
                kv_cache_groups=[
                    SimpleNamespace(
                        layer_names=["target"],
                        kv_cache_spec=SimpleNamespace(
                            block_size=32, page_size_bytes=128, dcp_sharded=True
                        ),
                    )
                ],
            )
        ),
    )
    encoder, decoder = MsgpackEncoder(), MsgpackDecoder()
    callback = decoder.decode(encoder.encode(worker_manifest))
    manifest = callback(worker)
    decoded = decoder.decode(encoder.encode(manifest))
    assert decoded == manifest
    assert decoded["physical_kv_bytes"] == 4096
    assert decoded["dcp"] == 2
