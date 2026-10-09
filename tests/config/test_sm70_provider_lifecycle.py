# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine isolation and retained parser/admission behavior without model weights."""

import multiprocessing
from contextvars import copy_context
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from tests.config.runtime_policy_utils import make_policy_defaults
from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.config.gdn_projection import projection_policy
from vllm.config.sm70_moe import unquantized_moe_policy
from vllm.config.sm70_sparse import Sm70SparseConfig, sparse_policy
from vllm.config.speculative_sampling import (
    SpeculativeSamplingPolicy,
    mtp_batch_enabled,
)
from vllm.diagnostics import diagnostic_history
from vllm.runtime_resources import release_runtime_resources, runtime_resources_for


def _engine(value):
    defaults = make_policy_defaults()
    cfg = defaults.cfg
    cfg.kernel_config.sm70_sparse.qsa_indexer_cublas = value
    cfg.kernel_config.sm70_sparse.qsa_score_tile_mb = 64 if value else 8
    cfg.kernel_config.sm70_sparse.active = True
    cfg.kernel_config.gdn.projection.input_batch = value
    cfg.kernel_config.gdn.projection.resolve()
    cfg.kernel_config.sm70_moe.unquantized.mtp_tuned = value
    cfg.kernel_config.sm70_moe.unquantized.resolve()
    cfg.speculative_config.sampling_policy = SpeculativeSamplingPolicy(
        shared_batch=value
    )
    cfg.speculative_config.sampling_policy.resolve()
    defaults.finish()
    return cfg


@pytest.mark.parametrize("order", [(True, False), (False, True)])
def test_provider_policies_and_resources_are_engine_local(monkeypatch, order):
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: NS(cuda_stream=123)
    )
    engines = [_engine(value) for value in order]
    hashes = [cfg.kernel_config.sm70_sparse.compute_hash() for cfg in engines]
    assert hashes[0] != hashes[1]
    aliases = set(Sm70SparseConfig.aliases.values())
    aliases.update(engines[0].kernel_config.gdn.projection.aliases.values())
    aliases.update(engines[0].kernel_config.sm70_moe.unquantized.aliases.values())
    aliases.update(SpeculativeSamplingPolicy.aliases.values())
    for alias in aliases:
        monkeypatch.setenv(alias, "invalid-after-initialization")
        monkeypatch.setitem(
            envs.environment_variables, alias, Mock(side_effect=AssertionError(alias))
        )
    pointers: dict[bool, int] = {}
    for cfg, expected in list(zip(engines, order)) * 2:
        with set_current_vllm_config(cfg):
            assert sparse_policy().qsa_indexer_cublas is expected
            assert projection_policy().input_batch is expected
            assert mtp_batch_enabled("shared_batch") is expected
            assert unquantized_moe_policy().mtp_tuned is expected
            workspace = qsa._qsa_xqa_page4_workspace(torch.zeros(3, 6, 256), 8, "auto")[
                0
            ]
            assert (
                pointers.setdefault(expected, workspace.data_ptr())
                == workspace.data_ptr()
            )
            history = diagnostic_history("glm_kda_seen")
            assert history.get("owner", expected) is expected
            history["owner"] = expected
    assert pointers[True] != pointers[False]
    second_resources = runtime_resources_for(engines[1])
    release_runtime_resources(engines[0])
    assert second_resources["attention_workspaces"].caches


@pytest.mark.parametrize(
    "raw,boolean",
    [
        (None, True),
        ("1", True),
        ("0", False),
        ("", False),
        ("true", False),
        ("2", False),
    ],
)
def test_sparse_boolean_dialect_and_typed_priority(monkeypatch, raw, boolean):
    alias = Sm70SparseConfig.aliases["qsa_xqa_page4"]
    if raw is None:
        monkeypatch.delenv(alias, raising=False)
    else:
        monkeypatch.setenv(alias, raw)
    legacy = Sm70SparseConfig()
    legacy.resolve()
    typed = Sm70SparseConfig(qsa_xqa_page4=not boolean)
    typed.resolve()
    assert legacy.value("qsa_xqa_page4") is boolean
    assert typed.value("qsa_xqa_page4") is not boolean
    assert typed.sources["qsa_xqa_page4"] == "typed"


def test_sparse_worker_transfer_preserves_errors_and_unused_hash(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QSA_INDEXER_SCORE_TILE_MB", "")
    policy = Sm70SparseConfig()
    policy.resolve()
    policy.validate_active()  # An unused sparse family retains its deferred error.
    assert (
        policy.compute_hash() == Sm70SparseConfig(qsa_score_tile_mb=12).compute_hash()
    )
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        sender.send(policy)
        transferred = receiver.recv()
    finally:
        sender.close()
        receiver.close()
    monkeypatch.setenv("VLLM_SM70_QSA_INDEXER_SCORE_TILE_MB", "64")
    transferred.resolve()
    with pytest.raises(ValueError, match="invalid literal"):
        transferred.value("qsa_score_tile_mb")
    transferred.active = True
    with pytest.raises(ValueError, match="invalid literal"):
        transferred.validate_active()


def test_warmup_override_is_context_local_and_restored_after_error():
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _get_sm70_mtp_moe_decode_config,
        force_sm70_mtp_moe_legacy_config,
    )

    cfg = _engine(True)
    shape = (2, 256, 128, 2048, 8)
    with set_current_vllm_config(cfg):
        independent = copy_context()
        with (
            pytest.raises(RuntimeError, match="warmup failed"),
            force_sm70_mtp_moe_legacy_config(),
        ):
            assert _get_sm70_mtp_moe_decode_config(*shape) is None
            assert independent.run(_get_sm70_mtp_moe_decode_config, *shape) is not None
            raise RuntimeError("warmup failed")
        assert _get_sm70_mtp_moe_decode_config(*shape) is not None


def test_sampler_captures_policy_before_later_environment_changes(monkeypatch):
    from vllm.v1.sample.sampler import Sampler

    cfg = _engine(True)
    cfg.kernel_config.layer_execution.compact_topk20 = True
    cfg.kernel_config.layer_execution.chunked_topk20_chunks = 8
    with set_current_vllm_config(cfg):
        sampler = Sampler()
    monkeypatch.setenv("VLLM_SM70_COMPACT_TOPK20_SAMPLER", "0")
    monkeypatch.setenv("VLLM_SM70_CHUNKED_TOPK20_CHUNKS", "invalid")
    assert sampler._layer_policy.value("compact_topk20")
    assert sampler._layer_policy.value("chunked_topk20_chunks") == 8
    # A rejected dtype/shape never parses the chunk count or invokes a native op.
    metadata = NS(
        max_num_logprobs=None,
        logprob_token_ids=None,
        all_random=True,
        top_k_cpu=(20,),
        top_p_cpu=(0.95,),
        temperature_cpu=(1.0,),
    )
    assert (
        Sampler._try_sm70_compact_topk20_sample(
            torch.zeros(1, 20), metadata, "raw_logprobs", sampler._layer_policy
        )
        is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capture lifecycle")
def test_qsa_capture_keeps_each_engine_buffers_after_growth_and_replay():
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    engines = [_engine(value) for value in (True, False)]
    records = []
    for cfg in engines:
        q = torch.ones(3, 6, 256, device="cuda", dtype=torch.float16)
        stream = torch.cuda.Stream()
        torch.accelerator.synchronize()
        with set_current_vllm_config(cfg), torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = qsa._qsa_xqa_page4_workspace(q, 8, "fp8_e4m3")[0]
                output.copy_(q.sum(dtype=torch.float32))
            enlarged = qsa._qsa_xqa_page4_workspace(
                torch.ones(7, 6, 256, device="cuda", dtype=torch.float16),
                8,
                "fp8_e4m3",
            )[0]
            assert enlarged.data_ptr() != output.data_ptr()
        records.append((q, graph, output))
    assert records[0][2].data_ptr() != records[1][2].data_ptr()
    for value, cfg, (q, graph, output) in zip((2, 3), engines, records):
        with set_current_vllm_config(cfg):
            q.fill_(value)
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(output, torch.full_like(output, value * q.numel()))
    records[0][1].reset()
    release_runtime_resources(engines[0])
    records[1][0].fill_(4)
    records[1][1].replay()
    torch.accelerator.synchronize()
    assert torch.equal(
        records[1][2], torch.full_like(records[1][2], 4 * records[1][0].numel())
    )
    records[1][1].reset()
    release_runtime_resources(engines[1])


def test_online_native_policy_hashes_without_enabling_serialized_fp8():
    from vllm.config import KernelConfig

    first, second = KernelConfig(), KernelConfig()
    second.sm70_fp8.native.fp8_tune_small_shapes = True
    assert first.compute_hash() == second.compute_hash()
    for cfg in (first, second):
        cfg.sm70_fp8.native.resolve("fp8")
        assert not cfg.sm70_fp8.resolved
    assert first.compute_hash() != second.compute_hash()
