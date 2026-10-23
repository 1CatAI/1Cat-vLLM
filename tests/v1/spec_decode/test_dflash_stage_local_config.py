# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.config import (
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    replace,
)
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

pytestmark = pytest.mark.cpu_test


def _target_config(monkeypatch, tmp_path, pp_size, depth):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Local metadata keeps the actual VllmConfig constructor and validators
    # active without downloading a checkpoint or allocating model weights.
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_key_value_heads=4,
                num_hidden_layers=14,
                max_position_embeddings=8192,
                vocab_size=128,
            )
        )
    )
    return VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            max_model_len=512,
            dtype="float32",
            enforce_eager=True,
        ),
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=pp_size,
            distributed_executor_backend="mp",
        ),
        scheduler_config=SchedulerConfig.default_factory(
            async_scheduling=True,
            pp_async_queue_depth=depth,
            max_num_seqs=3,
            max_num_batched_tokens=512,
        ),
    )


@pytest.mark.parametrize("speculator_cls", [DFlashSpeculator, DFlash2Speculator])
@pytest.mark.parametrize("pp_size,depth", [(1, None), (7, None), (7, 2), (7, 7)])
@pytest.mark.parametrize("non_causal", [False, True])
def test_local_draft_attention_clears_only_its_pipeline_depth(
    monkeypatch, tmp_path, speculator_cls, pp_size, depth, non_causal
):
    target = _target_config(monkeypatch, tmp_path, pp_size, depth)
    target_scheduler = target.scheduler_config
    spec = object.__new__(SpeculativeConfig)
    spec.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
        target.parallel_config,
        speculative_draft_tensor_parallel_size=1,
        speculative_draft_pipeline_parallel_size=1,
    )
    # Bypass GPU buffer allocation, then exercise the production property,
    # including its real Pydantic-safe replace and VllmConfig revalidation.
    speculator = object.__new__(speculator_cls)
    speculator.vllm_config = target
    speculator.speculative_config = spec
    speculator.draft_model_config = target.model_config
    speculator.requires_non_causal = non_causal

    draft = speculator.attn_vllm_config

    assert draft.is_speculative_draft
    assert draft.parallel_config.pipeline_parallel_size == 1
    assert draft.scheduler_config.pp_async_queue_depth is None
    assert draft.scheduler_config is not target_scheduler
    assert draft.scheduler_config.async_scheduling
    assert draft.scheduler_config.max_num_seqs == 3
    assert draft.scheduler_config.max_num_batched_tokens == 512
    assert draft.attention_config.use_non_causal is non_causal
    assert target.scheduler_config is target_scheduler
    assert target.scheduler_config.pp_async_queue_depth == depth
    assert target.parallel_config.pipeline_parallel_size == pp_size
    assert not target.attention_config.use_non_causal
    assert not target.is_speculative_draft


@pytest.mark.parametrize("depth", [2, 7])
def test_partitioned_draft_retains_target_pipeline_depth(monkeypatch, tmp_path, depth):
    target = _target_config(monkeypatch, tmp_path, 7, depth)
    spec = object.__new__(SpeculativeConfig)
    spec.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
        target.parallel_config, speculative_draft_tensor_parallel_size=1
    )
    spec.draft_model_config = target.model_config
    spec.moe_backend = None
    spec.attention_backend = None
    spec.kv_cache_dtype = None
    proposer = object.__new__(DraftModelProposer)
    proposer.vllm_config = target
    proposer.speculative_config = spec

    draft = proposer._create_draft_vllm_config()

    assert draft.parallel_config.pipeline_parallel_size == 7
    assert draft.scheduler_config.pp_async_queue_depth == depth
    assert target.scheduler_config.pp_async_queue_depth == depth


def test_target_pipeline_validation_still_rejects_local_depth(monkeypatch, tmp_path):
    target = _target_config(monkeypatch, tmp_path, 7, 2)
    with pytest.raises(ValueError, match="between 2 and pipeline_parallel_size"):
        replace(
            target,
            parallel_config=replace(target.parallel_config, pipeline_parallel_size=1),
        )
