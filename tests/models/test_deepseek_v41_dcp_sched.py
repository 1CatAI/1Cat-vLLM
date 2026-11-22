# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek-V4.1 TP4 KV-sharded DCP specs, configurable MM
encoder cache size, and PP decode-protected microbatch scheduling."""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.config.scheduler import SchedulerConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


def _make_stage_specs(kv_shard: bool, block: int = 16):
    """Simulate PP=4 (8,12,12,8) KVCacheSpecs for DeepSeek-V4.1 + DSpark."""
    tp = 4
    stage_ranges = [(0, 8), (8, 20), (20, 32), (32, 40)]
    kv_sources = {2: 2, 8: 2, 14: 2, 20: 1}
    specs_per_stage = []
    for stage_idx, (start, end) in enumerate(stage_ranges):
        worker_spec = {}
        # Mirror layer 20 on stage 3
        if stage_idx == 3:
            sources = [(20, 1)]
        else:
            sources = [(lid, r) for lid, r in kv_sources.items() if start <= lid < end]
        for layer_id, ratio in sources:
            for suffix, head_dim in (
                ("compressor.cache", 512),
                ("indexer.k_cache", 128),
            ):
                name = f"model.layers.{layer_id}.attn.{suffix}"
                if kv_shard:
                    mla_block = block * tp
                    worker_spec[name] = MLAAttentionSpec(
                        block_size=mla_block,
                        num_kv_heads=1,
                        head_size=head_dim,
                        dtype=torch.float16,
                        compress_ratio=ratio * tp,
                        alignment=(mla_block // tp) * head_dim * 2,
                    )
                else:
                    worker_spec[name] = MLAAttentionSpec(
                        block_size=block,
                        num_kv_heads=1,
                        head_size=head_dim,
                        dtype=torch.float16,
                        compress_ratio=ratio,
                        alignment=block * 512 * 2,
                    )
        for layer_id in range(start, end):
            worker_spec[f"model.layers.{layer_id}.attn.window_cache"] = (
                SlidingWindowMLASpec(
                    sliding_window=140,
                    block_size=block,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.float16,
                    compress_ratio=1,
                    alignment=block * 512 * 2,
                )
            )
        if stage_idx == 3:
            worker_spec["dspark.attn.window_cache"] = SlidingWindowMLASpec(
                sliding_window=140,
                block_size=block,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.float16,
                compress_ratio=1,
                alignment=block * 512 * 2,
            )
        specs_per_stage.append(worker_spec)
    return specs_per_stage


def test_v41_kv_shard_env_vars_registered(monkeypatch: pytest.MonkeyPatch):
    assert envs.VLLM_V41_KV_SHARD == "0"
    assert envs.VLLM_V41_DECODE_PROTECT == "1"
    monkeypatch.setenv("VLLM_V41_KV_SHARD", "1")
    monkeypatch.setenv("VLLM_V41_DECODE_PROTECT", "0")
    assert envs.VLLM_V41_KV_SHARD == "1"
    assert envs.VLLM_V41_DECODE_PROTECT == "0"


def test_scheduler_config_configurable_encoder_cache_size():
    cfg_default = SchedulerConfig(
        max_num_batched_tokens=4096,
        max_num_seqs=64,
        max_model_len=65536,
        is_multimodal_model=True,
        is_encoder_decoder=False,
    )
    assert cfg_default.encoder_cache_size == 32768
    cfg_text = SchedulerConfig(
        max_num_batched_tokens=4096,
        max_num_seqs=64,
        max_model_len=65536,
        is_multimodal_model=False,
        is_encoder_decoder=False,
    )
    assert cfg_text.encoder_cache_size == 4096

    cfg_custom = SchedulerConfig(
        max_num_batched_tokens=4096,
        max_num_seqs=64,
        max_model_len=65536,
        is_multimodal_model=True,
        is_encoder_decoder=False,
        encoder_cache_size=32768,
    )
    assert cfg_custom.encoder_cache_size == 32768

    # Lower than max_num_batched_tokens clamps up safely
    cfg_clamped = SchedulerConfig(
        max_num_batched_tokens=4096,
        max_num_seqs=64,
        max_model_len=65536,
        is_multimodal_model=True,
        is_encoder_decoder=False,
        encoder_cache_size=1024,
    )
    assert cfg_clamped.encoder_cache_size == 4096


def test_v41_dcp4_pp4_kv_capacity_6_4x_gain():
    vllm_cfg = SimpleNamespace(
        max_in_flight_tokens=4096,
        model_config=SimpleNamespace(
            max_model_len=1048576, original_max_model_len=1048576
        ),
        cache_config=SimpleNamespace(
            block_size=16,
            num_gpu_blocks_override=None,
            enable_prefix_caching=True,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=4096,
            max_num_seqs=64,
            disable_hybrid_kv_cache_manager=False,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=4,
            tensor_parallel_size=4,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        speculative_config=None,
    )
    avail = [4_750_000_000] * 4
    cfgs_unsharded = get_kv_cache_configs(vllm_cfg, _make_stage_specs(False), avail)
    cfgs_sharded = get_kv_cache_configs(vllm_cfg, _make_stage_specs(True), avail)

    # Unsharded: 72,479 blocks of 16 tokens (~1.10M tokens)
    # Sharded DCP=4 + 4096B index page: >115k blocks of 64 tokens (>7M, ~6.4x!)
    tokens_u = cfgs_unsharded[1].num_blocks * 16
    tokens_s = cfgs_sharded[1].num_blocks * 64
    assert tokens_s > tokens_u * 6.0, (
        f"Expected >6x KV capacity, got {tokens_s} vs {tokens_u}"
    )
