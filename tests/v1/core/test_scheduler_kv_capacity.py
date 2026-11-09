# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for startup admitting a request that can never finish."""

from dataclasses import replace

import pytest
import torch

from vllm.v1.core.kv_cache_utils import (
    _estimate_max_model_len_from_groups,
    _max_memory_usage_bytes_from_groups,
    _pool_bytes_per_block,
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    get_kv_cache_groups,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

SPAN = 1632
POOL_BLOCK_BYTES = 7_206_912
SMALL_POOL = 512 << 20


def make_config(*, dcp=2, max_model_len=50176, speculative_tokens=4):
    original = create_scheduler(
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        num_speculative_tokens=speculative_tokens or None,
        max_num_seqs=8,
        block_size=16,
        long_prefill_token_threshold=4896,
    )
    config = original.vllm_config
    # OPT supplies a cached CPU-only config; the scheduler tests use the
    # hybrid specs below, without loading weights or executing a model.
    config.model_config.max_model_len = max_model_len
    config.model_config.model_arch_config = replace(
        config.model_config.model_arch_config, total_num_kv_heads=4
    )
    config.cache_config.mamba_cache_mode = "align"
    config.parallel_config.tensor_parallel_size = 4
    config.parallel_config.decode_context_parallel_size = dcp
    # Constructing an MTP draft would load another model config. Changing the
    # already validated ngram config selects the real MTP scheduler semantics.
    if config.speculative_config is not None:
        config.speculative_config.method = "mtp"
    return config


def make_specs(*, dcp=2, speculative_tokens=4):
    specs = {}
    for layer in range(13):
        name = f"model.layers.{layer}.self_attn"
        specs[name] = FullAttentionSpec(
            block_size=SPAN // dcp if layer < 12 else SPAN,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.uint8,
            dcp_sharded=layer < 12,
        )
        specs[name + ".compressed"] = MLAAttentionSpec(
            block_size=SPAN,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            compress_ratio=4,
            dcp_sharded=False,
        )
        specs[name + ".compressor_state"] = CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float16,
            dcp_sharded=False,
        )
    for layer in range(36):
        specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
            block_size=SPAN,
            shapes=((1, 408576),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=speculative_tokens,
        )
    specs["model.layers.0.ple"] = MambaSpec(
        block_size=SPAN,
        shapes=((1, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=speculative_tokens,
        tp_replicated=True,
    )
    return specs


def make_scheduler(config, worker_cache):
    return Scheduler(
        vllm_config=config,
        kv_cache_config=generate_scheduler_kv_cache_config([worker_cache]),
        structured_output_manager=StructuredOutputManager(config),
        block_size=16,
        hash_block_size=16,
    )


def step(scheduler):
    output = scheduler.schedule()
    ids = list(output.num_scheduled_tokens)
    if ids:
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={rid: i for i, rid in enumerate(ids)},
                # A partial prefill produces no sampled token. Only the
                # completed prompt and decode steps return a model output.
                sampled_token_ids=[
                    [1]
                    if scheduler.requests[rid].num_computed_tokens
                    >= scheduler.requests[rid].num_prompt_tokens
                    else []
                    for rid in ids
                ],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    return output


def test_startup_rejects_observed_hybrid_livelock():
    config = make_config()
    specs = make_specs()
    groups = get_kv_cache_groups(config, specs)
    assert len(groups) == 9  # attention, compressor ring, seven Mamba groups
    assert _pool_bytes_per_block(groups) == POOL_BLOCK_BYTES
    assert _max_memory_usage_bytes_from_groups(config, groups) == 74 * POOL_BLOCK_BYTES
    assert SMALL_POOL // POOL_BLOCK_BYTES == 74
    # A 49K prompt + 64 outputs passes the request's length validation.
    assert config.model_config.max_model_len >= 49152 + 64
    with pytest.raises(ValueError, match="estimated maximum model length is 48960"):
        get_kv_cache_configs(config, [specs], [SMALL_POOL])


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_uniform_cache_also_reserves_null(prefix_caching):
    config = make_config(dcp=1, max_model_len=32, speculative_tokens=0)
    config.cache_config.enable_prefix_caching = prefix_caching
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    with pytest.raises(ValueError, match="estimated maximum model length is 16"):
        get_kv_cache_configs(config, [{"layer": spec}], [2 * spec.page_size_bytes])


@pytest.mark.parametrize("dcp", [1, 2])
@pytest.mark.parametrize("speculative_tokens", [0, 4])
def test_estimate_reserves_null_and_respects_owner_dcp(dcp, speculative_tokens):
    config = make_config(
        dcp=dcp, max_model_len=65536, speculative_tokens=speculative_tokens
    )
    groups = get_kv_cache_groups(
        config, make_specs(dcp=dcp, speculative_tokens=speculative_tokens)
    )
    stride = _pool_bytes_per_block(groups)
    request_blocks = _max_memory_usage_bytes_from_groups(config, groups) // stride
    # Memory that fits the request exactly has no room for the null block.
    estimated = _estimate_max_model_len_from_groups(
        config, groups, request_blocks * stride
    )
    assert estimated == 40 * SPAN
    assert config.model_config.max_model_len == 65536  # estimation restores config
    assert (
        _estimate_max_model_len_from_groups(
            config, groups, (request_blocks + 1) * stride
        )
        == 65536
    )


@pytest.mark.parametrize("extra_bytes", [-1, 0])
def test_exact_hybrid_pool_boundary(extra_bytes):
    config = make_config()
    specs = make_specs()
    budget = 75 * POOL_BLOCK_BYTES + extra_bytes
    if extra_bytes < 0:
        with pytest.raises(ValueError, match="max seq len"):
            get_kv_cache_configs(config, [specs], [budget])
    else:
        (cache,) = get_kv_cache_configs(config, [specs], [budget])
        assert cache.num_blocks == 75
        scheduler = make_scheduler(config, cache)
        (request,) = create_requests(
            1, num_tokens=49152, max_tokens=64, ignore_eos=True
        )
        scheduler.add_request(request)
        for _ in range(100):
            step(scheduler)
            if not scheduler.has_unfinished_requests():
                break
        assert not scheduler.has_unfinished_requests()
        assert request.num_preemptions == 0
        assert len(request.output_token_ids) == 64


def test_autofit_and_block_override_use_same_reservation():
    config = make_config(max_model_len=65536)
    config.model_config.original_max_model_len = -1
    config.cache_config.num_gpu_blocks_override = 74
    (cache,) = get_kv_cache_configs(config, [make_specs()], [1 << 30])
    assert cache.num_blocks == 74
    assert config.model_config.max_model_len == 48960


@pytest.mark.parametrize("warm_prefix", [False, True])
def test_cpu_reproduction_when_startup_guard_is_bypassed(warm_prefix):
    """Both revisions expose the same allocator failure if admission is bypassed."""
    config = make_config()
    groups = get_kv_cache_groups(config, make_specs())
    # This low-level allocator intentionally bypasses startup validation.
    cache = get_kv_cache_config_from_groups(config, groups, SMALL_POOL)
    scheduler = make_scheduler(config, cache)
    if warm_prefix:
        for i, prompt_len in enumerate([32768, 16, 16, 16]):
            (warmup,) = create_requests(
                1,
                num_tokens=prompt_len,
                max_tokens=64,
                ignore_eos=True,
                same_prompt=True,
                req_ids=[f"warmup-{i}"],
            )
            scheduler.add_request(warmup)
            for _ in range(100):
                step(scheduler)
                if not scheduler.has_unfinished_requests():
                    break
            assert not scheduler.has_unfinished_requests()
            assert len(warmup.output_token_ids) == 64
    (request,) = create_requests(
        1, num_tokens=49152, max_tokens=64, ignore_eos=True, same_prompt=True
    )
    scheduler.add_request(request)
    if warm_prefix:
        for _ in range(8):
            assert step(scheduler).num_scheduled_tokens == {}
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 73
    else:
        for _ in range(100):
            step(scheduler)
        assert request.num_preemptions > 1
    assert scheduler.has_unfinished_requests()


@pytest.mark.parametrize("blocks,max_model_len", [(111, 65536), (794, 524288)])
def test_larger_pool_finishes_warm_workload(blocks, max_model_len):
    config = make_config(max_model_len=max_model_len)
    (cache,) = get_kv_cache_configs(config, [make_specs()], [blocks * POOL_BLOCK_BYTES])
    assert cache.num_blocks == blocks
    scheduler = make_scheduler(config, cache)
    for i, prompt_len in enumerate([32768, 16, 16, 16, 49152]):
        (request,) = create_requests(
            1,
            num_tokens=prompt_len,
            max_tokens=64,
            ignore_eos=True,
            req_ids=[str(i)],
            same_prompt=True,
        )
        scheduler.add_request(request)
        for _ in range(110):
            output = step(scheduler)
            assert output.num_scheduled_tokens
            if not scheduler.has_unfinished_requests():
                break
        assert not scheduler.has_unfinished_requests()
        assert request.num_preemptions == 0
        assert len(request.output_token_ids) == 64
