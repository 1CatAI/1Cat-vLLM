# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

import vllm.envs as envs
from vllm import PoolingParams, SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    CrossAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

# Generation-config defaults that select sampling kernels.
_DEFAULT_SAMPLING_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repetition_penalty",
)


def _kernel_prefill_warmup_token_counts(
    model_runner: GPUModelRunner,
    default_prompt_len: int,
) -> tuple[int, ...]:
    """Collect per-request prefill sizes advertised by active kernel owners."""
    max_tokens = min(
        model_runner.scheduler_config.max_num_batched_tokens,
        model_runner.max_model_len,
    )
    token_counts = {default_prompt_len}
    static_context = model_runner.compilation_config.static_forward_context
    for layer in static_context.values():
        for token_count in getattr(layer, "kernel_warmup_prefill_token_counts", ()):
            if (
                isinstance(token_count, int)
                and not isinstance(token_count, bool)
                and default_prompt_len < token_count <= max_tokens
            ):
                token_counts.add(token_count)
    return tuple(sorted(token_counts))


def _coverage_prefill_token_counts(
    model_runner: GPUModelRunner,
    default_prompt_len: int,
    kv_cache_specs: list[KVCacheSpec],
) -> set[int]:
    """Prompt lengths that fill each power-of-two per-request query bucket.

    Kernels size per-request blocks from the next power of two of the query
    length, which a speculative drafter extends by 1 + num_speculative_steps
    tokens, and Triton specializes lengths divisible by 16 separately. A
    prefill chunk never crosses an align-mode Mamba block, so the longest
    profile is the largest chunk the scheduler can emit.
    """
    num_spec_steps = model_runner.num_speculative_steps
    lookahead = 1 + num_spec_steps if num_spec_steps > 0 else 0
    max_tokens = min(
        model_runner.scheduler_config.max_num_batched_tokens,
        model_runner.max_model_len,
    )
    for spec in kv_cache_specs:
        if isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align":
            max_tokens = min(max_tokens, spec.block_size)
    token_counts = {max_tokens} if max_tokens > default_prompt_len else set()
    bucket = 1
    while bucket - lookahead <= max_tokens:
        if bucket - lookahead > default_prompt_len:
            token_counts.add(bucket - lookahead)
        bucket *= 2
    return token_counts


def _warmup_request_counts(max_num_reqs: int) -> tuple[int, ...]:
    """Request counts that reach every per-batch kernel specialization.

    Triton specializes integer arguments equal to 1 or divisible by 16, and
    per-batch constexprs are padded to powers of two, so one request, every
    power of two, one other count and the maximum cover each class without
    running every batch size.
    """
    counts = {1, max_num_reqs}
    count = 2
    while count < max_num_reqs:
        counts.add(count)
        count *= 2
    if max_num_reqs > 3:
        counts.add(3)
    return tuple(sorted(counts))


def _warmup_sampling_profiles(model_runner: GPUModelRunner) -> list[SamplingParams]:
    """Greedy decoding and the model's default sampling parameters.

    Requests that leave sampling unset use the generation-config defaults.
    Those and greedy requests take sampler and rejection-sampler paths, such
    as compact top-k rejection without penalties, that
    ``SamplingParams.for_sampler_warmup()`` never selects.
    """
    profiles = [SamplingParams(temperature=0.0)]
    try:
        defaults = model_runner.model_config.get_diff_sampling_param()
    except AttributeError:
        defaults = {}
    fields = {
        name: defaults[name]
        for name in _DEFAULT_SAMPLING_FIELDS
        if defaults.get(name) is not None
    }
    if fields and fields.get("temperature", 1.0) > 0:
        profiles.append(SamplingParams(**fields))
    return profiles


def _reserved_block_count(
    num_tokens: int,
    kv_cache_spec: KVCacheSpec,
    *,
    num_lookahead_tokens: int,
    max_model_len: int,
    max_encoder_len: int,
) -> int:
    """Match the scheduler's block reservation in hand-built warmup batches."""
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        specs = tuple(kv_cache_spec.kv_cache_specs.values())
        assert specs
        kv_cache_spec = specs[0]
    if isinstance(kv_cache_spec, CircularBufferSpec):
        return 1
    if isinstance(kv_cache_spec, CrossAttentionSpec):
        return cdiv(max_encoder_len, kv_cache_spec.block_size)
    num_speculative_blocks = 0
    if isinstance(kv_cache_spec, MambaSpec):
        num_speculative_blocks = kv_cache_spec.num_speculative_blocks
        if kv_cache_spec.mamba_cache_mode == "align":
            return cdiv(num_tokens, kv_cache_spec.block_size) + num_speculative_blocks
    num_tokens = min(num_tokens + num_lookahead_tokens, max_model_len)
    return cdiv(num_tokens, kv_cache_spec.block_size) + num_speculative_blocks


def _warmup_block_counter(
    model_runner: GPUModelRunner,
) -> Callable[[int, KVCacheSpec], int]:
    def block_count(num_tokens: int, kv_cache_spec: KVCacheSpec) -> int:
        return _reserved_block_count(
            num_tokens,
            kv_cache_spec,
            num_lookahead_tokens=model_runner.vllm_config.num_lookahead_tokens,
            max_model_len=model_runner.max_model_len,
            max_encoder_len=getattr(model_runner.model_state, "max_encoder_len", 0),
        )

    return block_count


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run execute_model + sample_tokens iterations to JIT compile
    triton kernels. We must call the provided worker's execute_model for
    pipeline parallel coordination.

    Each iteration simulates a prefill of its requests, then a decode step
    with every request generating 1 + num_spec_steps tokens. The first
    iteration batches as many requests of 2 + num_spec_steps prompt tokens as
    fit; kernel-advertised prompt lengths run with one request. With
    ``VLLM_KERNEL_WARMUP_COVERAGE``, further iterations cover each request
    count class, greedy and default sampling, and the per-request query
    buckets that serving selects.
    """
    num_spec_steps = model_runner.num_speculative_steps
    # Use 1 + num_spec_steps + 1 tokens so the prefill batch's per-request
    # query length exceeds decode_query_len (= 1 + num_spec_steps), preventing
    # it from being misclassified as a uniform decode batch.
    default_prompt_len = 2 + num_spec_steps
    prompt_lengths = _kernel_prefill_warmup_token_counts(
        model_runner, default_prompt_len
    )

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    block_count = _warmup_block_counter(model_runner)
    kv_cache_specs = [group.kv_cache_spec for group in kv_cache_groups]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    def max_num_reqs(prompt_len: int) -> int:
        decode_len = prompt_len + 1 + num_spec_steps
        max_blocks_per_req = sum(
            block_count(decode_len, spec) for spec in kv_cache_specs
        )
        return min(
            model_runner.scheduler_config.max_num_seqs,
            model_runner.scheduler_config.max_num_batched_tokens
            // max(prompt_len, 1 + num_spec_steps),
            # Reserve block 0 (null block) and ensure enough blocks.
            max(
                1,
                (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req,
            ),
        )

    # (prompt length, number of requests, sampling params) of each iteration.
    batch_size = max_num_reqs(default_prompt_len)
    iterations = [(default_prompt_len, batch_size, sampling_params)]
    long_prompts = [(prompt_len, sampling_params) for prompt_len in prompt_lengths[1:]]
    if envs.VLLM_KERNEL_WARMUP_COVERAGE and not model_runner.is_pooling_model:
        request_counts = _warmup_request_counts(batch_size)
        iterations.extend(
            (default_prompt_len, count, sampling_params)
            for count in request_counts
            if count != batch_size
        )
        for profile in _warmup_sampling_profiles(model_runner):
            iterations.extend(
                (default_prompt_len, count, profile) for count in request_counts
            )
        # Only the prefill shape of these prompts matters. Greedy sampling
        # skips the full-vocabulary prompt logprobs of the all-features profile.
        greedy_params = SamplingParams(temperature=0.0)
        bucket_lengths = _coverage_prefill_token_counts(
            model_runner, default_prompt_len, kv_cache_specs
        ).difference(prompt_lengths)
        long_prompts.extend(
            (prompt_len, greedy_params) for prompt_len in sorted(bucket_lengths)
        )
    # Longer prompts run with one request so adding a profile does not
    # silently multiply startup work by max_num_seqs.
    iterations.extend(
        (prompt_len, min(max_num_reqs(prompt_len), 1), params)
        for prompt_len, params in long_prompts
    )

    # Disable KV connector for all warmup runs.
    model_runner.kv_connector.set_disabled(True)
    try:
        for profile_idx, (prompt_len, num_reqs, request_params) in enumerate(
            iterations
        ):
            if num_reqs <= 0:
                continue
            prompt_token_ids = list(range(prompt_len))
            decode_len = prompt_len + 1 + num_spec_steps
            prefill_block_counts = [
                block_count(prompt_len, spec) for spec in kv_cache_specs
            ]
            decode_block_counts = [
                block_count(decode_len, spec) for spec in kv_cache_specs
            ]
            decode_block_deltas = [
                d - p for d, p in zip(decode_block_counts, prefill_block_counts)
            ]

            req_ids = [f"_warmup_{profile_idx}_{i}_" for i in range(num_reqs)]
            next_block_id = 1

            def _alloc_blocks(num_blocks: int) -> list[int]:
                nonlocal next_block_id
                return list(
                    range(next_block_id, next_block_id := next_block_id + num_blocks)
                )

            new_reqs = [
                NewRequestData.from_request(
                    Request(
                        req_ids[i],
                        prompt_token_ids,
                        request_params,
                        pooling_params,
                    ),
                    block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
                    prefill_token_ids=prompt_token_ids,
                )
                for i in range(num_reqs)
            ]

            prefill_output = SchedulerOutput.make_empty()
            prefill_output.scheduled_new_reqs = new_reqs
            prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
            prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
            prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups
            worker_execute_model(prefill_output)

            if not model_runner.is_pooling_model:
                grammar_output = None
                if profile_idx == 0 and model_runner.is_last_pp_rank:
                    # Exercise the structured-output bitmask once; extra
                    # operator profiles only need the model path.
                    vocab_size = model_runner.model_config.get_vocab_size()
                    bitmask_width = (vocab_size + 31) // 32
                    grammar_bitmask = np.full(
                        (len(req_ids), bitmask_width),
                        fill_value=-1,
                        dtype=np.int32,
                    )
                    grammar_output = GrammarOutput(
                        structured_output_request_ids=req_ids,
                        grammar_bitmask=grammar_bitmask,
                    )
                worker_sample_tokens(grammar_output)

                cached_req_data = CachedRequestData.make_empty()
                cached_req_data.req_ids = list(req_ids)
                cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
                cached_req_data.num_output_tokens = [1] * num_reqs
                new_block = any(decode_block_deltas)
                cached_req_data.new_block_ids = [
                    (
                        tuple(_alloc_blocks(n) for n in decode_block_deltas)
                        if new_block
                        else None
                    )
                    for _ in range(num_reqs)
                ]

                decode_output = SchedulerOutput.make_empty()
                decode_output.scheduled_cached_reqs = cached_req_data
                decode_output.num_scheduled_tokens = {
                    req_id: 1 + num_spec_steps for req_id in req_ids
                }
                if num_spec_steps > 0:
                    decode_output.scheduled_spec_decode_tokens = {
                        req_id: [0] * num_spec_steps for req_id in req_ids
                    }
                decode_output.total_num_scheduled_tokens = sum(
                    decode_output.num_scheduled_tokens.values()
                )
                decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

                worker_execute_model(decode_output)
                worker_sample_tokens(None)

            cleanup_output = SchedulerOutput.make_empty()
            cleanup_output.finished_req_ids = set(req_ids)
            worker_execute_model(cleanup_output)
    finally:
        model_runner.kv_connector.set_disabled(False)
    torch.accelerator.synchronize()
