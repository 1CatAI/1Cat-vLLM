# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.logger import init_logger
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

logger = init_logger(__name__)


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


def _mtp_tail_warmup_profiles(
    max_num_reqs: int, num_spec_steps: int
) -> tuple[tuple[int, ...], ...]:
    """Bounded single/double-tail shapes; request order is canonicalized by V2."""
    if not 1 <= num_spec_steps <= 4:
        raise ValueError("MTP tail warmup supports 1 to 4 speculative tokens")
    max_num_reqs = min(max_num_reqs, 8)
    normal_query_len = num_spec_steps + 1
    profiles = []
    for num_reqs in range(1, max_num_reqs + 1):
        for tail_len in range(1, normal_query_len):
            profiles.append((tail_len,) + (normal_query_len,) * (num_reqs - 1))
        if num_reqs >= 2:
            for first in range(1, normal_query_len):
                for second in range(first, normal_query_len):
                    profiles.append(
                        (first, second) + (normal_query_len,) * (num_reqs - 2)
                    )
    return tuple(profiles)


def _warmup_mtp_tail_shapes(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Compile truncated verification shapes against the allocated, real KV cache.

    Opt-in because the bounded profile set can add minutes to a cold startup.
    Only the target runner participates; draft/PLE workers must not enter a
    target PP-group collective. Errors propagate and fail startup: trying to
    continue or drain an incomplete distributed forward could deadlock.
    """
    config = model_runner.vllm_config
    additional_config = config.additional_config
    if (
        not isinstance(additional_config, dict)
        or not additional_config.get("mtp_tail_warmup", False)
        or getattr(config, "is_speculative_draft", False)
    ):
        return
    if model_runner.is_pooling_model:
        raise ValueError("MTP tail warmup requires a generative target runner")
    spec_config = model_runner.speculative_config
    if spec_config is None or spec_config.method != "mtp":
        raise ValueError("MTP tail warmup requires method='mtp'")
    num_spec_steps = model_runner.num_speculative_steps
    # Validate before entering any collective. The target config is common to
    # all PP ranks; capacity below is deliberately reduced across local KV specs.
    _mtp_tail_warmup_profiles(0, num_spec_steps)
    normal_query_len = num_spec_steps + 1
    prompt_len = normal_query_len + 1
    if model_runner.max_model_len < prompt_len + normal_query_len:
        return

    from vllm.distributed.parallel_state import get_pp_group

    pp = get_pp_group()
    feedback_depth = config.max_concurrent_batches if pp.world_size > 1 else 1
    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    kv_cache_specs = [group.kv_cache_spec for group in kv_cache_groups]
    block_count = _warmup_block_counter(model_runner)
    prefill_counts = [block_count(prompt_len, spec) for spec in kv_cache_specs]
    decode_counts = [
        block_count(prompt_len + normal_query_len, spec) for spec in kv_cache_specs
    ]
    decode_deltas = [d - p for d, p in zip(decode_counts, prefill_counts)]
    max_blocks_per_req = sum(decode_counts)
    capacity = min(
        8,
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens // prompt_len,
    )
    if max_blocks_per_req:
        capacity = min(
            capacity,
            (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req,
        )
    # PP stages can have different projected KV groups. Every rank must execute
    # the same profiles/call sequence, including under a small KV budget.
    if pp.world_size > 1:
        capacity_tensor = torch.tensor(capacity, dtype=torch.int64, device="cpu")
        torch.distributed.all_reduce(
            capacity_tensor, op=torch.distributed.ReduceOp.MIN, group=pp.cpu_group
        )
        capacity = int(capacity_tensor.item())
    profiles = _mtp_tail_warmup_profiles(capacity, num_spec_steps)
    if not profiles:
        if pp.is_first_rank:
            logger.warning("MTP tail warmup skipped: insufficient batch/KV capacity")
        return

    def drain(num_slots: int) -> None:
        # Real empty executes advance PP feedback and apply deferred Mamba
        # commits. Never increment a CPU step counter or resize the PP ring.
        for _ in range(num_slots):
            worker_execute_model(SchedulerOutput.make_empty())

    # Earlier warmup profiles may have freed requests with pending PP packets.
    # Consume them under the existing generation mask before reusing blocks.
    drain(feedback_depth)
    torch.accelerator.synchronize()
    started = time.perf_counter()
    if pp.is_first_rank:
        logger.info(
            "MTP tail warmup starting: profiles=%d capacity=%d feedback_depth=%d",
            len(profiles),
            capacity,
            feedback_depth,
        )
    prompt_token_ids = list(range(prompt_len))
    sampling_params = SamplingParams.for_sampler_warmup()
    sampling_params.max_tokens = 2 * normal_query_len + 2
    sampling_params.ignore_eos = True

    for profile_idx, query_lengths in enumerate(profiles):
        num_reqs = len(query_lengths)
        req_ids = [f"_warmup_mtp_tail_{profile_idx}_{i}_" for i in range(num_reqs)]
        next_block_id = 1  # Block zero belongs to padding/null slots.

        def alloc_blocks(num_blocks: int) -> list[int]:
            nonlocal next_block_id
            block_ids = list(range(next_block_id, next_block_id + num_blocks))
            next_block_id += num_blocks
            return block_ids

        prefill = SchedulerOutput.make_empty()
        prefill.scheduled_new_reqs = [
            NewRequestData.from_request(
                Request(req_id, prompt_token_ids, sampling_params, None),
                block_ids=tuple(alloc_blocks(n) for n in prefill_counts),
                prefill_token_ids=prompt_token_ids,
            )
            for req_id in req_ids
        ]
        prefill.num_scheduled_tokens = dict.fromkeys(req_ids, prompt_len)
        prefill.total_num_scheduled_tokens = num_reqs * prompt_len
        prefill.num_common_prefix_blocks = [0] * len(kv_cache_groups)
        worker_execute_model(prefill)
        worker_sample_tokens(None)

        # The decode must see the prefill's sampled token and MTP proposal on
        # every rank. D-1 intervening slots put its execute at prefill_step + D.
        drain(feedback_depth - 1)
        cached = CachedRequestData.make_empty()
        cached.req_ids = req_ids
        cached.num_computed_tokens = [prompt_len] * num_reqs
        cached.num_output_tokens = [1] * num_reqs
        cached.new_block_ids = [
            tuple(alloc_blocks(n) for n in decode_deltas)
            if any(decode_deltas)
            else None
            for _ in req_ids
        ]
        decode = SchedulerOutput.make_empty()
        decode.scheduled_cached_reqs = cached
        decode.num_scheduled_tokens = dict(zip(req_ids, query_lengths))
        decode.scheduled_spec_decode_tokens = {
            req_id: [0] * (query_len - 1)
            for req_id, query_len in zip(req_ids, query_lengths)
            if query_len > 1
        }
        decode.total_num_scheduled_tokens = sum(query_lengths)
        decode.num_common_prefix_blocks = [0] * len(kv_cache_groups)
        worker_execute_model(decode)
        worker_sample_tokens(None)

        # Keep each request's index/generation and block ownership alive until
        # delayed feedback/state commits finish. Reclaim through the normal
        # runner path, which also removes sampler/model/PLE request state.
        drain(feedback_depth)
        cleanup = SchedulerOutput.make_empty()
        cleanup.finished_req_ids = set(req_ids)
        worker_execute_model(cleanup)
        torch.accelerator.synchronize()
        if pp.is_first_rank:
            logger.info(
                "MTP tail warmup complete: profile=%d/%d query_lengths=%s "
                "elapsed=%.2fs",
                profile_idx + 1,
                len(profiles),
                query_lengths,
                time.perf_counter() - started,
            )


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run two execute_model + sample_tokens iterations to JIT compile
    triton kernels. We must call the provided worker's execute_model for
    pipeline parallel coordination.

    The first iteration simulates a prefill with requests of
    2 + num_spec_steps prompt tokens each. The second iteration simulates
    a decode step with all requests generating 1 + num_spec_steps tokens.
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

    # Disable KV connector for all warmup runs. Kernel-advertised profiles run
    # with one request so adding an operator does not silently multiply startup
    # work by max_num_seqs.
    model_runner.kv_connector.set_disabled(True)
    try:
        for profile_idx, prompt_len in enumerate(prompt_lengths):
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
            max_blocks_per_req = sum(decode_block_counts)
            num_reqs = min(
                model_runner.scheduler_config.max_num_seqs,
                model_runner.scheduler_config.max_num_batched_tokens
                // max(prompt_len, 1 + num_spec_steps),
                # Reserve block 0 (null block) and ensure enough blocks.
                max(
                    1,
                    (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req,
                ),
            )
            if profile_idx:
                num_reqs = min(num_reqs, 1)
            if num_reqs <= 0:
                continue

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
                        sampling_params,
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
        _warmup_mtp_tail_shapes(
            model_runner, worker_execute_model, worker_sample_tokens
        )
    finally:
        model_runner.kv_connector.set_disabled(False)
    torch.accelerator.synchronize()
