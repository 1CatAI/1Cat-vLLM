# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy speculative verification from target argmax ids.

Equivalent to ``rejection_sample`` when every request is greedy: the output
row holds the target argmax for each position up to and including the first
draft mismatch (or the bonus position), and ``num_sampled`` counts the leading
matches plus one.
"""

from types import SimpleNamespace

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.output import SamplerOutput

logger = init_logger(__name__)


@triton.jit
def _greedy_verify_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    target_ptr,
    draft_ptr,
    cu_num_logits_ptr,
):
    req = tl.program_id(0)
    start = tl.load(cu_num_logits_ptr + req)
    end = tl.load(cu_num_logits_ptr + req + 1)
    accepted = True
    count = 0
    for i in range(end - start):
        target = tl.load(target_ptr + start + i)
        tl.store(sampled_ptr + req * sampled_stride + i, target)
        if i < end - start - 1:
            draft = tl.load(draft_ptr + start + i + 1)
            accepted &= target == draft
            count += accepted.to(tl.int32)
    tl.store(num_sampled_ptr + req, count + 1)


def greedy_verify(
    target_ids: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = target_ids.new_zeros(
        (num_reqs, num_speculative_steps + 1), dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    _greedy_verify_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_ids.to(torch.int64),
        draft_sampled.to(torch.int64),
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled


def greedy_capability(device, has_lora: bool) -> bool:
    return bool(
        device.type == "cuda"
        and current_platform.is_device_capability(70)
        and not has_lora
    )


def maybe_sample_greedy(
    model,
    sampler,
    rejection_sampler,
    capability_enabled: bool,
    verify_enabled: bool,
    sample_hidden_states,
    input_batch,
    grammar_output,
    sampler_output,
    cached_logits,
    *,
    speculator=None,
):
    sm70_greedy_decode = (
        sampler_output is None
        and input_batch.num_draft_tokens == 0
        and input_batch.num_reqs == 1
        and input_batch.num_tokens == 1
        and not input_batch.is_prefilling_np[0]
        and grammar_output is None
        and capability_enabled
        and hasattr(model, "get_top_tokens")
        and sampler is not None
        and sampler.can_use_sm70_greedy_token_fastpath(input_batch)
    )
    sm70_greedy_verify = (
        sampler_output is None
        and cached_logits is None
        and input_batch.num_draft_tokens > 0
        and grammar_output is None
        and capability_enabled
        and hasattr(model, "get_top_tokens")
        and rejection_sampler is not None
        and rejection_sampler.synthetic_conditional_rates is None
        and verify_enabled
        and rejection_sampler.sampler.can_use_sm70_greedy_token_fastpath(input_batch)
    )
    if sm70_greedy_verify:
        assert rejection_sampler is not None
        target_ids = model.get_top_tokens(sample_hidden_states).view(-1)
        sampled, num_sampled = greedy_verify(
            target_ids,
            input_batch.input_ids[input_batch.logits_indices],
            input_batch.cu_num_logits,
            rejection_sampler.num_speculative_steps,
        )
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
        )
        logger.info_once("SM70 greedy MTP verification from TP-local argmax.")
    if sm70_greedy_decode:
        sampled = model.get_top_tokens(sample_hidden_states)
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=input_batch.seq_lens.new_ones(input_batch.num_reqs),
        )
        logger.info_once("SM70 MRv2 greedy TP-local pair path enabled.")
    if sampler_output is None and cached_logits is None and capability_enabled:
        from vllm.v1.worker.gpu.sm70_runner_ops import try_target_sample

        # Keep the retained exact verifier behind the upstream sampling owner.
        # Capability admission already excludes LoRA and non-Volta execution.
        sampler_output = try_target_sample(
            SimpleNamespace(
                model=model,
                sampler=sampler,
                rejection_sampler=rejection_sampler,
                speculator=speculator,
                lora_config=None,
            ),
            sample_hidden_states,
            input_batch,
            grammar_output,
        )
    return sampler_output
