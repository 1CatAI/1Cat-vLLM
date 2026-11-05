# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission checks, before requests enter EngineCore or any TP collective."""

from vllm.exceptions import VLLMValidationError


def validate_prompt_logprobs_output(
    prompt_len: int,
    num_logprobs: int,
    vocab_size: int,
    max_output_bytes: int,
) -> None:
    if max_output_bytes <= 0:
        raise ValueError("VLLM_PROMPT_LOGPROBS_MAX_OUTPUT_BYTES must be positive")
    # The inherited small-subset Triton kernel expands next_power_of_2(k + 1)
    # lanes. Keep partial subsets bounded without changing the shared sampler.
    # Full-vocabulary output uses its existing dense torch fallback instead.
    if 128 < num_logprobs < vocab_size - 1:
        raise VLLMValidationError(
            "MRv2 prompt_logprobs supports partial top-k up to 128; "
            "use -1 for bounded full-vocabulary output.",
            parameter="prompt_logprobs",
            value=num_logprobs,
        )
    width = 1 + (vocab_size if num_logprobs == -1 else num_logprobs)
    # Each scored row returns int64 IDs, FP32 logprobs, and one int64 rank.
    output_bytes = max(0, prompt_len - 1) * (12 * width + 8)
    if output_bytes > max_output_bytes:
        raise VLLMValidationError(
            f"prompt_logprobs tensor output requires {output_bytes} bytes, "
            f"exceeding the per-request limit of {max_output_bytes} bytes. "
            "Use a shorter prompt or fewer prompt_logprobs.",
            parameter="prompt_logprobs",
            value=num_logprobs,
        )
