# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

# Smallest positive value produced by Triton's fp32 `tl.rand`. Used to clamp
# zero draws before the flipped Gumbel transform below.
#
# Triton requires globals accessed from `@triton.jit` functions to be wrapped
# in `tl.constexpr(...)`. We can only do that when Triton is actually
# available — on the CPU worker path `tl` is a placeholder whose `constexpr`
# attribute is `None`, and `tl.constexpr(...)` would crash at import time.
_TL_RAND_MIN = tl.constexpr(4.6566127342e-10) if HAS_TRITON else 4.6566127342e-10

# Keep draft proposals independent from target verification/resampling. Both
# streams key noise by (seed, position, token), so salt the draft position.
_DRAFT_NOISE_SALT = tl.constexpr(1 << 30) if HAS_TRITON else (1 << 30)


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def tl_rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def tl_rand32(seed, offset, includes_zero: tl.constexpr):
    u = tl.rand(seed, offset)
    if not includes_zero:
        u = tl.maximum(u, _TL_RAND_MIN)
    return u


@triton.jit
def gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    IS_DRAFTING: tl.constexpr,
    USE_FP64: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr = True,
):
    """Argmax under token-keyed Gumbel noise, or plain argmax at temp zero."""
    if temp != 0.0 and APPLY_TEMPERATURE:
        logits = logits / temp

    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        if IS_DRAFTING:
            pos = pos + _DRAFT_NOISE_SALT
        gumbel_seed = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            # Flip the transform so the deciding tail lies near zero, where
            # fp32 uniforms retain resolution. `log1p` avoids cancellation.
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_stride,
    processed_logits_col_ptr,
    vocab_size,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Preserve the established MRV2 contract: proposal logits handed to
        # rejection sampling have already had request temperature applied.
        logits = logits / temp

    if processed_logits_ptr is not None:
        # Store the temperature-applied proposal distribution.
        if processed_logits_col_ptr is not None:
            col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr
            + req_state_idx * processed_logits_stride
            + col * processed_logits_col_stride
            + block,
            logits,
            mask=mask & is_valid_req,
        )

    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    pos = tl.load(pos_ptr + token_idx)
    return gumbel_noised_argmax(
        logits,
        block,
        mask & is_valid_req,
        seed,
        pos,
        temp,
        IS_DRAFTING=IS_DRAFTING,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_stride,
    processed_logits_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        processed_logits_ptr,
        processed_logits_stride,
        processed_logits_col_stride,
        processed_logits_col_ptr,
        vocab_size,
        IS_DRAFTING=IS_DRAFTING,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    is_drafting: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        output_processed_logits,
        output_processed_logits.stride(0) if output_processed_logits is not None else 0,
        output_processed_logits.stride(1) if output_processed_logits is not None else 0,
        output_processed_logits_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_DRAFTING=is_drafting,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled
