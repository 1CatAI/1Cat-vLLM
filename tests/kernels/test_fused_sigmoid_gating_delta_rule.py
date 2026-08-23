# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fused_gdn_gating,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type


def _bitwise_diff_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[int, float, float]:
    diff = actual.float() - expected.float()
    return (
        torch.count_nonzero(actual != expected).item(),
        diff.abs().max().item(),
        diff.abs().mean().item(),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_verify_matches_fused_decode_beta_semantics(
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 requires a supported accelerator")
    set_random_seed(0)

    # Qwen3.8-27B TP4 per-rank head geometry (16 QK heads, 48 V heads).
    num_qk_heads = 4
    num_v_heads = 12
    head_dim = 128
    scale = head_dim**-0.5
    # Make the decay exactly zero in both kernels so this regression isolates
    # beta materialization instead of also measuring unrelated fused/split
    # decay rounding. softplus(-100) rounds to zero in the kernel.
    A_log = torch.zeros(num_v_heads, dtype=torch.float32)
    dt_bias = torch.zeros(num_v_heads, dtype=dtype)
    a = torch.full((1, num_v_heads), -100, dtype=dtype)
    # sigmoid(-0.5) is not exactly representable in FP16 or BF16, exposing
    # whether the split verifier keeps the fused decode FP32 beta semantics.
    b = torch.full((1, num_v_heads), -0.5, dtype=dtype)
    query = torch.rand(1, 1, num_qk_heads, head_dim, dtype=dtype)
    key = torch.rand_like(query)
    value = torch.rand(1, 1, num_v_heads, head_dim, dtype=dtype)
    # A zero state also removes decay and dot-product reduction-order effects
    # from this one-step comparison. The state transition then differs only
    # through the materialized beta value.
    initial_state = torch.zeros(
        1,
        num_v_heads,
        head_dim,
        head_dim,
        dtype=torch.float32,
    )
    state_indices = torch.zeros(1, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

    fused_state = initial_state.clone()
    fused_output, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        scale=scale,
        initial_state=fused_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=False,
    )

    packed_state = initial_state.clone()
    packed_output = torch.empty_like(fused_output)
    mixed_qkv = torch.cat(
        [
            query.squeeze(0).reshape(1, -1),
            key.squeeze(0).reshape(1, -1),
            value.squeeze(0).reshape(1, -1),
        ],
        dim=-1,
    ).contiguous()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=packed_state,
        out=packed_output,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=False,
    )

    def run_split(
        beta_dtype: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g, beta = fused_gdn_gating(
            A_log,
            a,
            b,
            dt_bias,
            beta_dtype=beta_dtype,
        )
        state = initial_state.clone()
        output, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=state,
            inplace_final_state=True,
            ssm_state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )
        return output, state, beta

    fixed_output, fixed_state, fixed_beta = run_split(torch.float32)
    legacy_output, legacy_state, legacy_beta = run_split(None)

    assert fixed_beta.dtype is torch.float32
    assert legacy_beta.dtype is dtype
    fixed_output_diff = _bitwise_diff_stats(fixed_output, fused_output)
    fixed_state_diff = _bitwise_diff_stats(fixed_state, fused_state)
    legacy_output_diff = _bitwise_diff_stats(legacy_output, fused_output)
    legacy_state_diff = _bitwise_diff_stats(legacy_state, fused_state)
    assert torch.equal(packed_output, fused_output), _bitwise_diff_stats(
        packed_output, fused_output
    )
    assert torch.equal(packed_state, fused_state), _bitwise_diff_stats(
        packed_state, fused_state
    )
    assert torch.equal(fixed_output, fused_output), fixed_output_diff
    assert torch.equal(fixed_state, fused_state), (
        "fixed",
        fixed_state_diff,
        "legacy",
        legacy_state_diff,
    )
    assert legacy_state_diff[0] > 0, legacy_state_diff
    assert legacy_output_diff[0] > 0, legacy_output_diff


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_non_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    seq_len = 1  # seq_len is 1 for decode
    num_tokens = num_reqs * seq_len
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    ssm_state = torch.rand(
        total_entries, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    state_indices = torch.randperm(total_entries, dtype=torch.int32)[:num_tokens]
    cu_seqlens = torch.arange(0, num_tokens + 1, dtype=torch.int32)

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    num_speculative_tokens: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    num_tokens = num_reqs * (num_speculative_tokens + 1)
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    ssm_state = torch.rand(
        total_entries, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    state_indices = torch.randperm(
        total_entries,
        dtype=torch.int32,
    )[:num_tokens].view(num_reqs, num_speculative_tokens + 1)
    num_accepted_tokens = torch.randint(
        1, num_speculative_tokens + 1, (num_reqs,), dtype=torch.int32
    )
    cu_seqlens = torch.arange(
        0, num_tokens + 1, num_speculative_tokens + 1, dtype=torch.int32
    )

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )
