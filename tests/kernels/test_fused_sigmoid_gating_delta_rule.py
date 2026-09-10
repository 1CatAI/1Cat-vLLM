# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
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


@pytest.mark.parametrize("num_reqs", [1, 2])
@pytest.mark.parametrize("num_speculative_tokens", [3, 7])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("runtime_bridge", [False, True])
def test_dflash2_packed_verify_matches_split_contract(
    num_reqs: int,
    num_speculative_tokens: int,
    state_dtype: torch.dtype,
    runtime_bridge: bool,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("the DFlash2 verifier fast path is SM70-only")

    torch.set_default_device(DEVICE)
    set_random_seed(11)
    dtype = torch.float16
    # Production GDN verifier shape used by the recorded TP4 parity evidence.
    num_k_heads, num_v_heads = 4, 12
    head_k_dim = head_v_dim = 128
    tokens_per_req = num_speculative_tokens + 1
    num_tokens = num_reqs * tokens_per_req
    qkv_width = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

    # Runtime Qwen3.5 retains the QKVZBA projection row stride after conv.
    padding = num_v_heads * head_v_dim + 2 * num_v_heads if runtime_bridge else 0
    projection = torch.randn(num_tokens, qkv_width + padding, dtype=dtype)
    mixed_qkv = projection[:, :qkv_width]
    projection_before = projection.clone()
    query, key, value = torch.split(
        mixed_qkv,
        [
            num_k_heads * head_k_dim,
            num_k_heads * head_k_dim,
            num_v_heads * head_v_dim,
        ],
        dim=-1,
    )
    query = query.contiguous().view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.contiguous().view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.contiguous().view(1, num_tokens, num_v_heads, head_v_dim)
    a = torch.randn(num_tokens, num_v_heads, dtype=dtype)
    b = torch.randn(num_tokens, num_v_heads, dtype=dtype)
    A_log = torch.randn(num_v_heads, dtype=torch.float32)
    dt_bias = torch.randn(num_v_heads, dtype=dtype)
    state_indices = torch.arange(num_tokens, dtype=torch.int32).view(
        num_reqs, tokens_per_req
    )
    num_accepted_tokens = torch.randint(
        1, tokens_per_req + 1, (num_reqs,), dtype=torch.int32
    )
    cu_seqlens = torch.arange(0, num_tokens + 1, tokens_per_req, dtype=torch.int32)
    initial_state = torch.randn(
        num_tokens + 3,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=state_dtype,
    )

    g, beta = fused_gdn_gating(
        A_log, a, b, dt_bias, beta_dtype=torch.float32 if runtime_bridge else dtype
    )
    reference_state = initial_state.clone()
    reference_out, _ = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state=reference_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    fused_state = initial_state.clone()
    fused_out = torch.empty(num_tokens, 1, num_v_heads, head_v_dim, dtype=dtype)
    if runtime_bridge:
        module = SimpleNamespace(
            A_log=A_log,
            dt_bias=dt_bias,
            num_k_heads=num_k_heads * 4,
            num_v_heads=num_v_heads * 4,
            tp_size=4,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
        )
        QwenGatedDeltaNetAttention._forward_dflash2_packed_gdn_verify(
            module,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            core_attn_out=fused_out.squeeze(1),
            ssm_state=fused_state,
            spec_query_start_loc=cu_seqlens,
            spec_state_indices_tensor=state_indices,
            spec_state_slot_selectors=num_accepted_tokens,
            num_spec_decodes=num_reqs,
        )
    else:
        fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            mixed_qkv=mixed_qkv,
            num_q_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            out=fused_out,
            initial_state=fused_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=True,
            precomputed_g=g,
            precomputed_beta=beta,
            match_recurrent_schedule=True,
            match_recurrent_numerics=True,
        )
    torch.accelerator.synchronize()

    assert torch.equal(
        projection.view(torch.uint8), projection_before.view(torch.uint8)
    )
    torch.testing.assert_close(fused_out.transpose(0, 1), reference_out, rtol=0, atol=0)
    torch.testing.assert_close(fused_state, reference_state, rtol=0, atol=0)


def test_dflash2_quantized_state_reload_matches_repeated_decode() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("the DFlash2 verifier fast path is SM70-only")

    torch.set_default_device(DEVICE)
    set_random_seed(12)
    dtype = torch.float16
    num_tokens = 4
    num_k_heads, num_v_heads = 2, 4
    head_k_dim = head_v_dim = 16
    qkv_width = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    mixed_qkv = torch.randn(num_tokens, qkv_width, dtype=dtype)
    a = torch.randn(num_tokens, num_v_heads, dtype=dtype)
    b = torch.randn(num_tokens, num_v_heads, dtype=dtype)
    A_log = torch.randn(num_v_heads, dtype=torch.float32)
    dt_bias = torch.randn(num_v_heads, dtype=dtype)
    initial_state = torch.randn(2, num_v_heads, head_v_dim, head_k_dim, dtype=dtype)
    shared_state_indices = torch.zeros((1, num_tokens), dtype=torch.int32)
    accepted = torch.ones(1, dtype=torch.int32)

    fused_state = initial_state.clone()
    fused_out = torch.empty(num_tokens, 1, num_v_heads, head_v_dim, dtype=dtype)
    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        mixed_qkv=mixed_qkv,
        num_q_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        out=fused_out,
        initial_state=fused_state,
        cu_seqlens=torch.tensor([0, num_tokens], dtype=torch.int32),
        ssm_state_indices=shared_state_indices,
        num_accepted_tokens=accepted,
        quantize_beta=True,
        quantize_state_each_step=True,
        match_recurrent_schedule=True,
        use_qk_l2norm_in_kernel=True,
    )

    sequential_state = initial_state.clone()
    sequential_outputs = []
    one_token_cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
    one_state_index = torch.zeros(1, dtype=torch.int32)
    for token_idx in range(num_tokens):
        token_out = torch.empty(1, 1, num_v_heads, head_v_dim, dtype=dtype)
        fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
            A_log=A_log,
            a=a[token_idx : token_idx + 1],
            b=b[token_idx : token_idx + 1],
            dt_bias=dt_bias,
            mixed_qkv=mixed_qkv[token_idx : token_idx + 1],
            num_q_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            out=token_out,
            initial_state=sequential_state,
            cu_seqlens=one_token_cu_seqlens,
            ssm_state_indices=one_state_index,
            num_accepted_tokens=accepted,
            quantize_beta=True,
            match_recurrent_schedule=True,
            use_qk_l2norm_in_kernel=True,
        )
        sequential_outputs.append(token_out)
    torch.accelerator.synchronize()

    assert torch.equal(fused_out, torch.cat(sequential_outputs, dim=0))
    assert torch.equal(fused_state, sequential_state)
