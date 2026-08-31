# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    dflash2_sparse_topk_rejection_sample,
    rejection_sample,
)

VOCAB_SIZE = 4096

# Skip if no CUDA - Triton kernel requires GPU
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for rejection sampler tests", allow_module_level=True)


def _build_rejection_sample_inputs(
    target_logits_1d: torch.Tensor,
    draft_logits_1d: torch.Tensor,
    num_speculative_steps: int,
    temperature: float,
    num_trials: int,
) -> dict:
    device = target_logits_1d.device
    vocab_size = target_logits_1d.shape[0]
    K = num_speculative_steps
    num_logits = num_trials * (K + 1)

    target_logits = target_logits_1d.unsqueeze(0).expand(num_logits, -1).contiguous()
    draft_logits = (
        draft_logits_1d.view(1, 1, vocab_size).expand(num_trials, K, -1).contiguous()
    )

    draft_probs = torch.softmax(draft_logits_1d, dim=0)
    draft_tokens = torch.multinomial(
        draft_probs.expand(num_trials, -1), K, replacement=True
    )
    draft_sampled_2d = torch.zeros(num_trials, K + 1, dtype=torch.int64, device=device)
    draft_sampled_2d[:, 1:] = draft_tokens
    draft_sampled = draft_sampled_2d.reshape(-1)

    cu_num_logits = torch.arange(num_trials + 1, dtype=torch.int32, device=device) * (
        K + 1
    )
    pos = torch.arange(num_logits, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(num_trials, dtype=torch.int32, device=device)
    expanded_idx_mapping = torch.arange(
        num_trials, dtype=torch.int32, device=device
    ).repeat_interleave(K + 1)
    expanded_local_pos = torch.arange(K + 1, dtype=torch.int32, device=device).repeat(
        num_trials
    )
    temp_tensor = torch.full(
        (num_trials,), temperature, dtype=torch.float32, device=device
    )
    seed = torch.arange(num_trials, dtype=torch.int64, device=device)

    return dict(
        target_logits=target_logits,
        draft_logits=draft_logits,
        draft_sampled=draft_sampled,
        cu_num_logits=cu_num_logits,
        pos=pos,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        temperature=temp_tensor,
        seed=seed,
    )


def _assert_distribution_match(
    sampled_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    device: str,
    label: str = "",
    min_expected: float = 5.0,
):
    """
    Assert sampled tokens match the target distribution via a
    chi-squared goodness-of-fit test. This is done by computing
    observed vs expected token counts (target_probs * num_samples),
    then checking that the chi-squared statistic is below a conservative
    threshold. The threshold is set at df + 10*sqrt(2*df), which
    corresponds to ~10 sigma under the chi-squared distribution's
    normal approximation, effectively disallowing false positives.

    NOTE: Tokens with expected count < min_expected are merged into
    a single "other" bin to minimize chi-squared noise.
    """
    num_samples = sampled_tokens.shape[0]
    vocab_size = target_probs.shape[0]

    observed = torch.zeros(vocab_size, device=device, dtype=torch.float32)
    observed.scatter_add_(0, sampled_tokens, torch.ones(num_samples, device=device))
    expected = target_probs * num_samples

    sufficient = expected >= min_expected
    obs_main = observed[sufficient]
    exp_main = expected[sufficient]

    obs_other = observed[~sufficient].sum().unsqueeze(0)
    exp_other = expected[~sufficient].sum().unsqueeze(0)

    if exp_other.item() >= min_expected:
        obs_all = torch.cat([obs_main, obs_other])
        exp_all = torch.cat([exp_main, exp_other])
    else:
        obs_all = obs_main
        exp_all = exp_main

    chi2 = ((obs_all - exp_all) ** 2 / exp_all).sum().item()
    df = obs_all.shape[0] - 1
    if df < 1:
        # All samples were merged into < 2 bins, which is too
        # few to evaluate.
        return

    threshold = df + 10 * math.sqrt(2 * df)
    prefix = f"[{label}] " if label else ""
    assert chi2 < threshold, (
        f"{prefix}Chi-squared test failed: chi2={chi2:.1f}, "
        f"df={df}, threshold={threshold:.1f}. "
        f"Output distribution does not match target distribution."
    )


@pytest.mark.parametrize("num_speculative_steps", [3, 7])
@pytest.mark.parametrize("top_p", [1.0, 0.95])
@pytest.mark.parametrize("temperature", [0.6, 1.0])
def test_dflash2_sparse_topk_matches_dense_rejection(
    num_speculative_steps: int,
    top_p: float,
    temperature: float,
):
    """Compact p/q support must preserve the dense DFlash2 decision path."""
    torch.manual_seed(20260823)
    device = torch.device("cuda")
    num_reqs = 4
    target_top_k = 20
    draft_top_k = 16
    rows_per_req = num_speculative_steps + 1
    num_logits = num_reqs * rows_per_req

    # Unique logits avoid making the reference depend on unspecified top-k
    # ordering for equal FP16 values. Adversarial tie behavior is a separate
    # candidate-set policy gate rather than a rejection-kernel property.
    target_topk_ids = torch.stack(
        [
            torch.randperm(VOCAB_SIZE, device=device)[:target_top_k]
            for _ in range(num_logits)
        ]
    )
    target_topk_logits = (
        torch.randn(num_logits, target_top_k, device=device)
        .sort(dim=-1, descending=True)
        .values
        + torch.arange(target_top_k, device=device).flip(0) * 1e-3
    )
    target_dense = torch.full(
        (num_logits, VOCAB_SIZE),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    target_dense.scatter_(1, target_topk_ids, target_topk_logits)

    top_k_tensor = torch.full(
        (num_logits,), target_top_k, dtype=torch.int32, device=device
    )
    top_p_rows = torch.full((num_logits,), top_p, dtype=torch.float32, device=device)
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    processed_target = apply_top_k_top_p(
        target_dense.clone() / temperature,
        top_k_tensor,
        top_p_rows,
    )

    draft_topk_ids = torch.stack(
        [
            torch.stack(
                [
                    torch.randperm(VOCAB_SIZE, device=device)[:draft_top_k]
                    for _ in range(num_speculative_steps)
                ]
            )
            for _ in range(num_reqs)
        ]
    )
    draft_topk_logits = (
        torch.randn(
            num_reqs,
            num_speculative_steps,
            draft_top_k,
            dtype=torch.float32,
            device=device,
        )
        / temperature
    )
    draft_dense = torch.full(
        (num_reqs, num_speculative_steps, VOCAB_SIZE),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    draft_dense.scatter_(2, draft_topk_ids, draft_topk_logits)

    draft_sampled_2d = torch.zeros(
        num_reqs, rows_per_req, dtype=torch.int64, device=device
    )
    draft_sampled_2d[:, 1:] = draft_topk_ids[:, :, 0]
    draft_sampled = draft_sampled_2d.reshape(-1)
    cu_num_logits = (
        torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * rows_per_req
    )
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    expanded_idx_mapping = idx_mapping.repeat_interleave(rows_per_req)
    expanded_local_pos = torch.arange(
        rows_per_req, dtype=torch.int32, device=device
    ).repeat(num_reqs)
    pos = torch.arange(num_logits, dtype=torch.int64, device=device) + 8192
    temperature_per_req = torch.full(
        (num_reqs,), temperature, dtype=torch.float32, device=device
    )
    top_p_per_req = torch.full((num_reqs,), top_p, dtype=torch.float32, device=device)
    seeds = torch.arange(101, 101 + num_reqs, dtype=torch.int64, device=device)

    dense_sampled, dense_num_sampled = rejection_sample(
        processed_target,
        draft_dense,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature_per_req,
        seeds,
        num_speculative_steps,
    )
    sparse_sampled, sparse_num_sampled = dflash2_sparse_topk_rejection_sample(
        target_topk_ids,
        target_topk_logits,
        draft_topk_ids,
        draft_topk_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        temperature_per_req,
        top_p_per_req,
        seeds,
        num_speculative_steps,
    )

    assert torch.equal(sparse_num_sampled, dense_num_sampled)
    steps = torch.arange(rows_per_req, device=device).unsqueeze(0)
    valid = steps < dense_num_sampled.unsqueeze(1)
    assert torch.equal(sparse_sampled[valid], dense_sampled[valid])


@pytest.mark.parametrize(
    "num_speculative_steps,temperature",
    [
        (1, 0.6),
        (3, 0.6),
        (1, 1.0),
        (3, 1.0),
    ],
)
def test_stochastic_rejection_sample(num_speculative_steps: int, temperature: float):
    """
    Verify that rejection sampling produces the target distribution.
    This is done by simulating many independent trials of speculative
    decoding (from a fixed target and draft distribution). We then
    run rejection sample on all of the trials (requests), and verify
    that the sampled tokens at every position follow the target
    distribution p(x).
    """

    torch.manual_seed(42)
    device = "cuda"
    num_trials = 10 * VOCAB_SIZE

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)

    if temperature > 0:
        target_logits_1d /= temperature
        draft_logits_1d /= temperature

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=temperature,
        num_trials=num_trials,
    )

    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps
    )

    target_probs = torch.softmax(target_logits_1d, dim=0)
    for pos in range(num_speculative_steps + 1):
        accepted_mask = num_sampled >= pos + 1
        _assert_distribution_match(
            sampled[accepted_mask, pos], target_probs, device, label=f"position {pos}"
        )


NARROW_VOCAB_SIZE = 16
NARROW_NUM_TRIALS = 200_000


def _gumbel_drafted_tokens(
    inputs: dict,
    draft_logits_1d: torch.Tensor,
    num_trials: int,
    num_speculative_steps: int,
) -> torch.Tensor:
    k = num_speculative_steps
    vocab_size = draft_logits_1d.shape[0]
    draft_tokens = gumbel_sample(
        draft_logits_1d.unsqueeze(0).expand(num_trials * k, vocab_size).float(),
        inputs["expanded_idx_mapping"]
        .view(num_trials, k + 1)[:, :k]
        .reshape(-1)
        .contiguous(),
        inputs["temperature"],
        inputs["seed"],
        inputs["pos"].view(num_trials, k + 1)[:, :k].reshape(-1).contiguous(),
        apply_temperature=True,
        is_drafting=True,
    )
    draft_sampled = torch.zeros(
        num_trials * (k + 1), dtype=torch.int64, device=draft_logits_1d.device
    )
    draft_sampled.view(num_trials, k + 1)[:, 1:] = draft_tokens.view(num_trials, k)
    return draft_sampled


@pytest.mark.parametrize("num_speculative_steps", [1, 3])
def test_gumbel_drafted_rejection_sample_is_unbiased(num_speculative_steps: int):
    """Draft proposals and residual resampling must use independent noise."""
    torch.manual_seed(42)
    device = "cuda"
    target_logits_1d = torch.randn(
        NARROW_VOCAB_SIZE, device=device, dtype=torch.float32
    )
    draft_logits_1d = -target_logits_1d
    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=1.0,
        num_trials=NARROW_NUM_TRIALS,
    )
    inputs["draft_sampled"] = _gumbel_drafted_tokens(
        inputs, draft_logits_1d, NARROW_NUM_TRIALS, num_speculative_steps
    )

    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps
    )

    target_probs = torch.softmax(target_logits_1d, dim=0)
    for pos in range(num_speculative_steps + 1):
        accepted_mask = num_sampled >= pos + 1
        _assert_distribution_match(
            sampled[accepted_mask, pos], target_probs, device, label=f"position {pos}"
        )


def test_calibrated_nucleus_draft_rejection_sample_is_unbiased():
    """A sharpened, truncated proposal must still sample the target."""
    torch.manual_seed(42)
    device = "cuda"
    num_speculative_steps = 3
    target_logits_1d = torch.randn(
        NARROW_VOCAB_SIZE, device=device, dtype=torch.float32
    )
    draft_logits_1d = (
        torch.randn(NARROW_VOCAB_SIZE, device=device, dtype=torch.float32) / 0.8
    )
    sorted_logits, sorted_indices = draft_logits_1d.sort(descending=True)
    sorted_probs = sorted_logits.softmax(dim=0)
    remove_sorted = sorted_probs.cumsum(dim=0) - sorted_probs >= 0.8
    remove = torch.zeros_like(remove_sorted).scatter(0, sorted_indices, remove_sorted)
    draft_logits_1d.masked_fill_(remove, -float("inf"))

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=1.0,
        num_trials=NARROW_NUM_TRIALS,
    )
    inputs["draft_sampled"] = _gumbel_drafted_tokens(
        inputs,
        draft_logits_1d,
        NARROW_NUM_TRIALS,
        num_speculative_steps,
    )

    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps
    )

    target_probs = torch.softmax(target_logits_1d, dim=0)
    for pos in range(num_speculative_steps + 1):
        accepted_mask = num_sampled >= pos + 1
        _assert_distribution_match(
            sampled[accepted_mask, pos], target_probs, device, label=f"position {pos}"
        )


@pytest.mark.parametrize("num_speculative_steps", [1, 3])
def test_greedy_rejection_sample(num_speculative_steps: int):
    """
    Verify that greedy (temperature=0) always outputs the target argmax
    at every accepted position.
    """

    torch.manual_seed(42)
    device = "cuda"
    num_trials = 10 * VOCAB_SIZE

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=0.0,
        num_trials=num_trials,
    )

    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps
    )

    target_argmax = target_logits_1d.argmax().item()

    steps = torch.arange(num_speculative_steps + 1, device=device).unsqueeze(0)
    accepted_mask = steps < num_sampled.unsqueeze(1)

    assert (sampled[accepted_mask] == target_argmax).all(), (
        "Greedy sampling produced tokens that are not the target argmax"
    )


@pytest.mark.parametrize(
    "num_speculative_steps,temperature,unconditional_rates",
    [
        (3, 1.0, [0.9, 0.5, 0.2]),
        (3, 0.0, [0.9, 0.5, 0.2]),
        (3, 1.0, [1.0, 1.0, 1.0]),
        (3, 0.0, [1.0, 1.0, 1.0]),
        (3, 1.0, [0.0, 0.0, 0.0]),
        (3, 0.0, [0.0, 0.0, 0.0]),
        (1, 1.0, [0.7]),
        (1, 0.0, [0.7]),
    ],
)
def test_synthetic_rejection_sample(
    num_speculative_steps: int,
    temperature: float,
    unconditional_rates: list[float],
):
    """
    Verify that synthetic rejection sampling produces the expected
    per-position acceptance rates. The unconditional rate at position i
    is P(all draft steps 0..i accepted) = product(conditional_rates[0:i+1]).
    This is approximately mean(num accepted >= i + 1) over many trials.
    """
    from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

    torch.manual_seed(42)
    device = "cuda"
    num_trials = 10 * VOCAB_SIZE
    deviation_tol = 1e-2

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)

    if temperature > 0:
        target_logits_1d /= temperature
        draft_logits_1d /= temperature

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=temperature,
        num_trials=num_trials,
    )

    conditional_rates = unconditional_to_conditional_rates(unconditional_rates)
    synthetic_conditional_rates = torch.tensor(
        conditional_rates, dtype=torch.float32, device=device
    )

    _, num_sampled = rejection_sample(
        **inputs,
        num_speculative_steps=num_speculative_steps,
        synthetic_conditional_rates=synthetic_conditional_rates,
    )

    # num_sampled includes the resampled/bonus token.
    num_accepted = num_sampled - 1
    for i, expected_rate in enumerate(unconditional_rates):
        observed_rate = (num_accepted >= i + 1).float().mean().item()
        assert abs(observed_rate - expected_rate) < deviation_tol, (
            f"Step {i}: observed rate {observed_rate:.4f} deviates from "
            f"expected rate {expected_rate:.4f} by more than {deviation_tol}."
        )
