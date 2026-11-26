# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-round verification length: controllers, config resolution, scheduler."""

import random
from types import SimpleNamespace

import pytest

from tests.v1.core.utils import create_requests, create_scheduler, pretend_gpu_count
from vllm.config.kernel import (
    AdaptiveVerifyConfig,
    KernelConfig,
    resolve_adaptive_verify,
)
from vllm.v1.core.sched.adaptive_verify import (
    FixedLengthController,
    MtpBandController,
    ThroughputController,
    best_verify_length,
    build_verify_controller,
)
from vllm.v1.outputs import ModelRunnerOutput

pytestmark = pytest.mark.cpu_test

# Cumulative acceptance A_j = P(at least j drafts accepted) per draft position.
SAMPLED_TEXT_ACCEPTANCE = [0.62, 0.37, 0.23, 0.15, 0.10, 0.07, 0.04]
GREEDY = [0.72, 0.50, 0.34, 0.23, 0.17, 0.11, 0.08]
HIGH_ACCEPTANCE = [0.85**j for j in range(1, 8)]
# A round cost ratio for the controller tables (GLM-5.3-Flash measured 0.119).
COST_RATIO = 0.108


def _kernel_config(**adaptive_verify) -> KernelConfig:
    return KernelConfig(adaptive_verify=AdaptiveVerifyConfig(**adaptive_verify))


def _dflash_plan(k_max: int = 7, **adaptive_verify):
    adaptive_verify.setdefault("verify_cost_ratio", COST_RATIO)
    plan = resolve_adaptive_verify(
        SimpleNamespace(method="dflash"),
        _kernel_config(enabled=True, **adaptive_verify),
        k_max,
    )
    assert plan is not None
    return plan


@pytest.mark.parametrize(
    ("acceptance", "expected_length"),
    [(SAMPLED_TEXT_ACCEPTANCE, 3), (GREEDY, 4), (HIGH_ACCEPTANCE, 7)],
)
def test_best_verify_length_follows_the_throughput_optimum(acceptance, expected_length):
    assert best_verify_length(acceptance, COST_RATIO, 1, 7) == expected_length


def test_best_verify_length_respects_the_bounds():
    assert best_verify_length(HIGH_ACCEPTANCE, COST_RATIO, 1, 5) == 5
    assert best_verify_length(SAMPLED_TEXT_ACCEPTANCE, COST_RATIO, 4, 7) == 4


def test_unobserved_positions_do_not_move():
    controller = ThroughputController(_dflash_plan(initial_tokens=3))
    state = controller.new_state()
    prior = controller.new_state()

    controller.update(state, num_draft_tokens=3, num_accepted_tokens=2)

    assert state.trials[:3] != prior.trials[:3]
    assert state.trials[3:] == prior.trials[3:]
    assert state.successes[3:] == prior.successes[3:]


def test_full_acceptance_probes_one_draft_further():
    # A pessimistic prior alone would keep n at the minimum.
    controller = ThroughputController(
        _dflash_plan(initial_tokens=3, prior_conditional_acceptance=0.05)
    )
    state = controller.new_state()
    state.length = 3
    controller.update(state, num_draft_tokens=3, num_accepted_tokens=3)
    assert state.length == 4


def test_position_after_the_first_rejection_is_not_counted_as_tried():
    controller = ThroughputController(_dflash_plan())
    state = controller.new_state()
    prior = controller.new_state()

    controller.update(state, num_draft_tokens=5, num_accepted_tokens=1)

    assert state.trials[0] != prior.trials[0]
    assert state.trials[1] != prior.trials[1]  # tried, rejected
    assert state.trials[2:] == prior.trials[2:]  # never reached
    assert state.successes[2:] == prior.successes[2:]


def test_low_acceptance_lowers_the_length_and_it_stays_in_bounds():
    controller = ThroughputController(_dflash_plan(initial_tokens=7, min_tokens=2))
    state = controller.new_state()
    for _ in range(30):
        controller.update(state, num_draft_tokens=state.length, num_accepted_tokens=0)
        assert 2 <= state.length <= 7
    assert state.length == 2


def test_forced_length_never_changes():
    plan = _dflash_plan(forced_tokens=5)
    controller = build_verify_controller(plan)
    assert isinstance(controller, FixedLengthController)
    state = controller.new_state()
    for accepted in (0, 5, 2):
        controller.update(state, num_draft_tokens=5, num_accepted_tokens=accepted)
        assert state.length == 5


def test_dispatch_picks_the_controller_of_the_rule():
    assert isinstance(build_verify_controller(_dflash_plan()), ThroughputController)
    assert isinstance(
        build_verify_controller(_dflash_plan(forced_tokens=3)),
        FixedLengthController,
    )


# Depth sequences of 1Cat's EntMTP controller recorded from its code as it was
# before this dispatch (async_scheduler.py of 1Cat main 2d218ab03): 200
# rounds with the acceptance probability redrawn every 25 rounds, seeds
# 99 + k_max. Each row is (k_max, k_min, depths, ema_successes, ema_trials,
# final depth). The k=7 row records 1Cat's behaviour as it is: its hysteresis
# table has no entry for depth 7, so that depth is never left.
MTP_RECORDINGS = [
    (
        7,
        2,
        "7" * 200,
        0.9744077310621213,
        1.9744077310621215,
        7,
    ),
    (
        6,
        3,
        (
            "65665443433433345445654545433333343433333343333333345434333334543344333333"
            "334344545445666655656566566666665666545434344545666666666556565654344433433"
            "433345654334343343334334334333343333333334334334343"
        ),
        1.6769947260808458,
        2.137515225409543,
        4,
    ),
    (
        5,
        2,
        (
            "54332232222222222222322222222222222223233322222222234555555555555555555555"
            "555443232222234323222222222323345544555555555555555433322222222222222222222"
            "223232333454343334544433455432232223222222222222222"
        ),
        0.3460011547533021,
        1.3440428527699013,
        2,
    ),
]


@pytest.mark.parametrize(
    ("k_max", "k_min", "depths", "ema_successes", "ema_trials", "final_depth"),
    MTP_RECORDINGS,
)
def test_mtp_bands_reproduce_1cat_bit_for_bit(
    monkeypatch, k_max, k_min, depths, ema_successes, ema_trials, final_depth
):
    speculative_config = SimpleNamespace(method="mtp")
    monkeypatch.setattr(
        "vllm.config.speculative_sampling.resolve_sampling_policy",
        lambda _config: SimpleNamespace(adaptive_mtp="on", adaptive_mtp_min_k=k_min),
    )
    plan = resolve_adaptive_verify(speculative_config, KernelConfig(), k_max)
    assert plan is not None
    controller = build_verify_controller(plan)
    assert isinstance(controller, MtpBandController)
    state = controller.new_state()
    rng = random.Random(99 + k_max)
    probability = 0.5
    chosen = []
    for round_index in range(200):
        if round_index % 25 == 0:
            probability = rng.choice([0.3, 0.55, 0.7, 0.78, 0.85, 0.92])
        draft = state.length
        accepted = 0
        while accepted < draft and rng.random() < probability:
            accepted += 1
        chosen.append(draft)
        controller.update(state, draft, accepted)

    assert "".join(map(str, chosen)) == depths
    assert state.ema_successes == ema_successes
    assert state.ema_trials == ema_trials
    assert state.length == final_depth


def _adaptive_scheduler(monkeypatch, plan, pipeline_parallel_size: int = 1):
    """An async scheduler as the V2 runner drives it.

    The CPU config cannot hold a DFlash drafter, so the speculative setup is
    what the scheduler derives from it: the block width and its placeholders.
    """
    pretend_gpu_count(monkeypatch, pipeline_parallel_size)
    scheduler = create_scheduler(
        async_scheduling=True, pipeline_parallel_size=pipeline_parallel_size
    )
    scheduler.use_v2_model_runner = True
    scheduler.num_spec_tokens = plan.max_tokens
    scheduler._spec_token_placeholders = (-1,) * plan.max_tokens
    scheduler._verify_controller = build_verify_controller(plan)
    return scheduler


def _run_rounds(scheduler, request, accepted_per_round, num_rounds):
    """Feed back `accepted_per_round` drafts each round; yield each round."""
    for round_index in range(num_rounds):
        output = scheduler.schedule()
        if not output.num_scheduled_tokens:
            continue  # pipeline bubble of the PP cadence
        drafts = output.scheduled_spec_decode_tokens.get(request.request_id, [])
        accepted = min(accepted_per_round(round_index), len(drafts))
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[7] * (accepted + 1)],
            ),
        )
        yield len(drafts)


def _scheduled_draft_counts(scheduler, request, accepted_per_round, num_rounds):
    return list(_run_rounds(scheduler, request, accepted_per_round, num_rounds))


def test_scheduler_shrinks_the_verify_length_when_nothing_is_accepted(monkeypatch):
    scheduler = _adaptive_scheduler(monkeypatch, _dflash_plan(initial_tokens=4))
    (request,) = create_requests(num_requests=1, num_tokens=37, max_tokens=400)
    scheduler.add_request(request)

    counts = _scheduled_draft_counts(scheduler, request, lambda _: 0, 14)

    # Prefill plans no drafts; the first decode round uses initial_tokens.
    assert counts[:2] == [0, 4]
    assert counts[-1] == 1
    assert counts[1:] == sorted(counts[1:], reverse=True)


def test_scheduler_grows_the_verify_length_when_everything_is_accepted(monkeypatch):
    scheduler = _adaptive_scheduler(monkeypatch, _dflash_plan(initial_tokens=2))
    (request,) = create_requests(num_requests=1, num_tokens=37, max_tokens=400)
    scheduler.add_request(request)

    counts = _scheduled_draft_counts(scheduler, request, lambda _: 7, 14)

    assert counts[:2] == [0, 2]
    assert counts[-1] == 7


def test_forced_length_is_planned_every_round(monkeypatch):
    scheduler = _adaptive_scheduler(monkeypatch, _dflash_plan(forced_tokens=5))
    (request,) = create_requests(num_requests=1, num_tokens=37, max_tokens=400)
    scheduler.add_request(request)

    counts = _scheduled_draft_counts(scheduler, request, lambda i: i % 6, 12)

    assert counts[0] == 0
    assert set(counts[1:]) == {5}


def test_varying_verify_length_keeps_pipeline_bookkeeping_consistent(monkeypatch):
    scheduler = _adaptive_scheduler(
        monkeypatch, _dflash_plan(initial_tokens=4), pipeline_parallel_size=2
    )
    (request,) = create_requests(num_requests=1, num_tokens=37, max_tokens=400)
    scheduler.add_request(request)
    pattern = [0, 4, 1, 2, 4, 4, 0, 3]

    counts = []
    for count in _run_rounds(scheduler, request, lambda i: pattern[i % 8], 40):
        counts.append(count)
        assert request.num_output_placeholders == 0
        assert request.num_computed_tokens == request.num_tokens - 1
    assert len(set(counts[1:])) > 1


def _scheduler_from_init(monkeypatch, plan, use_v2_model_runner: bool):
    """Build an async scheduler through its own __init__ with `plan` resolved.

    The CPU config holds no drafter, so the resolver is answered with a plan the
    real resolver produced; the wiring from plan to controller is the code under
    test."""
    from vllm.config import VllmConfig
    from vllm.v1.core.sched import async_scheduler

    monkeypatch.setattr(
        VllmConfig,
        "use_v2_model_runner",
        property(lambda _self: use_v2_model_runner),
    )
    monkeypatch.setattr(async_scheduler, "resolve_adaptive_verify", lambda *_args: plan)
    pretend_gpu_count(monkeypatch, 1)
    return create_scheduler(async_scheduling=True)


def _mtp_plan(monkeypatch, k_max: int = 5):
    monkeypatch.setattr(
        "vllm.config.speculative_sampling.resolve_sampling_policy",
        lambda _config: SimpleNamespace(adaptive_mtp="on", adaptive_mtp_min_k=2),
    )
    plan = resolve_adaptive_verify(SimpleNamespace(method="mtp"), KernelConfig(), k_max)
    assert plan is not None and plan.rule == "mtp_bands"
    return plan


def test_init_builds_the_controller_of_the_resolved_plan(monkeypatch):
    scheduler = _scheduler_from_init(
        monkeypatch, _dflash_plan(initial_tokens=4), use_v2_model_runner=True
    )
    assert isinstance(scheduler._verify_controller, ThroughputController)
    assert scheduler._verify_controller.initial_length == 4


def test_init_keeps_automatic_mtp_fixed_without_the_v2_runner(monkeypatch):
    scheduler = _scheduler_from_init(
        monkeypatch, _mtp_plan(monkeypatch), use_v2_model_runner=False
    )
    assert scheduler._verify_controller is None


def test_init_without_a_plan_keeps_the_fixed_length(monkeypatch):
    scheduler = _scheduler_from_init(monkeypatch, None, use_v2_model_runner=True)
    assert scheduler._verify_controller is None
    assert scheduler._select_adaptive_batch_k(["a"]) == scheduler.num_spec_tokens


def test_a_decode_batch_of_several_requests_verifies_the_full_block(monkeypatch):
    scheduler = _adaptive_scheduler(monkeypatch, _dflash_plan(initial_tokens=2))
    controller = scheduler._verify_controller
    state = controller.new_state()
    state.length = 3
    scheduler._verify_states["a"] = state

    # A lone request follows its own state, or initial_tokens before feedback.
    assert scheduler._select_adaptive_batch_k(["a"]) == 3
    assert scheduler._select_adaptive_batch_k(["b"]) == 2
    # Several requests share one captured graph shape: the full block.
    assert scheduler._select_adaptive_batch_k(["a", "b"]) == 7
