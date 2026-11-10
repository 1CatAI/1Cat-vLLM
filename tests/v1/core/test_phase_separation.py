# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import SchedulerConfig
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.phase_separation import Phase, PhaseConfig, PhaseCreditAllocator
from vllm.v1.outputs import ModelRunnerOutput


def test_credit_feedback_achieves_wall_time_share_without_a_stale_decision():
    control = PhaseCreditAllocator(PhaseConfig(max_decode_wait_ms=10000))
    now = decode_ms = elapsed_ms = 0.0
    for _ in range(1000):
        control.charge_elapsed(now)
        phase = control.next_phase(has_decode=True, has_prefill=True, now=now)
        control.begin_phase(phase, now)
        prefill = phase is Phase.PREFILL
        cost = 200 if prefill else 50
        control.observe_step_work(
            scheduled_tokens=1024 if prefill else 1,
            prefill_tokens=1024 if prefill else 0,
        )
        now += cost / 1000
        elapsed_ms += cost
        decode_ms += 0 if prefill else cost
    assert decode_ms / elapsed_ms == pytest.approx(1 / 3, abs=0.005)


def test_wait_guard_overrides_credit_at_a_step_boundary():
    control = PhaseCreditAllocator(
        PhaseConfig(decode_time_share=0, max_decode_wait_ms=100)
    )
    control.credit_ms = -1000
    control.begin_phase(Phase.PREFILL, 0)
    assert (
        control.next_phase(has_decode=True, has_prefill=True, now=0.099)
        is Phase.PREFILL
    )
    assert (
        control.next_phase(has_decode=True, has_prefill=True, now=0.100) is Phase.DECODE
    )


def test_empty_decode_attempt_overrides_fresh_decode_and_idle_clears_the_flip():
    control = PhaseCreditAllocator()
    control.note_empty_phase_step(Phase.DECODE)
    assert (
        control.next_phase(has_decode=True, has_prefill=True, fresh_decode=True, now=0)
        is Phase.PREFILL
    )
    control.note_empty_phase_step(Phase.DECODE)
    assert (
        control.next_phase(
            has_decode=False, has_prefill=False, has_waiting=False, now=1
        )
        is None
    )
    assert (
        control.next_phase(has_decode=True, has_prefill=True, fresh_decode=True, now=2)
        is Phase.DECODE
    )
    assert (
        control.next_phase(has_decode=False, has_prefill=False, has_waiting=True, now=3)
        is Phase.PREFILL
    )


@pytest.mark.parametrize(
    "tokens,prefill,expected", [(16, 4, 0), (1, 0, -100), (256, 256, 50)]
)
def test_feedback_uses_actual_recovery_work(tokens, prefill, expected):
    control = PhaseCreditAllocator()
    control.begin_phase(Phase.PREFILL, 0)
    control.observe_step_work(scheduled_tokens=tokens, prefill_tokens=prefill)
    control.charge_elapsed(0.1)
    assert control.credit_ms == pytest.approx(expected)
    if 0 < prefill < tokens:
        assert control.prefill_ms_per_token is control.decode_step_ms is None
        assert control.mixed_steps == 1


def test_idle_intervals_do_not_poison_cost_estimates():
    control = PhaseCreditAllocator()
    control.begin_phase(Phase.PREFILL, 0)
    control.observe_step_work(scheduled_tokens=1024, prefill_tokens=1024)
    control.charge_elapsed(60)
    assert control.credit_ms == 0 and control.prefill_ms_per_token is None


def test_chunk_bounds_remain_aligned_under_extreme_cost_estimates():
    control = PhaseCreditAllocator(
        PhaseConfig(block_size=2048, max_prefill_tokens=8193)
    )
    for cost in [None, 0.0001, 0.1, 0.2, 0.4, 10.0]:
        control.prefill_ms_per_token = cost
        cap = control.prefill_chunk_cap()
        assert 2048 <= cap <= 8192 and cap % 2048 == 0


@pytest.mark.parametrize(
    "values",
    [
        dict(decode_time_share=1),
        dict(decode_time_share=-1),
        dict(decode_time_share=float("nan")),
        dict(max_decode_wait_ms=0),
        dict(max_feedback_ms=float("inf")),
        dict(ema_alpha=0),
        dict(block_size=0),
        dict(block_size=2048, max_prefill_tokens=1024),
        dict(min_prefill_tokens=130, max_prefill_tokens=140, block_size=128),
        dict(decode_sweep_interval_steps=-1),
    ],
)
def test_invalid_phase_geometry_or_feedback_policy_is_rejected(values):
    with pytest.raises(ValueError):
        PhaseConfig(**values)


def test_sweep_equalizes_resident_budget_and_forgets_finished_requests():
    control = PhaseCreditAllocator(PhaseConfig(decode_sweep_interval_steps=2))
    control.mark_pending_decoders(["old", "new"], {"old"})
    assert not control.due_for_sweep()
    control.mark_pending_decoders(["old", "new"], {"old"})
    assert control.due_for_sweep()
    assert control.sweep_share(["old", "new"], 64) == 32
    control.mark_pending_decoders(["old"], {"old"})
    assert not control.due_for_sweep()


def test_typed_phase_policy_is_captured_and_unused_invalid_alias_is_deferred(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_SM70_PHASE_SEPARATION", "0")
    monkeypatch.setenv("VLLM_SM70_PHASE_DECODE_SHARE", "invalid")
    disabled = SchedulerConfig(
        max_model_len=2048,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        is_encoder_decoder=False,
    )
    assert not disabled.phase_separation and disabled.phase_decode_share is None
    captured = SchedulerConfig(
        max_model_len=2048,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        phase_separation=True,
        phase_decode_share=0.4,
        phase_max_wait_ms=100,
        phase_log_interval=0,
        is_encoder_decoder=False,
    )
    monkeypatch.setenv("VLLM_SM70_PHASE_DECODE_SHARE", "0.9")
    assert captured.phase_decode_share == 0.4
    assert captured.compute_hash() == disabled.compute_hash()


@pytest.fixture
def local_model_config(tmp_path):
    # Configuration only: exercise the real scheduler without weights/network.
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                vocab_size=128,
                max_position_embeddings=8192,
                torch_dtype="float16",
            )
        )
    )
    return str(tmp_path)


def _scheduler(monkeypatch, model, **kwargs):
    monkeypatch.setenv("VLLM_SM70_PHASE_SEPARATION", "1")
    monkeypatch.setenv("VLLM_SM70_PHASE_DECODE_SHARE", "0.4")
    instance = create_scheduler(
        model=model,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=kwargs.pop("max_model_len", 2048),
        skip_tokenizer_init=True,
        **kwargs,
    )
    clock = [100.0]
    monkeypatch.setattr(
        scheduler_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    return instance, clock


def _finish_step(scheduler, step):
    ids = list(step.num_scheduled_tokens)
    samples = [[] if scheduler.requests[rid].is_prefill_chunk else [7] for rid in ids]
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=samples,
        ),
    )


def _decoder(scheduler):
    request = create_requests(1, num_tokens=16, max_tokens=64, req_ids=["decode"])[0]
    scheduler.add_request(request)
    step = scheduler.schedule()
    _finish_step(scheduler, step)
    assert request.num_output_tokens == 1
    return request


def test_real_scheduler_keeps_both_resident_phases_separate(
    monkeypatch, local_model_config
):
    scheduler, clock = _scheduler(monkeypatch, local_model_config)
    _decoder(scheduler)
    prefill = create_requests(1, num_tokens=128, max_tokens=64, req_ids=["prefill"])[0]
    scheduler.add_request(prefill)
    scheduler.phase_separation.credit_ms = -1000
    clock[0] += 0.01
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"prefill": 64}
    _finish_step(scheduler, step)
    scheduler.phase_separation.credit_ms = 1000
    clock[0] += 0.01
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"decode": 1}
    _finish_step(scheduler, step)


def test_first_waiting_prefill_trains_feedback_and_ineligible_decode_does_not_block(
    monkeypatch, local_model_config
):
    scheduler, clock = _scheduler(monkeypatch, local_model_config)
    scheduler.phase_separation.config = replace(
        scheduler.phase_separation.config, min_prefill_tokens_for_ema=1
    )
    decoder = _decoder(scheduler)
    decoder.next_decode_eligible_step = scheduler.current_step + 1000
    request = create_requests(1, num_tokens=128, req_ids=["prefill"])[0]
    scheduler.add_request(request)
    clock[0] += 0.016
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"prefill": 64}
    assert scheduler.phase_separation.prefill_ms_per_token == pytest.approx(1)
    _finish_step(scheduler, step)


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("shared_prefix", [False, True])
def test_cached_preempted_decode_keeps_recovery_and_skips_mixed_feedback(
    monkeypatch, local_model_config, warm, shared_prefix
):
    monkeypatch.setenv("VLLM_KV_RETENTION_ENABLED", str(int(warm)))
    scheduler, clock = _scheduler(
        monkeypatch, local_model_config, enable_prefix_caching=True
    )
    decoder = _decoder(scheduler)
    prefill = create_requests(
        1 if shared_prefix else 2,
        num_tokens=96,
        req_ids=["prefill"] if shared_prefix else ["unused", "prefill"],
    )[-1]
    scheduler.add_request(prefill)
    scheduler.phase_separation.credit_ms = -1000
    clock[0] += 0.01
    prefill_step = scheduler.schedule()
    assert prefill_step.num_scheduled_tokens == {"prefill": 64}
    _finish_step(scheduler, prefill_step)
    remaining = 16 if shared_prefix else 32
    assert prefill.num_tokens - prefill.num_computed_tokens == remaining
    scheduler.running.remove(decoder)
    scheduler._preempt_request(decoder, clock[0])
    clock[0] += 0.01
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"prefill": remaining, "decode": 1}
    assert scheduler.phase_separation.mixed_steps == 1
    _finish_step(scheduler, step)
    credit = scheduler.phase_separation.credit_ms
    scheduler.phase_separation.charge_elapsed(clock[0] + 0.1)
    assert scheduler.phase_separation.credit_ms == credit


@pytest.mark.parametrize(
    "options",
    [dict(async_scheduling=True), dict(enable_chunked_prefill=False, max_model_len=64)],
)
def test_explicit_phase_admission_rejects_unsupported_modes(
    monkeypatch, local_model_config, options
):
    with pytest.raises(ValueError, match="phase_separation requires"):
        _scheduler(monkeypatch, local_model_config, **options)
