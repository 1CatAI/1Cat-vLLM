# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_PP_KEEP_BATCH: keep a running request across empty pipeline steps.

With pipeline parallelism the engine keeps ``pipeline_parallel_size`` batches
in flight (``EngineCore.step_with_batch_queue``). With one request and
synchronous scheduling the request is still in flight on every other engine
step, so that step schedules nothing. On such an empty step the V1 runner drops
the request from its persistent batch (it was not scheduled) and the scheduler
forgets that the request ran in the previous step. The next step then copies
and serializes the request's whole token history
(``CachedRequestData.all_token_ids``) to every worker, and every runner
rebuilds the request's persistent-batch row from its prompt and output lists.
All of that is O(context) host work on the critical path of each decode round.

With the knob on, a fully empty step (no request scheduled at all) changes
neither side:

- the scheduler keeps its previous-step record for the requests that are
  still running, so it does not resend their history;
- the runner keeps its persistent-batch rows (except a request preempted in
  that step), so their token ids, block table and previous drafts stay in
  place for the next step.

Both sides take the decision from the same config through
:func:`pp_keep_batch_decision`, so they agree. This is the V1-runner
counterpart of 1Cat-vLLM PR #1263, which removed the resend for the V2 runner
only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from vllm.config import VllmConfig

KNOB = "VLLM_PP_KEEP_BATCH"


def _refuse(reason: str) -> NoReturn:
    raise ValueError(f"{KNOB} / scheduler_config.pp_keep_batch requested but {reason}")


def pp_keep_batch_requested(vllm_config: VllmConfig) -> bool:
    """Consume the serialized scheduler setting, never the worker environment."""
    value = vllm_config.scheduler_config.pp_keep_batch
    if not isinstance(value, bool):
        raise TypeError(f"{KNOB}: policy must be resolved before execution")
    return value


def pp_keep_batch_decision(vllm_config: VllmConfig) -> bool:
    """Admit the covered V1 route; an explicit unsupported request raises."""
    if not pp_keep_batch_requested(vllm_config):
        return False

    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config
    speculative_config = vllm_config.speculative_config

    pp_size = parallel_config.pipeline_parallel_size
    if pp_size <= 1:
        _refuse("pipeline_parallel_size is 1, so there are no empty pipeline steps")
    if vllm_config.use_v2_model_runner:
        _refuse(
            "the V2 model runner keeps request state on the GPU and already "
            "skips the history resend (1Cat-vLLM #1263)"
        )
    if scheduler_config.async_scheduling:
        _refuse(
            "async scheduling is on; its PP path sends no new_token_ids and "
            "rebuilds requests from all_token_ids, which this knob does not cover"
        )
    if parallel_config.distributed_executor_backend == "external_launcher":
        _refuse("the external_launcher backend is not covered")
    if vllm_config.kv_transfer_config is not None:
        _refuse(
            "a KV connector is configured; its empty steps run "
            "kv_connector_no_forward, which this knob does not cover"
        )
    if vllm_config.ec_transfer_config is not None:
        _refuse("an encoder-cache connector is configured")
    if model_config is not None:
        if model_config.runner_type == "pooling":
            _refuse("pooling models are not covered")
        if model_config.is_hybrid:
            _refuse(
                "hybrid (Mamba/linear-attention) models read per-row accepted "
                "counts that a re-added row resets; not covered"
            )
    if (
        speculative_config is not None
        and not speculative_config.supports_pp_row_retention()
    ):
        _refuse(
            "the speculative configuration does not support persistent-row retention"
        )
    return True
