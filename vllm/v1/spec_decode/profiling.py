# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Legacy report layouts and static eligibility for speculative profiling."""

from logging import Logger
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from vllm.distributed.parallel_state import is_last_pp_first_tp_rank
from vllm.v1.worker.runtime.profiling import ProfileReport, StepProfiler

_REPORTS = {
    "runner": ProfileReport(
        label="SM70 spec runner profile",
        preferred=(
            "target_forward",
            "target_logits",
            "target_rejection_sample",
            "target_sample_no_spec",
            "state_update_wall_cpu",
            "state_update_validate_cpu",
            "state_update_attn_compact_cpu",
            "state_update_mamba_compact_cpu",
            "state_update_input_batch_cpu",
            "state_update_drafter_context_cpu",
            "draft_total",
            "draft_wall_cpu",
            "bookkeeping",
            "bookkeeping_wall_cpu",
        ),
        metadata=(("num_tokens", "num_tokens"), ("num_reqs", "num_reqs")),
        spec_steps=True,
    ),
    "runner_v2": ProfileReport(
        label="SM70 V2 MTP profile",
        preferred=(
            "target_verifier_wall_cpu",
            "target_verifier_gpu",
            "target_forward",
            "target_sample",
            "target_state_update",
            "draft_total",
            "total_gpu",
            "total_wall_cpu",
        ),
        metadata=(("tokens", "num_tokens"), ("drafts", "num_draft_tokens")),
        append_other=False,
        gpu_sums=(
            (
                "target_verifier_gpu",
                ("target_forward", "target_sample", "target_state_update"),
            ),
        ),
        cpu_fields=("target_verifier_wall_cpu",),
        wall_start="total_wall_start",
    ),
    "proposer": ProfileReport(
        label="SM70 MTP proposer profile",
        preferred=(
            "total_gpu",
            "total_wall_cpu",
            "first_setup_cpu",
            "first_forward",
            "first_sample",
            "loop_metadata_cpu",
            "loop0_forward",
            "loop0_sample",
            "loop1_forward",
            "loop1_sample",
            "loop2_forward",
            "loop2_sample",
        ),
        metadata=(("batch", "batch_size"), ("tokens", "num_tokens")),
        interval=False,
    ),
}


def create_step_profiler(
    config: "VllmConfig",
    device: torch.device | str,
    *,
    role: Literal["runner", "runner_v2", "proposer"],
    logger: Logger,
    is_last_pp_rank: bool = True,
) -> StepProfiler:
    policy = config.observability_config.step_profiler
    assert policy.interval is not None
    spec = config.speculative_config
    methods = (
        ("mtp", "dflash", "dflash_ddtree", "dspark")
        if role == "proposer"
        else ("mtp", "dflash", "dspark")
    )
    return StepProfiler(
        enabled=bool(
            policy.enabled
            and spec is not None
            and spec.method in methods
            and getattr(device, "type", device) == "cuda"
            and (role != "runner_v2" or is_last_pp_rank)
        ),
        interval=policy.interval,
        report=_REPORTS[role],
        logger=logger,
        report_rank=is_last_pp_first_tp_rank,
    )
