# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bind platform lifecycle policy once; dynamic inputs stay in runtime components."""

from vllm.platforms import current_platform
from vllm.sm70_decode_trace import DecodeEventTracer
from vllm.v1.worker.runtime.input_transfer import InputTransferSession


def create_input_transfer(config, device, *, logger) -> InputTransferSession:
    runtime = config.kernel_config.sm70_runtime
    trace = config.observability_config.runtime_trace
    use_async = config.scheduler_config.async_scheduling
    eligible = bool(
        runtime.staged_input
        and use_async
        and device.type == "cuda"
        and current_platform.is_device_capability(70)
        and config.speculative_config is None
        and not config.num_speculative_tokens
        and not config.model_config.is_encoder_decoder
    )
    return InputTransferSession(
        eligible=eligible,
        trace_enabled=bool(trace.async_cpu and use_async),
        trace_every=trace.async_every,
        logger=logger,
        trace_prefix="SM70 async worker trace",
        staged_message="SM70 async staged input prep enabled for no-MTP decode.",
        synchronize=DecodeEventTracer(trace).synchronize,
    )
