# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafter padding and finite warmup policy owned by the platform adapter."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import torch

from vllm.config.sm70_moe import unquantized_moe_policy
from vllm.model_executor.layers import sm70_draft47
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.eagle.prefill_moe_rows import PadRowOps


def extend_decode_graphs(manager, config):
    sm70_draft47.extend_draft_decode_graphs(
        manager, requested="d2b" in config.kernel_config.sm70_draft.units
    )


def capture_pad_ops(config) -> PadRowOps | None:
    if "d2a" not in config.kernel_config.sm70_draft.units:
        return None
    return PadRowOps(
        mask=sm70_draft47.mask_pad_rows_,
        note=lambda route: sm70_draft47.note_route("d2a", route),
    )


def decode_pad_rows(speculator):
    selector = speculator.prefill_moe_rows
    if selector is None or selector.pad_ops is None:
        if "d2a" in speculator.vllm_config.kernel_config.sm70_draft.units:
            sm70_draft47.note_route("d2a", "fallback:no_selector")
        return contextlib.nullcontext()
    if not selector.pad_rows_ready:
        selector.pad_ops.note("fallback:no_selector")
        return contextlib.nullcontext()
    num_valid = speculator.input_buffers.query_start_loc[
        speculator.max_num_reqs : speculator.max_num_reqs + 1
    ]
    return selector.pad_rows(num_valid)


def warmup_moe(
    self, dummy_run: Callable[..., Any], request_size_policy, logger
) -> tuple[str, ...]:
    """Warm the finite Qwen3.8 MTP-MoE prefill and decode signatures."""
    if (
        self.method != "mtp"
        or self.device.type != "cuda"
        or (not current_platform.is_device_capability(70))
        or (not unquantized_moe_policy(self.vllm_config).value("mtp_tuned"))
    ):
        return ()
    if getattr(self, "_sm70_mtp_moe_warmed", False):
        return ()
    text_config = self.draft_model_config.hf_text_config
    if not (
        self.draft_model_config.get_num_experts() == 512
        and int(getattr(text_config, "num_experts_per_tok", 0)) == 10
        and (self.draft_model_config.get_hidden_size() == 2560)
        and (int(getattr(text_config, "moe_intermediate_size", 0)) == 640)
        and (self.vllm_config.parallel_config.tensor_parallel_size == 4)
    ):
        return ()
    self._sm70_mtp_moe_warmed = True
    warmed: list[str] = []
    try:
        if self.max_num_tokens >= 16:
            dummy_run(16)
            warmed.append("mtp_draft_moe_prefill_m16")
        decode_query_len = 1 + self.num_speculative_steps
        request_sizes = request_size_policy(
            self.vllm_config.compilation_config.cudagraph_capture_sizes,
            decode_query_len,
            self.max_num_reqs,
        )
        executed_request_sizes = []
        for num_reqs in request_sizes:
            num_tokens = decode_query_len * num_reqs
            if num_tokens > self.max_num_tokens:
                continue
            dummy_run(num_tokens, uniform_decode=True)
            executed_request_sizes.append(num_reqs)
        torch.accelerator.synchronize()
        if executed_request_sizes:
            warmed.append(
                "mtp_draft_moe_decode_reqs_"
                + "_".join(str(size) for size in executed_request_sizes)
            )
    except Exception as err:
        logger.warning_once("SM70 V2 MTP MoE warmup skipped: %s", err)
        return ()
    return tuple(warmed)
