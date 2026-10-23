# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construction-time speculative attention policy calculations."""

from __future__ import annotations

from typing import Any

import torch

from vllm.config.sm70_dflash2 import capture_sm70_dflash2_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_v100 import config as _config

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def initialize_scalar_tail(self: Any, use_e4m3_fp32: bool) -> None:
    self._sm70_scalar_tail_attention = None
    from vllm.v1.attention.ops.sm70_grouped_scalar import (
        load_scalar_tail_attention,
        scalar_tail_attention_available,
    )

    if (
        use_e4m3_fp32
        and _config.registered("VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS")
        and (
            _config.registered("VLLM_SM70_DFLASH2_SCALAR_ATTENTION_MANIFEST")
            or scalar_tail_attention_available()
        )
        and not _config.raw("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
    ):
        # An empty name selects the operator compiled into this extension;
        # a manifest name keeps the explicit experimental override.
        self._sm70_scalar_tail_attention = load_scalar_tail_attention(
            _config.registered("VLLM_SM70_DFLASH2_SCALAR_ATTENTION_MANIFEST") or "",
            torch.device("cuda", torch.accelerator.current_device_index()),
        )


def initialize_verify_abi(self: Any, max_query_tokens, request_major_abi) -> None:
    self.dflash2_grouped_verify_max_query_tokens = max_query_tokens
    self.dflash2_grouped_verify_request_major_abi_version = request_major_abi


def configure_prefill(self: Any) -> None:
    self._flash_prefill_paged_supports_dflash2_bmhd = bool(
        getattr(self.flash_attn_prefill_paged, "_sm70_dflash2_direct_bmhd", False)
    )
    self._flash_prefill_paged_dflash2_split_pages = getattr(
        self.flash_attn_prefill_paged, "_sm70_dflash2_split_pages", ()
    )
    split_enabled = getattr(capture_sm70_dflash2_config(), "draft_window_split", True)
    if not split_enabled:
        self._flash_prefill_paged_dflash2_split_pages = ()
    if self.flash_attn_prefill_paged is not None and self.accepts_keyword(
        self.flash_attn_prefill_paged, "dflash2_window_split"
    ):
        from functools import partial

        self.flash_attn_prefill_paged = partial(
            self.flash_attn_prefill_paged, dflash2_window_split=split_enabled
        )
    logger.info_once(
        "FLASH_ATTN_V100 DFlash single-request window split pages=%s; "
        "page832 policy=%s; dtype/query/window guards apply at dispatch.",
        self._flash_prefill_paged_dflash2_split_pages,
        "enabled" if split_enabled else "disabled_by_configuration",
    )


def configure_verifier(self: Any) -> None:
    self.use_dflash2_grouped_verify = (
        self.flash_attn_grouped_verify_paged is not None
        and _config.registered("VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY")
        and current_platform.is_device_capability(70)
    )
    self.use_dflash2_batched_grouped_verify = (
        self.use_dflash2_grouped_verify
        and _config.registered("VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY")
    )
    self.dflash2_grouped_verify_min_model_len = _config.registered(
        "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY_MIN_MODEL_LEN"
    )
    if self.dflash2_grouped_verify_min_model_len < 1:
        raise ValueError(
            "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY_MIN_MODEL_LEN must be positive"
        )


POLICY_FIELDS = frozenset(
    {
        "_sm70_scalar_tail_attention",
        "dflash2_grouped_verify_max_query_tokens",
        "dflash2_grouped_verify_request_major_abi_version",
        "_flash_prefill_paged_supports_dflash2_bmhd",
        "_flash_prefill_paged_dflash2_split_pages",
        "use_dflash2_grouped_verify",
        "use_dflash2_batched_grouped_verify",
        "dflash2_grouped_verify_min_model_len",
    }
)


class SpecAttentionState:
    """Own feature policy and its two construction-time operator inputs."""

    def __init__(self, accepts_keyword):
        self.accepts_keyword = accepts_keyword

    initialize_scalar_tail = initialize_scalar_tail
    initialize_verify_abi = initialize_verify_abi

    def configure_prefill(self, operator):
        self.flash_attn_prefill_paged = operator
        configure_prefill(self)
        return self.flash_attn_prefill_paged

    def configure_verifier(self, operator):
        self.flash_attn_grouped_verify_paged = operator
        configure_verifier(self)
