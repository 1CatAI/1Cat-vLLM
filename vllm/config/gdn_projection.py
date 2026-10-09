# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialized projection and graph-boundary policy for the GDN adapter."""

from typing import ClassVar

from pydantic import Field

from vllm.config.execution_policy import ExecutionPolicy
from vllm.config.utils import config


@config
class GdnProjectionConfig(ExecutionPolicy):
    """Policy only; dynamic shapes, numerical stages and safety gates stay put."""

    input_projection: bool | None = None
    """Use the existing opaque input-projection boundary."""

    output_projection: bool | None = None
    """Use the existing opaque output-projection boundary."""

    input_core: bool | None = None
    """Request the combined input and recurrent-core boundary."""

    disable_input_core: bool | None = None
    """Retain the explicit override disabling the input/core boundary."""

    context_core: bool | None = None
    """Use the existing forward-context recurrent entry."""

    full_forward: bool | None = None
    """Explicitly request the whole-layer opaque boundary."""

    disable_full_forward: bool | None = None
    """Disable the automatic wrapper except when its safety guard is required."""

    spec_core_003: bool | None = None
    """Keep the experimental 0.0.3 speculative-core entry."""

    spec_allow_deep_mtp: bool | None = None
    """Retain the explicit override of the deep-MTP safety restriction."""

    mixed_qkv_contiguous: bool | None = None
    """Retain the projection QKV materialization policy."""

    z_contiguous: bool | None = None
    """Retain the projected gate materialization policy."""

    rmsnorm_onepass: bool | None = None
    """Admit the existing 12x128 one-pass gated norm."""

    qpn8_ba_split: bool | None = None
    """Admit the paired QPN8 b/a projection; validate native ops at binding."""

    batch_split_copy: bool | None = None
    """Use the existing native split/copy after fused input projection."""

    input_batch: bool | None = None
    """Admit the existing batched Qwen input projection at its shape gate."""

    aliases: ClassVar[dict[str, str]] = {
        "batch_split_copy": "VLLM_SM70_GDN_BATCH_SPLIT_COPY",
        "input_batch": "VLLM_SM70_QWEN38_GDN_INPUT_BATCH",
        "input_projection": "VLLM_SM70_QWEN_GDN_INPUT_PROJECTION_OP",
        "output_projection": "VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP",
        "input_core": "VLLM_SM70_QWEN_GDN_INPUT_CORE_OP",
        "disable_input_core": "VLLM_SM70_QWEN_GDN_DISABLE_INPUT_CORE_OP",
        "context_core": "VLLM_SM70_QWEN_GDN_CONTEXT_CORE",
        "full_forward": "VLLM_SM70_QWEN_GDN_FULL_FORWARD",
        "disable_full_forward": "VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD",
        "spec_core_003": "VLLM_SM70_QWEN_GDN_003_SPEC_CORE_OP",
        "spec_allow_deep_mtp": "VLLM_SM70_QWEN_GDN_003_SPEC_ALLOW_DEEP_MTP",
        "mixed_qkv_contiguous": "VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS",
        "z_contiguous": "VLLM_SM70_GDN_Z_CONTIGUOUS",
        "rmsnorm_onepass": "VLLM_SM70_GDN_RMSNORM_ONEPASS",
        "qpn8_ba_split": "VLLM_SM70_GDN_QPN8_BA_SPLIT",
    }

    provider_errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Retain dormant batch-projection parse failures until their admission gate."""

    def value(self, field):
        if field in self.provider_errors:
            raise ValueError(self.provider_errors[field])
        return getattr(self, field)

    def resolve(self) -> None:
        from vllm.config.sm70_runtime import resolve_legacy_fields

        batch_fields = ("batch_split_copy", "input_batch")
        resolve_legacy_fields(
            self,
            {
                field: self.aliases[field]
                for field in batch_fields
                if field not in self.sources
            },
            deferred_errors=self.provider_errors,
        )
        pending = {
            field: alias
            for field, alias in self.aliases.items()
            if field not in self.sources and field != "input_core"
        }
        resolve_legacy_fields(self, pending)
        if "input_core" not in self.sources:
            if self.disable_input_core and self.input_core is None:
                # The disabling override used to short-circuit this getter.
                self.input_core = False
                self.sources["input_core"] = "disabled by disable_input_core"
            else:
                resolve_legacy_fields(self, {"input_core": self.aliases["input_core"]})


def projection_policy(cfg=None) -> GdnProjectionConfig:
    from vllm.config.execution_policy import capture_execution_policy

    return capture_execution_policy(
        "kernel_config.gdn.projection", GdnProjectionConfig, cfg
    )
