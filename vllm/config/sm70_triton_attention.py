# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 fallback attention schedule, captured independently of Flash-V100."""

from pydantic import Field

from vllm.config.execution_policy_base import DeferredExecutionPolicy
from vllm.config.utils import config, hash_factors


def _validate_sm70_triton_attn_tile_size(tile_size: int, name: str) -> None:
    if tile_size == 0:
        return
    if tile_size <= 0 or tile_size & (tile_size - 1) != 0:
        raise ValueError(f"{name} must be a positive power of 2 or 0")
    if tile_size < 16 or tile_size > 128:
        raise ValueError(f"{name} must be in [16, 128] or 0")


def _validate_sm70_triton_attn_warps(num_warps: int) -> None:
    if num_warps == 0:
        return
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("VLLM_SM70_TRITON_ATTN_NUM_WARPS must be 1, 2, 4, 8, or 0")


def _validate_sm70_triton_attn_input_precision(value: str | None, name: str) -> str:
    if value is None or value == "":
        return ""
    if value not in ("tf32", "tf32x3", "ieee"):
        raise ValueError(f"{name} must be one of tf32, tf32x3, ieee, or empty")
    return value


@config
class Sm70TritonAttentionPolicy(DeferredExecutionPolicy):
    """Legacy schedule precedence; tensor-dependent tile defaults remain dynamic."""

    prefill_tile_size: int | None = None
    """Zero retains the shape-dependent prefill tile."""
    decode_tile_size: int | None = None
    """Zero retains the shape-dependent decode tile."""
    num_warps: int | None = None
    """Shared override, superseded by the per-stage overrides."""
    prefill_num_warps: int | None = None
    """Zero falls through to shared and then safe defaults."""
    decode_num_warps: int | None = None
    """Zero falls through to shared and then safe defaults."""
    safe_defaults: bool | None = None
    """Apply four/eight warps when no explicit stage schedule was provided."""
    qk_input_precision: str | None = None
    """Existing QK dot-product precision override."""
    pv_input_precision: str | None = None
    """Existing PV dot-product precision override."""
    schedule: tuple[int, int, int, int, str, str] | None = Field(
        default=None, init=False
    )
    """Validated effective schedule; independent of dynamic tensor shapes."""
    schedule_error: str | None = Field(default=None, init=False)
    """Retain the old SM70 admission checkpoint for malformed unused options."""

    aliases = {
        "prefill_tile_size": "VLLM_SM70_TRITON_ATTN_PREFILL_TILE_SIZE",
        "decode_tile_size": "VLLM_SM70_TRITON_ATTN_DECODE_TILE_SIZE",
        "num_warps": "VLLM_SM70_TRITON_ATTN_NUM_WARPS",
        "prefill_num_warps": "VLLM_SM70_TRITON_ATTN_PREFILL_NUM_WARPS",
        "decode_num_warps": "VLLM_SM70_TRITON_ATTN_DECODE_NUM_WARPS",
        "safe_defaults": "VLLM_SM70_TRITON_ATTN_SAFE_DEFAULTS",
        "qk_input_precision": "VLLM_SM70_TRITON_ATTN_QK_INPUT_PRECISION",
        "pv_input_precision": "VLLM_SM70_TRITON_ATTN_PV_INPUT_PRECISION",
    }

    def resolve(self):
        super().resolve()
        try:
            values = {field: self.value(field) for field in self.aliases}
            for field in ("qk_input_precision", "pv_input_precision"):
                values[field] = _validate_sm70_triton_attn_input_precision(
                    values[field], self.aliases[field]
                )
            for field in ("prefill_tile_size", "decode_tile_size"):
                _validate_sm70_triton_attn_tile_size(values[field], self.aliases[field])
            for field in ("num_warps", "prefill_num_warps", "decode_num_warps"):
                _validate_sm70_triton_attn_warps(values[field])
            prefill = values["prefill_num_warps"] or values["num_warps"]
            decode = values["decode_num_warps"] or values["num_warps"]
            if values["safe_defaults"]:
                prefill = prefill or 4
                decode = decode or 8
            self.schedule = (
                values["prefill_tile_size"],
                values["decode_tile_size"],
                prefill,
                decode,
                values["qk_input_precision"],
                values["pv_input_precision"],
            )
            self.schedule_error = None
        except ValueError as exc:
            self.schedule_error = str(exc)

    def resolved_schedule(self):
        if self.schedule_error is not None:
            raise ValueError(self.schedule_error)
        assert self.schedule is not None, (
            "attention policy must be resolved at initialization"
        )
        return self.schedule

    def compute_hash(self):
        return hash_factors({"schedule": self.schedule} if self.active else {})


def capture_triton_attention_policy():
    from vllm.config.execution_policy import capture_execution_policy

    return capture_execution_policy(
        "attention_config.sm70_triton", Sm70TritonAttentionPolicy
    )
