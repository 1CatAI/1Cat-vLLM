# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound Flash-V100 policy and independent deferred/loader compatibility."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any, Generic, TypeVar, cast, overload

import vllm.envs as envs

FP8_PREFILL_BRIDGE_PAGE_SIZE = 784


def options():
    from vllm.config.execution_policy import flash_v100_policy

    return flash_v100_policy().options


def trace():
    from vllm.config.sm70_runtime import capture_runtime_trace

    return capture_runtime_trace()


def observations(name, standalone):
    from vllm.diagnostics import diagnostics_for

    owner = diagnostics_for()
    return (
        standalone
        if owner is None
        else owner.histories.setdefault("flash_v100_" + name, {})
    )


def observed(name, key, standalone):
    """Remember an observation while preserving old standalone set aliases."""
    values = observations(name, standalone)
    if key in values:
        return True
    if isinstance(values, set):
        values.add(key)
    else:
        values[key] = True
    return False


def diagnostic_seen(name):
    from vllm.diagnostics import diagnostics_for
    from vllm.logger import log_once_seen

    owner = diagnostics_for()
    if owner is None:
        return log_once_seen(name)
    return name in owner.histories.setdefault("flash_v100_once", {})


def mark_diagnostic(name, value):
    from vllm.diagnostics import diagnostics_for
    from vllm.logger import set_log_once_state

    owner = diagnostics_for()
    if owner is None:
        set_log_once_state(name, value)
    elif value:
        owner.histories.setdefault("flash_v100_once", {})[name] = True
    else:
        owner.histories.setdefault("flash_v100_once", {}).pop(name, None)


def registered(name: str) -> Any:
    return getattr(envs, name)


@overload
def raw(name: str, default: str) -> str: ...


@overload
def raw(name: str, default: None = None) -> str | None: ...


def raw(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def env_is_set(name: str) -> bool:
    return name in os.environ


@dataclass(frozen=True)
class Fp8BridgeWorkspacePlan:
    """Construction-time bound for one stream's optional bridge workspace."""

    head_groups: int
    budget_bytes: int
    projected_bytes: int
    max_kv_tokens: int
    max_query_tokens: int

    @classmethod
    def build(
        cls,
        *,
        budget_bytes: int,
        max_model_len: int,
        max_query_tokens: int,
        block_size: int,
        num_kv_heads: int,
        num_heads: int,
        head_size: int,
    ) -> Fp8BridgeWorkspacePlan:
        if budget_bytes < 0:
            raise ValueError("fp8_bridge_workspace_bytes must be nonnegative")
        if min(
            max_model_len, max_query_tokens, block_size,
            num_kv_heads, num_heads, head_size,
        ) <= 0 or num_heads % num_kv_heads:
            raise ValueError("invalid FP8 bridge planning dimensions")
        # Input block-table capacity may exceed max_model_len by a partial page.
        # Conversion then rounds again to its native output pages.
        input_tokens = (
            (max_model_len + block_size - 1) // block_size * block_size
        )
        page_size = FP8_PREFILL_BRIDGE_PAGE_SIZE
        kv_tokens = (input_tokens + page_size - 1) // page_size * page_size
        query_tokens = min(max_model_len, max_query_tokens)
        padded_query_tokens = (query_tokens + 63) // 64 * 64
        table_bytes = kv_tokens // page_size * 4
        for groups in range(1, num_kv_heads + 1):
            if num_kv_heads % groups:
                continue
            # FP16 K/V, optional tail Q/output, and grouped Q/output copies.
            # Count retained tail buffers even for an aligned maximum chunk:
            # a shorter, unaligned chunk can have allocated them previously.
            projected = (
                4 * kv_tokens * (num_kv_heads // groups) * head_size
                + table_bytes
                + 4 * padded_query_tokens * (num_heads // groups) * head_size
                + (
                    4 * query_tokens * (num_heads // groups) * head_size
                    if groups > 1 else 0
                )
            )
            if projected <= budget_bytes:
                return cls(groups, budget_bytes, projected, kv_tokens, query_tokens)
        return cls(0, budget_bytes, 0, kv_tokens, query_tokens)


def configure_fp8_bridge_workspace_plans(engine, kv_cache_groups, layers) -> None:
    """Refine each layer from its final physical KV-group page, before use."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    for group in kv_cache_groups:
        for name in group.layer_names:
            layer = layers.get(name)
            configure = getattr(
                getattr(layer, "impl", None), "configure_fp8_bridge_workspace", None
            )
            if configure is None:
                continue
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            # Kernel pages divide physical group pages, so this covers their
            # maximum block-table padding without assuming a default page size.
            configure(engine, spec.block_size)


@dataclass(frozen=True)
class V100AttnConfig:
    """Resolved construction-time policy, separate from native ops and buffers."""

    allow_triton_fallback: bool
    compare_bhmd_out_dir: str | None
    compare_bhmd_out_max_calls: int
    compare_triton_out_dir: str | None
    compare_triton_out_max_calls: int
    compare_triton_tensor_dump_dir: str | None
    compare_triton_tensor_dump_max_tokens: int
    decode_strategy: str
    fp8_bridge_head_groups: int
    fp8_bridge_workspace_plan: Fp8BridgeWorkspacePlan | None
    prefill_bfla_mask_block_n: int
    prefill_bfla_min_kv: int
    prefill_bfla_min_q: int
    prefill_contig_dense_allow_copy: bool
    prefill_contig_dense_min_kv: int
    prefill_contig_dense_min_q: int
    prefill_gather_dense_min_kv: int
    prefill_gather_dense_min_q: int
    prefill_split_kv_max_q: int
    prefill_split_kv_min_kv: int
    prefill_split_kv_min_q: int
    prefill_split_kv_tokens: int
    prefix_anchored_decode_window: int | None
    smallq_decode_max_model_len: int
    smallq_decode_max_query_len: int
    use_decode_dense_cache: bool
    use_decode_dense_reference: bool
    use_decode_paged_prefill: bool
    use_decode_paged_prefill_bhmd_out: bool
    use_decode_scalar_paged: bool
    use_decode_wmma_wrapper: bool
    use_decode_xqa: bool
    use_flash_v100: bool
    use_flash_v100_decode: bool
    use_flash_v100_prefill_bfla: bool
    use_flash_v100_prefill_contig_dense: bool
    use_flash_v100_prefill_gather_dense: bool
    use_flash_v100_prefill_paged: bool
    use_flash_v100_prefill_splitkv: bool
    use_fp8_prefill_bridge: bool
    use_prefill_paged_cache: bool
    use_smallq_decode_xqa: bool
    use_triton_prefill: bool

    @classmethod
    def take_legacy_attributes(cls, attributes: dict[str, Any]) -> V100AttnConfig:
        """Transfer constructor values without re-reading or reordering policy."""
        return cls(**{field.name: attributes.pop(field.name) for field in fields(cls)})


T = TypeVar("T")


class ConfigField(Generic[T]):
    """Compatibility view of a policy field while executor call sites migrate.

    Old partial test fixtures and construction retain their original attributes.
    Once frozen, a legacy assignment replaces the snapshot instead of mutating it.
    Executors can consume the immutable snapshot directly.
    """

    def __init__(self, name: str):
        self.name = name

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> ConfigField[T]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None = None) -> T: ...

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        attributes = vars(instance)
        if "config" in attributes:
            return cast(T, getattr(attributes["config"], self.name))
        if self.name not in attributes:
            raise AttributeError(self.name)
        return cast(T, attributes[self.name])

    def __set__(self, instance, value: T) -> None:
        attributes = vars(instance)
        if "config" in attributes:
            instance.config = replace(attributes["config"], **{self.name: value})
        else:
            attributes[self.name] = value
