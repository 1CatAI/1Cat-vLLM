# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound Flash-V100 policy and independent deferred/loader compatibility."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any, Generic, TypeVar, cast, overload

import vllm.envs as envs


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
