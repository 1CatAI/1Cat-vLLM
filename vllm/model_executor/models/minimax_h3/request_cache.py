# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local H3 cache policies from Omni 7be014bce6374f06.

TeaCache preserves the official polynomial and residual update. Cache-DiT
executes the same pinned third-party Pattern_3 implementation as Omni.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

TEA_COEFFICIENTS = (
    2.283704065852778e03,
    -7.775977277886368e02,
    9.408414741359490e01,
    -4.232669906169421e00,
    2.173782527946167e-01,
)
CACHE_DIT_DEFAULTS = {
    "Fn_compute_blocks": 1,
    "Bn_compute_blocks": 0,
    "max_warmup_steps": 4,
    "max_cached_steps": -1,
    "residual_diff_threshold": 0.24,
    "max_continuous_cached_steps": 3,
}


def validate_cache_config(backend: str, config: dict[str, Any]) -> dict[str, Any]:
    from .config import H3InputError

    if backend not in ("none", "tea_cache", "cache_dit"):
        raise H3InputError("cache backend must be none, tea_cache or cache_dit")
    if not isinstance(config, dict):
        raise H3InputError("cache configuration must be an object")
    defaults = (
        {"rel_l1_thresh": 0.17}
        if backend == "tea_cache"
        else (CACHE_DIT_DEFAULTS if backend == "cache_dit" else {})
    )
    unknown = set(config) - set(defaults)
    if unknown:
        raise H3InputError(f"unsupported {backend} cache options: {sorted(unknown)}")
    result = {**defaults, **config}
    for key, value in result.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise H3InputError(f"{key} must be finite and numeric")
        if key in ("rel_l1_thresh", "residual_diff_threshold"):
            if value < 0:
                raise H3InputError(f"{key} must be nonnegative")
        elif not isinstance(value, int) or value < (
            -1 if key in ("max_cached_steps", "max_continuous_cached_steps") else 0
        ):
            raise H3InputError(f"invalid cache block/step count: {key}")
    return result


@dataclass(frozen=True)
class CachePlan:
    backend: str
    options: dict[str, Any]
    calls: int


def resolve_cache_plan(config, sampling, *, calls: int) -> CachePlan:
    from .config import H3InputError

    backend = config.cache_backend
    options = validate_cache_config(backend, config.cache_config)
    if sampling.quality == "lossless":
        backend, options = "none", {}
    elif sampling.quality == "high":
        if backend == "tea_cache":
            raise H3InputError(
                "TeaCache and the high Cache-DiT profile are mutually exclusive"
            )
        backend = "cache_dit"
        options = {
            **CACHE_DIT_DEFAULTS,
            "residual_diff_threshold": 0.04,
            "max_continuous_cached_steps": 1,
        }
    extra = sampling.extra_args
    hint, policy = (
        extra.get("force_refresh_step_hint"),
        extra.get("force_refresh_step_policy"),
    )
    if hint is not None or policy is not None:
        if backend != "cache_dit":
            raise H3InputError(
                "force-refresh arguments require an active Cache-DiT request"
            )
        if (
            isinstance(hint, bool)
            or not isinstance(hint, int)
            or not 1 <= hint <= calls
        ):
            raise H3InputError(f"force_refresh_step_hint must be between 1 and {calls}")
        policy = "once" if policy is None else policy
        if policy not in ("once", "repeat"):
            raise H3InputError("force_refresh_step_policy must be once or repeat")
        options.update(force_refresh_step_hint=hint, force_refresh_step_policy=policy)
    if backend == "tea_cache" and config.partition != "fl2va":
        raise H3InputError("official TeaCache is calibrated for FL2VA only")
    return CachePlan(backend, options, calls)


class TeaCacheState:
    """One request's official polynomial/residual state, including TP agreement."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.polynomial = np.poly1d(TEA_COEFFICIENTS)
        self.previous_input = None
        self.previous_residual = None
        self.accumulated = 0.0
        self.records: list[dict[str, Any]] = []

    def decide(self, modulated_input, group=None):
        compute = True
        distance = None
        if self.previous_input is not None:
            distance = float(
                (
                    (modulated_input - self.previous_input).abs().mean()
                    / (self.previous_input.abs().mean() + 1e-8)
                ).cpu()
            )
            self.accumulated += abs(float(self.polynomial(distance)))
            compute = self.accumulated >= self.threshold
            if not math.isfinite(self.accumulated):
                compute = True
        if group is not None and group.world_size > 1:
            decision = torch.tensor(
                int(compute), device=modulated_input.device, dtype=torch.int32
            )
            torch.distributed.all_reduce(
                decision, op=torch.distributed.ReduceOp.MAX, group=group.device_group
            )
            compute = bool(decision.item())
        if compute:
            self.accumulated = 0.0
        self.previous_input = modulated_input.detach()
        self.records.append({"computed": compute, "relative_l1": distance})
        return compute

    def execute(self, hidden, modulated_input, run_blocks, group=None):
        compute = self.decide(modulated_input, group)
        if not compute and self.previous_residual is not None:
            return hidden + self.previous_residual
        original = hidden.clone()
        output = run_blocks(hidden)
        self.previous_residual = (output - original).detach()
        return output

    def clear(self):
        self.previous_input = None
        self.previous_residual = None


@contextlib.contextmanager
def request_cache(model, plan: CachePlan):
    """Fresh cache per request; teardown runs after successful or failed sampling."""
    if getattr(model, "_h3_cache_active", False):
        raise RuntimeError(
            "H3 cache request cannot overlap another request on this model"
        )
    if plan.backend == "none":
        yield
        return
    model._h3_cache_active = True
    tea = None
    cache_dit = None
    original_class = None
    adapter = None
    try:
        if plan.backend == "tea_cache":
            tea = TeaCacheState(plan.options["rel_l1_thresh"])
            model._h3_tea_cache = tea
        else:
            if importlib.metadata.version("cache-dit") != "1.5.0":
                raise RuntimeError(
                    "H3 Cache-DiT requires the official cache-dit==1.5.0"
                )
            import cache_dit

            if plan.options["Fn_compute_blocks"] + plan.options[
                "Bn_compute_blocks"
            ] >= len(model.blocks):
                raise ValueError("Cache-DiT must leave at least one middle block")
            adapter = cache_dit.BlockAdapter(
                transformer=model,
                blocks=[model.blocks],
                forward_pattern=[cache_dit.ForwardPattern.Pattern_3],
                has_separate_cfg=False,
                check_forward_pattern=False,
            )
            # Avoid the package's process-wide wrapper-class cached marker.
            original_class = type(adapter.pipe)
            adapter.pipe.__class__ = type(
                "H3RequestCachePipe", (original_class,), {"_is_cached": False}
            )
            cache_dit.enable_cache(
                adapter,
                cache_config=cache_dit.DBCacheConfig(
                    num_inference_steps=plan.calls, **plan.options
                ),
                calibrator_config=None,
            )
            cache_dit.refresh_context(
                model, num_inference_steps=plan.calls, verbose=False
            )
        yield
    finally:
        try:
            if tea is not None:
                tea.clear()
                del model._h3_tea_cache
            if cache_dit is not None and getattr(model, "_is_cached", False):
                cache_dit.disable_cache(model)
        finally:
            if adapter is not None and original_class is not None:
                adapter.pipe.__class__ = original_class
            model._h3_cache_active = False
