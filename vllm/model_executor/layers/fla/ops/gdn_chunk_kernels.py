# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-owned autotuners; JIT functions and arithmetic are unchanged."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import torch

from vllm.config.gdn_schedule import GdnScheduleConfig
from vllm.platforms import current_platform
from vllm.runtime_resources import current_runtime_resources, runtime_resources_for
from vllm.triton_utils import triton


@dataclass
class GdnChunkKernels:
    kkt: object
    delta_h: object
    output: object


def _configs(dimensions, warps, stages):
    names = tuple(dimensions)
    return [
        triton.Config(dict(zip(names, values)), num_warps=w, num_stages=s)
        for values in product(*dimensions.values())
        for w in warps
        for s in stages
    ]


def _retune(kernel, configs, key, *, use_cuda_graph=False):
    # The old wrapper is heuristics -> autotune -> JIT. Reuse only immutable
    # heuristics/JIT code; winner and benchmark caches belong to this engine.
    tuned = triton.autotune(
        configs=configs,
        key=key,
        use_cuda_graph=use_cuda_graph,
    )(kernel.fn.fn)
    return triton.heuristics(kernel.values)(tuned)


def create_chunk_kernels(schedule: GdnScheduleConfig) -> GdnChunkKernels | None:
    if not torch.cuda.is_available():
        return None
    from .chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    from .chunk_o import BKV_LIST, NUM_WARPS, chunk_fwd_kernel_o
    from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel
    from .utils import use_cuda_graph

    sm70 = current_platform.is_device_capability((7, 0))
    kkt = (
        _configs({"BK": schedule.kkt_bk}, schedule.kkt_warps, schedule.kkt_stages)
        if sm70 and schedule.kkt_enabled
        else _configs({"BK": [32, 64, 128]}, [2, 4, 8], [2, 3, 4])
    )
    delta = (
        _configs(
            {"BV": schedule.delta_h_bv}, schedule.delta_h_warps, schedule.delta_h_stages
        )
        if sm70 and schedule.delta_h_enabled
        else [
            triton.Config({"BV": bv}, num_warps=w, num_stages=s)
            for w in [2, 4]
            for s in [2, 3, 4]
            for bv in [32, 64]
        ]
    )
    output = (
        _configs(
            {"BK": schedule.chunk_o_bk, "BV": schedule.chunk_o_bv},
            schedule.chunk_o_warps,
            schedule.chunk_o_stages,
        )
        if sm70 and schedule.chunk_o_enabled
        else _configs({"BK": BKV_LIST, "BV": BKV_LIST}, NUM_WARPS, [2, 3, 4])
    )
    return GdnChunkKernels(
        _retune(chunk_scaled_dot_kkt_fwd_kernel, kkt, ["H", "K", "BT", "IS_VARLEN"]),
        _retune(
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
            delta,
            ["H", "K", "V", "BT"],
            use_cuda_graph=use_cuda_graph,
        ),
        _retune(chunk_fwd_kernel_o, output, ["H", "K", "V", "BT"]),
    )


def bind_chunk_kernels(vllm_config, schedule):
    resources = runtime_resources_for(vllm_config)
    if "gdn_chunk_kernels" not in resources:
        schedule.resolve()
        resources["gdn_chunk_kernels"] = create_chunk_kernels(schedule)
    return resources["gdn_chunk_kernels"]


@lru_cache(maxsize=1)
def _legacy_chunk_kernels():
    # No-config operator clients retain a process-owned compatibility tuner.
    from vllm.config.gdn_schedule import resolve_schedule

    return create_chunk_kernels(resolve_schedule())


def resolve_chunk_kernels(kernels=None):
    if kernels is not None:
        return kernels
    resources = current_runtime_resources()
    if resources is not None:
        return resources["gdn_chunk_kernels"]
    return _legacy_chunk_kernels()


@dataclass
class KdaChunkKernels:
    recompute: object
    delta_h: object
    output: object


def create_kda_kernels(schedule, delta_h):
    if delta_h is None:
        return None
    from .kda import (
        _chunk_gla_o_configs,
        _recompute_w_u_configs,
        chunk_gla_fwd_kernel_o,
        recompute_w_u_fwd_kernel,
    )

    enabled = (
        current_platform.is_device_capability((7, 0)) and schedule.kda_prefill_enabled
    )
    recompute = _configs({}, [4, 8], [2]) if enabled else _recompute_w_u_configs
    output = (
        [
            triton.Config({"BK": bk, "BV": bv}, num_warps=w, num_stages=2)
            for bk in [32, 64]
            for bv in [64, 128]
            for w in [4, 8]
            if bv == 64 or w == 8
        ]
        if enabled
        else _chunk_gla_o_configs
    )
    return KdaChunkKernels(
        _retune(
            recompute_w_u_fwd_kernel,
            recompute,
            ["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
        ),
        delta_h,
        _retune(chunk_gla_fwd_kernel_o, output, ["BT"]),
    )


def bind_kda_kernels(vllm_config):
    resources = runtime_resources_for(vllm_config)
    if "kda_chunk_kernels" not in resources:
        schedule = vllm_config.kernel_config.gdn.schedule
        chunk = bind_chunk_kernels(vllm_config, schedule)
        resources["kda_chunk_kernels"] = create_kda_kernels(
            schedule, chunk.delta_h if chunk is not None else None
        )
    return resources["kda_chunk_kernels"]


@lru_cache(maxsize=1)
def _legacy_kda_kernels():
    from vllm.config.gdn_schedule import resolve_schedule

    chunk = _legacy_chunk_kernels()
    return create_kda_kernels(resolve_schedule(), chunk.delta_h if chunk else None)


def resolve_kda_kernels(kernels=None):
    if kernels is not None:
        return kernels
    resources = current_runtime_resources()
    if resources is not None:
        return resources["kda_chunk_kernels"]
    return _legacy_kda_kernels()
