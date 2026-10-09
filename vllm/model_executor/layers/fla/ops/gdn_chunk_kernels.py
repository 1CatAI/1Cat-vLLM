# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-owned autotuners; JIT functions and arithmetic are unchanged."""

from dataclasses import dataclass
from itertools import product

import torch

from vllm.config.gdn_schedule import GdnScheduleConfig
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

    sm70 = torch.cuda.get_device_capability() == (7, 0)
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
    if not hasattr(vllm_config, "_gdn_chunk_kernels"):
        vllm_config._gdn_chunk_kernels = create_chunk_kernels(schedule)
    return vllm_config._gdn_chunk_kernels
