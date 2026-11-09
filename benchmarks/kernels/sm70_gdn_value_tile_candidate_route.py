# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit TP4 q8 value-tile experiment for the existing packed GDN kernel.

The query/key reduction shape, one-warp schedule, recurrence arithmetic and
FP32 state remain unchanged. This module does not install a default route.
"""

import functools
import importlib

import torch


def install_gdn_value_tile_candidate() -> None:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    module = importlib.import_module(
        "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating"
    )
    original = module._select_fused_sigmoid_launch
    original_forward = QwenGatedDeltaNetAttention._forward_dflash2_packed_gdn_verify
    logged = False
    eligible = False

    @functools.wraps(original_forward)
    def forward(self, **kwargs):
        nonlocal eligible
        previous = eligible
        eligible = (
            self.tp_size == 4
            and kwargs["num_spec_decodes"] == 1
            and kwargs["mixed_qkv"].shape == (8, 2560)
            and kwargs["mixed_qkv"].dtype == torch.float16
            and kwargs["ssm_state"].dtype == torch.float32
        )
        try:
            return original_forward(self, **kwargs)
        finally:
            eligible = previous

    def select(V, N, HV, T, device, *, match_recurrent_schedule):
        nonlocal logged
        baseline = original(
            V, N, HV, T, device, match_recurrent_schedule=match_recurrent_schedule
        )
        if (
            eligible
            and (V, N, HV, T) == (128, 1, 12, 8)
            and match_recurrent_schedule
            and torch.cuda.is_current_stream_capturing()
            and torch.cuda.get_device_capability(device) == (7, 0)
        ):
            assert baseline[:2] == (8, 1), baseline
            if not logged:
                logged = True
                print(
                    f"GDN_VALUE_TILE_CAPTURE rank={torch.distributed.get_rank()} "
                    "TP4/B1/q8 HV=12 V=128 BV=2 warps=1 original_BV=8",
                    flush=True,
                )
            return 2, 1, baseline[2]
        return baseline

    module._select_fused_sigmoid_launch = select
    QwenGatedDeltaNetAttention._forward_dflash2_packed_gdn_verify = forward
