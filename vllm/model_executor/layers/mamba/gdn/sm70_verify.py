# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Legacy import adapter; active engines bind the explicit FLA provider."""

import torch

from vllm.model_executor.layers.fla.ops.gdn_stages import GdnHeadContract
from vllm.model_executor.layers.fla.ops.sm70.gdn_verify import verify_bound


def native_verifier():
    return verify if hasattr(torch.ops._C, "sm70_gdn_verify_out") else None


def verify(layer, *args, **kwargs):
    contract = GdnHeadContract(
        layer.num_k_heads,
        layer.num_v_heads,
        layer.head_k_dim,
        layer.head_v_dim,
        layer.tp_size,
    )
    return verify_bound(contract, layer.A_log, layer.dt_bias, *args, **kwargs)
