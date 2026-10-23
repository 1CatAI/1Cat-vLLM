# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Triton skinny MoE projection against the hand kernel and an FP32 reference."""

import pytest
import torch

from tests.kernels.utils import requires_tensor_core_triton
from vllm.model_executor.layers.fused_moe.experts.skinny_moe_triton import (
    skinny_moe_e8m0_m8n8k4_triton,
)
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    QpnFragmentOrder,
    grouped_splitk,
    qpn_prepack,
    rebase_e8m0_for_fp16,
)

EXPERTS, TOP_K = 16, 6
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
        reason="the M8N8K4 fragment order is Volta's",
    ),
    requires_tensor_core_triton,
]


def _expert_weights(n: int, k: int, gen: torch.Generator):
    codes = torch.randint(
        0, 256, (EXPERTS, n, k // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    scales = torch.randint(
        118, 128, (EXPERTS, n, k // 32), dtype=torch.uint8, device="cuda", generator=gen
    )
    nibbles = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(EXPERTS, n, k)
    dense = torch.tensor(E2M1, device="cuda")[nibbles.long()] * torch.exp2(
        scales.float() - 127.0
    ).repeat_interleave(32, dim=2)
    global_scales = rebase_e8m0_for_fp16(scales).cuda()
    for e in range(EXPERTS):
        packed_codes, packed_scales = qpn_prepack(
            codes[e], scales[e], 32, QpnFragmentOrder.M8N8K4
        )
        codes[e].view(-1).copy_(packed_codes)
        scales[e].view(-1).copy_(packed_scales)
    return codes, scales, global_scales, dense


def _routing(tokens: int, gen: torch.Generator, skew: float = 0.0):
    # Expert popularity ~ 1 / rank^skew: skewed routing gives groups of every
    # size, as real routers do.
    popularity = 1.0 / torch.arange(1, EXPERTS + 1, device="cuda").float() ** skew
    ids = torch.multinomial(
        popularity.expand(tokens, EXPERTS), TOP_K, replacement=False, generator=gen
    )
    flat = ids.reshape(-1).to(torch.int64)
    slots = flat.numel()
    perm = torch.argsort(flat, stable=True).to(torch.int32)
    sorted_experts = flat[perm.long()]
    new_group = torch.ones(slots, dtype=torch.bool, device="cuda")
    new_group[1:] = sorted_experts[1:] != sorted_experts[:-1]
    group_index = torch.cumsum(new_group, 0) - 1
    goff = torch.full((slots + 1,), slots, dtype=torch.int64, device="cuda")
    goff.scatter_reduce_(
        0,
        group_index,
        torch.arange(slots, dtype=torch.int64, device="cuda"),
        reduce="amin",
    )
    gids = torch.zeros(slots, dtype=torch.int64, device="cuda")
    gids.scatter_(0, group_index, sorted_experts)
    return ids, perm, gids.to(torch.int32), goff.to(torch.int32)


# W13 reads token-major x, W2 slot-major x; 8 tokens leave single rows per
# expert, 200 several row blocks per group.
@pytest.mark.parametrize(
    ("n", "k", "slot_major"), [(512, 1024, False), (512, 512, True)]
)
@pytest.mark.parametrize(("tokens", "skew"), [(8, 0.0), (200, 0.0), (120, 1.5)])
def test_triton_projection_matches_hand_kernel(n, k, slot_major, tokens, skew):
    gen = torch.Generator(device="cuda").manual_seed(tokens + n + k)
    codes, scales, global_scales, dense = _expert_weights(n, k, gen)
    ids, perm, gids, goff = _routing(tokens, gen, skew)
    if skew:
        # The case must reach all three row tiles (<= 16, 17-32, > 32 rows).
        rows = torch.bincount(ids.reshape(-1), minlength=EXPERTS)
        assert (rows <= 16).any() and ((rows > 16) & (rows <= 32)).any()
        assert (rows > 32).any()
    rows = tokens * TOP_K if slot_major else tokens
    x = (torch.randn(rows, k, device="cuda", generator=gen) * 0.5).half()
    flat = ids.reshape(-1)
    x_of_slot = x.float() if slot_major else x.float().repeat_interleave(TOP_K, 0)
    reference = torch.empty(flat.numel(), n, device="cuda")
    for e in range(EXPERTS):
        selected = (flat == e).nonzero().squeeze(1)
        reference[selected] = x_of_slot[selected] @ dense[e].T

    hand = torch.empty(flat.numel(), n, dtype=torch.float16, device="cuda")
    torch.ops._C.skinny_moe_qpn_sm70(
        x,
        codes,
        scales,
        global_scales,
        perm,
        gids,
        goff,
        TOP_K,
        hand,
        slot_major,
        tokens,
        grouped_splitk(k, QpnFragmentOrder.M8N8K4),
        1,
        QpnFragmentOrder.M8N8K4,
    )
    triton_out = torch.full_like(hand, float("nan"))
    skinny_moe_e8m0_m8n8k4_triton(
        x, codes, scales, global_scales, perm, gids, goff, TOP_K, triton_out, slot_major
    )

    scale = reference.abs().max()
    hand_error = (hand.float() - reference).abs().max() / scale
    triton_error = (triton_out.float() - reference).abs().max() / scale
    assert torch.isfinite(triton_out).all()
    # Both round the same products to fp16 once; only the accumulation order
    # differs.
    assert triton_error <= 1.5 * hand_error
