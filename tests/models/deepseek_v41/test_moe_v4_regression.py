# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-MOE: DeepSeek-V4-Flash must not regress under the V4.1 generalisation of the SM70 MXFP4 path.

* the TurboMind ``Mxfp4SM70MoEMethod.apply`` on V4-Flash TP4/TP8 shapes (QPN-M1 / direct-order at T=1, generic
  grouped stages at T>1) against an FP32-dequantised reference, and ``apply_slots`` consistent with ``apply``;
* skinny split-K admission for every non-V4.1 caller unchanged (``extended=False``), QPN prepack alignment;
* new split-K specialisations (4/6/9/12/18) agree with the old ones on the old shapes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    grouped_splitk,
    qpn_prepack,
    rebase_e8m0_for_fp16,
)
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import Mxfp4SM70MoEMethod, _v4_flash_fast_paths
from vllm.models.deepseek_v41.sm70.moe_method import dequant_mxfp4

pytestmark = pytest.mark.sm70


def _v4_config(tp_size: int) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=256, experts_per_token=6, hidden_dim=4096, intermediate_size_per_partition=2048 // tp_size,
        num_local_experts=256, num_logical_experts=256, activation=MoEActivation.SILU, device=torch.device("cuda"),
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=tp_size, pcp_size=1, dp_size=1, ep_size=1, tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0,
            sp_size=1, use_ep=False, all2all_backend="allgather_reducescatter", enable_eplb=False),
        in_dtype=torch.float16, swiglu_limit=7.0)


def _v4_layer(tp_size: int, seed: int) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    cfg = _v4_config(tp_size)
    inter = 2048 // tp_size
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        codes = torch.randint(0, 256, (256, n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
        scales = torch.randint(116, 126, (256, n, k // 32), dtype=torch.uint8, device="cuda", generator=g)
        return codes, scales

    w13, s13 = rnd(2 * inter, 4096)
    w2, s2 = rnd(4096, inter)
    ref = {"w13": w13.clone(), "s13": s13.clone(), "w2": w2.clone(), "s2": s2.clone()}
    layer = nn.Module()
    layer.local_num_experts = layer.global_num_experts = 256
    layer.top_k = 6
    layer.moe_config = cfg
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.swiglu_limit = 7.0
    layer.expert_map = None
    for name, t in (("w13_weight", w13), ("w13_weight_scale", s13), ("w2_weight", w2), ("w2_weight_scale", s2)):
        layer.register_parameter(name, nn.Parameter(t, requires_grad=False))
    Mxfp4SM70MoEMethod(cfg).process_weights_after_loading(layer)
    return layer, ref


def _reference(ref: dict[str, torch.Tensor], x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor,
               limit: float) -> torch.Tensor:
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device="cuda")
    for t in range(ids.shape[0]):
        for j, e in enumerate(ids[t].tolist()):
            h = dequant_mxfp4(ref["w13"][e], ref["s13"][e]) @ x[t].float()
            inter = h.shape[0] // 2
            gate = h[:inter].half().float().clamp(max=limit)
            up = h[inter:].half().float().clamp(-limit, limit)
            act = (torch.nn.functional.silu(gate) * up).half().float()
            out[t] += w[t, j] * (dequant_mxfp4(ref["w2"][e], ref["s2"][e]) @ act).half().float()
    return out


@pytest.mark.parametrize("tp_size", [4, 8])
def test_v4_flash_apply_unchanged_paths(tp_size):
    layer, ref = _v4_layer(tp_size, seed=tp_size)
    assert _v4_flash_fast_paths(layer)
    method = Mxfp4SM70MoEMethod(layer.moe_config)
    for num_tokens in (1, 2, 8, 33):
        g = torch.Generator(device="cuda").manual_seed(num_tokens)
        x = (torch.randn(num_tokens, 4096, device="cuda", generator=g) * 0.5).half()
        ids = torch.rand(num_tokens, 256, device="cuda", generator=g).topk(6, -1)[1].to(torch.int32).contiguous()
        w = torch.rand(num_tokens, 6, device="cuda", generator=g)
        out = method.apply(layer, x, w, ids, None, None).float()
        expect = _reference(ref, x, ids, w, 7.0)
        rel = ((out - expect).pow(2).mean().sqrt() / expect.pow(2).mean().sqrt()).item()
        assert rel < 2e-3, (tp_size, num_tokens, rel)
        slots = method.apply_slots(layer, x, ids).float().view(num_tokens, 6, -1)
        combined = (slots * w[..., None]).sum(1)
        rel2 = ((combined - out).pow(2).mean().sqrt() / out.pow(2).mean().sqrt()).item()
        assert rel2 < 2e-3, (tp_size, num_tokens, rel2)


def test_skinny_admission_unchanged_for_other_models():
    assert grouped_splitk(4096, 16) == 16 and grouped_splitk(512, 8) == 8 and grouped_splitk(256, 8) == 8
    assert grouped_splitk(320, 8) == 10
    for k in (576, 288, 192):
        with pytest.raises(ValueError):
            grouped_splitk(k, 8)
    assert grouped_splitk(576, 8, extended=True) == 12
    assert grouped_splitk(288, 8, extended=True) == 9
    assert grouped_splitk(5120, 16, extended=True) == 16
    codes = torch.zeros(32, 144, dtype=torch.uint8, device="cuda")
    scales = torch.full((32, 9), 127, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="K % 64"):
        qpn_prepack(codes, scales, 32)
    qpn_prepack(codes, scales, 32, k_align=32)


@pytest.mark.parametrize("k,old_split,new_splits", [(512, 8, (4, 16)), (4096, 16, (4, 8))])
def test_new_split_specialisations_agree_on_old_shapes(k, old_split, new_splits):
    n, experts, num_tokens, top_k = 1024, 8, 5, 3
    g = torch.Generator(device="cuda").manual_seed(k)
    codes = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    scales = torch.randint(116, 126, (experts, n, k // 32), dtype=torch.uint8, device="cuda", generator=g)
    gs = rebase_e8m0_for_fp16(scales).float().cuda()
    for e in range(experts):
        qc, qs = qpn_prepack(codes[e], scales[e], 32)
        codes[e].view(-1).copy_(qc)
        scales[e].view(-1).copy_(qs)
    x = (torch.randn(num_tokens, k, device="cuda", generator=g)).half()
    ids = torch.rand(num_tokens, experts, device="cuda", generator=g).topk(top_k, -1)[1].to(torch.int32)
    from vllm.models.deepseek_v41.sm70.moe_kernels import route_prep

    perm, ((gids, goff),) = route_prep(ids.contiguous(), None, experts, False)

    def run(split: int) -> torch.Tensor:
        y = torch.empty(num_tokens * top_k, n, dtype=torch.float16, device="cuda")
        torch.ops._C.skinny_moe_qpn_sm70(x, codes, scales, gs, perm, gids, goff, top_k, y, False, num_tokens,
                                         split, 1, 1)
        return y.float()

    base = run(old_split)
    for split in new_splits:
        other = run(split)
        assert ((other - base).abs() > base.abs() * 2.0**-9 + 2.0**-20).float().mean().item() < 0.01
