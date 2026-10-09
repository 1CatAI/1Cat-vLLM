# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 MXFP4 MoE beyond the DeepSeek-V4-Flash geometry.

* ``validate_mxfp4_sm70_moe_contract`` admits the V4-Flash and V4.1-Flash
  expert geometries and nothing else (CPU only).
* ``Mxfp4SM70MoEMethod.apply`` on V4-Flash TP4/TP8 still takes its fused
  decode paths and matches an FP32 dequantised reference; ``apply_slots``
  agrees with ``apply``.
* The generic grouped stages serve the V4.1 geometries (hidden 5120,
  intermediate 2304 at TP4; 384 experts top-6 and 128 experts top-3).
* The skinny split-K admission is unchanged for existing callers, and the new
  4/6/9/12/18 specialisations agree with the old ones and with the reference.
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
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import (
    Mxfp4SM70MoEMethod,
    _v4_flash_fast_paths,
    validate_mxfp4_sm70_moe_contract,
)

_E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

requires_sm70 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires NVIDIA V100/SM70",
)


def _dequant_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes [N, K/2] (low nibble = even k), E8M0 scales [N, K/32] -> FP32 [N, K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=codes.device)
    values = torch.stack(
        [lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], dim=-1
    ).flatten(-2)
    return values * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)


def _route_tables(topk_ids: torch.Tensor):
    """Routing tables: perm (slots sorted by expert), gids, goff (group offsets)."""
    key = topk_ids.reshape(-1).to(torch.int64)
    num_slots = key.numel()
    perm = torch.sort(key, stable=True)[1]
    skey = key[perm]
    first = torch.ones(num_slots, dtype=torch.bool, device=key.device)
    first[1:] = skey[1:] != skey[:-1]
    group = torch.cumsum(first.to(torch.int64), 0) - 1
    slots = torch.arange(num_slots, dtype=torch.int64, device=key.device)
    goff = torch.full((num_slots + 1,), num_slots, dtype=torch.int64, device=key.device)
    goff.scatter_reduce_(0, group, slots, reduce="amin")
    gids = torch.zeros(num_slots, dtype=torch.int64, device=key.device)
    gids.scatter_(0, group, skey)
    return perm.to(torch.int32), gids.to(torch.int32), goff.to(torch.int32)


# ---- contract (CPU) ----------------------------------------------------------


@pytest.mark.parametrize(
    "experts,top_k,hidden,inter_full,tp",
    [
        (256, 6, 4096, 2048, 4),  # DeepSeek-V4-Flash
        (256, 6, 4096, 2048, 8),
        (384, 6, 5120, 2304, 4),  # DeepSeek-V4.1-Flash backbone
        (128, 3, 5120, 2304, 4),  # DeepSeek-V4.1-Flash DSpark draft layers
    ],
)
def test_contract_admits_qualified_geometries(experts, top_k, hidden, inter_full, tp):
    validate_mxfp4_sm70_moe_contract(
        global_num_experts=experts,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size_per_partition=inter_full // tp,
        tp_size=tp,
    )


@pytest.mark.parametrize(
    "experts,top_k,hidden,inter_full,tp,match",
    [
        (128, 6, 4096, 2048, 4, "256 global experts"),
        (256, 8, 4096, 2048, 4, "top-k=6"),
        (256, 6, 5120, 2304, 4, "128/384 global experts"),
        (384, 3, 5120, 2304, 4, "top-k=6 for 384"),
        (128, 6, 5120, 2304, 4, "top-k=3 for 128"),
        (384, 6, 5120, 2048, 4, "intermediate size 2304"),
        (32, 4, 2880, 2880, 1, "hidden size"),
    ],
)
def test_contract_refuses_other_geometries(
    experts, top_k, hidden, inter_full, tp, match
):
    with pytest.raises(NotImplementedError, match=match):
        validate_mxfp4_sm70_moe_contract(
            global_num_experts=experts,
            top_k=top_k,
            hidden_size=hidden,
            intermediate_size_per_partition=inter_full // tp,
            tp_size=tp,
        )


# ---- TurboMind path (GPU) ----------------------------------------------------


def _config(
    experts: int, top_k: int, hidden: int, inter_full: int, tp_size: int
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=experts,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size_per_partition=inter_full // tp_size,
        num_local_experts=experts,
        num_logical_experts=experts,
        activation=MoEActivation.SILU,
        device=torch.device("cuda"),
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=tp_size,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        ),
        in_dtype=torch.float16,
        swiglu_limit=7.0,
    )


def _layer(
    experts: int, top_k: int, hidden: int, inter_full: int, tp_size: int, seed: int
) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    cfg = _config(experts, top_k, hidden, inter_full, tp_size)
    inter = inter_full // tp_size
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        codes = torch.randint(
            0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda", generator=g
        )
        scales = torch.randint(
            116,
            126,
            (experts, n, k // 32),
            dtype=torch.uint8,
            device="cuda",
            generator=g,
        )
        return codes, scales

    w13, s13 = rnd(2 * inter, hidden)
    w2, s2 = rnd(hidden, inter)
    ref = {"w13": w13.clone(), "s13": s13.clone(), "w2": w2.clone(), "s2": s2.clone()}
    layer = nn.Module()
    layer.local_num_experts = layer.global_num_experts = experts
    layer.top_k = top_k
    layer.moe_config = cfg
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.swiglu_limit = 7.0
    layer.expert_map = None
    for name, t in (
        ("w13_weight", w13),
        ("w13_weight_scale", s13),
        ("w2_weight", w2),
        ("w2_weight_scale", s2),
    ):
        layer.register_parameter(name, nn.Parameter(t, requires_grad=False))
    Mxfp4SM70MoEMethod(cfg).process_weights_after_loading(layer)
    return layer, ref


def _reference(
    ref: dict[str, torch.Tensor],
    x: torch.Tensor,
    ids: torch.Tensor,
    w: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device="cuda")
    for t in range(ids.shape[0]):
        for j, e in enumerate(ids[t].tolist()):
            h = _dequant_mxfp4(ref["w13"][e], ref["s13"][e]) @ x[t].float()
            inter = h.shape[0] // 2
            gate = h[:inter].half().float().clamp(max=limit)
            up = h[inter:].half().float().clamp(-limit, limit)
            act = (torch.nn.functional.silu(gate) * up).half().float()
            out[t] += (
                w[t, j]
                * (_dequant_mxfp4(ref["w2"][e], ref["s2"][e]) @ act).half().float()
            )
    return out


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def _check_apply(
    layer, ref, experts: int, top_k: int, hidden: int, token_counts: tuple[int, ...]
) -> None:
    method = Mxfp4SM70MoEMethod(layer.moe_config)
    for num_tokens in token_counts:
        g = torch.Generator(device="cuda").manual_seed(num_tokens)
        x = (torch.randn(num_tokens, hidden, device="cuda", generator=g) * 0.5).half()
        ids = (
            torch.rand(num_tokens, experts, device="cuda", generator=g)
            .topk(top_k, -1)[1]
            .to(torch.int32)
            .contiguous()
        )
        w = torch.rand(num_tokens, top_k, device="cuda", generator=g)
        out = method.apply(layer, x, w, ids, None, None).float()
        expect = _reference(ref, x, ids, w, 7.0)
        assert _rel(out, expect) < 2e-3, (num_tokens, _rel(out, expect))
        slots = method.apply_slots(layer, x, ids).float().view(num_tokens, top_k, -1)
        combined = (slots * w[..., None]).sum(1)
        assert _rel(combined, out) < 2e-3, (num_tokens, _rel(combined, out))


@requires_sm70
@pytest.mark.parametrize("tp_size", [4, 8])
def test_v4_flash_keeps_fast_paths_and_matches_reference(tp_size):
    layer, ref = _layer(256, 6, 4096, 2048, tp_size, seed=tp_size)
    assert _v4_flash_fast_paths(layer)
    _check_apply(layer, ref, 256, 6, 4096, (1, 2, 8, 33))


@requires_sm70
@pytest.mark.parametrize("experts,top_k", [(384, 6), (128, 3)])
def test_v41_geometry_runs_generic_stages(experts, top_k):
    layer, ref = _layer(experts, top_k, 5120, 2304, 4, seed=experts)
    assert not _v4_flash_fast_paths(layer)
    _check_apply(layer, ref, experts, top_k, 5120, (1, 3, 17))


# ---- skinny grouped kernel (GPU) --------------------------------------------


def test_skinny_admission_unchanged_for_existing_callers():
    assert grouped_splitk(4096, 16) == 16
    assert grouped_splitk(512, 8) == 8
    assert grouped_splitk(256, 8) == 8
    assert grouped_splitk(320, 8) == 10
    for k in (576, 288, 192):
        with pytest.raises(ValueError):
            grouped_splitk(k, 8)
    assert grouped_splitk(576, 8, extended=True) == 12
    assert grouped_splitk(288, 8, extended=True) == 9
    assert grouped_splitk(5120, 16, extended=True) == 16


@requires_sm70
def test_qpn_prepack_k_align():
    codes = torch.zeros(32, 144, dtype=torch.uint8, device="cuda")
    scales = torch.full((32, 9), 127, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="K % 64"):
        qpn_prepack(codes, scales, 32)
    qpn_prepack(codes, scales, 32, k_align=32)
    with pytest.raises(ValueError, match="k_align"):
        qpn_prepack(codes, scales, 32, k_align=16)


def _grouped_run(k: int, splits: tuple[int, ...], k_align: int):
    n, experts, num_tokens, top_k = 1024, 8, 5, 3
    g = torch.Generator(device="cuda").manual_seed(k)
    codes = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    scales = torch.randint(
        116, 126, (experts, n, k // 32), dtype=torch.uint8, device="cuda", generator=g
    )
    weights = [_dequant_mxfp4(codes[e], scales[e]) for e in range(experts)]
    gs = rebase_e8m0_for_fp16(scales).float().cuda()
    for e in range(experts):
        qc, qs = qpn_prepack(codes[e], scales[e], 32, k_align=k_align)
        codes[e].view(-1).copy_(qc)
        scales[e].view(-1).copy_(qs)
    x = torch.randn(num_tokens, k, device="cuda", generator=g).half()
    ids = (
        torch.rand(num_tokens, experts, device="cuda", generator=g)
        .topk(top_k, -1)[1]
        .to(torch.int32)
        .contiguous()
    )
    perm, gids, goff = _route_tables(ids)
    expect = torch.stack(
        [
            weights[e] @ x[s // top_k].float()
            for s, e in enumerate(ids.reshape(-1).tolist())
        ]
    )
    outs = {}
    for split in splits:
        y = torch.empty(num_tokens * top_k, n, dtype=torch.float16, device="cuda")
        torch.ops._C.skinny_moe_qpn_sm70(
            x,
            codes,
            scales,
            gs,
            perm,
            gids,
            goff,
            top_k,
            y,
            False,
            num_tokens,
            split,
            1,
            1,
        )
        outs[split] = y.float()
    return outs, expect


@requires_sm70
@pytest.mark.parametrize(
    "k,old_split,new_splits", [(512, 8, (4, 16)), (4096, 16, (4, 8))]
)
def test_new_splits_agree_with_old_on_old_shapes(k, old_split, new_splits):
    outs, expect = _grouped_run(k, (old_split, *new_splits), 64)
    base = outs[old_split]
    assert _rel(base, expect) < 1e-3
    for split in new_splits:
        other = outs[split]
        mismatch = (other - base).abs() > base.abs() * 2.0**-9 + 2.0**-20
        assert mismatch.float().mean().item() < 0.01, split


@requires_sm70
@pytest.mark.parametrize("k,splits", [(576, (4, 6, 9, 12, 18)), (288, (6, 9, 18))])
def test_new_splits_serve_k576_and_k288(k, splits):
    outs, expect = _grouped_run(k, splits, 32)
    for split, y in outs.items():
        assert torch.isfinite(y).all(), split
        assert _rel(y, expect) < 1e-3, (split, _rel(y, expect))
