# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-MOE: DeepseekV41Mxfp4MoEMethod on synthetic checkpoint-format experts at V4.1 shapes.

* contract generalisation (V4.1 shapes admitted, V4-Flash shapes unchanged);
* checkpoint loading with TP slicing (rank 0..3 of TP4, TP1) and I8 / E8M0 byte views;
* skinny and TurboMind backends vs the FP32-dequantised torch oracle (same FP16 rounding points: near-bitwise);
* expert spill (UVA pinned host memory) bitwise equal to all-resident;
* CUDA-graph capture/replay B1-B8 bitwise equal to eager.
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
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import validate_mxfp4_sm70_moe_contract
from vllm.models.deepseek_v41.sm70.moe_method import (
    DeepseekV41Mxfp4MoEMethod,
    ExpertSpillPlan,
    dequant_mxfp4,
    make_spill_plan,
)

pytestmark = pytest.mark.sm70

HIDDEN, INTER = 5120, 2304


def moe_config(num_experts: int, top_k: int, tp_size: int, tp_rank: int) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=num_experts, experts_per_token=top_k, hidden_dim=HIDDEN,
        intermediate_size_per_partition=INTER // tp_size, num_local_experts=num_experts,
        num_logical_experts=num_experts, activation=MoEActivation.SILU, device=torch.device("cuda"),
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=tp_size, pcp_size=1, dp_size=1, ep_size=1, tp_rank=tp_rank, pcp_rank=0, dp_rank=0, ep_rank=0,
            sp_size=1, use_ep=False, all2all_backend="allgather_reducescatter", enable_eplb=False),
        in_dtype=torch.float16, swiglu_limit=10.0)


class SyntheticCheckpoint:
    """Deterministic checkpoint-format experts: w1/w3 I8 [2304, 2560] + E8M0 [2304, 160], w2 I8 [5120, 1152] +
    E8M0 [5120, 72]; scale exponents in the measured V4.1 range (-12..-2)."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def expert(self, e: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        g = torch.Generator(device="cuda").manual_seed(self.seed * 1000 + e)
        out = {}
        for shard, (n, k) in (("w1", (INTER, HIDDEN)), ("w3", (INTER, HIDDEN)), ("w2", (HIDDEN, INTER))):
            codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
            scales = torch.randint(127 - 12, 127 - 1, (n, k // 32), dtype=torch.uint8, device="cuda", generator=g)
            out[shard] = (codes.view(torch.int8), scales.view(torch.float8_e8m0fnu))
        return out


def make_layer(num_experts: int, top_k: int, tp_size: int, tp_rank: int, backend: str,
               spill: ExpertSpillPlan | None, ckpt: SyntheticCheckpoint) -> tuple[nn.Module, DeepseekV41Mxfp4MoEMethod]:
    cfg = moe_config(num_experts, top_k, tp_size, tp_rank)
    method = DeepseekV41Mxfp4MoEMethod(cfg, backend=backend, spill=spill)
    layer = nn.Module()
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.swiglu_limit = 10.0
    layer.local_num_experts = layer.global_num_experts = num_experts
    layer.top_k = top_k
    layer.moe_config = cfg
    layer.expert_map = None
    with torch.device("cuda"):
        method.create_weights(layer, num_experts, HIDDEN, INTER // tp_size, torch.float16,
                              weight_loader=None, global_num_experts=num_experts)
    for e in range(num_experts):
        for shard, (codes, scales) in ckpt.expert(e).items():
            base = "w13" if shard in ("w1", "w3") else "w2"
            for suffix, tensor in (("weight", codes), ("weight_scale", scales)):
                param = getattr(layer, f"{base}_{suffix}")
                assert param.weight_loader(param, tensor, f"experts.{base}_{suffix}", shard_id=shard,
                                           expert_id=e, return_success=True)
    method.process_weights_after_loading(layer)
    return layer, method


def reference_slots(ckpt: SyntheticCheckpoint, x: torch.Tensor, ids: torch.Tensor, tp_size: int,
                    tp_rank: int) -> torch.Tensor:
    """Independent FP32-dequantised reference straight from the checkpoint tensors (TP slice by hand)."""
    inter = INTER // tp_size
    rows = slice(inter * tp_rank, inter * (tp_rank + 1))
    num_tokens, top_k = ids.shape
    y = torch.empty(num_tokens * top_k, HIDDEN, dtype=torch.float16, device="cuda")
    for slot, e in enumerate(ids.flatten().tolist()):
        ex = ckpt.expert(e)
        w1 = dequant_mxfp4(ex["w1"][0].view(torch.uint8)[rows], ex["w1"][1].view(torch.uint8)[rows])
        w3 = dequant_mxfp4(ex["w3"][0].view(torch.uint8)[rows], ex["w3"][1].view(torch.uint8)[rows])
        cols = slice(inter // 2 * tp_rank, inter // 2 * (tp_rank + 1))
        scols = slice(inter // 32 * tp_rank, inter // 32 * (tp_rank + 1))
        w2 = dequant_mxfp4(ex["w2"][0].view(torch.uint8)[:, cols], ex["w2"][1].view(torch.uint8)[:, scols])
        xr = x[slot // top_k].float()
        gate = (w1 @ xr).half().float().clamp(max=10.0)
        up = (w3 @ xr).half().float().clamp(-10.0, 10.0)
        act = (gate / (1.0 + torch.exp(-gate)) * up).half().float()
        y[slot] = (w2 @ act).half()
    return y


def _ids(num_tokens: int, top_k: int, num_experts: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(num_tokens, num_experts, device="cuda", generator=g).topk(top_k, -1)[1].to(torch.int32)


def _x(num_tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(num_tokens, HIDDEN, device="cuda", generator=g) * 0.5).half()


def _close(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """(fraction of elements that differ, rel-RMS, max |a - b| / max |b|).

    Kernel and reference round at the same points (W13 output, activation, W2 output); FP32 accumulation order
    can flip an FP16 rounding of one W13/activation element, which then moves that route's whole W2 row by a
    tiny amount, so chained outputs are compared by relative error, not per element."""
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    return ((diff > 0).float().mean().item(), (diff.pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item(),
            (diff.max() / b.abs().max()).item())


def _assert_near_bitwise(a: torch.Tensor, b: torch.Tensor, label: object) -> None:
    frac, rel_rms, rel_max = _close(a, b)
    assert frac < 0.05 and rel_rms < 2e-4 and rel_max < 2e-3, (label, frac, rel_rms, rel_max)


@pytest.mark.parametrize("tp,inter_local", [(1, 2304), (2, 1152), (4, 576), (8, 288)])
@pytest.mark.parametrize("experts,top_k", [(384, 6), (128, 3)])
def test_contract_admits_v41(tp, inter_local, experts, top_k):
    validate_mxfp4_sm70_moe_contract(global_num_experts=experts, top_k=top_k, hidden_size=HIDDEN,
                                     intermediate_size_per_partition=inter_local, tp_size=tp)


@pytest.mark.parametrize("kwargs,match", [({"global_num_experts": 256}, "384 global experts"),
                                          ({"top_k": 3}, "top-k=6 for 384"),
                                          ({"intermediate_size_per_partition": 512}, "intermediate size 2304"),
                                          ({"hidden_size": 6144}, "hidden size 4096")])
def test_contract_rejects_off_geometry_v41(kwargs, match):
    values = {"global_num_experts": 384, "top_k": 6, "hidden_size": HIDDEN, "intermediate_size_per_partition": 576,
              "tp_size": 4}
    values.update(kwargs)
    with pytest.raises(NotImplementedError, match=match):
        validate_mxfp4_sm70_moe_contract(**values)


def test_spill_plan():
    plan = make_spill_plan(116, 384)
    assert plan.spilled_expert_ids == tuple(range(268, 384)) and plan.mode == "uva"
    import numpy as np
    ranking = np.arange(384)[::-1].copy()  # expert 383 most routed ... expert 0 least
    assert make_spill_plan(3, 384, ranking).spilled_expert_ids == (0, 1, 2)
    with pytest.raises(ValueError):
        make_spill_plan(384, 384)


@pytest.fixture(scope="module")
def ckpt() -> SyntheticCheckpoint:
    return SyntheticCheckpoint(seed=11)


@pytest.mark.parametrize("backend", ["skinny", "turbomind", "torch"])
@pytest.mark.parametrize("tp_size,tp_rank", [(4, 0), (4, 3)])
def test_backends_match_reference_tp4(ckpt, backend, tp_size, tp_rank):
    layer, method = make_layer(384, 6, tp_size, tp_rank, backend, None, ckpt)
    for num_tokens in (1, 2, 8, 64):
        x = _x(num_tokens, seed=num_tokens)
        ids = _ids(num_tokens, 6, 384, seed=100 + num_tokens)
        if num_tokens >= 2:
            ids[1] = ids[0]
        y = method.expert_slots(layer, x, ids.contiguous())
        ref = reference_slots(ckpt, x, ids, tp_size, tp_rank)
        _assert_near_bitwise(y, ref, (backend, num_tokens))


def test_dspark_geometry_128_top3(ckpt):
    layer, method = make_layer(128, 3, 4, 1, "skinny", None, ckpt)
    x = _x(5, seed=5)
    ids = _ids(5, 3, 128, seed=55)
    _assert_near_bitwise(method.expert_slots(layer, x, ids), reference_slots(ckpt, x, ids, 4, 1), "dspark")


def test_spill_is_bitwise_equal_to_resident(ckpt):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    resident, m_res = make_layer(384, 6, 4, 2, "skinny", None, ckpt)
    torch.cuda.synchronize()
    resident_bytes = torch.cuda.memory_allocated() - base
    plan = make_spill_plan(116, 384)
    spilled, m_sp = make_layer(384, 6, 4, 2, "skinny", plan, ckpt)
    torch.cuda.synchronize()
    spilled_bytes = torch.cuda.memory_allocated() - base - resident_bytes
    assert spilled.w13_weight.shape[0] == 268 and spilled.w13_weight_spill.shape[0] == 116
    per_expert = 4_700_160  # PORT_DESIGN §3.4: bytes of one TP4 expert shard
    # the spilled experts never occupy HBM (caching-allocator rounding aside)
    assert resident_bytes - spilled_bytes >= 116 * per_expert * 0.99, (resident_bytes, spilled_bytes)
    for num_tokens in (1, 3, 8, 64):
        x = _x(num_tokens, seed=7 * num_tokens)
        ids = _ids(num_tokens, 6, 384, seed=9 * num_tokens)
        ids[0, 0] = 383  # surely spilled
        ids[0, 1] = 0    # surely resident
        torch.testing.assert_close(m_sp.expert_slots(spilled, x, ids.contiguous()),
                                   m_res.expert_slots(resident, x, ids.contiguous()), rtol=0, atol=0)


def test_spill_ranked_and_torch_oracle(ckpt):
    import numpy as np
    ranking = np.random.default_rng(0).permutation(384)
    plan = make_spill_plan(50, 384, ranking)
    layer, method = make_layer(384, 6, 4, 0, "skinny", plan, ckpt)
    oracle, m_or = make_layer(384, 6, 4, 0, "torch", plan, ckpt)
    x = _x(4, seed=4)
    ids = _ids(4, 6, 384, seed=44)
    ids[0, :3] = torch.tensor(plan.spilled_expert_ids[:3], dtype=torch.int32, device="cuda")
    _assert_near_bitwise(method.expert_slots(layer, x, ids.contiguous()),
                         m_or.expert_slots(oracle, x, ids.contiguous()), "ranked spill")


@pytest.mark.parametrize("backend", ["skinny", "turbomind"])
def test_cuda_graph_replay_bitwise(ckpt, backend):
    layer, method = make_layer(384, 6, 4, 1, backend, None, ckpt)
    for num_tokens in range(1, 9):
        x = _x(num_tokens, seed=num_tokens)
        ids = _ids(num_tokens, 6, 384, seed=num_tokens + 50).contiguous()
        eager = method.expert_slots(layer, x, ids)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            method.expert_slots(layer, x, ids)  # warm-up outside the graph
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = method.expert_slots(layer, x, ids)
        x.copy_(_x(num_tokens, seed=1000 + num_tokens))
        ids.copy_(_ids(num_tokens, 6, 384, seed=2000 + num_tokens))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, method.expert_slots(layer, x, ids), rtol=0, atol=0)
        assert eager.shape == out.shape


def test_loader_rejects_bad_shapes(ckpt):
    cfg = moe_config(384, 6, 4, 0)
    method = DeepseekV41Mxfp4MoEMethod(cfg, backend="skinny")
    layer = nn.Module()
    with torch.device("cuda"):
        method.create_weights(layer, 384, HIDDEN, 576, torch.float16, weight_loader=None)
    bad = torch.zeros(2000, 2560, dtype=torch.int8, device="cuda")
    with pytest.raises(ValueError, match="does not shard"):
        layer.w13_weight.weight_loader(layer.w13_weight, bad, "x", shard_id="w1", expert_id=0)
    with pytest.raises(TypeError, match="unexpected checkpoint dtype"):
        layer.w13_weight.weight_loader(layer.w13_weight, bad.float(), "x", shard_id="w1", expert_id=0)
