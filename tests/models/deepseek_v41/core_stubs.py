# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEST-ONLY stand-ins for the L-ATTN / L-MOE / L-ENGRAM modules (PORT_DESIGN §3.3, §3.4, §3.6 signatures).

They carry the real parameter names, shapes, dtypes and TP slicing of §3.7, so L-CORE's loader, the FP8 dequant
fallback and the per-GPU memory accounting are exercised with real checkpoint tensors. Their forward passes are
NOT the model: attention skips the token mixing (q -> o directly), Engram is the identity, and the MoE is a
slow per-token torch MXFP4 implementation. ``install()`` registers them in ``sys.modules`` under the real module
names; never import this file from vllm/.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.models.deepseek_v41.common import contracts as C

ALLOCATE_EXPERTS = True     # unit tests on CPU switch the 384-expert byte tensors off (7.2 GB at TP1)
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


class _Norm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (self.weight * x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + C.NORM_EPS)).to(x.dtype)


# ------------------------------------------------------------------ attention
class _Compressor(nn.Module):
    def __init__(self, ratio: int) -> None:
        super().__init__()
        if ratio == 2:
            self.fused_wkv_wgate = MergedColumnParallelLinear(C.HIDDEN, [512, 512], bias=False, quant_config=None,
                                                              disable_tp=True)
        else:
            self.wkv = ReplicatedLinear(C.HIDDEN, 512, bias=False, quant_config=None)
        self.norm = _Norm(512)


class _Indexer(nn.Module):
    def __init__(self, quant_config, owns_keys: bool) -> None:
        super().__init__()
        self.wq_b = ReplicatedLinear(C.Q_LORA, C.IDX_HEADS * C.IDX_DIM, bias=False, quant_config=quant_config)
        self.weights_proj = ReplicatedLinear(C.HIDDEN, C.IDX_HEADS, bias=False, quant_config=None)
        if owns_keys:
            self.wk = ReplicatedLinear(512, C.IDX_DIM, bias=False, quant_config=None)
            self.k_norm = _Norm(C.IDX_DIM)


class DeepseekV41Attention(nn.Module):
    def __init__(self, vllm_config, prefix, topo, stage, shared, aux_streams=None) -> None:
        super().__init__()
        qc = vllm_config.quant_config
        tp = get_tensor_model_parallel_world_size()
        self.n_local_heads = C.N_HEADS // tp
        self.n_local_groups = C.O_GROUPS // tp
        self.fused_wqa_wkv = MergedColumnParallelLinear(C.HIDDEN, [C.Q_LORA, C.HEAD_DIM], bias=False,
                                                        quant_config=qc, disable_tp=True,
                                                        prefix=f"{prefix}.fused_wqa_wkv")
        self.q_norm = _Norm(C.Q_LORA)
        self.kv_norm = _Norm(C.HEAD_DIM)
        self.wq_b = ColumnParallelLinear(C.Q_LORA, C.N_HEADS * C.HEAD_DIM, bias=False, quant_config=qc,
                                         prefix=f"{prefix}.wq_b")
        self.wo_a = ColumnParallelLinear(C.N_HEADS * C.HEAD_DIM // C.O_GROUPS, C.O_GROUPS * C.O_LORA, bias=False,
                                         quant_config=qc, prefix=f"{prefix}.wo_a")
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(C.O_GROUPS * C.O_LORA, C.HIDDEN, bias=False, quant_config=qc,
                                      reduce_results=False, prefix=f"{prefix}.wo_b")
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32), requires_grad=False)
        if topo.owns_compressor:
            self.compressor = _Compressor(topo.compress_ratio)
        if topo.owns_indexer:
            self.indexer = _Indexer(qc, topo.owns_compressor)

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        qr_kv, _ = self.fused_wqa_wkv(x)
        qr = self.q_norm(qr_kv[:, : C.Q_LORA])
        q, _ = self.wq_b(qr)                                        # [T, local_heads * 512]
        o = q.view(-1, self.n_local_groups, (self.n_local_heads // self.n_local_groups) * C.HEAD_DIM)
        o = self.wo_a.quant_method.apply(self.wo_a, o)              # [T, groups, 1024]
        out, _ = self.wo_b(o.flatten(1))
        out = out.float()
        return tensor_model_parallel_all_reduce(out) if get_tensor_model_parallel_world_size() > 1 else out


# ------------------------------------------------------------------ MoE
@dataclass(frozen=True)
class ExpertSpillPlan:
    spilled_expert_ids: tuple[int, ...]
    mode: str = "uva"


def make_spill_plan(num_spilled_per_layer: int, n_routed_experts: int = 384,
                    ranking: np.ndarray | None = None) -> ExpertSpillPlan:
    return ExpertSpillPlan(tuple(range(n_routed_experts - num_spilled_per_layer, n_routed_experts)))


class _Gate(nn.Module):
    def __init__(self, n_experts: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_experts, C.HIDDEN, dtype=torch.float16), requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(n_experts, dtype=torch.float32),
                                                    requires_grad=False)


class _SharedExpert(nn.Module):
    def __init__(self, qc) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(C.HIDDEN, [C.MOE_INTER] * 2, bias=False, quant_config=qc)
        self.down_proj = RowParallelLinear(C.MOE_INTER, C.HIDDEN, bias=False, quant_config=qc, reduce_results=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu, _ = self.gate_up_proj(x)
        gate, up = gu.float().chunk(2, dim=-1)
        h = F.silu(gate.clamp(max=C.SWIGLU_LIMIT)) * up.clamp(-C.SWIGLU_LIMIT, C.SWIGLU_LIMIT)
        out, _ = self.down_proj(h.half())
        return out.float()


class _Experts(nn.Module):
    """MXFP4 bytes per rank: w13 [E, 2*inter/tp, 2560] + E8M0 [E, 2*inter/tp, 160]; w2 [E, 5120, inter/tp/2] + [.., /32]."""

    def __init__(self, n_experts: int) -> None:
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.inter = C.MOE_INTER // tp
        self.tp_rank = get_tensor_model_parallel_rank()
        shapes = {"w13_weight": (n_experts, 2 * self.inter, C.HIDDEN // 2),
                  "w13_weight_scale": (n_experts, 2 * self.inter, C.HIDDEN // 32),
                  "w2_weight": (n_experts, C.HIDDEN, self.inter // 2),
                  "w2_weight_scale": (n_experts, C.HIDDEN, self.inter // 32)}
        for name, shape in shapes.items():
            p = nn.Parameter(torch.empty(shape if ALLOCATE_EXPERTS else (0,), dtype=torch.uint8), requires_grad=False)
            p.weight_loader = self.weight_loader
            self.register_parameter(name, p)

    def weight_loader(self, param, loaded, name, shard_id, expert_id, return_success=False) -> bool:
        r, inter = self.tp_rank, self.inter
        loaded = loaded.view(torch.uint8)
        if shard_id in ("w1", "w3"):
            rows = loaded[inter * r: inter * (r + 1)]
            off = 0 if shard_id == "w1" else inter
            param.data[expert_id, off: off + inter].copy_(rows)
        else:
            per = loaded.shape[1] // (C.MOE_INTER // inter)            # packed / scale cols per rank
            param.data[expert_id].copy_(loaded[:, per * r: per * (r + 1)])
        return True

    @staticmethod
    def _dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        table = E2M1.to(packed.device)
        lo, hi = table[(packed & 0xF).long()], table[(packed >> 4).long()]
        vals = torch.stack((lo, hi), dim=-1).flatten(-2)             # low nibble = even K index
        s = torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.long() - 127)
        return vals * s.repeat_interleave(32, dim=-1)

    def expert(self, e: int, x: torch.Tensor) -> torch.Tensor:
        w13 = self._dequant(self.w13_weight[e], self.w13_weight_scale[e])
        gu = x.float() @ w13.t()
        gate, up = gu[:, : self.inter], gu[:, self.inter:]
        h = F.silu(gate.clamp(max=C.SWIGLU_LIMIT)) * up.clamp(-C.SWIGLU_LIMIT, C.SWIGLU_LIMIT)
        return h @ self._dequant(self.w2_weight[e], self.w2_weight_scale[e]).t()


class DeepseekV41MoE(nn.Module):
    def __init__(self, vllm_config, prefix, layer_id, *, n_routed_experts=384, top_k=6, spill=None) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = _Gate(n_routed_experts)
        self.shared_experts = _SharedExpert(vllm_config.quant_config)
        self.experts = _Experts(n_routed_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.shared_experts(x)
        if ALLOCATE_EXPERTS:
            scores = torch.sqrt(F.softplus(x.float() @ self.gate.weight.float().t()))
            ids = torch.topk(scores + self.gate.e_score_correction_bias, self.top_k, dim=-1).indices
            w = scores.gather(1, ids)
            w = w / (w.sum(-1, keepdim=True) + 1e-20) * C.ROUTED_SCALE
            for t in range(x.shape[0]):
                for k in range(self.top_k):
                    out[t] += w[t, k] * self.experts.expert(int(ids[t, k]), x[t: t + 1])[0]
        return tensor_model_parallel_all_reduce(out) if get_tensor_model_parallel_world_size() > 1 else out

    @staticmethod
    def expert_params_mapping(n_routed_experts: int) -> list[tuple[str, str, int, str]]:
        return [("experts.w13_" if shard in ("w1", "w3") else "experts.w2_", f"experts.{e}.{shard}.", e, shard)
                for e in range(n_routed_experts) for shard in ("w1", "w2", "w3")]


def make_v41_moe_method(moe):
    raise NotImplementedError("test stub: the stub MoE does not use FusedMoE")


# ------------------------------------------------------------------ Engram
class EngramHostService:
    def __init__(self, hf_config, layers, tp_rank, tp_size, row_dir, tokenizer_path, max_num_batched_tokens,
                 device, io_threads=4) -> None:
        self.layers, self.tp_rank, self.tp_size, self.io_threads = layers, tp_rank, tp_size, io_threads
        self.plans: list = []
        self.layouts: list = []

    def begin_step(self, plan) -> None:
        self.plans.append(plan)

    def bind_batch(self, layout) -> None:
        self.layouts.append(layout)

    def wait_rows(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("test stub")

    def end_step(self, step_id: int) -> None:
        pass

    def stats(self) -> dict[str, float]:
        return {}

    def shutdown(self) -> None:
        pass


class DeepseekV41Engram(nn.Module):
    """Identity forward; loads wkv (biased 2^10, sub-table columns of this rank) and q*k like the real module."""

    def __init__(self, vllm_config, prefix, layer_id, service) -> None:
        super().__init__()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        n_sub = C.ENGRAM_SUBTABLES // self.tp_size
        self.wkv_r = nn.Parameter(torch.empty(C.HC * C.HIDDEN + C.HIDDEN, n_sub * C.ENGRAM_HEAD_DIM,
                                              dtype=torch.float16), requires_grad=False)
        self.qk = nn.Parameter(torch.empty(C.HC, C.HIDDEN, dtype=torch.float32), requires_grad=False)
        self.loaded_names: list[str] = []

    def load_weights(self, weights) -> set[str]:
        got = dict(weights)
        self.loaded_names = sorted(got)
        w, s = got["wkv.weight"], got["wkv.scale"]
        cols = torch.cat([torch.arange(st * C.ENGRAM_HEAD_DIM, (st + 1) * C.ENGRAM_HEAD_DIM)
                          for st in range(self.tp_rank, C.ENGRAM_SUBTABLES, self.tp_size)])
        full = w.float() * s.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
        self.wkv_r.data.copy_((full[:, cols] * 1024.0).to(torch.float16))
        self.qk.data.copy_(got["q_weight"].float() * got["k_weight"].float())
        return {"wkv.weight", "wkv.scale", "q_weight", "k_weight"}       # AM-1: consumed checkpoint names

    def forward(self, stream: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        # AM-2 stand-in: record which kind of step reached Engram (the real module raises on unbound real steps)
        self.last_is_dummy = is_forward_context_available() and get_forward_context().is_dummy_run
        return stream


# ------------------------------------------------------------------ mirror
class DeepseekV41KVSourceMirror(nn.Module):
    def __init__(self, vllm_config, source_layer, shared) -> None:
        super().__init__()
        self.shared = shared
        self.ingested: list = []

        self.crcs: list = []

    def ingest(self, positions, ckv, ik, cand, crc=None) -> None:
        self.shared.candidate_blocks[: cand.shape[0]].copy_(cand)
        self.ingested.append((ckv.shape, ik.shape, cand.shape))
        self.crcs.append(None if crc is None else crc.clone())


MIRROR_CHECK_ENV = "VLLM_DS41_ATTN_MIRROR_CHECK"
PP_KEY_KV20_CRC = "kv20_crc"


def kv20_crc(ckv, ik):
    """Same formula as ds41/attn kv_mirror.kv20_crc."""
    return ckv.view(torch.int16).sum(dim=-1, dtype=torch.int32) + ik.view(torch.int16).sum(dim=-1, dtype=torch.int32)


_MODULES = {
    "vllm.models.deepseek_v41.attention": ("DeepseekV41Attention",),
    "vllm.models.deepseek_v41.sm70.moe": ("DeepseekV41MoE", "ExpertSpillPlan", "make_spill_plan",
                                          "make_v41_moe_method"),
    "vllm.models.deepseek_v41.common.engram": ("DeepseekV41Engram",),
    "vllm.models.deepseek_v41.common.engram_host": ("EngramHostService",),
    "vllm.models.deepseek_v41.kv_mirror": ("DeepseekV41KVSourceMirror", "MIRROR_CHECK_ENV", "PP_KEY_KV20_CRC",
                                           "kv20_crc"),
}


def install(setitem=None) -> None:
    """Register the stand-ins under the real module names (``setitem`` = monkeypatch.setitem for tests)."""
    this = sys.modules[__name__]
    for module_name, names in _MODULES.items():
        mod = types.ModuleType(module_name)
        mod.__dict__["__ds41_test_stub__"] = True
        for name in names:
            setattr(mod, name, getattr(this, name))
        if setitem is not None:
            setitem(sys.modules, module_name, mod)
        else:
            sys.modules[module_name] = mod
