# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts of DeepSeek-V4.1 from an EXL3 (exllamav3 trellis) checkpoint on SM70 (gpubox lane, not upstream).

Checkpoint: per-expert bit widths K = 2..5 (``exl3_moe_plan.json``: ``layers[L]["k"]`` = 384 digits), codebook
``mul1``, exllamav3's default sm80 tile order, tensors ``layers.L.ffn.experts.E.w{1,2,3}.{trellis,suh,svh,mul1}``
(trellis int16 [in/16, out/16, 16K], y = x @ W). Kernels: ``csrc/sm70_turbomind/ops/exl3_moe_sm70.cu`` (mixed-K
launches decode the trellis inside the matmul; nothing is dequantized to memory).

Expert parallelism inside the TP group: the EXL3 Hadamard blocks are 128 wide and V4.1's expert intermediate 2304 /
TP4 = 576 is not a multiple of 128, so experts are not sliced. Each rank owns whole experts (load-balanced by the
routing histogram when one is given, else by bytes) and computes only the routes that land on them; the other routes
come out as exact zeros. The activations are already replicated and ``DeepseekV41MoE`` all-reduces the routed +
shared output, so the TP-partial contract of ``expert_slots`` holds with no extra collective.

Per rank and layer, every owned expert's strip-major trellis sits in one flat buffer per projection (gate / up share
offsets and K); ``offs`` are u32 word offsets, ``kbits`` the per-expert K, ``glob2loc`` maps global expert ids to
rank-local ones (-1 = owned elsewhere).
"""

from __future__ import annotations

import contextvars
import functools
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase, FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.exl3 import GROUP_MIN_TOKENS, MUL1_MULT, exl3_group_slots, exl3_strip_major_
from vllm.models.deepseek_v41 import knobs

logger = init_logger(__name__)

PLAN_KNOB = "VLLM_DS41_MOE_EXL3_PLAN"          # path of exl3_moe_plan.json; set = routed experts are EXL3
BALANCE_KNOB = "VLLM_DS41_MOE_EXL3_BALANCE"    # dir of layer<L>.npy routing counts (VLLM_DS41_MOE_ROUTE_HIST)
MUL1_MARKER = MUL1_MULT - (1 << 32)
CODEBOOK_MUL1 = 1
_PARAMS = ("trellis", "suh", "svh", "mul1")
_LAYER: contextvars.ContextVar[int | None] = contextvars.ContextVar("ds41_moe_exl3_layer", default=None)


@contextmanager
def expert_layer_context(layer_id: int) -> Iterator[None]:
    """Hands the decoder layer id to the factory call of the FusedMoE built inside the block (the plan is per layer)."""
    token = _LAYER.set(layer_id)
    try:
        yield
    finally:
        _LAYER.reset(token)


def plan_path() -> str:
    return knobs.env_str(PLAN_KNOB, "")


@functools.cache
def load_plan(path: str) -> tuple[str, dict[int, np.ndarray]]:
    plan = json.load(open(path))
    if plan.get("format") != "exl3_moe_plan" or plan.get("codebook") != "mul1":
        raise ValueError(f"{path}: expected an exl3_moe_plan with the mul1 codebook, got format "
                         f"{plan.get('format')!r} codebook {plan.get('codebook')!r}")
    ks = {int(layer): np.frombuffer(row["k"].encode(), dtype=np.uint8).astype(np.int64) - ord("0")
          for layer, row in plan["layers"].items()}
    for layer, k in ks.items():
        if k.min() < 2 or k.max() > 5:
            raise ValueError(f"{path}: layer {layer} has bit widths outside 2..5")
    return plan["codebook"], ks


def assign_experts(k: np.ndarray, counts: np.ndarray | None, ranks: int) -> np.ndarray:
    """Owner rank of each expert. Greedy longest-processing-time on routing share (when a histogram is given) plus
    weight bytes, so both the decode work and the memory stay even across the TP group."""
    n = k.shape[0]
    size = k / k.sum()
    work = counts / counts.sum() if counts is not None and counts.sum() > 0 else size
    cost = 0.5 * work + 0.5 * size
    owner = np.empty(n, dtype=np.int64)
    load = np.zeros(ranks)
    for e in np.argsort(-cost, kind="stable"):
        r = int(np.argmin(load))
        owner[e] = r
        load[r] += cost[e]
    return owner


def _words(k_in: int, n_out: int, bits: int) -> int:
    return k_in * n_out * bits // 32


class DeepseekV41Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig, layer_id: int, path: str):
        super().__init__(moe)
        if moe.moe_parallel_config.use_ep or moe.has_bias:
            raise NotImplementedError("DeepSeek-V4.1 EXL3 experts: vLLM EP / expert bias are not supported")
        _, ks = load_plan(path)
        if layer_id not in ks or ks[layer_id].shape[0] != moe.num_experts:
            raise ValueError(f"{path}: no {moe.num_experts}-expert plan for layer {layer_id}")
        self.layer_id = layer_id
        self.k_global = ks[layer_id]
        self.tp_rank = moe.moe_parallel_config.tp_rank
        self.tp_size = moe.moe_parallel_config.tp_size
        counts = None
        balance = knobs.env_str(BALANCE_KNOB, "")
        if balance:
            hist = np.load(os.path.join(balance, f"layer{layer_id:02d}.npy"))
            counts = hist[1] if hist.ndim == 2 else hist
        self.owner = assign_experts(self.k_global, counts, self.tp_size)
        self.local = np.flatnonzero(self.owner == self.tp_rank)        # global ids owned here, ascending
        self.weight_dtype = "exl3"

    # ---- FusedMoEMethodBase plumbing: driven by DeepseekV41MoE.expert_slots, never by FusedMoE.forward ----
    @property
    def supports_internal_mk(self) -> bool:
        return True

    @property
    def is_monolithic(self) -> bool:
        return False

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return None

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):  # type: ignore[override]
        raise NotImplementedError("DeepSeek-V4.1 experts run through DeepseekV41MoE (expert_slots + FP32 combine)")

    # ---- weights ----
    def create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int, params_dtype: torch.dtype,
                       **extra_weight_attrs) -> None:
        if not ops.exl3_moe_available():
            raise RuntimeError("EXL3 MoE kernels are not built into this vLLM (compute capability 7.0)")
        H = hidden_size
        inter = intermediate_size_per_partition * self.tp_size     # whole experts: no TP slicing
        if H % 128 or inter % 128:
            raise ValueError(f"EXL3 experts need hidden / intermediate multiples of 128, got {H} / {inter}")
        kl = self.k_global[self.local]
        n_loc = len(self.local)
        w13 = np.array([_words(H, inter, int(k)) for k in kl], dtype=np.int64)
        w2 = np.array([_words(inter, H, int(k)) for k in kl], dtype=np.int64)
        offs13 = np.concatenate([[0], np.cumsum(w13)[:-1]]) if n_loc else np.zeros(0, np.int64)
        offs2 = np.concatenate([[0], np.cumsum(w2)[:-1]]) if n_loc else np.zeros(0, np.int64)
        layer.exl3_H, layer.exl3_inter = H, inter
        layer.exl3_offs13_host, layer.exl3_offs2_host = offs13, offs2
        loc_of = np.full(num_experts, -1, dtype=np.int64)
        loc_of[self.local] = np.arange(n_loc)
        layer.exl3_loc_of_host = loc_of
        spec = {
            "w13_trellis": ((2, int(w13.sum()) * 2), torch.int16),
            "w13_suh": ((2, n_loc, H), torch.float16), "w13_svh": ((2, n_loc, inter), torch.float16),
            "w13_mul1": ((2, n_loc), torch.int32),
            "w2_trellis": ((int(w2.sum()) * 2,), torch.int16),
            "w2_suh": ((n_loc, inter), torch.float16), "w2_svh": ((n_loc, H), torch.float16),
            "w2_mul1": ((n_loc,), torch.int32),
        }
        attrs = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        loader = self._make_weight_loader(layer)
        for name, (shape, dtype) in spec.items():
            param = Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, param)
            for k, v in attrs.items():
                setattr(param, k, v)
            param.weight_loader = loader  # type: ignore[attr-defined]
            param.ds41_name = name        # type: ignore[attr-defined]
        nbytes = sum(getattr(layer, n).numel() * getattr(layer, n).element_size() for n in spec)
        logger.info_once("DeepSeek-V4.1 EXL3 experts: expert-parallel in the TP group, %d experts on this rank in "
                         "layer %d (%.2f GiB)", n_loc, self.layer_id, nbytes / 2**30)

    def _make_weight_loader(self, layer: torch.nn.Module):
        method = self

        def ds41_exl3_weight_loader(param: Parameter, loaded_weight: torch.Tensor, weight_name: str = "",
                                    shard_id: str = "", expert_id: int = -1, return_success: bool = False):
            method._load(layer, param, loaded_weight, shard_id, int(expert_id))
            return True if return_success else None

        ds41_exl3_weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
        return ds41_exl3_weight_loader

    def _load(self, layer: torch.nn.Module, param: Parameter, t: torch.Tensor, shard: str, expert: int) -> None:
        name = param.ds41_name  # type: ignore[attr-defined]
        loc = int(layer.exl3_loc_of_host[expert])
        if loc < 0:
            return                                   # owned by another rank of the TP group
        gate_up = name.startswith("w13")
        if gate_up != (shard in ("w1", "w3")) or (not gate_up and shard != "w2"):
            raise ValueError(f"{name}: shard {shard!r} does not belong here")
        g = (0 if shard == "w1" else 1) if gate_up else None
        H, inter, K = layer.exl3_H, layer.exl3_inter, int(self.k_global[expert])
        if name.endswith("_trellis"):
            k_in, n_out = (H, inter) if gate_up else (inter, H)
            want = (k_in // 16, n_out // 16, 16 * K)
            if tuple(t.shape) != want or t.dtype != torch.int16:
                raise ValueError(f"layer {self.layer_id} expert {expert} {shard}.trellis: {tuple(t.shape)} {t.dtype}, "
                                 f"expected {want} int16 (K={K} from the plan)")
            off = int((layer.exl3_offs13_host if gate_up else layer.exl3_offs2_host)[loc]) * 2
            dst = (param.data[g] if gate_up else param.data)[off: off + t.numel()]
            dst.copy_(exl3_strip_major_(t.clone()).reshape(-1))
            return
        dst = param.data[g][loc] if gate_up else param.data[loc]
        if tuple(t.shape) != tuple(dst.shape) or t.dtype != dst.dtype:
            raise ValueError(f"layer {self.layer_id} expert {expert} {shard}.{name.split('_', 1)[1]}: "
                             f"{tuple(t.shape)} {t.dtype}, expected {tuple(dst.shape)} {dst.dtype}")
        dst.copy_(t)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.activation.name.lower() != "silu" or layer.apply_router_weight_on_input:
            raise NotImplementedError("DeepSeek-V4.1 experts are SwiGLU with routing weights applied to outputs")
        if layer.swiglu_limit is None:
            raise ValueError("DeepSeek-V4.1 experts need swiglu_limit (10.0)")
        for name in ("w13_mul1", "w2_mul1"):
            marker = getattr(layer, name)
            if marker.numel() and not bool((marker == MUL1_MARKER).all()):
                raise ValueError(f"layer {self.layer_id}: {name} markers are not the mul1 codebook "
                                 "(an expert may be missing from the checkpoint)")
            delattr(layer, name)
        device = layer.w13_trellis.device
        layer.exl3_glob2loc = torch.as_tensor(layer.exl3_loc_of_host, dtype=torch.int32, device=device)
        layer.exl3_offs13 = torch.as_tensor(layer.exl3_offs13_host, dtype=torch.int64, device=device)
        layer.exl3_offs2 = torch.as_tensor(layer.exl3_offs2_host, dtype=torch.int64, device=device)
        layer.exl3_kbits = torch.as_tensor(self.k_global[self.local], dtype=torch.int32, device=device)
        layer.exl3_limit = float(layer.swiglu_limit)

    # ---- compute ----
    def expert_slots(self, layer: torch.nn.Module, x: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """x [T, H] fp16, topk_ids [T, k] int32 -> y [T * k, H] fp16 (route order; zero for other ranks' experts)."""
        num_tokens, top_k = topk_ids.shape
        if x.dtype != torch.float16 or x.ndim != 2 or x.shape[0] != num_tokens or not x.is_contiguous():
            raise TypeError(f"expert_slots: x must be contiguous fp16 [{num_tokens}, H], got {x.dtype} {tuple(x.shape)}")
        H, inter = layer.exl3_H, layer.exl3_inter
        S = num_tokens * top_k
        f16 = dict(dtype=torch.float16, device=x.device)
        f32 = dict(dtype=torch.float32, device=x.device)
        loc = layer.exl3_glob2loc[topk_ids.long()].contiguous()            # [T, k], -1 = another rank's expert
        flat = loc.view(-1)
        xg, xu = torch.empty((S, H), **f16), torch.empty((S, H), **f16)
        yg, yu = torch.empty((S, inter), **f32), torch.empty((S, inter), **f32)
        hd, yd = torch.empty((S, inter), **f16), torch.empty((S, H), **f32)
        out = torch.empty((S, H), **f32)
        ops.exl3_moe_pre_out(xg, xu, x, loc, layer.w13_suh[0], layer.w13_suh[1])
        if num_tokens >= GROUP_MIN_TOKENS:
            ids, rows = exl3_group_slots(loc, layer.exl3_kbits.shape[0] + 1)
            ids, rows = ids.contiguous(), rows.contiguous()
        else:
            ids, rows = flat, None
        ops.exl3_moe_gemv_mixed_out(yg, yu, xg, xu, layer.w13_trellis[0], layer.w13_trellis[1], layer.exl3_offs13,
                                    layer.exl3_kbits, ids, rows, False, CODEBOOK_MUL1, inter)
        ops.exl3_moe_mid_out(hd, yg, yu, loc, layer.w13_svh[0], layer.w13_svh[1], layer.w2_suh, layer.exl3_limit)
        ops.exl3_moe_gemv_mixed_out(yd, None, hd, None, layer.w2_trellis, None, layer.exl3_offs2, layer.exl3_kbits,
                                    ids, rows, False, CODEBOOK_MUL1, H)
        ops.exl3_moe_post_out(out, yd, flat.view(S, 1), torch.ones((S, 1), **f32), layer.w2_svh)
        return out.to(torch.float16)
