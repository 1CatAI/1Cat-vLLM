# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE block for SM70 (lane L-MOE; PORT_DESIGN §3.4, §4.1, §5.2; reference ref:m.py:792-903).

``DeepseekV41MoE.forward(x)`` takes the FFN input ``x`` [T, 5120] fp16 and returns ``[T, 5120]`` float32 =
``sum_k w_k * expert_k(x) + shared(x)``, all-reduced over TP. Per rank:

* router: logits in FP32 from an exponent-biased FP16 copy of the BF16 gate weight (``W * 2^e`` is exact in FP16
  for the real checkpoint; FP16 x FP16 products are exact in FP32, so the logits equal the reference's FP32
  GEMV up to summation order); ``sqrt(softplus)``, top-k of ``score + bias`` (bias selects only), weights =
  ``score / (sum + 1e-20) * 1.5``; no hash layers, ``bias_vl`` not loaded (text only);
* routed experts: ``DeepseekV41Mxfp4MoEMethod.expert_slots`` (per-route FP16 outputs, TP-partial);
* shared expert: FP16 weights (dense FP8 dequantised by the quant config's fallback); W13 output and the SwiGLU
  rounded to FP16 like a routed expert, W2 output kept FP32;
* combine: ``sum_j w[t, j] * y[t, j]`` (route order, FP32, no FMA) + shared, then one FP32 TP all-reduce.

Decode (T <= 8) runs the gate and the shared expert through the GEMVs of ``sm70/gemv.py`` (SwiGLU fused into the
gate-up GEMV, the combine fused into the down GEMV); larger T uses cuBLAS with FP32 outputs and the Triton combine.

Knobs (``knobs.py`` helpers): ``VLLM_DS41_MOE_IMPL`` = sm70 | torch (per-expert FP32-dequantised oracle, §7.1);
``VLLM_DS41_MOE_BACKEND`` = skinny | turbomind; ``VLLM_DS41_MOE_DECODE_GEMV`` (default 1);
``VLLM_DS41_MOE_TOPK_CHECK`` (default 0: all-gathers a router-id checksum across TP each step and asserts
equality, §4.1; debug only — it syncs the host, so it refuses to run under CUDA-graph capture: use enforce_eager).
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

import vllm._custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group, tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig, FusedMoEMethodBase
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common import contracts as C

from . import gemv as v41_gemv
from . import moe_kernels as mk
from .moe_method import DeepseekV41Mxfp4MoEMethod, ExpertSpillPlan, make_spill_plan

__all__ = ["DeepseekV41MoE", "ExpertSpillPlan", "make_spill_plan", "make_v41_moe_method", "expert_spill_context"]

IMPL_KNOB = "VLLM_DS41_MOE_IMPL"
BACKEND_KNOB = "VLLM_DS41_MOE_BACKEND"
DECODE_GEMV_KNOB = "VLLM_DS41_MOE_DECODE_GEMV"
TOPK_CHECK_KNOB = "VLLM_DS41_MOE_TOPK_CHECK"
_GATE_TARGET_MAX = 32768.0  # largest |W| * 2^e kept in FP16 (half the FP16 range, leaves headroom)

_SPILL: contextvars.ContextVar[ExpertSpillPlan | None] = contextvars.ContextVar("ds41_moe_spill", default=None)


@contextlib.contextmanager
def expert_spill_context(spill: ExpertSpillPlan | None) -> Iterator[None]:
    """Hands ``spill`` to the ``make_v41_moe_method`` call of the FusedMoE built inside the block (the quant
    config's factory only sees the FusedMoEConfig, and the spill must be known before weights are allocated)."""
    token = _SPILL.set(spill)
    try:
        yield
    finally:
        _SPILL.reset(token)


def _impl() -> str:
    return knobs.env_str(IMPL_KNOB, "sm70", choices=("sm70", "torch"))


def make_v41_moe_method(moe: FusedMoEConfig) -> FusedMoEMethodBase:
    """Factory used by DeepseekV41FP8Config for every FusedMoE of the model (backbone and DSpark)."""
    backend = "torch" if _impl() == "torch" else knobs.env_str(BACKEND_KNOB, "skinny",
                                                                 choices=("skinny", "turbomind"))
    return DeepseekV41Mxfp4MoEMethod(moe, backend=backend, spill=_SPILL.get())


class DeepseekV41Gate(nn.Module):
    """Router weight [E, 5120] stored as FP16 ``W * 2^weight_exp`` (exact; chosen at load) + FP32 bias."""

    def __init__(self, n_experts: int, hidden: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.zeros(n_experts, hidden, dtype=torch.float16), requires_grad=False)
        self.e_score_correction_bias = Parameter(torch.zeros(n_experts, dtype=torch.float32), requires_grad=False)
        self.weight_exp = 0
        set_weight_attrs(self.weight, {"weight_loader": self._load_weight})

    def _load_weight(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        if tuple(loaded_weight.shape) != tuple(param.shape):
            raise ValueError(f"gate weight {tuple(loaded_weight.shape)} != {tuple(param.shape)}")
        w = loaded_weight.to(device=param.device, dtype=torch.float32)
        amax = float(w.abs().max().item())
        exp = 0 if amax == 0.0 else max(0, min(24, math.floor(math.log2(_GATE_TARGET_MAX / amax))))
        scaled = w * (2.0**exp)
        stored = scaled.to(torch.float16)
        inexact = int((stored.float() != scaled).sum().item())
        if inexact:
            raise ValueError(f"gate weight is not exact in FP16 with bias 2^{exp} ({inexact} elements, "
                             f"max |w| {amax:.3e}); PORT_DESIGN §4.1 requires exact router weights")
        param.data.copy_(stored)
        self.weight_exp = exp

    def true_weight_fp32(self) -> torch.Tensor:
        return self.weight.float() * (2.0 ** -self.weight_exp)


class DeepseekV41SharedExpert(nn.Module):
    """Shared expert (TP-sharded on the intermediate dim; ``reduce_results=False``: combined before the AR)."""

    def __init__(self, hidden: int, inter: int, quant_config, prefix: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden, [inter] * 2, bias=False, quant_config=quant_config,
                                                       prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(inter, hidden, bias=False, quant_config=quant_config,
                                           reduce_results=False, prefix=f"{prefix}.down_proj")

    def fp16_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        w13, w2 = self.gate_up_proj.weight, self.down_proj.weight
        for name, w in (("gate_up_proj", w13), ("down_proj", w2)):
            if w.dtype != torch.float16 or w.ndim != 2 or not w.is_contiguous():
                raise TypeError(f"shared expert {name} must hold a contiguous FP16 [N, K] weight after loading "
                                f"(dense FP8 dequant fallback); got {w.dtype} {tuple(w.shape)}")
        return w13, w2


class DeepseekV41MoE(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, layer_id: int, *, n_routed_experts: int = 384,
                 top_k: int = 6, spill: ExpertSpillPlan | None = None) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.layer_id = layer_id
        self.hidden = int(config.hidden_size)
        self.n_experts = int(n_routed_experts)
        self.top_k = int(top_k)
        self.swiglu_limit = float(config.swiglu_limit)
        self.routed_scale = float(config.routed_scaling_factor)
        checks = {
            "hidden_size": (self.hidden, C.HIDDEN), "moe_intermediate_size": (config.moe_intermediate_size,
                                                                               C.MOE_INTER),
            "swiglu_limit": (self.swiglu_limit, C.SWIGLU_LIMIT), "routed_scaling_factor": (self.routed_scale,
                                                                                         C.ROUTED_SCALE),
            "scoring_func": (getattr(config, "scoring_func", None), "sqrtsoftplus"),
            "topk_method": (getattr(config, "topk_method", None), "noaux_tc"),
            "norm_topk_prob": (bool(getattr(config, "norm_topk_prob", False)), True),
            "n_shared_experts": (getattr(config, "n_shared_experts", None), 1),
            "num_hash_layers": (getattr(config, "num_hash_layers", 0) or 0, 0),
        }
        for name, (got, want) in checks.items():
            if got != want:
                raise ValueError(f"DeepseekV41MoE layer {layer_id}: config {name}={got!r}, expected {want!r}")
        if (self.n_experts, self.top_k) not in ((C.N_EXPERTS, C.TOP_K), (128, 3)):
            raise ValueError(f"DeepseekV41MoE: unsupported experts/top-k {self.n_experts}/{self.top_k}")
        self.impl = _impl()
        self.decode_gemv = knobs.env_bool(DECODE_GEMV_KNOB, True)
        self.topk_check = knobs.env_bool(TOPK_CHECK_KNOB, False)
        self.tp_size = get_tensor_model_parallel_world_size()
        quant_config = vllm_config.quant_config
        self.gate = DeepseekV41Gate(self.n_experts, self.hidden)
        with expert_spill_context(spill):
            self.experts = FusedMoE(
                num_experts=self.n_experts, top_k=self.top_k, hidden_size=self.hidden,
                intermediate_size=int(config.moe_intermediate_size), params_dtype=torch.float16, renormalize=True,
                quant_config=quant_config, prefix=f"{prefix}.experts", scoring_func="sqrtsoftplus",
                routed_scaling_factor=self.routed_scale, e_score_correction_bias=self.gate.e_score_correction_bias,
                swiglu_limit=self.swiglu_limit, router_logits_dtype=torch.float32)
        if not isinstance(self.experts.quant_method, DeepseekV41Mxfp4MoEMethod):
            raise TypeError(f"{prefix}.experts got {type(self.experts.quant_method).__name__}; the quant config must "
                            "route DeepSeek-V4.1 FusedMoE layers to make_v41_moe_method")
        self.shared_experts = DeepseekV41SharedExpert(self.hidden, int(config.moe_intermediate_size), quant_config,
                                                      prefix=f"{prefix}.shared_experts")

    @staticmethod
    def expert_params_mapping(n_routed_experts: int) -> list[tuple[str, str, int, str]]:
        """(param_name, weight_name, expert_id, shard_id) in FusedMoE's form; checkpoint expert ``.scale`` must
        already be renamed ``.weight_scale`` (as the V4 weights mapper does)."""
        return [("experts.w13_" if shard in ("w1", "w3") else "experts.w2_", f"experts.{e}.{shard}.", e, shard)
                for e in range(n_routed_experts) for shard in ("w1", "w2", "w3")]

    @property
    def method(self) -> DeepseekV41Mxfp4MoEMethod:
        return self.experts.quant_method  # type: ignore[return-value]

    # ---- router ----
    def gate_logits(self, x: torch.Tensor) -> torch.Tensor:
        alpha = 2.0 ** -self.gate.weight_exp
        if self.impl == "torch":
            return x.float() @ self.gate.true_weight_fp32().t()
        if self.decode_gemv and x.shape[0] <= v41_gemv.MAX_GEMV_ROWS:
            return v41_gemv.gemv(x, self.gate.weight, alpha=alpha)
        return torch.mm(x, self.gate.weight.t(), out_dtype=torch.float32).mul_(alpha)

    def route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP32 logits [T, E] -> (weights [T, k] f32, ids [T, k] int32)."""
        bias = self.gate.e_score_correction_bias
        if self.impl == "torch":
            scores = F.softplus(logits).sqrt()
            ids = (scores + bias).topk(self.top_k, dim=-1)[1]
            w = scores.gather(1, ids)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * self.routed_scale
            return w.contiguous(), ids.to(torch.int32).contiguous()
        num_tokens = logits.shape[0]
        w = torch.empty((num_tokens, self.top_k), dtype=torch.float32, device=logits.device)
        ids = torch.empty((num_tokens, self.top_k), dtype=torch.int32, device=logits.device)
        token_expert = torch.empty((num_tokens, self.top_k), dtype=torch.int32, device=logits.device)
        ops.topk_hash_softplus_sqrt(w, ids, token_expert, logits.contiguous(), True, self.routed_scale, bias,
                                    None, None)
        return w, ids

    # ---- shared expert ----
    def shared_and_combine(self, x: torch.Tensor, y_slots: torch.Tensor, topk_w: torch.Tensor) -> torch.Tensor:
        w13, w2 = self.shared_experts.fp16_weights()
        if self.impl == "torch":
            h = (x.float() @ w13.float().t()).half()
            shared = mk.swiglu_fp32_reference(h, self.swiglu_limit).float() @ w2.float().t()
            return mk.combine_reference(y_slots, topk_w, shared)
        if self.decode_gemv and x.shape[0] <= v41_gemv.MAX_GEMV_ROWS:
            act = v41_gemv.gate_up_swiglu(x, w13, self.swiglu_limit)
            return v41_gemv.down_combine(act, w2, y_slots, topk_w)
        h = torch.mm(x, w13.t(), out_dtype=torch.float32).half()
        shared = torch.mm(mk.swiglu_fp32(h, self.swiglu_limit), w2.t(), out_dtype=torch.float32)
        return mk.combine_fp32(y_slots, topk_w, shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != self.hidden:
            raise TypeError(f"DeepseekV41MoE expects fp16 [T, {self.hidden}], got {x.dtype} {tuple(x.shape)}")
        x = x.contiguous()
        if x.shape[0] == 0:
            return x.new_empty((0, self.hidden), dtype=torch.float32)
        topk_w, topk_ids = self.route(self.gate_logits(x))
        if self.topk_check and self.tp_size > 1:
            self._check_topk_consistent(topk_ids)
        y_slots = self.method.expert_slots(self.experts, x, topk_ids)
        out = self.shared_and_combine(x, y_slots, topk_w)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _check_topk_consistent(self, topk_ids: torch.Tensor) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{TOPK_CHECK_KNOB}=1 syncs the host and cannot run under CUDA-graph capture; "
                               "run the debug check with enforce_eager")
        pos = torch.arange(1, topk_ids.numel() + 1, device=topk_ids.device, dtype=torch.int64)
        checksum = (topk_ids.reshape(-1).to(torch.int64) * pos).sum().view(1)
        gathered = get_tp_group().all_gather(checksum, dim=0)
        if not bool((gathered == gathered[0]).all().item()):
            raise RuntimeError(f"DeepseekV41MoE layer {self.layer_id}: router ids differ across TP ranks "
                               f"(checksums {gathered.tolist()}); TP-replicated inputs must be bitwise equal (§4.1)")


def spill_plan_from_ranking(num_spilled_per_layer: int, counts: np.ndarray) -> ExpertSpillPlan:
    """Spill the least-routed experts given per-expert routing counts (L-REF histogram)."""
    order = np.argsort(-np.asarray(counts, dtype=np.int64), kind="stable")
    return make_spill_plan(num_spilled_per_layer, len(order), ranking=order)
