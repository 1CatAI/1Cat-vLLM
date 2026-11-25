# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Inkling2 routed MoE for SM70.

Router: fp32 sigmoid scores plus a selection-only correction bias, top-8,
weights normalized and scaled by `route_scale`.

Experts (TP-sharded along the intermediate dimension), by checkpoint format:
- NVFP4 (layers 0-19): the fork's TurboMind W4A16 method
  (`ModelOptNvFp4SM70MoEMethod`) on a `FusedMoE` layer. At load, w13's
  interleaved gate/up rows become `[gate; up]` halves and the per-expert
  global scale is `amax / outer_scale_denominator` (ModelOpt
  `weight_scale_2`). The activation is the softcapped SwiGLU, plugged into
  the method's FP4 codec (`sm70_moe_activation`).
- MXFP8 (layers 20-23): dequantized to fp16 once at load (exact for this
  checkpoint); decode-size batches run `moe_gemv`, larger ones per-expert
  cuBLAS GEMMs.
`VLLM_INKLING2_SM70_MOE=ref` keeps every layer on the packed reference path
(`ops.moe_ref`, decode-on-the-fly), the checker of the fast paths.
"""

import os
from functools import partial
from typing import Literal

import torch
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.utils import set_weight_attrs

from ..configs import Inkling2ModelConfig
from .ops import dense_moe_prefill as _dmp
from .ops import moe_fused as _moe_fused
from .ops.moe import moe_gemv, moe_gemv_swiglu, moe_weighted_sum, softcapped_swiglu_
from .ops.moe_ref import ExpertWeights, dequant_mxfp8, experts_forward
from .ops.router import fused_route
from .ops.router import supported as _fused_router_supported

QuantFormat = Literal["nvfp4", "mxfp8"]

_BACKEND = os.environ.get("VLLM_INKLING2_SM70_MOE", "auto")
# Batches up to this many tokens run the dense experts as GEMVs.
_DENSE_GEMV_MAX_TOKENS = 8
# Order of the 8-slot weighted sum of `forward_decode` (0 sequential, 1 four interleaved partials):
# whichever is bitwise torch's `sum(1)`, found by test_mxfp8.py.
_WSUM_ORDER = int(os.environ.get("VLLM_INKLING2_SM70_MXFP8_WSUM_ORDER", "1"))


def _rows_to_halves(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """`[E, 2I, X]` with gate/up rows interleaved -> this rank's `[E, 2I/tp, X]`
    as `[gate; up]` halves."""
    gate, up = t[:, 0::2], t[:, 1::2]
    n = gate.shape[1] // tp_size
    sl = slice(tp_rank * n, (tp_rank + 1) * n)
    return torch.cat((gate[:, sl], up[:, sl]), dim=1)


def _shard_last(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    n = t.shape[-1] // tp_size
    return t[..., tp_rank * n : (tp_rank + 1) * n]


def _copy(param: torch.Tensor, loaded: torch.Tensor) -> None:
    assert tuple(loaded.shape) == tuple(param.shape), (
        tuple(loaded.shape),
        tuple(param.shape),
    )
    if loaded.dtype != param.dtype and loaded.element_size() == param.element_size():
        loaded = loaded.view(param.dtype)
    param.data.copy_(loaded)


class Inkling2Gate(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int) -> None:
        super().__init__()
        # Routing runs in fp32 (the checkpoint stores the weight in bf16).
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32), requires_grad=False
        )


class Inkling2Experts(nn.Module):
    """Packed checkpoint experts of one layer (reference path, and the load
    staging of the MXFP8 dense path)."""

    def __init__(
        self, num_experts: int, hidden: int, inter: int, fmt: QuantFormat
    ) -> None:
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        assert inter % tp == 0
        self.fmt = fmt
        self.inter = inter // tp
        rows13 = 2 * self.inter

        def param(shape, dtype, shard_dim):
            p = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(p, {"weight_loader": self._loader(shard_dim)})
            return p

        if fmt == "nvfp4":
            self.w13_weight = param((num_experts, rows13, hidden // 2), torch.uint8, 1)
            self.w13_weight_scale = param(
                (num_experts, rows13, hidden // 16), torch.float8_e4m3fn, 1
            )
            self.w2_weight = param(
                (num_experts, hidden, self.inter // 2), torch.uint8, 2
            )
            self.w2_weight_scale = param(
                (num_experts, hidden, self.inter // 16), torch.float8_e4m3fn, 2
            )
            self.w13_weight_amax = param((num_experts,), torch.float32, None)
            self.w2_weight_amax = param((num_experts,), torch.float32, None)
            self.w13_weight_outer_scale_denominator = param((1,), torch.float32, None)
            self.w2_weight_outer_scale_denominator = param((1,), torch.float32, None)
        else:
            self.w13_weight = param(
                (num_experts, rows13, hidden), torch.float8_e4m3fn, 1
            )
            self.w13_weight_scale = param(
                (num_experts, rows13, hidden // 32), torch.uint8, 1
            )
            self.w2_weight = param(
                (num_experts, hidden, self.inter), torch.float8_e4m3fn, 2
            )
            self.w2_weight_scale = param(
                (num_experts, hidden, self.inter // 32), torch.uint8, 2
            )
        self._weights: ExpertWeights | None = None

    def _loader(self, shard_dim: int | None):
        def load(param: nn.Parameter, loaded: torch.Tensor) -> None:
            if (
                shard_dim is not None
                and loaded.shape[shard_dim] != param.shape[shard_dim]
            ):
                size = param.shape[shard_dim]
                loaded = loaded.narrow(shard_dim, self.tp_rank * size, size)
            _copy(param, loaded)

        return load

    def inkling_post_load(self) -> None:
        if self.fmt == "nvfp4":
            self._weights = ExpertWeights(
                "nvfp4",
                self.w13_weight,
                self.w13_weight_scale,
                self.w2_weight,
                self.w2_weight_scale,
                self.w13_weight_amax / self.w13_weight_outer_scale_denominator,
                self.w2_weight_amax / self.w2_weight_outer_scale_denominator,
            )
        else:
            self._weights = ExpertWeights(
                "mxfp8",
                self.w13_weight,
                self.w13_weight_scale,
                self.w2_weight,
                self.w2_weight_scale,
            )

    @property
    def weights(self) -> ExpertWeights:
        if self._weights is None:
            self.inkling_post_load()
        assert self._weights is not None
        return self._weights


class Inkling2DenseExperts(Inkling2Experts):
    """MXFP8 experts dequantized once to resident fp16 `[gate; up]` halves."""

    def __init__(self, num_experts: int, hidden: int, inter: int) -> None:
        super().__init__(num_experts, hidden, inter, "mxfp8")
        self.w13_fp16: torch.Tensor | None = None
        self.w2_fp16: torch.Tensor | None = None

    def inkling_post_load(self) -> None:
        dtype = torch.float16
        w13 = torch.empty(
            (
                self.w13_weight.shape[0],
                self.w13_weight.shape[1],
                self.w13_weight.shape[2],
            ),
            dtype=dtype,
            device=self.w13_weight.device,
        )
        w2 = torch.empty(
            self.w2_weight.shape, dtype=dtype, device=self.w2_weight.device
        )
        for e in range(self.w13_weight.shape[0]):  # bounded temporaries
            full = dequant_mxfp8(
                self.w13_weight[e : e + 1], self.w13_weight_scale[e : e + 1], dtype
            )
            w13[e] = torch.cat((full[0, 0::2], full[0, 1::2]), dim=0)
            w2[e] = dequant_mxfp8(
                self.w2_weight[e : e + 1], self.w2_weight_scale[e : e + 1], dtype
            )[0]
        self.w13_fp16, self.w2_fp16 = w13, w2
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            getattr(self, name).data = torch.empty(0, device=w13.device)
        torch.accelerator.empty_cache()

    def forward_decode(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor:
        """`forward` for decode batches (VLLM_INKLING2_SM70_MXFP8_FUSE): W13 + SwiGLU in one launch,
        W2, then the weighted sum + fp16 cast in one launch (the same bits as the 7-launch path)."""
        assert self.w13_fp16 is not None and self.w2_fp16 is not None
        num_tokens, hidden = x.shape
        flat = ids.reshape(-1)
        slots = flat.shape[0]
        act = x.new_empty((slots, self.w2_fp16.shape[2]))
        # Tile / warps measured on cold weights (test_mxfp8.py BN / NW): every choice is bitwise the same.
        small = num_tokens <= 3
        moe_gemv_swiglu(
            x,
            self.w13_fp16,
            flat,
            act,
            ids.shape[1],
            gate_cap,
            up_cap,
            block_n=8 if small else 32,
            num_warps=4 if small else 8,
        )
        down = torch.empty((slots, hidden), dtype=torch.float32, device=x.device)
        moe_gemv(act, self.w2_fp16, flat, down, 1)
        return moe_weighted_sum(
            down, weights, x.new_empty((num_tokens, hidden)), _WSUM_ORDER
        )

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor:
        assert self.w13_fp16 is not None and self.w2_fp16 is not None
        num_tokens, hidden = x.shape
        top_k = ids.shape[1]
        inter = self.w2_fp16.shape[2]
        if num_tokens <= _DENSE_GEMV_MAX_TOKENS:
            flat = ids.reshape(-1)
            slots = flat.shape[0]
            gate_up = x.new_empty((slots, 2 * inter))
            moe_gemv(x, self.w13_fp16, flat, gate_up, top_k)
            act = x.new_empty((slots, inter))
            softcapped_swiglu_(act, gate_up, gate_cap, up_cap)
            down = torch.empty((slots, hidden), dtype=torch.float32, device=x.device)
            moe_gemv(act, self.w2_fp16, flat, down, 1)
            out = (down.view(num_tokens, top_k, hidden) * weights.unsqueeze(-1)).sum(1)
            return out.to(x.dtype)
        if _dmp.ENABLED and _dmp.supported(x, self.w13_fp16, self.w2_fp16, ids):
            # VLLM_INKLING2_SM70_DENSE_MOE_PREFILL: GPU-side routing + grouped tensor-core GEMMs.
            return _dmp.dense_moe_prefill(
                x,
                weights,
                ids.to(torch.int32),
                self.w13_fp16,
                self.w2_fp16,
                gate_cap,
                up_cap,
            )
        # Prefill: per-expert cuBLAS GEMMs over the tokens routed to each expert.
        out = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=x.device)
        flat = ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        token_of = order // top_k
        weight_of = weights.reshape(-1)[order]
        counts = torch.bincount(flat, minlength=self.w13_fp16.shape[0]).tolist()
        start = 0
        for e, c in enumerate(counts):
            if not c:
                continue
            toks = token_of[start : start + c]
            gate_up = nn.functional.linear(x[toks], self.w13_fp16[e])
            act = x.new_empty((c, inter))
            softcapped_swiglu_(act, gate_up, gate_cap, up_cap)
            down = nn.functional.linear(act, self.w2_fp16[e]).float()
            out.index_add_(0, toks, down * weight_of[start : start + c, None])
            start += c
        return out.to(x.dtype)


class Inkling2MoE(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        fmt: QuantFormat,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
        self.fmt = fmt
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.gate = Inkling2Gate(config.n_routed_experts, config.hidden_size)
        self._fused: _moe_fused.FusedState | None = None
        self._route: _moe_fused.RouteState | None = None
        num_experts, hidden, inter = (
            config.n_routed_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        if _BACKEND == "ref":
            self.impl = "ref"
            self.experts = Inkling2Experts(num_experts, hidden, inter, fmt)
        elif fmt == "mxfp8":
            self.impl = "dense"
            self.experts = Inkling2DenseExperts(num_experts, hidden, inter)
        else:
            from vllm.model_executor.layers.fused_moe import FusedMoE

            self.impl = "turbomind"
            assert quant_config is not None
            self.experts = FusedMoE(
                num_experts=num_experts,
                top_k=self.top_k,
                hidden_size=hidden,
                intermediate_size=inter,
                params_dtype=torch.float16,
                renormalize=False,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
                activation="silu",
            )
            self.experts.sm70_moe_activation = partial(
                softcapped_swiglu_, gate_cap=self.gate_cap, up_cap=self.up_cap
            )
            self._wire_nvfp4_loading()

    # ---- NVFP4 checkpoint -> ModelOpt NVFP4 layout of the FusedMoE layer ----
    def _wire_nvfp4_loading(self) -> None:
        experts = self.experts
        num_experts = experts.w13_weight.shape[0]
        for name in ("w13_weight_amax", "w2_weight_amax"):
            p = nn.Parameter(
                torch.empty(num_experts, dtype=torch.float32), requires_grad=False
            )
            experts.register_parameter(name, p)
        for name in (
            "w13_weight_outer_scale_denominator",
            "w2_weight_outer_scale_denominator",
        ):
            p = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
            experts.register_parameter(name, p)
        rank, tp = self.tp_rank, self.tp_size
        loaders = {
            "w13_weight": lambda p, w: _copy(p, _rows_to_halves(w, rank, tp)),
            "w13_weight_scale": lambda p, w: _copy(p, _rows_to_halves(w, rank, tp)),
            "w2_weight": lambda p, w: _copy(p, _shard_last(w, rank, tp)),
            "w2_weight_scale": lambda p, w: _copy(p, _shard_last(w, rank, tp)),
            "w13_weight_amax": _copy,
            "w2_weight_amax": _copy,
            "w13_weight_outer_scale_denominator": _copy,
            "w2_weight_outer_scale_denominator": _copy,
        }
        for name, loader in loaders.items():
            getattr(experts, name).weight_loader = loader
        # Derived from amax / denominator in `inkling_post_load`.
        for name in (
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            if hasattr(experts, name):
                getattr(experts, name).inkling_derived = True

    def inkling_post_load(self) -> None:
        if self.impl == "dense":
            # VLLM_INKLING2_SM70_MXFP8_FUSE: scratch of the route-only launch.
            self._route = _moe_fused.make_route_state(self.gate.weight, self.top_k)
            return
        if self.impl != "turbomind":
            return
        e = self.experts
        g13 = (e.w13_weight_amax / e.w13_weight_outer_scale_denominator).float()
        g2 = (e.w2_weight_amax / e.w2_weight_outer_scale_denominator).float()
        e.w13_weight_scale_2.data.copy_(g13.view(-1, 1).expand_as(e.w13_weight_scale_2))
        e.w2_weight_scale_2.data.copy_(g2.view_as(e.w2_weight_scale_2))
        for name in ("w13_input_scale", "w2_input_scale"):
            if hasattr(e, name):
                getattr(e, name).data.fill_(1.0)
        # VLLM_INKLING2_SM70_MOE_FUSED: scratch of the one-launch decode visit.
        self._fused = _moe_fused.make_state(
            self.gate.weight, self.top_k, 2 * e.w2_weight.shape[2]
        )

    def forward(
        self, x: torch.Tensor, reduce: bool = True, pass_idx: int = 0
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_empty(x.shape)
        fused = self._fused
        if (
            fused is not None
            and x.shape[0] <= _moe_fused.MAX_TOKENS
            and x.dtype == torch.float16
            and x.is_contiguous()
            and _moe_fused.ready(fused, self.experts)
        ):
            # NVFP4 layers: router + experts + weighted reduce in one launch (M <= 8).
            out = _moe_fused.moe_fused(
                x,
                self.gate.weight,
                self.gate.bias,
                self.route_scale,
                self.experts,
                self.gate_cap,
                self.up_cap,
                fused,
            )
            if reduce and self.tp_size > 1:
                out = tensor_model_parallel_all_reduce(out)
            return out
        route = self._route
        if (
            route is not None
            and x.shape[0] <= _moe_fused.MAX_TOKENS
            and x.dtype == torch.float16
            and x.is_contiguous()
            and self.impl == "dense"
        ):
            # MXFP8 layers (VLLM_INKLING2_SM70_MXFP8_FUSE): route-only launch, then the fused experts.
            weights, ids = _moe_fused.route(
                x, self.gate.weight, self.gate.bias, self.route_scale, route
            )
            out = self.experts.forward_decode(
                x, weights, ids, self.gate_cap, self.up_cap
            )
            if reduce and self.tp_size > 1:
                out = tensor_model_parallel_all_reduce(out)
            return out
        if _fused_router_supported(x, self.gate.weight, self.top_k):
            weights, ids = fused_route(
                x, self.gate.weight, self.gate.bias, self.route_scale
            )
        else:
            weights, ids = route(
                x, self.gate.weight, self.gate.bias, self.top_k, self.route_scale
            )
        if self.impl == "turbomind":
            out = self.experts.quant_method.apply(
                self.experts, x, weights, ids.to(torch.int32), None, None
            )
        elif self.impl == "dense":
            out = self.experts(x, weights, ids, self.gate_cap, self.up_cap)
        else:
            out = experts_forward(
                x, weights, ids, self.experts.weights, self.gate_cap, self.up_cap
            )
        if reduce and self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out
