# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 (Volta) Inkling2 routed experts for NVFP4 (L0..19) and MXFP8 (L20..23).

Uses `1Cat-vLLM`'s native TurboMind `Fp4MoECodec` / `nvfp4_moe_dense_stage_sm70_out`
for NVFP4 layers and FP16 dequantized weights + Triton `grouped_gemm` for the four
MXFP8 coda layers, with EP=TP sharding via `moe_permute_with_scratch` and a
CUDA-graph-safe `valid_num`-checked Triton unpermute reduction.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm import _sm70_ops as sm70_ops
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.sm70.fp4_codec import Fp4MoECodec
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    NVFP4_GROUP_SIZE,
    unpack_mxfp4_weight,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton

from .activation import softcapped_situ_sm70
from .grouped_gemm import grouped_gemm

_GRAPH_MAX_TOKENS = 18
_COMPACT_MAX_SLOTS = 80


@triton.jit
def _prepare_compact_ep_slots_kernel(
    sorted_expert_ids_ptr,
    compact_offsets_ptr,
    active_expert_ids_ptr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    TOTAL_SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < TOTAL_SLOTS
    expert_ids = tl.load(sorted_expert_ids_ptr + offsets, mask=valid, other=0)
    clamped = tl.where(
        (expert_ids >= 0) & (expert_ids < NUM_LOCAL_EXPERTS), expert_ids, 0
    )
    tl.store(compact_offsets_ptr + offsets, offsets, mask=offsets <= TOTAL_SLOTS)
    tl.store(active_expert_ids_ptr + offsets, clamped, mask=valid)


def _prepare_compact_ep_slots(
    sorted_expert_ids: torch.Tensor,
    compact_offsets: torch.Tensor,
    active_expert_ids: torch.Tensor,
    num_local_experts: int,
) -> None:
    total_slots = sorted_expert_ids.numel()
    block = triton.next_power_of_2(total_slots + 1)
    _prepare_compact_ep_slots_kernel[(1,)](
        sorted_expert_ids,
        compact_offsets,
        active_expert_ids,
        NUM_LOCAL_EXPERTS=num_local_experts,
        TOTAL_SLOTS=total_slots,
        BLOCK=block,
        num_warps=1,
    )


@triton.jit
def _fp16_moe_gemv_sm70_kernel(
    X,
    W,
    Y,
    ExpertIds,
    ValidNumPtr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Per-slot FP16 GEMV for small-batch MXFP8/FP16 MoE layers on SM70."""
    r = tl.program_id(0)
    valid_num = tl.load(ValidNumPtr)
    if r >= valid_num:
        return
    n_tile = tl.program_id(1)
    e = tl.load(ExpertIds + r)
    cols = n_tile * BN + tl.arange(0, BN)
    cmask = cols < N
    ks = tl.arange(0, BK)
    acc = tl.zeros((BN,), dtype=tl.float32)
    x_ptr = X + r * K + ks
    w_base = W + (e.to(tl.int64) * N + cols[:, None]) * K + ks[None, :]
    for k in range(0, K, BK):
        km = (k + ks) < K
        xv = tl.load(x_ptr, mask=km, other=0.0).to(tl.float32)
        wv = tl.load(w_base, mask=cmask[:, None] & km[None, :], other=0.0).to(
            tl.float32
        )
        acc += tl.sum(wv * xv[None, :], axis=1)
        x_ptr += BK
        w_base += BK
    tl.store(Y + r * N + cols, acc.to(Y.dtype.element_ty), mask=cmask)


def _fp16_moe_gemv_sm70(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    expert_ids: torch.Tensor,
    valid_num_ptr: torch.Tensor,
) -> None:
    slots, k = x.shape
    n = weight.shape[1]
    if slots == 0:
        return
    bn, bk = 16, 128
    _fp16_moe_gemv_sm70_kernel[(slots, triton.cdiv(n, bn))](
        x,
        weight,
        out,
        expert_ids,
        valid_num_ptr,
        N=n,
        K=k,
        BN=bn,
        BK=bk,
        num_warps=4,
    )


@triton.jit
def _unpermute_reduce_sm70_kernel(
    SortedOut,
    Weights,
    InvPerm,
    ValidNumPtr,
    Out,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum local expert rows per token weighted by `Weights` in FP32."""
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    valid_num = tl.load(ValidNumPtr)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        idx = token * TOPK + k
        row = tl.load(InvPerm + idx).to(tl.int64)
        w = tl.load(Weights + idx).to(tl.float32)
        in_range = (row >= 0) & (row < valid_num)
        safe_row = tl.where(in_range, row, 0)
        val = tl.load(
            SortedOut + safe_row * H + cols,
            mask=mask & in_range,
            other=0.0,
        ).to(tl.float32)
        acc += val * w
    tl.store(Out + token * H + cols, acc.to(Out.dtype.element_ty), mask=mask)


def _unpermute_reduce_sm70(
    sorted_output: torch.Tensor,
    weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_offsets64: torch.Tensor,
    num_local_experts: int,
    out: torch.Tensor,
) -> None:
    tokens, hidden = out.shape
    topk = inv_permuted_idx.shape[1]
    if tokens == 0:
        return
    block = 256 if hidden % 256 == 0 else 128
    _unpermute_reduce_sm70_kernel[(tokens, triton.cdiv(hidden, block))](
        sorted_output,
        weights,
        inv_permuted_idx,
        expert_offsets64[num_local_experts:],
        out,
        H=hidden,
        TOPK=topk,
        BLOCK=block,
        num_warps=4,
    )


class Inkling2Sm70Experts(nn.Module):
    """Volta SM70 routed experts for Inkling2 (EP=TP over NVLink)."""

    supports_lora = False

    def __init__(self, config, quant_format: str) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.global_num_experts = int(config.n_routed_experts)
        self.num_experts = self.global_num_experts // self.ep_size
        self.topk = int(config.num_experts_per_tok)
        self.gate_cap = float(config.mlp_gate_softcap)
        self.up_cap = float(config.mlp_up_softcap)
        self.nvfp4 = quant_format == "nvfp4"
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.intermediate_size)

        pack, group = (2, 16) if self.nvfp4 else (1, 32)
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        self.shapes = (
            (2 * self.intermediate_size, self.hidden_size),
            (self.hidden_size, self.intermediate_size),
        )
        for name, (n, k) in zip(("w13", "w2"), self.shapes):
            self._register(
                f"{name}_weight",
                (n, k // pack),
                weight_dtype,
                loader=self._load_expert_shard,
            )
            self._register(
                f"{name}_weight_scale",
                (n, k // group),
                scale_dtype,
                loader=self._load_expert_shard,
            )
            if self.nvfp4:
                self._register(f"{name}_weight_amax", (), torch.float32)
                self._register(
                    f"{name}_weight_outer_scale_denominator",
                    (1,),
                    torch.float32,
                    per_expert=False,
                )

        start = self.ep_rank * self.num_experts
        expert_map = torch.full((self.global_num_experts,), -1, dtype=torch.int32)
        expert_map[start : start + self.num_experts] = torch.arange(
            self.num_experts, dtype=torch.int32
        )
        self.register_buffer("expert_map", expert_map, persistent=False)
        self.register_buffer(
            "expert_bounds",
            torch.tensor([start, start + self.num_experts], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "dense_expert_ids",
            torch.arange(self.num_experts, dtype=torch.int32),
            persistent=False,
        )
        self._prepared = False
        self._scratch: dict[str, torch.Tensor] | None = None

    def _register(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        per_expert: bool = True,
        loader=None,
    ) -> None:
        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        if loader is None:
            loader = self._load_expert_shard if per_expert else self._load_full
        set_weight_attrs(param, {"weight_loader": loader})
        self.register_parameter(name, param)

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        start = self.ep_rank * self.num_experts
        self._load_full(param, loaded[start : start + self.num_experts])

    @staticmethod
    def _load_full(param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded)

    def _allocate_scratch(self, device: torch.device) -> None:
        max_slots = _GRAPH_MAX_TOKENS * self.topk
        ws_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            max_slots, self.global_num_experts
        )
        self._scratch = {
            "output": torch.empty(
                _GRAPH_MAX_TOKENS, self.hidden_size, dtype=torch.float16, device=device
            ),
            "permuted_input": torch.empty(
                max_slots, self.hidden_size, dtype=torch.float16, device=device
            ),
            "gate_up": torch.empty(
                max_slots,
                2 * self.intermediate_size,
                dtype=torch.float16,
                device=device,
            ),
            "intermediate": torch.empty(
                max_slots, self.intermediate_size, dtype=torch.float16, device=device
            ),
            "sorted_output": torch.empty(
                max_slots, self.hidden_size, dtype=torch.float16, device=device
            ),
            "expert_offsets": torch.empty(
                self.num_experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                self.num_experts + 1, dtype=torch.int64, device=device
            ),
            "inv_permuted_idx": torch.empty(
                _GRAPH_MAX_TOKENS, self.topk, dtype=torch.int32, device=device
            ),
            "topk_ids": torch.empty(
                _GRAPH_MAX_TOKENS, self.topk, dtype=torch.int32, device=device
            ),
            "token_expert_indices": torch.arange(
                max_slots, dtype=torch.int32, device=device
            ).view(_GRAPH_MAX_TOKENS, self.topk),
            "permuted_idx": torch.empty(max_slots, dtype=torch.int32, device=device),
            "permuted_experts_id": torch.empty(
                max_slots, dtype=torch.int32, device=device
            ),
            "sorted_row_idx": torch.empty(max_slots, dtype=torch.int32, device=device),
            "topk_ids_for_sort": torch.empty(
                max_slots, dtype=torch.int32, device=device
            ),
            "sort_workspace": torch.empty(ws_size, dtype=torch.int8, device=device),
            "compact_offsets": torch.arange(
                max_slots + 1, dtype=torch.int32, device=device
            ),
            "active_expert_ids": torch.empty(
                max_slots, dtype=torch.int32, device=device
            ),
        }

    def _get_buffers(
        self, num_tokens: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        slots = num_tokens * self.topk
        if num_tokens <= _GRAPH_MAX_TOKENS and self._scratch is not None:
            s = self._scratch
            return {
                "output": s["output"][:num_tokens],
                "permuted_input": s["permuted_input"][:slots],
                "gate_up": s["gate_up"][:slots],
                "intermediate": s["intermediate"][:slots],
                "sorted_output": s["sorted_output"][:slots],
                "expert_offsets": s["expert_offsets"],
                "expert_offsets64": s["expert_offsets64"],
                "inv_permuted_idx": s["inv_permuted_idx"][:num_tokens],
                "topk_ids": s["topk_ids"][:num_tokens],
                "token_expert_indices": s["token_expert_indices"][:num_tokens],
                "permuted_idx": s["permuted_idx"][:slots],
                "permuted_experts_id": s["permuted_experts_id"][:slots],
                "sorted_row_idx": s["sorted_row_idx"][:slots],
                "topk_ids_for_sort": s["topk_ids_for_sort"][:slots],
                "sort_workspace": s["sort_workspace"],
                "compact_offsets": s["compact_offsets"][: slots + 1],
                "active_expert_ids": s["active_expert_ids"][:slots],
            }
        ws_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            slots, self.global_num_experts
        )
        return {
            "output": torch.empty(
                num_tokens, self.hidden_size, dtype=torch.float16, device=device
            ),
            "permuted_input": torch.empty(
                slots, self.hidden_size, dtype=torch.float16, device=device
            ),
            "gate_up": torch.empty(
                slots, 2 * self.intermediate_size, dtype=torch.float16, device=device
            ),
            "intermediate": torch.empty(
                slots, self.intermediate_size, dtype=torch.float16, device=device
            ),
            "sorted_output": torch.empty(
                slots, self.hidden_size, dtype=torch.float16, device=device
            ),
            "expert_offsets": torch.empty(
                self.num_experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                self.num_experts + 1, dtype=torch.int64, device=device
            ),
            "inv_permuted_idx": torch.empty(
                num_tokens, self.topk, dtype=torch.int32, device=device
            ),
            "topk_ids": torch.empty(
                num_tokens, self.topk, dtype=torch.int32, device=device
            ),
            "token_expert_indices": torch.arange(
                slots, dtype=torch.int32, device=device
            ).view(num_tokens, self.topk),
            "permuted_idx": torch.empty(slots, dtype=torch.int32, device=device),
            "permuted_experts_id": torch.empty(slots, dtype=torch.int32, device=device),
            "sorted_row_idx": torch.empty(slots, dtype=torch.int32, device=device),
            "topk_ids_for_sort": torch.empty(slots, dtype=torch.int32, device=device),
            "sort_workspace": torch.empty(ws_size, dtype=torch.int8, device=device),
            "compact_offsets": torch.arange(
                slots + 1, dtype=torch.int32, device=device
            ),
            "active_expert_ids": torch.empty(slots, dtype=torch.int32, device=device),
        }

    def process_weights_after_loading(self) -> None:
        if self._prepared:
            return
        device = self.w13_weight.device
        if self.nvfp4:
            for name in ("w13", "w2"):
                raw_w = getattr(self, f"{name}_weight").data
                raw_s = getattr(self, f"{name}_weight_scale").data
                amax = getattr(self, f"{name}_weight_amax").data.float()
                denom = getattr(
                    self, f"{name}_weight_outer_scale_denominator"
                ).data.float()
                global_scale = amax / denom

                tm_weights: list[torch.Tensor] = []
                tm_scales: list[torch.Tensor] = []
                metas: list[torch.Tensor] = []
                for eid in range(self.num_experts):
                    packed = unpack_mxfp4_weight(raw_w[eid])
                    scales = (
                        (raw_s[eid].float() * global_scale[eid]).half().t().contiguous()
                    )
                    tw, ts, meta = Fp4MoECodec.prepare_weights(
                        "nvfp4",
                        packed,
                        scales,
                        NVFP4_GROUP_SIZE,
                        interleave_gated_silu=False,
                    )
                    tm_weights.append(tw)
                    tm_scales.append(ts)
                    metas.append(meta)

                stacked_w = torch.stack(tm_weights)
                stacked_s = torch.stack(tm_scales)
                k_ld = int(metas[0][0].item())
                q_ld = int(metas[0][1].item())
                ptrs = sm70_ops.awq_moe_build_strided_ptrs(
                    stacked_w, stacked_s, k_ld, q_ld, self.num_experts
                )
                self.register_buffer(f"{name}_tm_weight", stacked_w, persistent=False)
                self.register_buffer(f"{name}_tm_scales", stacked_s, persistent=False)
                self.register_buffer(f"{name}_ptrs_w", ptrs[0], persistent=False)
                self.register_buffer(f"{name}_ptrs_s", ptrs[1], persistent=False)

                delattr(self, f"{name}_weight")
                delattr(self, f"{name}_weight_scale")
                delattr(self, f"{name}_weight_amax")
                delattr(self, f"{name}_weight_outer_scale_denominator")
        else:
            for name in ("w13", "w2"):
                raw_w = getattr(self, f"{name}_weight").data
                raw_s = getattr(self, f"{name}_weight_scale").data
                exp2 = torch.exp2(raw_s.float() - 127.0).repeat_interleave(32, dim=-1)
                fp16_w = (raw_w.float() * exp2).to(torch.float16).contiguous()
                self.register_buffer(f"{name}_fp16", fp16_w, persistent=False)
                delattr(self, f"{name}_weight")
                delattr(self, f"{name}_weight_scale")

        self._allocate_scratch(device)
        torch.accelerator.empty_cache()
        self._prepared = True

    def forward(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        if not self._prepared:
            self.process_weights_after_loading()
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return x.new_empty((0, self.hidden_size))
        assert x.dtype == torch.float16, f"SM70 experts expect float16, got {x.dtype}"
        buffers = self._get_buffers(num_tokens, x.device)
        slots = num_tokens * self.topk
        topk_ids_i32 = buffers["topk_ids"]
        topk_ids_i32.copy_(ids, non_blocking=True)
        buffers["permuted_idx"].fill_(slots)

        torch.ops._moe_C.moe_permute_with_scratch(
            x,
            topk_ids_i32,
            buffers["token_expert_indices"],
            self.expert_map,
            self.global_num_experts,
            self.num_experts,
            self.topk,
            buffers["permuted_input"],
            buffers["expert_offsets64"],
            buffers["inv_permuted_idx"],
            buffers["permuted_idx"],
            buffers["sort_workspace"],
            buffers["permuted_experts_id"],
            buffers["sorted_row_idx"],
            buffers["topk_ids_for_sort"],
        )
        buffers["expert_offsets"].copy_(buffers["expert_offsets64"], non_blocking=True)

        if self.nvfp4:
            if slots <= _COMPACT_MAX_SLOTS:
                _prepare_compact_ep_slots(
                    buffers["permuted_experts_id"],
                    buffers["compact_offsets"],
                    buffers["active_expert_ids"],
                    self.num_experts,
                )
                offsets = buffers["compact_offsets"]
                expert_ids = buffers["active_expert_ids"]
                stage_experts = slots
            else:
                offsets = buffers["expert_offsets"]
                expert_ids = self.dense_expert_ids
                stage_experts = self.num_experts

            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                offsets,
                expert_ids,
                self.w13_ptrs_w,
                self.w13_ptrs_s,
                stage_experts,
                self.hidden_size,
                2 * self.intermediate_size,
                NVFP4_GROUP_SIZE,
            )
            softcapped_situ_sm70(
                buffers["gate_up"],
                buffers["intermediate"],
                self.gate_cap,
                self.up_cap,
            )
            sm70_ops.nvfp4_moe_dense_stage_sm70_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                offsets,
                expert_ids,
                self.w2_ptrs_w,
                self.w2_ptrs_s,
                stage_experts,
                self.intermediate_size,
                self.hidden_size,
                NVFP4_GROUP_SIZE,
            )
        else:
            if slots <= 64:
                valid_num_ptr = buffers["expert_offsets64"][self.num_experts :]
                _fp16_moe_gemv_sm70(
                    buffers["permuted_input"],
                    self.w13_fp16,
                    buffers["gate_up"],
                    buffers["permuted_experts_id"],
                    valid_num_ptr,
                )
                softcapped_situ_sm70(
                    buffers["gate_up"],
                    buffers["intermediate"],
                    self.gate_cap,
                    self.up_cap,
                )
                _fp16_moe_gemv_sm70(
                    buffers["intermediate"],
                    self.w2_fp16,
                    buffers["sorted_output"],
                    buffers["permuted_experts_id"],
                    valid_num_ptr,
                )
            else:
                grouped_gemm(
                    buffers["permuted_input"],
                    self.w13_fp16,
                    buffers["sorted_row_idx"],
                    buffers["expert_offsets"],
                    slots,
                    self.topk,
                    gather=False,
                    out=buffers["gate_up"],
                    accumulate=False,
                )
                softcapped_situ_sm70(
                    buffers["gate_up"],
                    buffers["intermediate"],
                    self.gate_cap,
                    self.up_cap,
                )
                grouped_gemm(
                    buffers["intermediate"],
                    self.w2_fp16,
                    buffers["sorted_row_idx"],
                    buffers["expert_offsets"],
                    slots,
                    self.topk,
                    gather=False,
                    out=buffers["sorted_output"],
                    accumulate=False,
                )

        _unpermute_reduce_sm70(
            buffers["sorted_output"],
            weights,
            buffers["inv_permuted_idx"],
            buffers["expert_offsets64"],
            self.num_experts,
            buffers["output"],
        )
        out = buffers["output"]
        if not reduce or self.ep_size == 1:
            return out
        return tensor_model_parallel_all_reduce(out)
