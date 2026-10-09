# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-expert method of the DeepSeek-V4.1 SM70 MoE (lane L-MOE, PORT_DESIGN §3.4, §5.2).

``DeepseekV41Mxfp4MoEMethod`` owns the expert tensors of one ``FusedMoE`` layer (checkpoint layout E2M1 nibble
pairs + E8M0 per 32, TP-sharded on the intermediate dimension) and computes per-route expert outputs
``y[t * top_k + j] = expert_{ids[t, j]}(x[t])`` (FP16, TP-partial) through ``expert_slots``; ``DeepseekV41MoE``
weights and combines them in FP32 with the shared expert. ``FusedMoE.forward`` is never used.

Backends (``backend``):

* ``skinny`` (default): the grouped QPN tensor-core kernels (``skinny_moe_qpn_sm70``) on fragment-order weights,
  permuted in place at load (no second copy). Per-expert E8M0 rebase keeps the FP16 decode exact.
* ``turbomind``: the generic grouped TurboMind MXFP4 stages of ``Mxfp4SM70MoEMethod`` (``apply_slots``).
* ``torch``: per-expert loop over FP32-dequantised weights (the in-tree oracle, PORT_DESIGN §7.1). Same rounding
  points as the kernels: W13 output, activation and W2 output each rounded to FP16 once.

Expert spill (Phase A, §5.2): experts of ``ExpertSpillPlan.spilled_expert_ids`` live in mapped pinned host memory
and are read by the kernels in place through their UVA device view. ``create_weights`` builds each spilled tensor as
a pageable zero tensor that ``get_accelerator_view_from_cpu_tensor`` copies into an exact-size ``cudaHostAlloc``
(torch ``pin_memory`` would round to 2^k), so host RAM transiently holds two copies of one tensor (<= 342 MB at TP4);
a failed ``cudaHostAlloc`` raises. Physical expert ids put the resident experts first; the resident and spilled
partitions run one launch each over one shared routing permutation. The spilled weights are never allocated in HBM,
not even during loading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    grouped_splitk,
    qpn_prepack,
    rebase_e8m0_for_fp16,
)
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import (
    Mxfp4SM70MoEMethod,
    validate_mxfp4_sm70_moe_contract,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from . import moe_kernels as mk

logger = init_logger(__name__)

BACKENDS = ("skinny", "turbomind", "torch")
MXFP4_BLOCK = 32
SKINNY_SCALE_MODE_MXFP4 = 1
SKINNY_GRID_Y_LIMIT = 65535
# split-K of the grouped skinny launches: W13 K = 5120 (320 16-code groups), W2 K = 576 at TP4 (36 groups);
# chosen by the decode microbenchmark (L-MOE progress, sub-item 4). grouped_splitk falls back if K differs.
W13_SPLITK = 16
W2_SPLITK = 6
_PARAMS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@dataclass(frozen=True)
class ExpertSpillPlan:
    spilled_expert_ids: tuple[int, ...]   # global expert ids kept in pinned host memory; identical on all TP ranks
    mode: str = "uva"                      # zero-copy UVA view of cudaHostAlloc memory (1C:get_accelerator_view_from_cpu_tensor)


def make_spill_plan(num_spilled_per_layer: int, n_routed_experts: int = 384,
                    ranking: np.ndarray | None = None) -> ExpertSpillPlan:
    """ranking None -> the highest ids are spilled; else ``ranking`` lists expert ids from most to least routed
    (L-REF routing histogram, §5.2) and the ``num_spilled_per_layer`` least-routed ones are spilled."""
    if not 0 <= num_spilled_per_layer < n_routed_experts:
        raise ValueError(f"spill of {num_spilled_per_layer} experts must be in [0, {n_routed_experts})")
    if ranking is None:
        spilled = range(n_routed_experts - num_spilled_per_layer, n_routed_experts)
    else:
        order = np.asarray(ranking).astype(np.int64).reshape(-1)
        if sorted(order.tolist()) != list(range(n_routed_experts)):
            raise ValueError(f"spill ranking must be a permutation of 0..{n_routed_experts - 1}")
        spilled = order[n_routed_experts - num_spilled_per_layer:].tolist()
    return ExpertSpillPlan(spilled_expert_ids=tuple(sorted(int(e) for e in spilled)))


def _as_bytes(t: torch.Tensor, what: str) -> torch.Tensor:
    """Checkpoint I8 nibble pairs / F8_E8M0 scales / uint8 -> uint8 bytes (bit-preserving view)."""
    if t.dtype == torch.uint8:
        return t
    if t.dtype in (torch.int8, torch.float8_e8m0fnu):
        return t.view(torch.uint8)
    raise TypeError(f"DeepSeek-V4.1 MXFP4 {what}: unexpected checkpoint dtype {t.dtype}")


def dequant_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes [N, K/2] uint8 (low nibble = even k), scales [N, K/32] uint8 E8M0 -> exact FP32 [N, K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=codes.device)
    values = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], dim=-1).flatten(-2)
    return values * torch.exp2(scales.float() - 127.0).repeat_interleave(MXFP4_BLOCK, dim=-1)


class _V41TurboMindStages(Mxfp4SM70MoEMethod):
    """TurboMind grouped stages with the V4.1 FP32 SwiGLU (one FP16 rounding, §4.1)."""

    @staticmethod
    def _apply_swiglu(layer, out: torch.Tensor, gate_up: torch.Tensor) -> None:  # type: ignore[override]
        mk.swiglu_fp32(gate_up, float(layer.swiglu_limit), out=out)


class DeepseekV41Mxfp4MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig, *, backend: str = "skinny", spill: ExpertSpillPlan | None = None):
        super().__init__(moe)
        if backend not in BACKENDS:
            raise ValueError(f"DeepSeek-V4.1 MoE backend {backend!r} not in {BACKENDS}")
        if moe.moe_parallel_config.use_all2all_kernels or moe.moe_parallel_config.use_ep:
            raise NotImplementedError("DeepSeek-V4.1 SM70 MoE shards experts by TP only (no EP / all-to-all)")
        if moe.has_bias:
            raise NotImplementedError("DeepSeek-V4.1 experts have no bias")
        validate_mxfp4_sm70_moe_contract(
            global_num_experts=moe.num_experts, top_k=moe.experts_per_token, hidden_size=moe.hidden_dim,
            intermediate_size_per_partition=moe.intermediate_size_per_partition, tp_size=moe.tp_size)
        if spill is not None and spill.spilled_expert_ids:
            ids = spill.spilled_expert_ids
            if spill.mode != "uva":
                raise NotImplementedError(f"expert spill mode {spill.mode!r} (only 'uva')")
            if list(ids) != sorted(set(ids)) or ids[0] < 0 or ids[-1] >= moe.num_experts:
                raise ValueError(f"spilled expert ids must be sorted, unique, in [0, {moe.num_experts})")
            if len(ids) >= moe.num_experts:
                raise ValueError("at least one expert must stay resident")
            if backend == "turbomind":
                raise NotImplementedError("expert spill runs on the skinny or torch backend")
        else:
            spill = None
        self.backend = backend
        self.spill = spill
        self.weight_dtype = "mxfp4"
        self._tm: _V41TurboMindStages | None = None

    # ---- FusedMoEMethodBase plumbing: this method is driven by DeepseekV41MoE, never by FusedMoE.forward ----
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
        if num_experts != self.moe.num_experts:
            raise ValueError(f"local experts {num_experts} != global {self.moe.num_experts} (TP-only sharding)")
        hidden, inter = hidden_size, intermediate_size_per_partition
        shapes = {"w13_weight": (2 * inter, hidden // 2), "w13_weight_scale": (2 * inter, hidden // MXFP4_BLOCK),
                  "w2_weight": (hidden, inter // 2), "w2_weight_scale": (hidden, inter // MXFP4_BLOCK)}
        spilled = self.spill.spilled_expert_ids if self.spill is not None else ()
        spilled_set = set(spilled)
        resident = [e for e in range(num_experts) if e not in spilled_set]
        phys = np.empty(num_experts, dtype=np.int64)
        phys[resident] = np.arange(len(resident))
        if spilled:
            phys[list(spilled)] = len(resident) + np.arange(len(spilled))
        layer.ds41_n_resident = len(resident)
        layer.ds41_n_spilled = len(spilled)
        layer.ds41_phys_of_global = phys          # host copy, used by the loader
        loader = self._make_weight_loader(layer)
        attrs = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        for name, shape in shapes.items():
            param = Parameter(torch.zeros((len(resident), *shape), dtype=torch.uint8), requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, attrs)
            set_weight_attrs(param, {"weight_loader": loader, "ds41_name": name})
            if spilled:
                host = torch.zeros((len(spilled), *shape), dtype=torch.uint8, device="cpu")
                if name.endswith("_scale"):
                    host.fill_(127)  # 2^0: a dummy-loaded spill stays decodable
                layer.register_buffer(f"{name}_spill", get_accelerator_view_from_cpu_tensor(host), persistent=False)
        if spilled:
            nbytes = sum(getattr(layer, f"{n}_spill").numel() for n in _PARAMS)
            logger.info_once("DeepSeek-V4.1 MoE: %d experts/layer spilled to pinned host memory (%.1f MiB/layer)",
                             len(spilled), nbytes / 2**20)

    def _make_weight_loader(self, layer: torch.nn.Module):
        method = self

        def ds41_expert_weight_loader(param: Parameter, loaded_weight: torch.Tensor, weight_name: str = "",
                                      shard_id: str = "", expert_id: int = -1, return_success: bool = False):
            method._load_expert(layer, param, loaded_weight, shard_id, int(expert_id))
            return True if return_success else None

        ds41_expert_weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
        return ds41_expert_weight_loader

    def _load_expert(self, layer: torch.nn.Module, param: Parameter, loaded: torch.Tensor, shard_id: str,
                     expert_id: int) -> None:
        name = param.ds41_name  # type: ignore[attr-defined]
        if not 0 <= expert_id < self.moe.num_experts:
            raise ValueError(f"{name}: expert id {expert_id} out of range")
        phys = int(layer.ds41_phys_of_global[expert_id])
        n_res = int(layer.ds41_n_resident)
        dest = param.data[phys] if phys < n_res else getattr(layer, f"{name}_spill")[phys - n_res]
        loaded = _as_bytes(loaded, name)
        tp = self.moe.moe_parallel_config
        if name.startswith("w13"):
            if shard_id not in ("w1", "w3"):
                raise ValueError(f"{name}: shard {shard_id!r} is not w1/w3")
            inter = dest.shape[0] // 2
            if loaded.ndim != 2 or loaded.shape[0] != inter * tp.tp_size or loaded.shape[1] != dest.shape[1]:
                raise ValueError(f"{name} expert {expert_id} {shard_id}: checkpoint {tuple(loaded.shape)} does not "
                                 f"shard into {inter} rows x {dest.shape[1]} at tp={tp.tp_size}")
            src = loaded.narrow(0, inter * tp.tp_rank, inter)
            dest = dest.narrow(0, 0 if shard_id == "w1" else inter, inter)
        else:
            if shard_id != "w2":
                raise ValueError(f"{name}: shard {shard_id!r} is not w2")
            cols = dest.shape[1]
            if loaded.ndim != 2 or loaded.shape[0] != dest.shape[0] or loaded.shape[1] != cols * tp.tp_size:
                raise ValueError(f"{name} expert {expert_id}: checkpoint {tuple(loaded.shape)} does not shard into "
                                 f"{dest.shape[0]} x {cols} at tp={tp.tp_size}")
            src = loaded.narrow(1, cols * tp.tp_rank, cols)
        dest.copy_(src)

    def _partitions(self, layer: torch.nn.Module) -> list[tuple[torch.Tensor, ...]]:
        parts = [tuple(getattr(layer, n).data for n in _PARAMS)]
        if int(layer.ds41_n_spilled):
            parts.append(tuple(getattr(layer, f"{n}_spill") for n in _PARAMS))
        return parts

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.activation.name.lower() != "silu" or layer.apply_router_weight_on_input:
            raise NotImplementedError("DeepSeek-V4.1 experts are SwiGLU with routing weights applied to outputs")
        if layer.swiglu_limit is None:
            raise ValueError("DeepSeek-V4.1 experts need swiglu_limit (10.0)")
        device = layer.w13_weight.device
        phys = torch.as_tensor(layer.ds41_phys_of_global, dtype=torch.int32, device=device)
        layer.ds41_phys_map = phys if int(layer.ds41_n_spilled) else None
        if self.backend == "torch":
            return
        if self.backend == "turbomind":
            self._tm = _V41TurboMindStages(self.moe)
            self._tm.process_weights_after_loading(layer)
            return
        hidden = int(self.moe.hidden_dim)
        inter = int(self.moe.intermediate_size_per_partition)
        layer.ds41_splitk13 = W13_SPLITK if (hidden // 16) % W13_SPLITK == 0 else grouped_splitk(
            hidden, 16, extended=True)
        layer.ds41_splitk2 = W2_SPLITK if (inter // 16) % W2_SPLITK == 0 else grouped_splitk(
            inter, 8, extended=True)
        gscales = []
        for w13, s13, w2, s2 in self._partitions(layer):
            g13 = rebase_e8m0_for_fp16(s13)
            g2 = rebase_e8m0_for_fp16(s2)
            for e in range(w13.shape[0]):
                for codes, scales in ((w13[e], s13[e]), (w2[e], s2[e])):
                    qc, qs = qpn_prepack(codes.to(device), scales.to(device), MXFP4_BLOCK, k_align=32)
                    codes.view(-1).copy_(qc)
                    scales.view(-1).copy_(qs)
            gscales.append((g13.to(device=device, dtype=torch.float32).contiguous(),
                            g2.to(device=device, dtype=torch.float32).contiguous()))
        layer.ds41_gscales = gscales
        torch.accelerator.empty_cache()

    # ---- compute ----
    def expert_slots(self, layer: torch.nn.Module, x: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """x [T, H] fp16, topk_ids [T, k] int32 -> y [T * k, H] fp16 (route order, TP-partial)."""
        num_tokens, top_k = topk_ids.shape
        if x.dtype != torch.float16 or x.ndim != 2 or x.shape[0] != num_tokens or not x.is_contiguous():
            raise TypeError(f"expert_slots: x must be contiguous fp16 [{num_tokens}, H], got {x.dtype} "
                            f"{tuple(x.shape)}")
        if top_k != int(self.moe.experts_per_token) or topk_ids.dtype != torch.int32:
            raise TypeError(f"expert_slots: topk_ids must be int32 [T, {self.moe.experts_per_token}]")
        if self.backend == "turbomind":
            assert self._tm is not None, "process_weights_after_loading has not run"
            return self._tm.apply_slots(layer, x, topk_ids.contiguous())
        if self.backend == "torch":
            return self._torch_expert_slots(layer, x, topk_ids)
        y = torch.empty((num_tokens * top_k, x.shape[1]), dtype=torch.float16, device=x.device)
        chunk = SKINNY_GRID_Y_LIMIT // top_k
        for start in range(0, num_tokens, chunk):
            end = min(num_tokens, start + chunk)
            self._skinny_chunk(layer, x[start:end], topk_ids[start:end].contiguous(), y[start * top_k:end * top_k])
        return y

    def _skinny_chunk(self, layer: torch.nn.Module, x: torch.Tensor, topk_ids: torch.Tensor,
                      y: torch.Tensor) -> None:
        num_tokens, top_k = topk_ids.shape
        inter = int(self.moe.intermediate_size_per_partition)
        spill = int(layer.ds41_n_spilled) > 0
        perm, tables = mk.route_prep(topk_ids, layer.ds41_phys_map, int(layer.ds41_n_resident), spill)
        y13 = torch.empty((num_tokens * top_k, 2 * inter), dtype=torch.float16, device=x.device)
        parts = self._partitions(layer)
        for (w13, s13, _, _), (g13, _), (gids, goff) in zip(parts, layer.ds41_gscales, tables):
            torch.ops._C.skinny_moe_qpn_sm70(x, w13, s13, g13, perm, gids, goff, top_k, y13, False, num_tokens,
                                             layer.ds41_splitk13, 1, SKINNY_SCALE_MODE_MXFP4)
        act = mk.swiglu_fp32(y13, float(layer.swiglu_limit))
        for (_, _, w2, s2), (_, g2), (gids, goff) in zip(parts, layer.ds41_gscales, tables):
            torch.ops._C.skinny_moe_qpn_sm70(act, w2, s2, g2, perm, gids, goff, top_k, y, True, num_tokens,
                                             layer.ds41_splitk2, 1, SKINNY_SCALE_MODE_MXFP4)

    def _torch_expert_slots(self, layer: torch.nn.Module, x: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        num_tokens, top_k = topk_ids.shape
        y = torch.empty((num_tokens * top_k, x.shape[1]), dtype=torch.float16, device=x.device)
        flat = topk_ids.reshape(-1)
        n_res = int(layer.ds41_n_resident)
        for expert in torch.unique(flat).tolist():
            phys = int(layer.ds41_phys_of_global[expert])
            w13, s13, w2, s2 = self._partitions(layer)[0 if phys < n_res else 1]
            idx = phys if phys < n_res else phys - n_res
            slots = torch.nonzero(flat == expert).flatten()
            rows = x.index_select(0, slots // top_k).float()
            h = (rows @ dequant_mxfp4(w13[idx].to(x.device), s13[idx].to(x.device)).t()).half()
            act = mk.swiglu_fp32_reference(h, float(layer.swiglu_limit))
            out = act.float() @ dequant_mxfp4(w2[idx].to(x.device), s2[idx].to(x.device)).t()
            y.index_copy_(0, slots, out.half())
        return y
