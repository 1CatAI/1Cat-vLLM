# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 (exllamav3 trellis-coded) checkpoints.

Routed MoE experts stored as EXL3 run on kernels that decode the trellis inside
the matmul; the weights are never dequantized to memory. Supported: the mcg
codebook, K = 2, 3 or 4 bits (per layer, read from the checkpoint), and both
tile orders exllamav3 can write ("sm80", its default, and "colmajor"). The
kernels are built for compute capability 7.0.

Every other linear layer is expected unquantized. The few dense EXL3 tensors a
converter may emit besides the experts (an MTP head, for example) are
dequantized to FP16 while loading, see ``Exl3Config.dequantize_dense``.

EXL3 tensor format, per linear (in_features k, out_features n, y = x @ W):
  trellis  int16 [k/16, n/16, 16K]  one 16x16 tile per cell, K bits per weight
  suh      fp16  [k]                input sign/scale vector
  svh      fp16  [n]                output sign/scale vector
  mcg/mul1 int32 scalar marker      codebook (no marker: 3inst)
Tile decode: a tile is a tail-biting stream of 256K bits stored as uint16 pairs
half-swapped (MSB-first as big-endian u32). Position p's state is the 16 bits
ending at bit (p + 1) K, its value codebook(state), and it holds tile element
perm[p] (element = row * 16 + col). W = had_l(had_r(T) ...) with blockwise
128-point Sylvester Hadamards: W = H T H * suh[:, None] * svh[None, :] in the
order of ``exl3_reconstruct``.
"""

import json
import math
import os
from collections.abc import Iterable, Iterator
from functools import cache
from typing import Any

import regex as re
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = init_logger(__name__)

MCG_MULT = 0xCBAC1FED
MUL1_MULT = 0x83DCD12D
MCG_MARKER = MCG_MULT - (1 << 32)  # the .mcg tensor: the multiplier as int32
COLMAJOR_MARKER = 0x70C0  # the .tile_order tensor of a colmajor tensor
HAD_DIM = 128
TILE_ORDERS = ("sm80", "colmajor")

_EXPERT_KEY = re.compile(r"\.experts\.\d+\.")
_EXPERT_TRELLIS = re.compile(
    r"layers\.(\d+)\..*\.experts\.\d+\.(?:gate|up|down)_proj\.trellis$"
)
_LAYER_INDEX = re.compile(r"layers\.(\d+)(?:\.|$)")


def normalize_tile_order(tile_order: str) -> str:
    """Tile order name; a platform prefix ("<arch>_colmajor") is accepted."""
    name = (
        tile_order.rpartition("_")[2] if tile_order.endswith("colmajor") else tile_order
    )
    if name not in TILE_ORDERS:
        raise ValueError(f"Unknown EXL3 tile order {tile_order!r}")
    return name


# ---------------------------------------------------------------------------
# Reference decoder (pure torch). Used to dequantize dense EXL3 tensors at load
# time and as the oracle for the kernel tests.
# ---------------------------------------------------------------------------


@cache
def tile_perm(tile_order: str) -> torch.Tensor:
    """Position -> tile element (row * 16 + col) for a tile order."""
    if normalize_tile_order(tile_order) == "colmajor":
        return torch.tensor([(p % 16) * 16 + p // 16 for p in range(256)])
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        for j, c in enumerate((t // 4, t // 4 + 8)):
            for i, r in enumerate((r0, r0 + 1, r0 + 8, r0 + 9)):
                perm[t * 8 + j * 4 + i] = r * 16 + c
    return torch.tensor(perm)


def _trellis_states(trellis: torch.Tensor, K: int) -> torch.Tensor:
    """int16 [..., 16K] -> int64 states [..., 256]."""
    w = trellis.to(torch.int64) & 0xFFFF
    w = w.view(*w.shape[:-1], 8 * K, 2).flip(-1).reshape(*w.shape[:-1], 16 * K)
    shifts = torch.arange(15, -1, -1, device=w.device)
    bits = ((w.unsqueeze(-1) >> shifts) & 1).reshape(*w.shape[:-1], 256 * K)
    ends = (torch.arange(256, device=w.device) + 1) * K
    idx = (ends.unsqueeze(1) - 16 + torch.arange(16, device=w.device)) % (256 * K)
    weights = 1 << torch.arange(15, -1, -1, device=w.device)
    return (bits[..., idx] * weights).sum(-1)


def _halves_sum(x: torch.Tensor) -> torch.Tensor:
    lo = (x & 0xFFFF).to(torch.int16).view(torch.float16).float()
    hi = ((x >> 16) & 0xFFFF).to(torch.int16).view(torch.float16).float()
    return (lo + hi).to(torch.float16)


def _decode_states(states: torch.Tensor, codebook: str) -> torch.Tensor:
    if codebook == "3inst":
        x = (states * 89226354 + 64248484) & 0xFFFFFFFF
        return _halves_sum((x & 0x8FFF8FFF) ^ 0x3B603B60)
    if codebook == "mcg":
        x = (states * MCG_MULT) & 0xFFFFFFFF
        return _halves_sum((x & 0x8FFF8FFF) ^ 0x3B603B60)
    if codebook == "mul1":
        x = (states * MUL1_MULT) & 0xFFFFFFFF
        s = 0x6400 + sum((x >> (8 * i)) & 0xFF for i in range(4))
        h = (s & 0xFFFF).to(torch.int16).view(torch.float16).float()
        k_inv = torch.tensor(0x1EEE, dtype=torch.int16).view(torch.float16).float()
        k_bias = torch.tensor(0xC931 - 0x10000, dtype=torch.int16).view(torch.float16)
        return (h * k_inv.item() + k_bias.float().item()).to(torch.float16)
    raise ValueError(f"Unknown EXL3 codebook {codebook!r}")


def exl3_decode_tiles(
    trellis: torch.Tensor, codebook: str, tile_order: str
) -> torch.Tensor:
    """int16 [k/16, n/16, 16K] -> fp16 [k, n] in the trellis (rotated) basis."""
    tk, tn, words = trellis.shape
    K = words // 16
    values = _decode_states(_trellis_states(trellis, K), codebook)
    tiles = torch.empty_like(values)
    tiles[..., tile_perm(tile_order).to(values.device)] = values
    return tiles.view(tk, tn, 16, 16).permute(0, 2, 1, 3).reshape(tk * 16, tn * 16)


@cache
def _hadamard(device: torch.device) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < HAD_DIM:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(HAD_DIM)).float().to(device)


def exl3_reconstruct(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    codebook: str,
    tile_order: str,
) -> torch.Tensor:
    """Full fp32 weight W [in_features, out_features] (y = x @ W)."""
    t = exl3_decode_tiles(trellis, codebook, tile_order).float()
    k, n = t.shape
    h = _hadamard(t.device)
    t = (h @ t.view(k // HAD_DIM, HAD_DIM, n)).view(k, n) * suh.float().unsqueeze(1)
    t = (t.view(k, n // HAD_DIM, HAD_DIM) @ h).view(k, n)
    return t * svh.float().unsqueeze(0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _read_safetensors_shapes(model_name: str, revision: str | None) -> dict:
    """Tensor name -> shape from the checkpoint's safetensors headers."""
    shapes: dict[str, list[int]] = {}
    if os.path.isdir(model_name):
        for fn in sorted(os.listdir(model_name)):
            if not fn.endswith(".safetensors"):
                continue
            with open(os.path.join(model_name, fn), "rb") as f:
                header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
            for key, info in header.items():
                if key != "__metadata__":
                    shapes[key] = info["shape"]
        return shapes
    from huggingface_hub import get_safetensors_metadata

    metadata = get_safetensors_metadata(model_name, revision=revision)
    for file_meta in metadata.files_metadata.values():
        for key, info in file_meta.tensors.items():
            shapes[key] = info.shape
    return shapes


def _layer_index(prefix: str) -> int | None:
    """Decoder layer index of a module or tensor name (the roots differ between
    checkpoint and model: "model.", "model.language_model.", ...)."""
    found = _LAYER_INDEX.findall(prefix)
    return int(found[-1]) if found else None


class Exl3Config(QuantizationConfig):
    """Config for EXL3 checkpoints (``quant_method: exl3``)."""

    def __init__(
        self, bits: float = 2, codebook: str = "mul1", tile_order: str = "sm80"
    ):
        super().__init__()
        self.bits = bits
        self.codebook = codebook
        self.tile_order = normalize_tile_order(tile_order)
        # Routed-expert bit width per decoder layer index, filled from the
        # checkpoint headers by maybe_update_config.
        self.expert_bits: dict[int, int] = {}

    def __repr__(self) -> str:
        return (
            f"Exl3Config(bits={self.bits}, codebook={self.codebook}, "
            f"tile_order={self.tile_order})"
        )

    @classmethod
    def get_name(cls):
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(
            bits=config.get("bits", 2),
            codebook=config.get("codebook", "mul1"),
            tile_order=config.get("tile_order", "sm80"),
        )

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None):
        shapes = _read_safetensors_shapes(model_name, revision)
        bits: dict[int, set[int]] = {}
        for key, shape in shapes.items():
            m = _EXPERT_TRELLIS.search(key)
            if m:
                bits.setdefault(int(m.group(1)), set()).add(shape[-1] // 16)
        mixed = {k: sorted(v) for k, v in bits.items() if len(v) > 1}
        if mixed:
            raise ValueError(
                f"EXL3: experts of one MoE layer use mixed bit widths: {mixed}"
            )
        self.expert_bits = {k: next(iter(v)) for k, v in bits.items()}
        expert_keys = [k for k in shapes if _EXPERT_KEY.search(k)]
        n_trellis = sum(k.endswith(".trellis") for k in expert_keys)
        n_mcg = sum(k.endswith(".mcg") for k in expert_keys)
        if n_trellis and (
            n_mcg != n_trellis
            or any(k.endswith((".mul1", ".su", ".sv")) for k in expert_keys)
        ):
            raise ValueError(
                "EXL3 routed experts must use the mcg codebook with unpacked sign "
                f"vectors (suh/svh); this checkpoint's codebook is {self.codebook!r}"
            )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            key = _layer_index(prefix)
            if key not in self.expert_bits:
                raise ValueError(f"EXL3: no routed-expert tensors found for {prefix}")
            return Exl3MoEMethod(
                layer.moe_config, self.expert_bits[key], self.tile_order == "colmajor"
            )
        return None

    def dequantize_dense(self, weights: Iterable[tuple]) -> Iterator[tuple]:
        """Pass a model's weight iterator through, replacing each dense EXL3
        linear (trellis/suh/svh + codebook marker) by its FP16 ``.weight`` in
        checkpoint orientation (out_features, in_features). Expert tensors pass
        through unchanged. The dequantized tensors are yielded at the end."""
        parts: dict[str, dict[str, torch.Tensor]] = {}
        suffixes = (".trellis", ".suh", ".svh", ".mcg", ".mul1", ".tile_order")
        for item in weights:
            name = item[0]
            if name.endswith(suffixes) and not _EXPERT_KEY.search(name):
                prefix, _, kind = name.rpartition(".")
                parts.setdefault(prefix, {})[kind] = item[1]
                continue
            yield item
        for prefix, p in parts.items():
            if not {"trellis", "suh", "svh"} <= p.keys():
                raise ValueError(f"EXL3 tensor {prefix} is incomplete: {sorted(p)}")
            codebook = "mcg" if "mcg" in p else "mul1" if "mul1" in p else "3inst"
            w = exl3_reconstruct(
                p["trellis"].cuda(),
                p["suh"].cuda(),
                p["svh"].cuda(),
                codebook,
                self.tile_order,
            )
            yield prefix + ".weight", w.T.contiguous().half().cpu()


# ---------------------------------------------------------------------------
# Routed experts
# ---------------------------------------------------------------------------


GROUP_MIN_TOKENS = 65  # from here on, decode each expert once per <= 8 slots


def exl3_group_slots(
    topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group slots (s = token * topk + j) by expert, up to 8 per group, without
    host syncs (CUDA-graph safe). Returns the expert of each group [G] and its
    slots [G, 8] (-1 = empty) for the worst case G = S / 8 + min(E, S); the
    groups beyond the real count are empty."""
    ids = topk_ids.reshape(-1).long()
    S = ids.numel()
    G = S // 8 + min(num_experts, S)
    es, order = torch.sort(ids, stable=True)
    pos = torch.arange(S, device=ids.device)
    first = torch.ones(S, dtype=torch.bool, device=ids.device)
    first[1:] = es[1:] != es[:-1]
    start = torch.cummax(torch.where(first, pos, torch.zeros_like(pos)), 0).values
    in_run = pos - start
    gid = torch.cumsum((in_run % 8 == 0).long(), 0) - 1
    rows = torch.full((G * 8,), -1, dtype=torch.int32, device=ids.device)
    rows[gid * 8 + in_run % 8] = order.int()
    gexp = torch.zeros(G, dtype=torch.int32, device=ids.device)
    gexp[gid] = es.int()
    return gexp, rows.view(G, 8)


def exl3_moe_forward(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_trellis: torch.Tensor,
    w13_suh: torch.Tensor,
    w13_svh: torch.Tensor,
    w2_trellis: torch.Tensor,
    w2_suh: torch.Tensor,
    w2_svh: torch.Tensor,
    colmajor: bool,
) -> torch.Tensor:
    """Routed experts: x [T, H] fp16, topk_ids [T, topk] int32, topk_weights
    [T, topk] fp32 -> this rank's partial output [T, H] fp32. Weights are laid
    out as in Exl3MoEMethod."""
    T, H = x.shape
    S = T * topk_ids.shape[1]
    inter = w2_trellis.shape[1] * 16
    f16 = dict(dtype=torch.float16, device=x.device)
    f32 = dict(dtype=torch.float32, device=x.device)
    xg, xu = torch.empty(S, H, **f16), torch.empty(S, H, **f16)
    yg, yu = torch.empty(S, inter, **f32), torch.empty(S, inter, **f32)
    hd, yd = torch.empty(S, inter, **f16), torch.empty(S, H, **f32)
    out = torch.empty(T, H, **f32)
    ops.exl3_moe_pre_out(xg, xu, x, topk_ids, w13_suh[0], w13_suh[1])
    if T >= GROUP_MIN_TOKENS:
        gexp, rows = exl3_group_slots(topk_ids, w2_trellis.shape[0])
    else:
        gexp, rows = topk_ids.reshape(-1), None
    ops.exl3_moe_gemv_out(
        yg, yu, xg, xu, w13_trellis[0], w13_trellis[1], gexp, rows, colmajor
    )
    ops.exl3_moe_mid_out(hd, yg, yu, topk_ids, w13_svh[0], w13_svh[1], w2_suh)
    ops.exl3_moe_gemv_out(yd, None, hd, None, w2_trellis, None, gexp, rows, colmajor)
    ops.exl3_moe_post_out(out, yd, topk_ids, topk_weights, w2_svh)
    return out


class Exl3MoEMethod(FusedMoEMethodBase):
    """Routed experts decoded from the EXL3 trellis inside the expert matmuls.

    Per layer the experts are stacked (inter = this rank's intermediate slice):
      w13_trellis [2, E, H/16, inter/16, 16K]  gate (0) and up (1)
      w13_suh     [2, E, H],  w13_svh [2, E, inter]
      w2_trellis  [E, inter/16, H/16, 16K],  w2_suh [E, inter],  w2_svh [E, H]
    Tensor parallelism slices the intermediate dimension at 128-column
    boundaries, which keeps the Hadamard blocks rank-local.
    """

    MAX_GRID_Y = 65535

    def __init__(self, moe: FusedMoEConfig, bits: int, colmajor: bool):
        super().__init__(moe)
        if bits not in (2, 3, 4):
            raise ValueError(f"EXL3 MoE supports K = 2, 3 or 4 bits, got {bits}")
        self.bits = bits
        self.colmajor = colmajor

    @property
    def supports_eplb(self) -> bool:
        return False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if not ops.exl3_moe_available():
            raise RuntimeError(
                "EXL3 MoE kernels are not built for this GPU (compute capability 7.0)."
            )
        if layer.ep_size > 1:
            raise ValueError("EXL3 MoE does not support expert parallelism.")
        E, H, inter = num_experts, hidden_size, intermediate_size_per_partition
        if H % 512 or inter % 512:
            raise ValueError(
                f"EXL3 MoE needs hidden and per-rank intermediate sizes divisible "
                f"by 512, got {H} and {inter}"
            )
        K, rank = self.bits, layer.tp_rank
        layer.exl3_loaded = set()

        def register(name, shape, dtype, tp_dim, gate_up):
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype), requires_grad=False
            )

            def loader(
                param, loaded, weight_name=None, shard_id=None, expert_id=None, **kwargs
            ) -> bool:
                dst = (
                    param.data[0 if shard_id == "w1" else 1] if gate_up else param.data
                )
                dst = dst[expert_id]
                if tp_dim is not None:
                    unit = dst.shape[tp_dim]
                    loaded = loaded.narrow(tp_dim, rank * unit, unit)
                if dst.shape != loaded.shape or dst.dtype != loaded.dtype:
                    raise ValueError(
                        f"{weight_name}: checkpoint tensor {tuple(loaded.shape)} "
                        f"{loaded.dtype} != expected {tuple(dst.shape)} {dst.dtype}"
                    )
                dst.copy_(loaded)
                layer.exl3_loaded.add((name, shard_id, expert_id))
                return True

            param.weight_loader = loader  # type: ignore[attr-defined]
            layer.register_parameter(name, param)

        register(
            "w13_trellis", (2, E, H // 16, inter // 16, 16 * K), torch.int16, 1, True
        )
        register("w13_suh", (2, E, H), torch.float16, None, True)
        register("w13_svh", (2, E, inter), torch.float16, 0, True)
        register("w13_mcg", (2, E), torch.int32, None, True)
        register("w2_trellis", (E, inter // 16, H // 16, 16 * K), torch.int16, 0, False)
        register("w2_suh", (E, inter), torch.float16, 0, False)
        register("w2_svh", (E, H), torch.float16, None, False)
        register("w2_mcg", (E,), torch.int32, None, False)
        # Optional per-tensor tile order markers (checked against the config).
        register("w13_tile_order", (2, E), torch.int32, None, True)
        register("w2_tile_order", (E,), torch.int32, None, False)
        layer.exl3_num_experts = E

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # With a quantization config, vLLM does not check for parameters the
        # checkpoint never filled; an absent tensor would stay uninitialized.
        E = layer.exl3_num_experts
        for name in ("trellis", "suh", "svh", "mcg"):
            for shard in ("w1", "w3"):
                got = sum(
                    (f"w13_{name}", shard, e) in layer.exl3_loaded for e in range(E)
                )
                if got != E:
                    raise ValueError(f"EXL3: {E - got} experts lack {shard} {name}")
            got = sum((f"w2_{name}", "w2", e) in layer.exl3_loaded for e in range(E))
            if got != E:
                raise ValueError(f"EXL3: {E - got} experts lack w2 {name}")
        if not (
            bool((layer.w13_mcg == MCG_MARKER).all())
            and bool((layer.w2_mcg == MCG_MARKER).all())
        ):
            raise ValueError("EXL3 MoE expects the mcg codebook marker on every expert")
        marked = {
            (n, s, e) for n, s, e in layer.exl3_loaded if n.endswith("tile_order")
        }
        if marked:
            if not self.colmajor or len(marked) != 3 * E:
                raise ValueError(
                    "EXL3: per-tensor tile order markers disagree with "
                    "quantization_config.tile_order"
                )
            if not (
                bool((layer.w13_tile_order == COLMAJOR_MARKER).all())
                and bool((layer.w2_tile_order == COLMAJOR_MARKER).all())
            ):
                raise ValueError("EXL3: unknown per-tensor tile order marker")
        for name in ("w13_mcg", "w2_mcg", "w13_tile_order", "w2_tile_order"):
            delattr(layer, name)
        del layer.exl3_loaded
        logger.info_once(
            "EXL3 routed experts: mcg, K=%d, %s tile order, decoded in the matmul.",
            self.bits,
            "colmajor" if self.colmajor else "sm80",
        )

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # This method owns permutation, both expert matmuls and the reduction.
        return None

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def _forward(self, layer, x, topk_ids, topk_weights) -> torch.Tensor:
        return exl3_moe_forward(
            x,
            topk_ids,
            topk_weights,
            layer.w13_trellis,
            layer.w13_suh,
            layer.w13_svh,
            layer.w2_trellis,
            layer.w2_suh,
            layer.w2_svh,
            self.colmajor,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2:
            raise TypeError("EXL3 MoE requires CUDA FP16 activations [tokens, hidden].")
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return x.new_empty((0, x.shape[1]))
        x = x.contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        w = topk_weights.to(torch.float32).contiguous()
        # Each launch covers 2 * tokens * topk grid rows (gate + up fused).
        chunk = self.MAX_GRID_Y // (2 * ids.shape[1])
        if num_tokens <= chunk:
            return self._forward(layer, x, ids, w).to(x.dtype)
        out = torch.empty_like(x)
        for s in range(0, num_tokens, chunk):
            e = min(s + chunk, num_tokens)
            out[s:e] = self._forward(layer, x[s:e], ids[s:e], w[s:e]).to(x.dtype)
        return out

    def apply_monolithic(self, layer, x, router_logits, input_ids=None) -> torch.Tensor:
        raise NotImplementedError("EXL3 MoE is not monolithic.")
