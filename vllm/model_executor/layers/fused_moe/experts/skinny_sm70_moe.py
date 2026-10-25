# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped NVFP4 and MXFP4 MoE over the skinny QPN kernels (SM70/SM75).

The backend oracle admits compatible configurations by default. Expert weights
are permuted in place at load time into the fragment order of the card's
tensor-core MMA: mma.m8n8k4 on SM70, mma.m16n8k8 from SM75 on, which Turing
runs at twice its m8n8k4 rate. Each layer's weights live on one card, so one
order per card covers every serving path. Device-side
routing handles both scale formats; chunking bounds native CUDA grid dimensions
and temporary bytes without a fixed token or concurrency admission limit.

Activations stay fp16 (w4a16). NVFP4 carries one fp8-e4m3 scale per 16
codes, MXFP4 one E8M0 scale per 32; the codes, their packing and the fragment
order are the same, so both formats share the kernels.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    Nvfp4QuantizationEmulationTritonExperts,
)
from vllm.model_executor.layers.fused_moe.experts.skinny_moe_triton import (
    TRITON_MIN_ROWS_PER_EXPERT,
    skinny_moe_e8m0_m8n8k4_triton,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.sm70.reduction import _mtp_weighted_reduce
from vllm.model_executor.layers.fused_moe.sm70.routing import group_routing
from vllm.model_executor.layers.quantization.utils.csf32_expand import (
    CsfExpertScales,
    expand_csf32_experts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.s5_scales import (
    S5Scales,
    encode_s5,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# The grouped operator uses slots as its grid-y dimension.
_CUDA_GRID_Y_LIMIT = 65535
_QPN_MAX_M = 16


class QpnFragmentOrder(IntEnum):
    """Register order of the packed expert weights, one per MMA shape.

    The values are the native operator's ``fragment_order`` argument.
    """

    M8N8K4 = 0
    M16N8K8 = 1


def qpn_fragment_order(device_id: int) -> QpnFragmentOrder:
    """The order the given card's kernels read: m16n8k8 where the card has it."""
    if current_platform.has_device_capability(75, device_id=device_id):
        return QpnFragmentOrder.M16N8K8
    return QpnFragmentOrder.M8N8K4


def grouped_splitk(k: int, order: QpnFragmentOrder) -> int:
    """Split-K warps per CTA for a GEMM that reduces over ``k``.

    Picks the most preferred specialization the native kernel instantiates
    that divides the actual K groups. Both kernels prefer 8 warps (tuning
    sweeps 2026-10-21); Turing's m16n8k8 kernel takes 4 where 8 would leave a
    warp fewer than 16 groups, Volta's m8n8k4 kernel 16 or 10 where 8 does
    not divide K.
    """
    if order == QpnFragmentOrder.M8N8K4:
        preference: tuple[int, ...] = (8, 16, 10)
    else:
        preference = (8, 4) if k >= 2048 else (4, 8)
    for split in preference:
        if k > 0 and k % 64 == 0 and (k // 16) % split == 0:
            return split
    raise ValueError(f"skinny grouped MoE has no split-K specialization for K={k}")


def _fragment_tables(
    order: QpnFragmentOrder, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Where each packed code and scale byte of a tile comes from.

    Returns the output column [32 lanes, 16 codes] and the K offset inside the
    16-code group [32, 16] of every code a lane holds (codes in nibble order,
    low nibble first), and the column [32] of each of a scale group's 32
    scale bytes. The m16n8k8 tables mirror m16n8k8_fp4_column and
    m16n8k8_fp4_k in csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh.
    """
    lane = torch.arange(32, device=device).view(32, 1)
    if order == QpnFragmentOrder.M8N8K4:
        # Lane holds one column; its 16 K values interleaved so the decoder's
        # (i, i + 4) output is the adjacent-k B register pair.
        col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
        korder = torch.tensor(
            [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15], device=device
        )
        return col.expand(32, 16), korder.view(1, 16).expand(32, 16), col.view(32)
    # Code word w covers A tile w; nibble n sits at column 16w + l/4 + 8(n&1)
    # and K 4(l%4) + 2((n>>1)&1) + (n>>2). Scale byte r is column r/4 + 8(r%4).
    position = torch.arange(16, device=device).view(1, 16)
    word, nibble = position >> 3, position & 7
    col = 16 * word + (lane >> 2) + 8 * (nibble & 1)
    k = 4 * (lane & 3) + 2 * ((nibble >> 1) & 1) + (nibble >> 2)
    entry = torch.arange(32, device=device)
    return col, k, (entry >> 2) + 8 * (entry & 3)


def nvfp4_skinny_scale_reason(g1: torch.Tensor, g2: torch.Tensor) -> str | None:
    """Check the actual FP16 rebiased global factors before weight mutation."""
    if g1.ndim == 2 and g1.shape[1] == 2 and not torch.equal(g1[:, 0], g1[:, 1]):
        return "gate and up global scales require a shared factor"
    for scales in (g1, g2):
        if not torch.isfinite(scales).all() or bool((scales < 0).any()):
            return "global NVFP4 scales must be finite and nonnegative"
        if not torch.isfinite((scales.float() * 2**14).half()).all():
            return "global NVFP4 scale rebias exceeds FP16 range"
    return None


def qpn_scale_entries(order: QpnFragmentOrder, device: torch.device) -> torch.Tensor:
    """Raster position of each tile row's scale byte, int32 [32].

    The inverse of the scale column table of ``_fragment_tables``; the CSF
    expansion writes compressed expert scales through it.
    """
    _, _, scale_col = _fragment_tables(order, device)
    entries = torch.empty(32, dtype=torch.int32, device=device)
    entries[scale_col] = torch.arange(32, dtype=torch.int32, device=device)
    return entries


def qpn_prepack(
    codes: torch.Tensor,
    scales: torch.Tensor | None,
    scale_group: int,
    order: QpnFragmentOrder,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Permute one weight matrix into the fragment order the QPN kernels read.

    Codes become [tile N/32][group K/16][lane 32] x 8 bytes, with the nibbles
    placed so that the decoder's (i, i + 4) output pairs are exactly the MMA
    operand registers of the given order; scales become
    [tile N/32][group K/scale_group][32 bytes]. A pure permutation of the
    checkpoint bytes.

    Args:
        codes: [N, K/2] uint8, two e2m1 codes per byte, low nibble first.
        scales: [N, K/scale_group] uint8 scale bytes, or None to permute only
            the codes (compressed scales are expanded into this order later).
        scale_group: codes per scale, 16 for NVFP4 (fp8-e4m3) and 32 for
            MXFP4 (E8M0). The code layout is the same for both: a 32-code
            MXFP4 block is two adjacent 16-code groups sharing one scale.
        order: the MMA fragment order of the card the weights serve on.

    Returns:
        The permuted codes and scales, both flat and contiguous.
    """
    n, k2 = codes.shape
    k = k2 * 2
    if n % 32 or k % 64 or k % scale_group:
        raise ValueError(
            f"QPN prepack needs N % 32 == 0 and K % 64 == 0, got N={n} K={k}"
        )
    dev = codes.device
    tiles, groups = n // 32, k // scale_group
    code_col, code_k, scale_col = _fragment_tables(order, dev)
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k)
    cgroups = k // 16  # code groups: always 16 codes per fragment pair
    g = torch.arange(groups, device=dev)
    cg = torch.arange(cgroups, device=dev)
    kidx = cg.view(cgroups, 1, 1) * 16 + code_k.view(1, 32, 16)
    qc = torch.empty(tiles, cgroups, 32, 8, dtype=torch.uint8, device=dev)
    qs = (
        None
        if scales is None
        else torch.empty(tiles, groups, 32, dtype=torch.uint8, device=dev)
    )
    # Chunk the gather: the int64 index intermediates are 16x the payload,
    # so cap the transients at ~300 MB.
    chunk = max(1, 36864 // max(groups, cgroups))
    for t0 in range(0, tiles, chunk):
        t1 = min(t0 + chunk, tiles)
        tt = t1 - t0
        tile_col = torch.arange(t0, t1, device=dev).view(tt, 1, 1) * 32
        ncol = tile_col + code_col.view(1, 32, 16)
        nb = nib[
            ncol.view(tt, 1, 32, 16).expand(tt, cgroups, 32, 16),
            kidx.view(1, cgroups, 32, 16).expand(tt, cgroups, 32, 16),
        ]
        qc[t0:t1] = nb[..., 0::2] | (nb[..., 1::2] << 4)
        if qs is not None and scales is not None:
            scol = tile_col.view(tt, 1) + scale_col.view(1, 32)
            qs[t0:t1] = scales[
                scol.view(tt, 1, 32).expand(tt, groups, 32),
                g.view(1, groups, 1).expand(tt, groups, 32),
            ]
    del nib
    return qc.view(-1).contiguous(), None if qs is None else qs.view(-1).contiguous()


def _expert_gemm(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    gscale: float,
    n: int,
) -> torch.Tensor:
    """One expert's GEMM at any M, on QPN-prepacked weight fragments."""
    m = x.size(0)
    if m <= _QPN_MAX_M:
        return torch.ops._C.skinny_qpn_gemm_sm70(x, codes, scales, gscale, n)
    return torch.cat(
        [
            torch.ops._C.skinny_qpn_gemm_sm70(
                x[i : i + _QPN_MAX_M], codes, scales, gscale, n
            )
            for i in range(0, m, _QPN_MAX_M)
        ]
    )


def skinny_moe_forward(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scales_u8: torch.Tensor,
    w2_scales_u8: torch.Tensor,
    g1: list[float],
    g2: list[float],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inter_dim: int,
    activation_fn,
) -> None:
    """Per-expert MoE forward; module-level so tests can drive it directly.

    ``activation_fn(out, inp)`` fills ``out`` [m, inter_dim] from ``inp``
    [m, N]; ``g1``/``g2`` are the per-expert multiplicative global scales
    (the reciprocal of the checkpoint's ``weight_global_scale``).
    """
    device = hidden_states.device
    top_k = topk_ids.size(1)
    w_flat = topk_weights.reshape(-1)

    # One host sync for the routing table instead of one per expert.
    slots_per_expert: dict[int, list[int]] = {}
    for slot, expert in enumerate(topk_ids.cpu().flatten().tolist()):
        slots_per_expert.setdefault(expert, []).append(slot)

    output.zero_()
    for expert, slots in slots_per_expert.items():
        slot_idx = torch.tensor(slots, dtype=torch.long, device=device)
        rows = slot_idx // top_k
        x_e = hidden_states.index_select(0, rows)

        y13 = _expert_gemm(
            x_e, w1[expert], w1_scales_u8[expert], g1[expert], w1.size(1)
        )
        inter = torch.empty((x_e.size(0), inter_dim), dtype=y13.dtype, device=device)
        activation_fn(inter, y13)
        y = _expert_gemm(
            inter, w2[expert], w2_scales_u8[expert], g2[expert], w2.size(1)
        )

        y.mul_(w_flat.index_select(0, slot_idx).unsqueeze(1).to(y.dtype))
        output.index_add_(0, rows, y)


# The kernels decode an E8M0 scale byte b as the fp16 bit pattern (b - 112)
# << 10, which is exact only while that exponent field stays inside fp16's
# normal range 1..30.
_E8M0_FP16_MIN = 113
_E8M0_FP16_MAX = 142
# They then multiply that block scale by the expert's global scale times
# 2^14 (the code decoder's exponent re-bias) in fp16, so every true scale
# times 2^14 has to be a power of two fp16 holds, subnormals included.
_DECODE_REBIAS_EXP = 14
_FP16_POW2_MIN_EXP = -24
_FP16_POW2_MAX_EXP = 15


def rebase_e8m0_for_fp16(
    scales_u8: torch.Tensor, *, dry_run: bool = False
) -> torch.Tensor:
    """Shift each expert's E8M0 scales, in place, into the window the kernels
    decode.

    MXFP4 scales are bare powers of two, 2^(b - 127), and span far more than
    fp16 can hold. Adding the same d to every byte of one expert and handing
    2^-d back as that expert's global scale changes nothing numerically, so
    the rebased raster reproduces the checkpoint exactly. Working on the bytes
    in place keeps the load-time peak at the raster itself (a 256-expert
    DeepSeek-V4 layer carries 128 MiB of w13 scales).

    Args:
        scales_u8: E8M0 bytes, [num_experts, ...]; rewritten in place.

    Returns:
        The per-expert global scales (float32, [num_experts]).

    Raises:
        ValueError: if a scale is NaN (0xFF), if an expert's scales span more
            exponents than fp16 decodes exactly, or if a scale lies outside
            the range the kernels' fp16 scale product represents. The raster
            is left untouched then.
    """
    flat = scales_u8.view(scales_u8.size(0), -1)
    if bool((flat == 0xFF).any()):
        raise ValueError("MXFP4 scale raster contains NaN (E8M0 0xFF)")
    lo = flat.amin(dim=1).to(torch.int32)
    hi = flat.amax(dim=1).to(torch.int32)
    product_lo = int(lo.min()) - 127 + _DECODE_REBIAS_EXP
    product_hi = int(hi.max()) - 127 + _DECODE_REBIAS_EXP
    if product_lo < _FP16_POW2_MIN_EXP or product_hi > _FP16_POW2_MAX_EXP:
        raise ValueError(
            f"MXFP4 scales 2^{product_lo - _DECODE_REBIAS_EXP}.."
            f"2^{product_hi - _DECODE_REBIAS_EXP} fall outside what the fp16 "
            f"skinny kernels represent exactly"
        )
    # Lift the smallest scale to the bottom of the decode window, but never
    # so far down that the global scale times 2^14 leaves fp16.
    shift = torch.clamp(
        _E8M0_FP16_MIN - lo, min=_DECODE_REBIAS_EXP - _FP16_POW2_MAX_EXP
    )
    too_wide = hi + shift > _E8M0_FP16_MAX
    if bool(too_wide.any()):
        expert = int(torch.nonzero(too_wide)[0])
        raise ValueError(
            f"MXFP4 scales of expert {expert} span "
            f"{int(hi[expert] - lo[expert])} exponents; the fp16 skinny "
            f"kernels decode at most {_E8M0_FP16_MAX - _E8M0_FP16_MIN} exactly"
        )
    # uint8 addition wraps modulo 256, so a shift of -1 is added as 255; the
    # checks above guarantee every result lands inside 113..142.
    if not dry_run:
        flat.add_((shift % 256).to(torch.uint8).unsqueeze(1))
    return torch.pow(2.0, -shift.to(torch.float32))


@dataclass(frozen=True)
class _ScaleCaches:
    """Per-expert scales in the forms the two serving paths read."""

    # Global scales as host floats (per-expert loop) and as a device tensor
    # (grouped kernel).
    g1: list[float]
    g2: list[float]
    g1_t: torch.Tensor
    g2_t: torch.Tensor
    # Block scale rasters, viewed as the bytes the kernels decode.
    w1_scales_u8: torch.Tensor
    w2_scales_u8: torch.Tensor


class Nvfp4SkinnySm70Experts(Nvfp4QuantizationEmulationTritonExperts):
    """NVFP4 MoE over the skinny QPN kernels on fragment-order expert weights."""

    def __init__(self, moe_config, quant_config):
        # Skip the emulation __init__ (its "dequantize on the fly" warnings
        # would be wrong here) but keep its scale stashing: the uint8 scale
        # rasters move out of the quant config so no base-class path
        # mistakes them for fp scales.
        TritonExperts.__init__(self, moe_config, quant_config)
        logger.info_once(
            "Using %s MoE backend: skinny QPN FP4 kernels on fragment-order "
            "expert weights, fp16 activations.",
            type(self).__name__,
        )
        w1_scale, w2_scale = self.quant_config.w1_scale, self.quant_config.w2_scale
        assert w1_scale is not None and w2_scale is not None, (
            "skinny NVFP4 MoE needs the checkpoint's block scales"
        )
        self._w1_block_scales: torch.Tensor = w1_scale
        self._w2_block_scales: torch.Tensor = w2_scale
        # 0 = NVFP4 (fp8-e4m3 per 16 codes), 1 = MXFP4 (E8M0 per 32)
        self._scale_mode: int = 0
        # From this worker's own card: a node may mix card generations.
        self._fragment_order = qpn_fragment_order(
            torch.accelerator.current_device_index()
        )
        # Prefill chunks with many rows per expert can take the tensor-core
        # Triton projection; decided once, for this worker's card.
        self._triton_tensor_cores = (
            get_current_vllm_config().kernel_config.use_triton_sm7x_tensor_cores()
        )
        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None
        self.quantization_emulation = False
        # Built on first apply (weights live on the GPU then).
        self._scale_caches: _ScaleCaches | None = None
        # csf_32x2_v1 planes (w13, w2): the block scales above are then shared
        # rasters that each grouped call fills for its routed experts.
        self._csf_planes: tuple[CsfExpertScales, CsfExpertScales] | None = None
        self._csf_scale_entries: torch.Tensor | None = None
        # The same scales as five-bit planes (s5_scales.py), which the kernels
        # read in place; they replace the csf planes once the weights are loaded.
        self._s5: tuple[S5Scales, S5Scales] | None = None

    def attach_csf_scales(self, w13: CsfExpertScales, w2: CsfExpertScales) -> None:
        """Serve compressed expert scales; call before process_weights_after_loading."""
        self._csf_planes = (w13, w2)
        self._csf_scale_entries = qpn_scale_entries(
            self._fragment_order, w13.bases.device
        )

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        # w4a16: activations are never quantized.
        return None

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return weight_key == kNvfp4Static and activation_key in (None, kNvfp4Dynamic)

    @staticmethod
    def _mxfp4_foldable(scales_u8: torch.Tensor) -> bool:
        """True when the NVFP4 scale raster is MXFP4 in disguise.

        NVFP4 ships one fp8-e4m3 scale per 16 codes. A checkpoint converted
        from MXFP4 has every scale duplicated across the two halves of its
        32-code block, and all scales are exact powers of two -- that is what
        an E8M0 exponent becomes when written as fp8. In that case only half
        the raster carries information and the kernel can read it as MXFP4.
        """
        # Byte arithmetic only: converting the whole raster to float costs
        # gigabytes of transients mid-load (256 experts per layer).
        # fp8-e4m3 is [sign:1][exp:4][mantissa:3]; a power of two has zero
        # mantissa, and the scales are positive.
        b = scales_u8.reshape(-1)
        if b.numel() % 2 or b.numel() == 0:
            return False
        if bool((b & 0x87).any()) or bool((b == 0).any()):
            # Zero is not an E8M0 power of two. Folding it would turn a zero
            # scale into 2^-7 and change the checkpoint's weights.
            return False
        pairs = b.view(-1, 2)
        return bool(torch.equal(pairs[:, 0], pairs[:, 1]))

    @staticmethod
    def _release_by_storage(
        layer: torch.nn.Module, tensors: tuple[torch.Tensor, ...]
    ) -> None:
        """Empty every layer parameter/buffer backed by these tensors."""
        ptrs = {t.untyped_storage().data_ptr() for t in tensors}
        freed = 0
        for holder in (layer._parameters, layer._buffers):
            for name, val in list(holder.items()):
                if val is None:
                    continue
                data = val.data if isinstance(val, torch.nn.Parameter) else val
                if data.numel() and data.untyped_storage().data_ptr() in ptrs:
                    freed += data.numel()
                    empty = data.new_empty(0)
                    if isinstance(val, torch.nn.Parameter):
                        val.data = empty
                    else:
                        holder[name] = empty
        for attr in ("w1_scale", "w2_scale"):
            if getattr(layer, attr, None) is not None:
                setattr(layer, attr, None)
        if freed:
            torch.accelerator.empty_cache()
            logger.info_once(
                "Skinny MoE: released %.2f MiB of full-length scale rasters",
                freed / 2**20,
            )

    @staticmethod
    def _fold_to_e8m0(scales_u8: torch.Tensor) -> torch.Tensor:
        """Drop the duplicate half and rewrite as E8M0 (value = 2^(b-127)).

        With a zero mantissa, fp8-e4m3 is 2^(e-7) where e = byte >> 3, so the
        E8M0 byte is e + 120. Pure integer work, no float transients.
        """
        half = scales_u8[..., ::2]
        return ((half >> 3) + 120).to(torch.uint8).contiguous()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Re-permute expert weights + scale rasters into QPN fragment
        order, in place (byte-equal permutation; shapes and footprint
        unchanged). Runs once per layer during weight loading, before the
        KV-cache pool is reserved, so the per-expert transients are safe.
        Every serving path below reads this layout; there is no
        checkpoint-layout copy left afterwards.
        """
        w13, w2 = layer.w13_weight.data, layer.w2_weight.data
        for name, w in (("w13", w13), ("w2", w2)):
            if w.size(1) % 32 or (w.size(2) * 2) % 64:
                raise RuntimeError(
                    f"skinny NVFP4 MoE requires QPN-eligible expert shapes "
                    f"(N % 32, K % 64); got {name} N={w.size(1)} "
                    f"K={w.size(2) * 2}"
                )
        if self._csf_planes is not None:
            # The shared rasters take the expanded scales in this card's order.
            for e in range(w13.size(0)):
                w13[e].view(-1).copy_(
                    qpn_prepack(w13[e], None, 16, self._fragment_order)[0]
                )
                w2[e].view(-1).copy_(
                    qpn_prepack(w2[e], None, 16, self._fragment_order)[0]
                )
            self._s5 = self._csf_to_s5()
            logger.info_once(
                "Skinny MoE: expert codes re-permuted to QPN fragment order; "
                "csf_32x2_v1 scales re-encoded as five-bit planes (%d experts)",
                w13.size(0),
            )
            return
        s13, s2, sg = self._scale_rasters(layer)
        order = self._fragment_order
        for e in range(w13.size(0)):
            qc, qs = qpn_prepack(w13[e], s13[e], sg, order)
            w13[e].view(-1).copy_(qc)
            s13[e].view(-1).copy_(qs)
            qc, qs = qpn_prepack(w2[e], s2[e], sg, order)
            w2[e].view(-1).copy_(qc)
            s2[e].view(-1).copy_(qs)
        if sg == 32:
            self._w1_block_scales = s13
            self._w2_block_scales = s2
            self._scale_mode = 1
        logger.info_once(
            "Skinny MoE: expert weights re-permuted to %s fragment order in "
            "place (%d experts, one scale per %d codes)",
            order.name.lower(),
            w13.size(0),
            sg,
        )

    def _csf_to_s5(self) -> tuple[S5Scales, S5Scales]:
        """Re-encode the csf planes as five-bit planes and drop the csf planes.

        Each projection set is expanded for all experts into its shared raster
        (the expansion kernel the csf planes always went through), then packed.
        """
        assert self._csf_planes is not None and self._csf_scale_entries is not None
        scales = self._get_scale_caches()
        converted = []
        for planes, raster in zip(
            self._csf_planes, (scales.w1_scales_u8, scales.w2_scales_u8)
        ):
            experts = raster.shape[0]
            routed = torch.arange(experts, dtype=torch.int32, device=raster.device)
            expand_csf32_experts(
                raster.view(experts, -1),
                planes,
                routed,
                torch.arange(experts + 1, dtype=torch.int32, device=raster.device),
                self._csf_scale_entries,
            )
            converted.append(encode_s5(raster.view(experts, -1), planes.groups))
        self._csf_planes = None
        self._csf_scale_entries = None
        return converted[0], converted[1]

    def _scale_rasters(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return the w13/w2 scale rasters as bytes and their group size."""
        s13 = self._w1_block_scales.view(torch.uint8)
        s2 = self._w2_block_scales.view(torch.uint8)
        # MXFP4 keeps one scale per 32 codes where NVFP4 has one per 16. When
        # the raster is MXFP4 in disguise, half of it is redundant: fold it and
        # hand the kernel the shorter table. The codes are untouched either way
        # -- a 32-code MXFP4 block is two adjacent 16-code groups.
        if self._mxfp4_foldable(s13) and self._mxfp4_foldable(s2):
            full13, full2 = s13, s2
            s13 = self._fold_to_e8m0(s13)
            s2 = self._fold_to_e8m0(s2)
            # Release the full-length rasters. Dropping our own reference is
            # not enough: the layer still holds the parameters they came from,
            # so the folded copy would sit on top of the original instead of
            # replacing it (measured: 1.9 GiB per stage LOST, not gained).
            # Match by storage pointer rather than by attribute name.
            self._release_by_storage(layer, (full13, full2))
            del full13, full2
            logger.info_once(
                "Skinny MoE: scale rasters folded to MXFP4 (one E8M0 scale per "
                "32 codes), halving scale memory for %d experts",
                s13.size(0),
            )
            return s13, s2, 32
        return s13, s2, 16

    @staticmethod
    def _supports_current_device() -> bool:
        # The kernels run mma.m8n8k4 on SM70 and m16n8k8 on SM75, tested on
        # both. The capability comes from this worker's own device: on a node
        # that mixes card generations, device 0 answers for another card.
        if not current_platform.is_cuda() or not torch.accelerator.is_available():
            return False
        device_id = torch.accelerator.current_device_index()
        return any(
            current_platform.is_device_capability(cap, device_id=device_id)
            for cap in ((7, 0), (7, 5))
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        from vllm.config import get_current_vllm_config

        if not cls._supports_current_device():
            return False, "requires an SM70 or SM75 CUDA device"
        kernel_config = get_current_vllm_config().kernel_config
        # Record consultation before the disable check: enabled and disabled
        # executions of a loaded MoE family require distinct compiled graphs.
        kernel_config.sm70_skinny_moe_applicable = True
        if not kernel_config.sm70_skinny_moe:
            return False, "disabled by kernel_config.sm70_skinny_moe"
        order = qpn_fragment_order(torch.accelerator.current_device_index())
        for name, k in (
            ("hidden", moe_config.hidden_dim),
            ("intermediate", moe_config.intermediate_size_per_partition),
        ):
            try:
                grouped_splitk(k, order)
            except ValueError as exc:
                return False, f"{name} dimension: {exc}"
        if moe_config.in_dtype != torch.float16:
            return False, "requires FP16 activations"
        if moe_config.has_bias:
            return False, "expert bias is unsupported"
        if moe_config.is_lora_enabled:
            return False, "LoRA is unsupported"
        if moe_config.apply_router_weight_on_input:
            return False, "input-side router weighting is unsupported"
        if not 0 < moe_config.experts_per_token <= _CUDA_GRID_Y_LIMIT:
            return False, "requires a positive routing width"
        if moe_config.moe_parallel_config.ep_size > 1:
            return False, "expert-parallel routing is unsupported"
        if moe_config.moe_parallel_config.enable_eplb:
            return False, "expert load balancing is unsupported"
        if not hasattr(torch.ops._C, "skinny_moe_qpn_sm70"):
            return False, "native skinny MoE operator is missing"
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The per-expert loop allocates its own small transients; only the
        # fused output buffer is needed from the framework.
        return ((8,), (8,), (M, K))

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        # Grid-y is the native routing limit, independent of model, TP or
        # concurrency. Chunking also bounds slot-major temporary allocations.
        top_k = topk_ids.size(1)
        if not 0 < top_k <= _CUDA_GRID_Y_LIMIT:
            raise ValueError("routing width exceeds the native CUDA grid-y limit")
        bytes_per_token = top_k * 2 * (w1.size(1) + w2.size(1))
        chunk = min(_CUDA_GRID_Y_LIMIT // top_k, max(1, (128 << 20) // bytes_per_token))
        for start in range(0, hidden_states.size(0), chunk):
            end = start + chunk
            self._apply_grouped(
                output[start:end],
                hidden_states[start:end].contiguous(),
                w1,
                w2,
                topk_weights[start:end],
                topk_ids[start:end],
                activation,
                expert_map,
                apply_router_weight_on_input,
            )

    def _check_apply_args(
        self, w1, hidden_states, expert_map, apply_router_weight_on_input
    ):
        assert w1.dtype == torch.uint8
        assert hidden_states.dtype == torch.float16, (
            f"skinny NVFP4 kernels take fp16 activations, got {hidden_states.dtype}"
        )
        if expert_map is not None:
            raise NotImplementedError(
                "Per-expert skinny NVFP4 MoE does not support expert parallelism."
            )
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "Per-expert skinny NVFP4 MoE does not support "
                "apply_router_weight_on_input."
            )

    def _get_scale_caches(self) -> _ScaleCaches:
        if self._scale_caches is None:
            g1, g2 = self.quant_config.g1_alphas, self.quant_config.g2_alphas
            assert g1 is not None and g2 is not None, (
                "skinny NVFP4 MoE needs the checkpoint's global scales"
            )
            self._scale_caches = _ScaleCaches(
                g1=g1.cpu().tolist(),
                g2=g2.cpu().tolist(),
                g1_t=g1.to(torch.float32).contiguous(),
                g2_t=g2.to(torch.float32).contiguous(),
                w1_scales_u8=self._w1_block_scales.view(torch.uint8),
                w2_scales_u8=self._w2_block_scales.view(torch.uint8),
            )
        return self._scale_caches

    def _apply_grouped(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        expert_map,
        apply_router_weight_on_input,
    ):
        self._check_apply_args(
            w1, hidden_states, expert_map, apply_router_weight_on_input
        )
        scales = self._get_scale_caches()
        num_tokens, top_k = topk_ids.shape
        inter_dim = self.adjust_N_for_activation(w1.size(1), activation)
        device = hidden_states.device
        # Compact device-side routing (all fixed shapes, CUDA-graph safe): the
        # kernel launches min(S, experts) group rows and skips the empty ones.
        num_slots = topk_ids.numel()
        perm, gids32, goff32 = group_routing(topk_ids)
        s5_w13, s5_w2 = self._s5 if self._s5 is not None else (None, None)
        y13 = torch.empty(
            (num_slots, w1.size(1)), dtype=hidden_states.dtype, device=device
        )
        self._grouped_projection(
            hidden_states,
            w1,
            scales.w1_scales_u8,
            scales.g1_t,
            perm,
            gids32,
            goff32,
            top_k,
            y13,
            False,
            num_tokens,
            s5_w13,
        )
        inter = torch.empty((num_slots, inter_dim), dtype=y13.dtype, device=device)
        self.activation(activation, inter, y13)
        y2 = torch.empty((num_slots, w2.size(1)), dtype=y13.dtype, device=device)
        self._grouped_projection(
            inter,
            w2,
            scales.w2_scales_u8,
            scales.g2_t,
            perm,
            gids32,
            goff32,
            top_k,
            y2,
            True,
            num_tokens,
            s5_w2,
        )
        # One pass over the slot rows straight into the output (the reduction
        # moe_unpermute computes) instead of writing and re-reading two FP32
        # copies of y2.
        _mtp_weighted_reduce(y2, topk_weights, output)

    def _grouped_projection(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        scales_u8: torch.Tensor,
        global_scales: torch.Tensor,
        perm: torch.Tensor,
        gids: torch.Tensor,
        goff: torch.Tensor,
        top_k: int,
        y: torch.Tensor,
        x_slot_major: bool,
        num_tokens: int,
        s5: S5Scales | None = None,
    ) -> None:
        """y[slot] = x[row] @ W[expert].T over the expert groups of the slots.

        With five-bit scale planes ``scales_u8`` is only the raster the op fills
        for launches that do not decode the planes in place.
        """
        # Volta's hand kernel decodes every weight tile again for each 8-row
        # block; past 13 rows per expert on average the Triton kernel, which
        # decodes a tile once per row block, is faster.
        if (
            self._triton_tensor_cores
            and self._fragment_order == QpnFragmentOrder.M8N8K4
            and self._scale_mode == 1
            and y.size(0) > TRITON_MIN_ROWS_PER_EXPERT * w.size(0)
        ):
            skinny_moe_e8m0_m8n8k4_triton(
                x, w, scales_u8, global_scales, perm, gids, goff, top_k, y, x_slot_major
            )
            return
        torch.ops._C.skinny_moe_qpn_sm70(
            x,
            w,
            scales_u8,
            global_scales,
            perm,
            gids,
            goff,
            top_k,
            y,
            x_slot_major,
            num_tokens,
            grouped_splitk(x.size(1), self._fragment_order),
            self._scale_mode,
            self._fragment_order,
            None if s5 is None else s5.base,
            None if s5 is None else s5.nib,
            None if s5 is None else s5.hi,
        )


class Mxfp4SkinnySm70Experts(Nvfp4SkinnySm70Experts):
    """MXFP4 MoE through the same skinny kernels, in their E8M0 scale mode.

    MXFP4 differs from NVFP4 only in the scale raster: one E8M0 exponent per
    32 codes instead of an fp8 scale per 16. The codes, their packing and
    the fragment order are identical. The oracle hands over rasters already
    rebased by :func:`rebase_e8m0_for_fp16`, with the per-expert global
    scales as ``g1_alphas``/``g2_alphas``.
    """

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, None)

    def _scale_rasters(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        return (
            self._w1_block_scales.view(torch.uint8),
            self._w2_block_scales.view(torch.uint8),
            32,
        )


def skinny_backend_admitted(moe_config, *, mxfp4: bool = False) -> bool:
    """Reuse the oracle's capability contract before an SM70 legacy override."""
    from vllm.config import get_current_vllm_config

    cls = Mxfp4SkinnySm70Experts if mxfp4 else Nvfp4SkinnySm70Experts
    if not cls._supports_current_device():
        return False
    supported, reason = cls.is_supported_config(
        cls,
        moe_config,
        kMxfp4Static if mxfp4 else kNvfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )
    key = (
        f"{cls.__name__}:{moe_config.hidden_dim}:"
        f"{moe_config.intermediate_size_per_partition}"
    )
    get_current_vllm_config().kernel_config.moe_kernel_selections[key] = {
        "kernel": cls.__name__,
        "enabled": supported,
        "reason": reason,
        "scope": "configured_moe_capability",
    }
    if not supported:
        logger.info_once("Skinny MoE fallback: %s", reason)
    return supported
