# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Inkling2 MoE reference path for SM70: routing, exact NVFP4 / MXFP8 expert
decoding and per-expert cuBLAS GEMMs.

Correctness baseline for the port (and the checker of faster kernels): the
active experts' weights are decoded to fp16 on the fly, so only the packed
checkpoint formats stay resident.

- NVFP4: value = e2m1[code] * e4m3(block scale, per 16 along K)
  * amax / outer_scale_denominator; the low nibble holds the even column.
- MXFP8: value = e4m3 * 2^(e8m0 - 127), one scale per 32 along K.
- w13 rows interleave gate and up (row 2i gate_i, row 2i+1 up_i).
- Activation: softcapped SwiGLU, `gcap tanh(g / gcap) * sigmoid(g) * ucap tanh(u / ucap)`.
"""

import torch

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
_lut_cache: dict[torch.device, torch.Tensor] = {}


def _e2m1_lut(device: torch.device) -> torch.Tensor:
    lut = _lut_cache.get(device)
    if lut is None:
        lut = _lut_cache[device] = torch.tensor(
            _E2M1, dtype=torch.float32, device=device
        )
    return lut


def dequant_nvfp4(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor | float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """`codes` uint8 `[..., N, K/2]`, `scales` float8_e4m3fn `[..., N, K/16]`."""
    lut = _e2m1_lut(codes.device)
    *lead, n, half = codes.shape
    k = half * 2
    values = torch.stack((lut[(codes & 15).long()], lut[(codes >> 4).long()]), dim=-1)
    values = values.view(*lead, n, k // 16, 16) * scales.float().unsqueeze(-1)
    values = values.view(*lead, n, k)
    if isinstance(global_scale, torch.Tensor):
        values = values * global_scale.view(
            *global_scale.shape, *([1] * (values.dim() - global_scale.dim()))
        )
    else:
        values = values * global_scale
    return values.to(dtype)


def dequant_mxfp8(
    codes: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """`codes` float8_e4m3fn `[..., N, K]`, `scales` uint8 E8M0 `[..., N, K/32]`."""
    *lead, n, k = codes.shape
    values = codes.float().view(*lead, n, k // 32, 32)
    values = values * torch.exp2(scales.float() - 127.0).unsqueeze(-1)
    return values.view(*lead, n, k).to(dtype)


def route(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 sigmoid routing with the correction bias: `(weights, ids)`.

    The bias only selects; weights are the selected sigmoid scores normalized
    to sum to one, times `route_scale`.
    """
    logits = torch.nn.functional.linear(x.float(), gate_weight.float())
    scores = torch.sigmoid(logits)
    ids = torch.topk(scores + gate_bias, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * route_scale, ids


def softcapped_swiglu(
    gate: torch.Tensor, up: torch.Tensor, gate_cap: float, up_cap: float
) -> torch.Tensor:
    g, u = gate.float(), up.float()
    gf = gate_cap * torch.tanh(g / gate_cap) if gate_cap > 0 else g
    if up_cap > 0:
        u = up_cap * torch.tanh(u / up_cap)
    return (gf * torch.sigmoid(g) * u).to(gate.dtype)


class ExpertWeights:
    """Decoder of one layer's packed experts into fp16 `(w13 [2I, H], w2 [H, I])`
    with gate/up de-interleaved (gate rows first)."""

    def __init__(
        self,
        fmt: str,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_global: torch.Tensor | None = None,
        w2_global: torch.Tensor | None = None,
    ):
        self.fmt = fmt
        self.w13, self.w13_scale, self.w2, self.w2_scale = w13, w13_scale, w2, w2_scale
        self.w13_global, self.w2_global = w13_global, w2_global

    def decode(
        self, experts: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the experts `experts` (1-D long) to `([n, 2I, H], [n, H, I])`."""
        if self.fmt == "nvfp4":
            w13 = dequant_nvfp4(
                self.w13[experts],
                self.w13_scale[experts],
                self.w13_global[experts],
                dtype,
            )
            w2 = dequant_nvfp4(
                self.w2[experts], self.w2_scale[experts], self.w2_global[experts], dtype
            )
        else:
            w13 = dequant_mxfp8(self.w13[experts], self.w13_scale[experts], dtype)
            w2 = dequant_mxfp8(self.w2[experts], self.w2_scale[experts], dtype)
        # rows 2i / 2i+1 -> [gate; up]
        w13 = torch.cat((w13[:, 0::2], w13[:, 1::2]), dim=1)
        return w13, w2


def experts_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    experts: ExpertWeights,
    gate_cap: float,
    up_cap: float,
    chunk: int = 16,
) -> torch.Tensor:
    """Sum over each token's top-k of `w_k * down(act(gate_up(x)))`, in fp32,
    returned in `x.dtype`."""
    num_tokens, hidden = x.shape
    out = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=x.device)
    flat_ids = ids.reshape(-1)
    order = torch.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids[order]
    token_of = order // ids.shape[1]
    weight_of = weights.reshape(-1)[order]
    present, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    bounds = torch.cumsum(counts, 0).tolist()
    starts = [0] + bounds[:-1]
    present_list = present.tolist()
    for c in range(0, len(present_list), chunk):
        group = present[c : c + chunk]
        w13, w2 = experts.decode(group, x.dtype)
        inter = w2.shape[2]
        for j, e in enumerate(present_list[c : c + chunk]):
            lo, hi = starts[c + j], bounds[c + j]
            toks = token_of[lo:hi]
            gate_up = torch.nn.functional.linear(x[toks], w13[j])
            act = softcapped_swiglu(
                gate_up[:, :inter], gate_up[:, inter:], gate_cap, up_cap
            )
            down = torch.nn.functional.linear(act, w2[j]).float()
            out.index_add_(0, toks, down * weight_of[lo:hi, None])
    return out.to(x.dtype)
