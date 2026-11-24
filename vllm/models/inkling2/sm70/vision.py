# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Inkling2 vision encoder on SM70 (V100).

Port of upstream `vllm/models/inkling2/nvidia/vision.py` (a PE-Lang-style ViT
with a CLS token per image and a 2x2 merge adapter) in eager PyTorch. The
upstream Triton kernels (`nvidia/ops/vision.py`) round to bf16 with an sm_80
instruction; here the same arithmetic runs as torch ops with the same bf16
rounding points (fp32 math in between). Linear layers stay bf16 as in the
checkpoint. Attention runs per image through fp16 SDPA (memory-efficient
kernel, fp32 accumulation): every bf16 value in fp16's range is exact in fp16,
and the V100 has no bf16 tensor cores.

Module names mirror the checkpoint, so `model.visual.<name>` loads into
`Inkling2VisionEncoder.<name>`, except that each layer's `q_proj`/`k_proj`/
`v_proj` load into one fused `qkv_proj`. Runs replicated on every TP rank.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from vllm.model_executor.model_loader.weight_utils import default_weight_loader

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# The image processor's CLIP std (float32-rounded constants): pixels are first
# CLIP-normalized with it and rounded to bf16, then mapped to [-1, 1].
_PROCESSOR_CLIP_STD = (0.26862954, 0.2613026, 0.2757771)

_BF16 = torch.bfloat16


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 to bf16 and back (round to nearest even)."""
    return x.to(_BF16).float()


def _layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """fp32 LayerNorm statistics and affine (bf16 weights), fp32 result."""
    return F.layer_norm(
        x, norm.normalized_shape, norm.weight.float(), norm.bias.float(), norm.eps
    )


def _linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=bias, dtype=_BF16)


class Inkling2VisionAttention(nn.Module):
    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv_proj = _linear(width, 3 * width)
        self.out_proj = _linear(width, width)

    def forward(
        self,
        normed: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        lengths: list[int],
    ) -> torch.Tensor:
        tokens, width = normed.shape
        qkv = self.qkv_proj(normed).view(tokens, 3, self.num_heads, self.head_dim)
        # Interleaved 2D RoPE on q and k in fp32, rounded back to bf16 (upstream
        # `apply_rotary_qk_`): pair p = (2p, 2p + 1) of each head.
        cos, sin = rope  # fp32 [T, 1, 1, head_dim / 2]
        qk = qkv[:, :2].float()
        even, odd = qk[..., 0::2], qk[..., 1::2]
        rot = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
        q, k = rot.flatten(-2).to(_BF16).to(torch.float16).unbind(1)
        v = qkv[:, 2].to(torch.float16)
        out = torch.empty(
            (tokens, self.num_heads, self.head_dim),
            dtype=torch.float16,
            device=normed.device,
        )
        start = 0
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            for n in lengths:
                sl = slice(start, start + n)
                o = F.scaled_dot_product_attention(
                    q[sl].transpose(0, 1).unsqueeze(0),
                    k[sl].transpose(0, 1).unsqueeze(0),
                    v[sl].transpose(0, 1).unsqueeze(0),
                    scale=self.head_dim**-0.5,
                )
                out[sl] = o[0].transpose(0, 1)
                start += n
        return self.out_proj(out.view(tokens, width).to(_BF16))


class Inkling2VisionMLP(nn.Module):
    def __init__(self, width: int, intermediate_size: int) -> None:
        super().__init__()
        self.c_fc = _linear(width, intermediate_size)
        self.c_proj = _linear(intermediate_size, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(hidden)))


class Inkling2VisionLayerScale(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(width, dtype=_BF16))


class Inkling2VisionLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width, eps = config["hidden_size"], config["layer_norm_eps"]
        self.ln_1 = nn.LayerNorm(width, eps=eps, dtype=_BF16)
        self.self_attn = Inkling2VisionAttention(width, config["num_attention_heads"])
        self.ln_2 = nn.LayerNorm(width, eps=eps, dtype=_BF16)
        self.mlp = Inkling2VisionMLP(width, int(width * config["mlp_ratio"]))
        self.ls_1 = Inkling2VisionLayerScale(width)
        self.ls_2 = Inkling2VisionLayerScale(width)

    @staticmethod
    def _residual(
        hidden: torch.Tensor, update: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        # upstream `scaled_residual_norm_`: bf16(hidden + bf16(update * gamma)), fp32 math.
        return _bf16(hidden + _bf16(update.float() * gamma.float()))

    def forward(self, hidden: torch.Tensor, rope, lengths: list[int]) -> torch.Tensor:
        """`hidden`: the fp32-held, bf16-valued residual stream; returns the new one."""
        attn = self.self_attn(_layer_norm(hidden, self.ln_1).to(_BF16), rope, lengths)
        hidden = self._residual(hidden, attn, self.ls_1.gamma)
        mlp = self.mlp(_layer_norm(hidden, self.ln_2).to(_BF16))
        return self._residual(hidden, mlp, self.ls_2.gamma)


class Inkling2VisionEmbeddings(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width, grid = config["hidden_size"], config["posemb_grid_size"]
        self.patch_embedding = _linear(
            3 * config["native_patch_size"] ** 2, width, bias=False
        )
        self.position_embedding = nn.Parameter(
            torch.empty(grid * grid, width, dtype=_BF16)
        )
        self.class_embedding = nn.Parameter(torch.empty(width, dtype=_BF16))
        self.cls_position_embedding = nn.Parameter(torch.empty(width, dtype=_BF16))


class Inkling2VisionBackbone(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"], dtype=_BF16
        )
        self.layers = nn.ModuleList(
            Inkling2VisionLayer(config) for _ in range(config["num_hidden_layers"])
        )


def _source_coords(target: int, source: int, device: torch.device):
    """align_corners=False bilinear source coordinates for `target` outputs,
    clamped to the source grid (upstream `_source_coords`)."""
    index = torch.arange(target, device=device, dtype=torch.float32)
    coord = (float(source) * (index + 0.5)) / float(target) - 0.5
    coord = coord.clamp(0.0, source - 1.0)
    src0 = coord.floor().long()
    src1 = (src0 + 1).clamp(max=source - 1)
    w1 = coord - src0.float()
    return src0, src1, 1.0 - w1, w1


def _lerp(
    a: torch.Tensor, b: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor
) -> torch.Tensor:
    return a * wa + b * wb


class Inkling2VisionEncoder(nn.Module):
    """Maps raw `[N, 40, 40, 3]` bf16 pixel cells (0-255, -1 = padding) to decoder
    embeddings `[N, text_hidden_size]` (bf16): a ViT over the native 20x20
    patches with a CLS token per image, then a 2x2 merge projection."""

    def __init__(self, config: Mapping[str, Any], text_hidden_size: int) -> None:
        super().__init__()
        self.patch_size = config["native_patch_size"]
        self.merge_size = config["adapter_merge_size"]
        self.posemb_grid_size = config["posemb_grid_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        self.text_hidden_size = text_hidden_size
        self.backbone = Inkling2VisionBackbone(config)
        self.proj_in = _linear(
            self.merge_size**2 * config["hidden_size"],
            config["projection_hidden_dim"],
            bias=config["projector_bias"],
        )
        self.proj_out = _linear(
            config["projection_hidden_dim"],
            text_hidden_size,
            bias=config["projector_bias"],
        )
        self._inv_freq: torch.Tensor | None = None

    # ------------------------------------------------------------ pieces

    def _patches(self, cells: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """Native patches of one image's cells as bf16 `(py, px, c)` rows in [-1, 1]
        (upstream `gather_patches`): `2 * (bf16((v / 255 - mean) / clip_std) * std + mean) - 1`,
        each op in fp32."""
        m, p = self.merge_size, self.patch_size
        x = cells.view(rows, cols, m, p, m, p, 3).permute(0, 2, 1, 4, 3, 5, 6)
        x = x.reshape(rows * m * cols * m, p * p * 3).float()
        dev = x.device
        channel = torch.arange(p * p * 3, device=dev) % 3
        mean = torch.tensor(_CLIP_MEAN, dtype=torch.float32, device=dev)[channel]
        clip_std = torch.tensor(_PROCESSOR_CLIP_STD, dtype=torch.float32, device=dev)[
            channel
        ]
        std = torch.tensor(_CLIP_STD, dtype=torch.float32, device=dev)[channel]
        x = _bf16((x / 255.0 - mean) / clip_std)
        x = 2.0 * (x * std + mean) - 1.0
        return x.to(_BF16)

    def _positions(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Position embeddings resized to the native `height x width` grid, bf16-rounded
        fp32 `[height * width, D]` (upstream `_embed_kernel`)."""
        g = self.posemb_grid_size
        grid = self.backbone.embeddings.position_embedding.float().view(g, g, -1)
        y0, y1, wy0, wy1 = _source_coords(height, g, device)
        x0, x1, wx0, wx1 = _source_coords(width, g, device)
        wx0, wx1 = wx0[None, :, None], wx1[None, :, None]
        wy0, wy1 = wy0[:, None, None], wy1[:, None, None]
        top = _lerp(grid[y0][:, x0], grid[y0][:, x1], wx0, wx1)
        bottom = _lerp(grid[y1][:, x0], grid[y1][:, x1], wx0, wx1)
        return _bf16(_lerp(top, bottom, wy0, wy1)).reshape(height * width, -1)

    def _rope(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cos / sin `[1 + height * width, 1, 1, head_dim / 2]` (CLS first: angle 0). The first
        half of the pairs rotate by x + 1, the rest by y + 1 (upstream `_rotary_qk_kernel`)."""
        if self._inv_freq is None or self._inv_freq.device != device:
            # Rounded from float64 to float32, matching the reference constants.
            values = [
                10000.0 ** (-dim / self.head_dim) for dim in range(0, self.head_dim, 4)
            ]
            self._inv_freq = torch.tensor(values, dtype=torch.float32, device=device)
        inv = self._inv_freq  # [head_dim / 4]
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        x = torch.cat([torch.full((1,), -1, device=device), x.reshape(-1)]) + 1
        y = torch.cat([torch.full((1,), -1, device=device), y.reshape(-1)]) + 1
        angle = torch.cat([x[:, None].float() * inv, y[:, None].float() * inv], dim=1)
        return angle.cos()[:, None, None, :], angle.sin()[:, None, None, :]

    # ------------------------------------------------------------ forward

    @torch.inference_mode()
    def forward(self, pixels: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        side = self.patch_size * self.merge_size
        assert pixels.shape[1:] == (side, side, 3), pixels.shape
        grid = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        assert all(t == 1 for t, _, _ in grid)
        assert sum(h * w for _, h, w in grid) == pixels.shape[0]
        pixels = pixels.to(_BF16)
        emb = self.backbone.embeddings
        m = self.merge_size
        dev = pixels.device

        hidden_parts, cos_parts, sin_parts, lengths = [], [], [], []
        start = 0
        cls = _bf16(emb.class_embedding.float() + emb.cls_position_embedding.float())[
            None
        ]
        for _, rows, cols in grid:
            cells = pixels[start : start + rows * cols]
            start += rows * cols
            height, width = rows * m, cols * m
            patch = emb.patch_embedding(self._patches(cells, rows, cols)).float()
            tokens = _bf16(patch + self._positions(height, width, dev))
            hidden_parts += [cls, tokens]
            cos, sin = self._rope(height, width, dev)
            cos_parts.append(cos)
            sin_parts.append(sin)
            lengths.append(1 + height * width)
        hidden = _bf16(_layer_norm(torch.cat(hidden_parts), self.backbone.ln_pre))
        rope = (torch.cat(cos_parts), torch.cat(sin_parts))
        for layer in self.backbone.layers:
            hidden = layer(hidden, rope, lengths)

        # 2x2 merge: per image, the patch rows (after its CLS row) in
        # [rows, cols, m, m] order, then the projector.
        pooled, start = [], 0
        for (_, rows, cols), n in zip(grid, lengths):
            patches = hidden[start + 1 : start + n].view(rows, m, cols, m, -1)
            pooled.append(patches.permute(0, 2, 1, 3, 4).reshape(rows * cols, -1))
            start += n
        pooled = torch.cat(pooled).to(_BF16)
        return self.proj_out(F.gelu(self.proj_in(pooled)))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load `model.visual.*` checkpoint tensors with that prefix stripped."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        qkv_shards: Counter[str] = Counter()
        for name, weight in weights:
            for shard, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                if f".self_attn.{proj}." in name:
                    name = name.replace(proj, "qkv_proj")
                    param = params[name]
                    size = param.shape[0] // 3
                    default_weight_loader(
                        param.data[shard * size : (shard + 1) * size], weight
                    )
                    qkv_shards[name] += 1
                    break
            else:
                if name not in params:
                    raise ValueError(f"Unexpected vision checkpoint tensor: {name}")
                default_weight_loader(params[name], weight)
                loaded.add(name)
        return loaded | {name for name, count in qkv_shards.items() if count == 3}
