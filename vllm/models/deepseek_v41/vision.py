# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 vision input (VLLM_V41_VISION=1).

GGUF route: the tower's weights come from llama.cpp's vision projector
(``VLLM_V41_VISION_MMPROJ``, projector type deepseek41v) and the MoE image
routing bias from its F32 sidecar (``VLLM_V41_VISION_BIAS_VL``, the
``exp_probs_b_vl.f32`` of the GGUF repository: [layers, experts]).

A port of the checkpoint's reference ``inference/vision.py`` and
``inference/image_processor.py``. An image becomes a ``n_vit_h x n_vit_w``
patch grid for the ViT and a ``n_llm_h x n_llm_w`` token grid after the 3x3
aligner downsample, which the language model sees as

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every position of that span carries ``image_token_id`` in the prompt; the
embedding of each position (learned delimiters, aligner rows) comes from the
vision tower. The language model tells image positions apart by that token id:
they take the VL routing bias in every MoE gate and no Engram n-gram.
"""

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from transformers import BatchFeature

import vllm.envs as envs
from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

VISION = envs.VLLM_V41_VISION == "1"
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
DEFAULT_IMAGE_TOKEN_ID = 129264


# ---------------------------------------------------------------- geometry


class V41VisionGeometry:
    """The checkpoint's vision_config fields used by the preprocessing."""

    def __init__(self, vision_config: Mapping[str, Any] | None):
        v = dict(vision_config or {})
        self.patch_size = int(v.get("patch_size", 14))
        self.downsample_ratio = int(v.get("downsample_ratio", 3))
        self.max_n_token = int(v.get("max_image_tokens", 1024))
        self.min_pixels = int(v.get("min_pixels", 544 * 544))
        ratio = v.get("max_wh_ratio")
        self.max_wh_ratio = None if ratio is None else int(ratio)
        self.dim = int(v.get("hidden_size", 1024))
        self.n_heads = int(v.get("num_attention_heads", 16))
        self.inter_dim = int(v.get("intermediate_size", 2816))
        self.n_layers = int(v.get("num_hidden_layers", 32))
        self.rope_theta = float(v.get("rope_theta", 10000.0))


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def _llm_grid(best_height, best_width, patch_size, downsample_ratio):
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def _solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(
        math.floor(max_w_float) * cell / width,
        math.floor(max_h_float) * cell / height,
    )
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def _safe_resize(
    height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token
):
    n_llm_h, n_llm_w = _llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = _solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = _llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, g: V41VisionGeometry):
    """(n_llm_h, n_llm_w, best_height, best_width) of an image of this size."""
    p = g.patch_size
    if g.max_wh_ratio is not None and width > height * g.max_wh_ratio:
        width = height * g.max_wh_ratio
    if 0 < width * height < g.min_pixels:
        ratio = (g.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return _safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        g.downsample_ratio,
        g.max_n_token,
    )


def image_patches(image: Image.Image, g: V41VisionGeometry):
    """(patches [n_vit_h * n_vit_w, 3 * p * p] FP16, n_vit_h, n_vit_w).

    The reference rounds the normalized pixels to BF16; FP16 holds every BF16
    value in [-1, 1] exactly, so the ViT sees the same inputs."""
    p = g.patch_size
    image = image.convert("RGB")
    _, _, best_height, best_width = plan_image_grid(image.width, image.height, g)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if g.max_wh_ratio is not None and image.width >= g.max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3 * p * p)
    )
    return patches.to(torch.float16), n_vit_h, n_vit_w


# ---------------------------------------------------------------- tower


@lru_cache(8)
def _vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float, device: str):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return (
        freqs.cos().unsqueeze(1).to(device),
        freqs.sin().unsqueeze(1).to(device),
    )


def _rotary(x, cos, sin):
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(dim, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * x


def _linear(i, o, bias):
    layer = nn.Linear(i, o, bias=bias)
    layer.weight.requires_grad_(False)
    if bias:
        layer.bias.requires_grad_(False)
    return layer


class _PatchEmbed(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.proj = _linear(3 * g.patch_size**2, g.dim, True)


class _Attention(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.n_heads = g.n_heads
        self.head_dim = g.dim // g.n_heads
        self.wqkv = _linear(g.dim, 3 * g.dim, True)
        self.wo = _linear(g.dim, g.dim, True)

    def forward(self, x, cos, sin):
        n = x.size(0)
        q, k, v = (
            t.reshape(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
        )[0]
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class _MLP(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.w1 = _linear(g.dim, 2 * g.inter_dim, False)
        self.w2 = _linear(g.inter_dim, g.dim, False)

    def forward(self, x):
        gate, up = self.w1(x).float().chunk(2, dim=-1)
        return self.w2((F.silu(gate) * up).to(x.dtype))


class _Block(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.norm1 = _RMSNorm(g.dim)
        self.attn = _Attention(g)
        self.norm2 = _RMSNorm(g.dim)
        self.mlp = _MLP(g)


class _ViT(nn.Module):
    """Full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, g):
        super().__init__()
        self.g = g
        self.patch_embed = _PatchEmbed(g)
        self.blocks = nn.ModuleList([_Block(g) for _ in range(g.n_layers)])
        self.norm = _RMSNorm(g.dim)

    def forward(self, patches, n_h, n_w):
        # The residual stream stays FP32 (the reference keeps BF16 range);
        # each sublayer takes FP16 inputs (FP32 tensor-core accumulation).
        dtype = self.patch_embed.proj.weight.dtype
        x = self.patch_embed.proj(patches.to(dtype)).float()
        cos, sin = _vision_cos_sin(
            n_h,
            n_w,
            self.g.dim // self.g.n_heads // 2,
            self.g.rope_theta,
            str(x.device),
        )
        for block in self.blocks:
            x = x + block.attn(block.norm1(x).to(dtype), cos, sin).float()
            x = x + block.mlp(block.norm2(x).to(dtype)).float()
        return self.norm(x)


class _Aligner(nn.Module):
    def __init__(self, g, out_dim):
        super().__init__()
        self.r = g.downsample_ratio
        self.w1 = _linear(g.dim * self.r**2, out_dim, True)
        self.w2 = _linear(out_dim, out_dim, True)

    def forward(self, x, n_h, n_w):
        r, dtype = self.r, self.w1.weight.dtype
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x.to(dtype)).float()).to(dtype))


def mmproj_state_dict(path: str) -> dict[str, torch.Tensor]:
    """llama.cpp mmproj (deepseek41v) -> the checkpoint's tower names (FP32)."""
    from gguf import GGUFReader

    reader = GGUFReader(path)
    field = reader.fields.get("clip.projector_type")
    kind = field.contents() if field is not None else None
    if kind not in ("deepseek41v", "deepseek4v"):
        raise ValueError(f"{path}: not a DeepSeek V4.1 vision projector ({kind})")
    tensors = {t.name: t for t in reader.tensors}

    def get(name):
        t = tensors[name]
        a = np.asarray(t.data)
        if t.tensor_type.name == "BF16":
            a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif t.tensor_type.name in ("F32", "F16"):
            a = a.astype(np.float32)
        else:
            raise ValueError(
                f"{path}: {name} is {t.tensor_type.name} (F32/F16/BF16 only)"
            )
        shape = [int(d) for d in reversed(t.shape.tolist())]
        return torch.from_numpy(a.reshape(shape).copy())

    sd = {
        # conv2d [dim, 3, p, p] == the reference Linear over flattened patches
        "vision.patch_embed.proj.weight": get("v.patch_embd.weight").flatten(1),
        "vision.patch_embed.proj.bias": get("v.patch_embd.bias"),
        "vision.norm.weight": get("v.post_ln.weight"),
        "aligner.w1.weight": get("mm.1.weight"),
        "aligner.w1.bias": get("mm.1.bias"),
        "aligner.w2.weight": get("mm.2.weight"),
        "aligner.w2.bias": get("mm.2.bias"),
        "image_start": get("v.token_embd.img_start"),
        "image_end": get("v.token_embd.img_end"),
        "image_newline": get("v.image_newline"),
    }
    n = 0
    while f"v.blk.{n}.ln1.weight" in tensors:
        b, o = f"v.blk.{n}.", f"vision.blocks.{n}."
        sd[o + "norm1.weight"] = get(b + "ln1.weight")
        sd[o + "norm2.weight"] = get(b + "ln2.weight")
        sd[o + "attn.wqkv.weight"] = get(b + "attn_qkv.weight")
        sd[o + "attn.wqkv.bias"] = get(b + "attn_qkv.bias")
        sd[o + "attn.wo.weight"] = get(b + "attn_out.weight")
        sd[o + "attn.wo.bias"] = get(b + "attn_out.bias")
        # llama.cpp splits the fused SwiGLU w1 (gate first, then up)
        sd[o + "mlp.w1.weight"] = torch.cat(
            [get(b + "ffn_gate.weight"), get(b + "ffn_up.weight")]
        )
        sd[o + "mlp.w2.weight"] = get(b + "ffn_down.weight")
        n += 1
    return sd


def load_bias_vl(layers, gate_type, config) -> None:
    """Fill every local MoE gate's bias_vl from VLLM_V41_VISION_BIAS_VL."""
    path = envs.VLLM_V41_VISION_BIAS_VL
    if not path:
        raise ValueError(
            "VLLM_V41_VISION=1 on the GGUF route needs VLLM_V41_VISION_BIAS_VL "
            "(exp_probs_b_vl.f32, F32 [layers, experts])"
        )
    n_layers, n_experts = config.num_hidden_layers, config.n_routed_experts
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != n_layers * n_experts or not np.isfinite(raw).all():
        raise ValueError(f"{path}: expected {n_layers} x {n_experts} finite F32")
    table = torch.from_numpy(raw.reshape(n_layers, n_experts).copy())
    for index, layer in enumerate(layers):
        gate = getattr(getattr(layer, "ffn", None), "gate", None)
        if isinstance(gate, gate_type) and gate.bias_vl is not None:
            gate.bias_vl.copy_(table[index].to(gate.bias_vl.device))


class V41VisionTower(nn.Module):
    """ViT + aligner + learned span delimiters; parameter names follow the
    checkpoint (``vision.*``, ``aligner.*``, ``image_start`` ...)."""

    def __init__(self, vision_config, hidden_size, dtype=torch.float16):
        super().__init__()
        g = self.geometry = V41VisionGeometry(vision_config)
        self.vision = _ViT(g)
        self.aligner = _Aligner(g, hidden_size)
        for name in ("image_start", "image_end", "image_newline"):
            self.register_parameter(
                name, nn.Parameter(torch.empty(hidden_size), requires_grad=False)
            )
        self.to(dtype)
        # RMSNorm weights stay FP32 (as the reference).
        for module in self.modules():
            if isinstance(module, _RMSNorm):
                module.weight.data = module.weight.data.float()

    @torch.no_grad()
    def load_mmproj(self, path: str | None = None) -> None:
        """Weights from llama.cpp's vision projector (VLLM_V41_VISION_MMPROJ)."""
        path = path or envs.VLLM_V41_VISION_MMPROJ
        if not path:
            raise ValueError(
                "VLLM_V41_VISION=1 on the GGUF route needs VLLM_V41_VISION_MMPROJ "
                "(mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf)"
            )
        state = mmproj_state_dict(path)
        own = dict(self.named_parameters())
        missing, extra = own.keys() - state.keys(), state.keys() - own.keys()
        if missing or extra:
            raise ValueError(
                f"{path}: tower mismatch, missing {sorted(missing)[:4]}, "
                f"extra {sorted(extra)[:4]}"
            )
        for name, value in state.items():
            parameter = own[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"{path}: {name} {tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            converted = value.to(parameter.dtype)
            if not torch.isfinite(converted).all():
                raise ValueError(f"{path}: {name} overflows {parameter.dtype}")
            parameter.data.copy_(converted.to(parameter.device))

    @torch.inference_mode()
    def span(self, patches, n_vit_h, n_vit_w):
        """The [num_image_tokens, hidden] embeddings of one image span."""
        g = self.geometry
        rows = self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)
        n_llm_h = math.ceil(n_vit_h / g.downsample_ratio)
        n_llm_w = math.ceil(n_vit_w / g.downsample_ratio)
        hidden = rows.shape[-1]
        dtype = self.image_start.dtype
        body = torch.cat(
            [
                rows.to(dtype).view(n_llm_h, n_llm_w, hidden),
                self.image_newline.view(1, 1, hidden).expand(n_llm_h, 1, hidden),
            ],
            dim=1,
        ).reshape(-1, hidden)
        return torch.cat([self.image_start[None], body, self.image_end[None]])


# ---------------------------------------------------------------- processor


def _geometry(info) -> V41VisionGeometry:
    return V41VisionGeometry(getattr(info.get_hf_config(), "vision_config", None))


def _image_token_id(info) -> int:
    return int(getattr(info.get_hf_config(), "image_token_id", DEFAULT_IMAGE_TOKEN_ID))


class DeepseekV41ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Empty: text-only (the runner keeps token-id inputs and graphs).
        return {"image": None} if VISION else {}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": _geometry(self).max_n_token}

    def get_hf_processor(self, **kwargs: object):
        raise NotImplementedError("V4.1 vision preprocessing runs in vLLM")

    def get_num_image_tokens(self, width: int, height: int) -> int:
        n_llm_h, n_llm_w, _, _ = plan_image_grid(width, height, _geometry(self))
        return num_image_tokens(n_llm_h, n_llm_w)


class DeepseekV41DummyInputsBuilder(BaseDummyInputsBuilder[DeepseekV41ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "\n\n".join([IMAGE_PLACEHOLDER] * mm_counts.get("image", 0))

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        # A square image at the token cap (1024 -> a 31 x 31 LLM grid).
        g = _geometry(self.info)
        side = 31 * g.patch_size * g.downsample_ratio
        overrides = mm_options.get("image")
        assert overrides is None or isinstance(overrides, ImageDummyOptions)
        return {
            "image": self._get_dummy_images(
                width=side,
                height=side,
                num_images=mm_counts.get("image", 0),
                overrides=overrides,
            )
        }


class DeepseekV41MultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV41ProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        images = mm_data.get("images") or []
        if not isinstance(images, list):
            images = [images]
        g = _geometry(self.info)
        patches, sizes, grids = [], [], []
        for image in images:
            p, n_h, n_w = image_patches(image, g)
            patches.append(p)
            sizes.append(p.shape[0])
            grids.append((n_h, n_w))
        data: dict[str, Any] = {"input_ids": [ids]}
        if images:
            data["pixel_patches"] = torch.cat(patches)
            data["patches_per_image"] = torch.tensor(sizes, dtype=torch.int64)
            data["vit_grid"] = torch.tensor(grids, dtype=torch.int64)
        return BatchFeature(data, tensor_type="pt")

    def _hf_processor_applies_updates(self, *args, **kwargs) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        sizes = hf_inputs.get("patches_per_image", torch.empty(0, dtype=torch.int64))
        return dict(
            pixel_patches=MultiModalFieldConfig.flat_from_sizes("image", sizes),
            patches_per_image=MultiModalFieldConfig.batched("image"),
            vit_grid=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_token_id = _image_token_id(self.info)

        def replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            size = images.get_image_size(item_idx)
            count = self.info.get_num_image_tokens(size.width, size.height)
            return [image_token_id] * count

        return [
            PromptReplacement(
                modality="image", target=[image_token_id], replacement=replacement
            )
        ]
