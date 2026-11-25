# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 image preprocessing and multimodal processing on SM70.

Port of upstream `vllm/models/inkling2/processing.py` to this fork's
multimodal processor API. Images are not resized: each one is padded to a
40-pixel grid (padding pixels at -1) and cut into one 40x40 cell per decoder
token; the encoder normalizes the raw RGB values itself. Images above
`VLLM_INKLING2_SM70_MAX_IMAGE_TOKENS` cells (default 4096, e.g. 2560x2560) are
first scaled down to fit, which upstream does not do.

The chat template renders each image as `<|content_image|><|unused_200054|>`
(upstream's own renderer emits only the marker); the processor replaces that
pair with the marker followed by `rows * cols` pad tokens, the same ids as
upstream. Only the pad positions receive vision embeddings.

`VLLM_INKLING2_SM70_VISION=0` offers no image modality, so the model runs as
text-only (token-id inputs).
"""

from __future__ import annotations

import io
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

VISION = os.environ.get("VLLM_INKLING2_SM70_VISION", "1") == "1"
MAX_IMAGE_TOKENS = int(os.environ.get("VLLM_INKLING2_SM70_MAX_IMAGE_TOKENS", "4096"))

# <|content_image|>: block marker, kept verbatim.
IMAGE_MARKER_ID = 200005
IMAGE_MARKER = "<|content_image|>"
# <|unused_200054|>: one per decoder image token, overwritten by embeddings.
IMAGE_PAD_ID = 200054
IMAGE_PAD = "<|unused_200054|>"
# What the chat template renders for one image.
IMAGE_PLACEHOLDER = IMAGE_MARKER + IMAGE_PAD

PATCH_SIZE = 40


def _composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    rgba = rgba.astype(np.uint16)
    height, width, _ = rgba.shape
    y, x = np.ogrid[:height, :width]
    background = np.where((x // 8 + y // 8) % 2 == 0, 204, 153).astype(np.uint16)
    alpha = rgba[:, :, 3:4]
    blended = rgba[:, :, :3] * alpha + background[:, :, None] * (255 - alpha) + 127
    return (blended // 255).astype(np.uint8)


def _pil_to_rgb(image: Image.Image) -> np.ndarray:
    if not image.has_transparency_data:
        return np.array(image.convert("RGB"), dtype=np.uint8)
    return _composite_on_checkerboard(np.array(image.convert("RGBA")))


def decode_image(image: Any) -> np.ndarray:
    """Decode an image to HWC uint8 RGB, compositing alpha on a checkerboard.

    Encoded bytes (including `MediaWithBytes.original_bytes`) are preferred
    over decoded media, so no vLLM-side mode conversion leaks into the pixels.
    """
    if isinstance(image, MediaWithBytes):
        image = image.original_bytes if image.original_bytes else image.media
    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if data.startswith(b"\xff\xd8") and not data.endswith(b"\xff\xd9"):
            data += b"\xff\xd9"
        with Image.open(io.BytesIO(data)) as pil:
            pil.load()
            return _pil_to_rgb(pil)
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return _pil_to_rgb(image)


def image_grid(width: int, height: int) -> tuple[int, int]:
    return math.ceil(height / PATCH_SIZE), math.ceil(width / PATCH_SIZE)


def fitted_size(
    width: int, height: int, max_tokens: int = MAX_IMAGE_TOKENS
) -> tuple[int, int]:
    """`(width, height)` unchanged when its grid has at most `max_tokens` cells,
    else the largest same-aspect size whose grid fits."""
    rows, cols = image_grid(width, height)
    if rows * cols <= max_tokens:
        return width, height
    scale = math.sqrt(max_tokens * PATCH_SIZE * PATCH_SIZE / (width * height))
    while True:
        w, h = max(1, int(width * scale)), max(1, int(height * scale))
        rows, cols = image_grid(w, h)
        if rows * cols <= max_tokens:
            return w, h
        scale *= 0.99


def num_image_tokens(width: int, height: int) -> int:
    rows, cols = image_grid(*fitted_size(width, height))
    return rows * cols


def preprocess(images: Sequence[Any]) -> dict[str, torch.Tensor]:
    """`vision_pixels` (bf16 `[sum(rows*cols), 40, 40, 3]` raw RGB, padding at -1)
    and `vision_grid_thw` (`[n, 3]` = 1, rows, cols)."""
    decoded = []
    for image in images:
        rgb = decode_image(image)
        height, width, _ = rgb.shape
        w, h = fitted_size(width, height)
        if (w, h) != (width, height):
            rgb = np.array(
                Image.fromarray(rgb).resize((w, h), Image.Resampling.BICUBIC),
                dtype=np.uint8,
            )
        decoded.append(torch.from_numpy(rgb))
    grids = [image_grid(p.shape[1], p.shape[0]) for p in decoded]
    out = torch.empty(
        (sum(r * c for r, c in grids), PATCH_SIZE, PATCH_SIZE, 3), dtype=torch.bfloat16
    )
    start = 0
    for pixels, (rows, cols) in zip(decoded, grids):
        height, width, _ = pixels.shape
        full = pixels.to(torch.bfloat16)
        if (height, width) != (rows * PATCH_SIZE, cols * PATCH_SIZE):
            full = torch.full(
                (rows * PATCH_SIZE, cols * PATCH_SIZE, 3), -1, dtype=torch.bfloat16
            )
            full[:height, :width] = pixels
        cells = full.view(rows, PATCH_SIZE, cols, PATCH_SIZE, 3)
        out[start : start + rows * cols].view(
            rows, cols, PATCH_SIZE, PATCH_SIZE, 3
        ).copy_(cells.permute(0, 2, 1, 3, 4))
        start += rows * cols
    return {
        "vision_pixels": out,
        "vision_grid_thw": torch.tensor(
            [[1, r, c] for r, c in grids], dtype=torch.int64
        ),
    }


class Inkling2ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Empty: text-only (the runner keeps token-id inputs).
        return {"image": None} if VISION else {}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        return {"image": MAX_IMAGE_TOKENS}

    def get_hf_processor(self, **kwargs: object):
        raise NotImplementedError("Inkling2 image preprocessing runs in vLLM")


class Inkling2DummyInputsBuilder(BaseDummyInputsBuilder[Inkling2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        # A square image at the token cap.
        side = PATCH_SIZE * math.isqrt(MAX_IMAGE_TOKENS)
        return {
            "image": self._get_dummy_images(
                width=side,
                height=side,
                num_images=mm_counts.get("image", 0),
                overrides=(mm_options or {}).get("image"),
            )
        }


class Inkling2MultiModalProcessor(BaseMultiModalProcessor[Inkling2ProcessingInfo]):
    def _get_hf_mm_data(self, mm_items: MultiModalDataItems):
        processor_data, passthrough_data = super()._get_hf_mm_data(mm_items)
        images = mm_items.get("image")
        if isinstance(images, ImageProcessorItems):
            # Raw items keep MediaWithBytes.original_bytes for exact decoding.
            processor_data = {**processor_data, "images": list(images.data)}
        return processor_data, passthrough_data

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        data: dict[str, Any] = {
            "input_ids": [tokenizer.encode(prompt, add_special_tokens=False)]
        }
        images = mm_data.get("images") or []
        if not isinstance(images, list):
            images = [images]
        if images:
            data.update(preprocess(images))
        return BatchFeature(data, tensor_type="pt")

    def _hf_processor_applies_updates(self, *args, **kwargs) -> bool:
        return False

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        grid_thw = hf_inputs.get(
            "vision_grid_thw", torch.empty((0, 3), dtype=torch.int64)
        )
        return dict(
            vision_pixels=MultiModalFieldConfig.flat_from_sizes(
                "image", grid_thw.prod(-1)
            ),
            vision_grid_thw=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out_items = out_mm_kwargs.get("image", [])

        def replacement(item_idx: int) -> PromptUpdateDetails:
            item = out_items[item_idx] if item_idx < len(out_items) else None
            if item is not None:
                num_tokens = int(item["vision_grid_thw"].data.prod())
            else:
                size = mm_items.get_items("image", ImageProcessorItems).get_image_size(
                    item_idx
                )
                num_tokens = num_image_tokens(size.width, size.height)
            return PromptUpdateDetails.select_token_id(
                [IMAGE_MARKER_ID] + [IMAGE_PAD_ID] * num_tokens, IMAGE_PAD_ID
            )

        return [
            PromptReplacement(
                modality="image",
                target=[IMAGE_MARKER_ID, IMAGE_PAD_ID],
                replacement=replacement,
            )
        ]
