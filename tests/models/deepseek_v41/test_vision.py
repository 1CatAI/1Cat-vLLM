# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 vision input: geometry, span layout, chat placeholders."""

import pytest
import torch
from PIL import Image

from vllm.models.deepseek_v41.vision import (
    V41VisionGeometry,
    V41VisionTower,
    image_patches,
    num_image_tokens,
    plan_image_grid,
)
from vllm.tokenizers.deepseek_v41 import _image_blocks, _text_only

CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "intermediate_size": 2816,
    "patch_size": 14,
    "rope_theta": 10000,
    "downsample_ratio": 3,
    "max_image_tokens": 1024,
    "min_pixels": 295936,
    "max_wh_ratio": None,
}


@pytest.mark.parametrize(
    ("size", "vit", "tokens"),
    [
        # Grids of the checkpoint's reference image_processor.
        ((200, 120), (31, 51), 200),
        ((2400, 300), (22, 172), 474),
    ],
)
def test_grid_matches_reference(size, vit, tokens):
    g = V41VisionGeometry(CONFIG)
    n_llm_h, n_llm_w, height, width = plan_image_grid(*size, g)
    assert (height // 14, width // 14) == vit
    assert num_image_tokens(n_llm_h, n_llm_w) == tokens
    patches, n_h, n_w = image_patches(Image.new("RGB", size, (1, 2, 3)), g)
    assert (n_h, n_w) == vit
    assert patches.shape == (n_h * n_w, 3 * 14 * 14)
    assert patches.dtype == torch.float16


@pytest.mark.parametrize("size", [(1, 1), (13, 7777), (7777, 13), (4000, 3000)])
def test_token_cap(size):
    g = V41VisionGeometry(CONFIG)
    n_llm_h, n_llm_w, _, _ = plan_image_grid(*size, g)
    assert 0 < num_image_tokens(n_llm_h, n_llm_w) <= 1024


def test_span_layout():
    config = dict(
        CONFIG,
        num_hidden_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=16,
    )
    tower = V41VisionTower(config, 8, torch.float32)
    for p in tower.parameters():
        p.data.normal_()
    span = tower.span(torch.randn(4 * 7, 3 * 14 * 14), 4, 7)
    n_llm_h, n_llm_w = 2, 3  # ceil(4 / 3), ceil(7 / 3)
    assert span.shape == (num_image_tokens(n_llm_h, n_llm_w), 8)
    assert torch.equal(span[0], tower.image_start)
    assert torch.equal(span[-1], tower.image_end)
    for row in range(n_llm_h):
        assert torch.equal(span[1 + row * (n_llm_w + 1) + n_llm_w], tower.image_newline)


def test_chat_placeholders_become_image_blocks():
    image = "<｜deepseek_image｜>"
    blocks = _image_blocks(f"first\n{image}\nsecond\n{image}\nquestion")
    assert [b["type"] for b in blocks] == ["text", "image", "text", "image", "text"]
    assert [b.get("text") for b in blocks if b["type"] == "text"] == [
        "first",
        "second",
        "question",
    ]
    assert [b["type"] for b in _image_blocks(f"{image}\ndescribe")] == [
        "image",
        "text",
    ]


def test_typed_placeholders_are_refused():
    with pytest.raises(ValueError):
        _text_only("look <｜deepseek_image｜>", images=True)
    with pytest.raises(ValueError):
        _text_only([{"type": "image_url", "image_url": {"url": "x"}}])
    _text_only([{"type": "image_url", "image_url": {"url": "x"}}], images=True)
