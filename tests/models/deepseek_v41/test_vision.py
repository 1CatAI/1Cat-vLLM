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
from vllm.renderers.deepseek_v41 import DeepseekV41Renderer
from vllm.tokenizers.deepseek_v41 import (
    _IMAGE,
    _IMAGE_ESCAPED,
    _image_blocks,
    _sanitize_message,
    _text_only,
)

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


def test_typed_placeholders_are_sanitized(monkeypatch):
    # Literal <image> is ordinary ASCII text in V4.1 and passes verbatim.
    assert _text_only("compact <image> tag </image>", images=False) == (
        "compact <image> tag </image>"
    )
    assert _text_only("compact <image> tag", images=True) == "compact <image> tag"

    # Literal <｜deepseek_image｜> in text is neutralized so it never collides
    # with real multimodal placeholders or emits image_token_id (129264).
    assert _text_only(f"look {_IMAGE} here", images=True) == (
        f"look {_IMAGE_ESCAPED} here"
    )
    assert _text_only(f"look {_IMAGE} here", images=False) == (
        f"look {_IMAGE_ESCAPED} here"
    )

    # Unsupported block types and image blocks with vision disabled still fail.
    with pytest.raises(ValueError):
        _text_only([{"type": "image_url", "image_url": {"url": "x"}}])
    with pytest.raises(ValueError):
        _text_only([{"type": "audio", "url": "x"}], images=True)

    # Real image blocks pass when vision is enabled; mixed text+image blocks
    # sanitize only the literal text without mutating caller inputs.
    orig_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"saw <image> and {_IMAGE}"},
            {"type": "image_url", "image_url": {"url": "x"}},
            {
                "type": "tool_result",
                "content": [{"type": "text", "text": f"tool {_IMAGE}"}],
            },
        ],
        "reasoning_content": f"think {_IMAGE}",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "f", "arguments": f'{{"x":"{_IMAGE}"}}'},
            }
        ],
    }
    cleaned = _sanitize_message(orig_msg, images=True)
    assert cleaned is not orig_msg
    assert orig_msg["content"][0]["text"] == f"saw <image> and {_IMAGE}"
    assert cleaned["content"][0]["text"] == f"saw <image> and {_IMAGE_ESCAPED}"
    assert cleaned["content"][1] == {"type": "image_url", "image_url": {"url": "x"}}
    assert cleaned["content"][2]["content"][0]["text"] == f"tool {_IMAGE_ESCAPED}"
    assert cleaned["reasoning_content"] == f"think {_IMAGE_ESCAPED}"
    assert (
        cleaned["tool_calls"][0]["function"]["arguments"]
        == f'{{"x":"{_IMAGE_ESCAPED}"}}'
    )

    # Renderer validates and sanitizes before parse_chat_messages.
    monkeypatch.setattr("vllm.renderers.deepseek_v41._vision", lambda: True)
    msgs = [
        {"role": "system", "content": f"sys <image> {_IMAGE}"},
        {"role": "user", "content": "plain text"},
    ]
    out = DeepseekV41Renderer._validate_messages(msgs)
    assert out[0]["content"] == f"sys <image> {_IMAGE_ESCAPED}"
    assert out[1] is msgs[1]
    assert msgs[0]["content"] == f"sys <image> {_IMAGE}"
