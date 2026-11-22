# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 vision input: geometry, span layout, chat placeholders,
GGUF route loaders (llama.cpp mmproj, VL bias sidecar)."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vllm.models.deepseek_v41 import vision as v41_vision
from vllm.models.deepseek_v41.vision import (
    V41VisionGeometry,
    V41VisionTower,
    image_patches,
    load_bias_vl,
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


def _small_tower():
    config = dict(
        CONFIG,
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=16,
    )
    return V41VisionTower(config, 8, torch.float32)


def _write_mmproj(path, tower):
    """A deepseek41v mmproj with llama.cpp's tensor names and layouts."""
    gguf = pytest.importorskip("gguf")
    sd = {k: v.detach().float().numpy() for k, v in tower.state_dict().items()}
    p = tower.geometry.patch_size
    tensors = {
        "v.patch_embd.weight": sd["vision.patch_embed.proj.weight"].reshape(
            -1, 3, p, p
        ),
        "v.patch_embd.bias": sd["vision.patch_embed.proj.bias"],
        "v.post_ln.weight": sd["vision.norm.weight"],
        "mm.1.weight": sd["aligner.w1.weight"],
        "mm.1.bias": sd["aligner.w1.bias"],
        "mm.2.weight": sd["aligner.w2.weight"],
        "mm.2.bias": sd["aligner.w2.bias"],
        "v.token_embd.img_start": sd["image_start"],
        "v.token_embd.img_end": sd["image_end"],
        "v.image_newline": sd["image_newline"],
    }
    n = 0
    while f"vision.blocks.{n}.norm1.weight" in sd:
        b, o = f"v.blk.{n}.", f"vision.blocks.{n}."
        gate, up = np.split(sd[o + "mlp.w1.weight"], 2)
        tensors.update(
            {
                b + "ln1.weight": sd[o + "norm1.weight"],
                b + "ln2.weight": sd[o + "norm2.weight"],
                b + "attn_qkv.weight": sd[o + "attn.wqkv.weight"],
                b + "attn_qkv.bias": sd[o + "attn.wqkv.bias"],
                b + "attn_out.weight": sd[o + "attn.wo.weight"],
                b + "attn_out.bias": sd[o + "attn.wo.bias"],
                b + "ffn_gate.weight": gate,
                b + "ffn_up.weight": up,
                b + "ffn_down.weight": sd[o + "mlp.w2.weight"],
            }
        )
        n += 1
    writer = gguf.GGUFWriter(str(path), "clip")
    writer.add_string("clip.projector_type", "deepseek41v")
    for name, value in tensors.items():
        writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_mmproj_loads_the_tower(tmp_path):
    source = _small_tower()
    for p in source.parameters():
        p.data.normal_()
    path = tmp_path / "mmproj.gguf"
    _write_mmproj(path, source)
    tower = _small_tower()
    tower.load_mmproj(str(path))
    for (name, a), (_, b) in zip(source.named_parameters(), tower.named_parameters()):
        assert torch.equal(a, b), name
    patches = torch.randn(4 * 7, 3 * 14 * 14)
    assert torch.equal(source.span(patches, 4, 7), tower.span(patches, 4, 7))


class _Gate(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.register_buffer("bias_vl", torch.zeros(experts), persistent=False)


class _Layer(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.ffn = nn.Module()
        self.ffn.gate = _Gate(experts)


def test_bias_vl_sidecar(tmp_path, monkeypatch):
    config = type("C", (), {"num_hidden_layers": 3, "n_routed_experts": 4})()
    table = np.arange(12, dtype="<f4").reshape(3, 4)
    path = tmp_path / "exp_probs_b_vl.f32"
    table.tofile(path)
    monkeypatch.setattr(v41_vision.envs, "VLLM_V41_VISION_BIAS_VL", str(path))
    # A later PP stage: the first layer is not local.
    layers = [nn.Identity(), _Layer(4), _Layer(4)]
    load_bias_vl(layers, _Gate, config)
    for index in (1, 2):
        assert torch.equal(
            layers[index].ffn.gate.bias_vl, torch.from_numpy(table[index])
        )
    np.zeros(5, dtype="<f4").tofile(path)
    with pytest.raises(ValueError):
        load_bias_vl(layers, _Gate, config)
