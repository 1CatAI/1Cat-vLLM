# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN: RoPE theta / YaRN per layer type (PORT_DESIGN §7.3: layers 0, 2, 20) and rotation semantics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.common import rope as v41rope

REF_CONFIG = Path("/mnt/nvme2/scratch/ds41/model-ref/config.json")


@pytest.fixture(scope="module")
def cfg() -> SimpleNamespace:
    path = REF_CONFIG
    if not path.is_file():
        path = Path("/mnt/nvme2/models/DeepSeek-V4.1-Flash/config.json")
    raw = json.loads(path.read_text())
    merged = {k: v for k, v in raw.items() if k != "text_config"}
    merged.update(raw["text_config"])
    return SimpleNamespace(**merged)


# --- transcription of ref:m.py:369-406 (precompute_freqs_cis / apply_rotary_emb) ---
def _ref_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen, device=device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _ref_apply(x, freqs_cis, inverse=False):
    y = x.clone()
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1) // 2)
    y.copy_(torch.view_as_real(xc * freqs_cis).flatten(-2))
    return y


def _ref_layer_freqs(cfg, layer: int, seqlen: int, device):
    ratio = cfg.compress_ratios[layer]
    rs = cfg.rope_scaling
    if ratio:
        return _ref_freqs_cis(cfg.qk_rope_head_dim, seqlen, rs["original_max_position_embeddings"],
                              cfg.compress_rope_theta, rs["factor"], rs["beta_fast"], rs["beta_slow"], device)
    return _ref_freqs_cis(cfg.qk_rope_head_dim, seqlen, 0, cfg.rope_theta, rs["factor"], rs["beta_fast"],
                          rs["beta_slow"], device)


@pytest.mark.parametrize("layer,theta,yarn", [(0, 10000.0, False), (1, 10000.0, False), (2, 160000.0, True),
                                              (14, 160000.0, True), (20, 160000.0, True), (39, 160000.0, True)])
def test_params_per_layer_type(cfg, layer: int, theta: float, yarn: bool) -> None:
    p = v41rope.v41_rope_params(cfg, cfg.compress_ratios[layer])
    assert (p.theta, p.yarn, p.rope_dim) == (theta, yarn, 64)
    if yarn:   # ratio 1 (layer 20+) must get YaRN: the V4 code keyed on ratio > 1 and would drop it
        assert (p.factor, p.original_max_position, p.beta_fast, p.beta_slow) == (16.0, 65536, 32.0, 1.0)


def test_rejects_bad_ratio_and_rope(cfg) -> None:
    with pytest.raises(ValueError):
        v41rope.v41_rope_params(cfg, 4)
    bad = SimpleNamespace(**{**vars(cfg), "rope_scaling": {"rope_type": "default"}})
    with pytest.raises(ValueError):
        v41rope.v41_rope_params(bad, 2)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.sm70)])
@pytest.mark.parametrize("layer", [0, 2, 20])
def test_cache_matches_reference(cfg, layer: int, device: str) -> None:
    n = 70000          # crosses original_max_position 65536
    rope = v41rope.DeepseekV41RotaryEmbedding(v41rope.v41_rope_params(cfg, cfg.compress_ratios[layer]), n, device)
    ref = _ref_layer_freqs(cfg, layer, n, device)
    half = 32
    assert rope.cos_sin_cache.shape == (n, 64) and rope.cos_sin_cache.dtype == torch.float32
    # frequencies are bitwise the reference's; cos/sin of the same FP32 angles
    assert torch.equal(rope.cos_sin_cache[:, :half], ref.real)
    assert torch.equal(rope.cos_sin_cache[:, half:], ref.imag)


@pytest.mark.parametrize("layer", [0, 2, 20])
@pytest.mark.parametrize("inverse", [False, True])
def test_rotation_matches_reference(cfg, layer: int, inverse: bool) -> None:
    torch.manual_seed(layer)
    T, H, D = 37, 3, 512
    positions = torch.randint(0, 200000, (T,))
    n = int(positions.max()) + 1
    rope = v41rope.DeepseekV41RotaryEmbedding(v41rope.v41_rope_params(cfg, cfg.compress_ratios[layer]), n, "cpu")
    x = torch.randn(T, H, D, dtype=torch.float32)
    out = rope.rotate(x, positions, inverse=inverse)
    freqs = _ref_layer_freqs(cfg, layer, n, "cpu")[positions]            # [T, 32]
    ref_rot = _ref_apply(x[..., -64:].unsqueeze(0).transpose(0, 1).reshape(1, T, H, 64), freqs, inverse)
    ref_rot = ref_rot.reshape(T, H, 64)
    assert torch.equal(out[..., :-64], x[..., :-64])
    torch.testing.assert_close(out[..., -64:], ref_rot, rtol=1e-6, atol=1e-6)
    back = rope.rotate(out, positions, inverse=not inverse)
    torch.testing.assert_close(back, x, rtol=1e-5, atol=1e-5)


def test_build_is_shared_and_capped(cfg) -> None:
    a = v41rope.build_v41_rope(cfg, 2, max_positions=4096, device="cpu")
    b = v41rope.build_v41_rope(cfg, 1, max_positions=4096, device="cpu")
    c = v41rope.build_v41_rope(cfg, 0, max_positions=4096, device="cpu")
    assert a is b and a is not c
    assert a.cos_sin_cache.shape == (4096, 64)
