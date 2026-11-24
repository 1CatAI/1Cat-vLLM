# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGML NVFP4 -> official NVFP4 expert conversion is exact (CPU only).

Set INKLING2_GGUF_Q4 to an Unsloth Inkling-2-Mini GGUF to also check real
expert banks; otherwise that case is skipped.
"""

import os

import pytest
import torch

from vllm.models.inkling2.gguf_weights import (
    dequant_ggml_nvfp4,
    dequant_official_nvfp4,
    ggml_nvfp4_to_official,
)


def _random_ggml_nvfp4(experts: int, rows: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    nsb = k // 64
    raw = torch.randint(
        0, 256, (experts, rows, nsb, 36), dtype=torch.uint8, generator=g
    )
    # Finite, positive UE4M3 block scales (exponent field 1..14, no NaN).
    exp = torch.randint(1, 15, (experts, rows, nsb, 4), dtype=torch.uint8, generator=g)
    man = torch.randint(0, 8, (experts, rows, nsb, 4), dtype=torch.uint8, generator=g)
    raw[..., :4] = (exp << 3) | man
    return raw.reshape(experts, rows, nsb * 36)


def test_layout_shapes():
    raw = _random_ggml_nvfp4(2, 3, 128)
    packed, scales = ggml_nvfp4_to_official(raw)
    assert packed.shape == (2, 3, 64) and packed.dtype == torch.uint8
    assert scales.shape == (2, 3, 8) and scales.dtype == torch.float8_e4m3fn


def test_conversion_is_exact_random():
    raw = _random_ggml_nvfp4(4, 32, 2304)
    gscale = torch.rand(4) + 0.5
    packed, scales = ggml_nvfp4_to_official(raw)
    ref = dequant_ggml_nvfp4(raw, gscale)
    out = dequant_official_nvfp4(packed, scales, gscale)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_rejects_partial_blocks():
    with pytest.raises(ValueError):
        ggml_nvfp4_to_official(torch.zeros(2, 35, dtype=torch.uint8))


@pytest.mark.skipif(
    not os.path.exists(os.environ.get("INKLING2_GGUF_Q4", "")),
    reason="set INKLING2_GGUF_Q4 to an Inkling-2-Mini GGUF",
)
def test_conversion_is_exact_real_experts():
    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(os.environ["INKLING2_GGUF_Q4"])
    tensors = {t.name: t for t in reader.tensors}
    for name in ("blk.0.ffn_gate_up_exps", "blk.11.ffn_down_exps"):
        t = tensors[f"{name}.weight"]
        assert t.tensor_type == gguf.GGMLQuantizationType.NVFP4
        raw = torch.from_numpy(t.data[:3].copy())  # first three experts
        gscale = torch.from_numpy(tensors[f"{name}.scale"].data[:3].copy())
        packed, scales = ggml_nvfp4_to_official(raw)
        ref = dequant_ggml_nvfp4(raw, gscale)
        out = dequant_official_nvfp4(packed, scales, gscale)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert torch.isfinite(out).all() and out.abs().max() > 0
