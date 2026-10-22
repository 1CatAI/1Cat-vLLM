# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""csf_32x2_v1 expert scales: decoder, reference encoder and QPN expansion."""

import hashlib

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    QpnFragmentOrder,
    qpn_prepack,
    qpn_scale_entries,
)
from vllm.model_executor.layers.quantization.utils.csf32_expand import (
    CsfExpertScales,
    expand_csf32_experts,
)
from vllm.model_executor.layers.quantization.utils.csf32_scales import (
    decode_csf32,
    encode_csf32,
)

_PARTS = ("bases", "codes", "page_bases", "offsets", "exceptions")


def _sample(i: int, row: int) -> int:
    """The reference encoder's test generator (TensorFold nvfp4_csf32.zig)."""
    mask = 0xFFFFFFFF
    x = (i * 0x9E3779B1 + 0x7F4A7C15) & mask
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & mask
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & mask
    x ^= x >> 16
    if x % 11 == 0:
        return (x >> 8) & 0xFF
    return 96 + (row * 37) % 64 + (x >> 4) % 18


def _torch_parts(parts: dict[str, np.ndarray]) -> list[torch.Tensor]:
    return [torch.from_numpy(parts[name]) for name in _PARTS]


def _banded(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    band = 100 + (np.arange(rows * cols).reshape(rows, cols) // cols) % 50
    scale = band + rng.integers(0, 18, (rows, cols))
    stray = rng.integers(0, 8, (rows, cols)) == 0
    return np.where(stray, rng.integers(0, 256, (rows, cols)), scale).astype(np.uint8)


# Each part's sha256 as the reference encoder (encode_csf.py) writes it.
_GOLDEN = [
    (
        32,
        1,
        (
            "22398d175284dd029a1d821b2246c4dd8a5f26792f18288d4b40a20b2ac81f93",
            "5ac6a5945f16500911219129984ba8b387a06f24fe383ce4e81a73294065461b",
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
    ),
    (
        160,
        3,
        (
            "9d619ac307c9069040829a6706abc714a70e211e4cc11a7ef2f8f061711f15db",
            "fa7e29965dd9df08cb889d6d92b0b43c7f7653160a0f49b3900f7e380e354816",
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
            "38cbb6ef752ceaff2bc461ca353b7b1c207de8bddcc50509b54921b0cb572895",
            "b70b64afbf3d1fbcf9024540a08bce3d5d1e387f39604fa3e4c6ffab49b049bb",
        ),
    ),
    (
        160,
        67,
        (
            "8bfb5a39f0230d3c2010cb2ebd404712c3a599349be267090f1ee3e13fc0443e",
            "17fc8131793cf5760f182531ce1de42e492a7016f957952784f973c01363fb09",
            "724141ec21c423374babec5b754ad8556c45639c0183aee003976853f41919a4",
            "1679e3a939225bcf8ea373a4238cd81696b022eaac1f45e71b66668c63d41534",
            "e318269691664c405d4ddde7868983f3bf85ed73ca0eae3f956c40d55dad76e0",
        ),
    ),
]


@pytest.mark.cpu_test
@pytest.mark.parametrize(("rows", "cols", "digests"), _GOLDEN)
def test_encoder_matches_reference_bytes(rows, cols, digests):
    scale = np.array(
        [[_sample(r * cols + c, r) for c in range(cols)] for r in range(rows)],
        dtype=np.uint8,
    )
    parts = encode_csf32(scale)
    got = tuple(hashlib.sha256(parts[name].tobytes()).hexdigest() for name in _PARTS)
    assert got == digests
    assert np.array_equal(decode_csf32(*_torch_parts(parts), rows, cols).numpy(), scale)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("rows", "cols"),
    [(32, 1), (32, 3), (160, 67), (128, 132), (1056, 33), (2048, 256), (4096, 128)],
)
def test_decode_inverts_encode(rows, cols):
    scale = _banded(rows, cols, seed=rows + cols)
    parts = encode_csf32(scale)
    assert np.array_equal(decode_csf32(*_torch_parts(parts), rows, cols).numpy(), scale)


@pytest.mark.cpu_test
def test_decode_rejects_malformed_parts():
    parts = encode_csf32(_banded(64, 4, seed=3))
    bad_base = dict(parts, bases=parts["bases"].copy())
    bad_base["bases"][5] = 241
    with pytest.raises(ValueError, match="base"):
        decode_csf32(*_torch_parts(bad_base), 64, 4)
    short = dict(parts, codes=parts["codes"][:-1])
    with pytest.raises(ValueError, match="disagree"):
        decode_csf32(*_torch_parts(short), 64, 4)
    if parts["exceptions"].size:
        lost = dict(parts, exceptions=parts["exceptions"][:-1])
        with pytest.raises(ValueError, match="exception index"):
            decode_csf32(*_torch_parts(lost), 64, 4)


def _planes(matrices: list[dict[str, np.ndarray]], rows, groups, projections, device):
    def stack(name, dtype):
        return torch.stack(
            [torch.from_numpy(m[name].astype(dtype)) for m in matrices]
        ).to(device)

    counts = [m["exceptions"].size for m in matrices]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return CsfExpertScales(
        rows=rows,
        groups=groups,
        projections=projections,
        bases=stack("bases", np.uint8),
        codes=stack("codes", np.uint8),
        page_bases=stack("page_bases", np.int32),
        offsets=stack("offsets", np.int32),
        exceptions=torch.from_numpy(
            np.concatenate([m["exceptions"] for m in matrices]).astype(np.int32)
        ).to(device),
        exception_start=torch.tensor(starts, dtype=torch.int64, device=device),
        exception_count=torch.tensor(counts, dtype=torch.int32, device=device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("order", list(QpnFragmentOrder))
@pytest.mark.parametrize(
    ("rows", "groups", "projections"), [(256, 64, 2), (512, 32, 1)]
)
# 5 slots: fewer than the 6 experts (decode); 14 slots: more (prefill, where
# only the first group rows can hold groups).
@pytest.mark.parametrize("num_slots", [5, 14])
def test_expansion_matches_qpn_prepack(rows, groups, projections, order, num_slots):
    device = torch.device("cuda")
    num_experts = 6
    scales = [
        [_banded(rows, groups, seed=17 * e + p) for p in range(projections)]
        for e in range(num_experts)
    ]
    planes = _planes(
        [encode_csf32(s) for expert in scales for s in expert],
        rows,
        groups,
        projections,
        device,
    )
    raster = torch.zeros(
        num_experts, projections * rows * groups, dtype=torch.uint8, device=device
    )
    # Groups for experts 4, 1 and 3, then empty slots that must be skipped.
    gids = torch.zeros(num_slots, dtype=torch.int32, device=device)
    gids[:3] = torch.tensor([4, 1, 3], dtype=torch.int32)
    goff = torch.full((num_slots + 1,), num_slots, dtype=torch.int32, device=device)
    goff[:3] = torch.tensor([0, 2, 3], dtype=torch.int32)
    expand_csf32_experts(raster, planes, gids, goff, qpn_scale_entries(order, device))

    codes = torch.zeros(
        projections * rows, groups * 8, dtype=torch.uint8, device=device
    )
    for expert in range(num_experts):
        stacked = torch.from_numpy(np.concatenate(scales[expert])).to(device)
        _, expected = qpn_prepack(codes, stacked, 16, order)
        if expert in (4, 1, 3):
            assert torch.equal(raster[expert], expected), expert
        else:
            assert not raster[expert].any(), expert
