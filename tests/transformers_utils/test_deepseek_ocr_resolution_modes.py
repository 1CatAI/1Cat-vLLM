# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-OCR resolution modes (R4): per-request ``image_mode`` selection,
crop parametrization, the multi-image safeguard, and — critically — the
producer/counter consistency contract (`tokenize_with_images` versus
`count_image_tokens`), whose violation is the "N multimodal tokens vs M
placeholders" crash class."""

import os

import pytest

from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE,
    CROP_MODE,
    IMAGE_SIZE,
    RESOLUTION_MODES,
    DeepseekOCRProcessor,
    count_image_tokens_for,
)

DSOCR_PATH = os.environ.get("DSOCR_CHECKPOINT", "/mnt/models/DeepSeek-OCR")


# ---------------------------------------------------------------------------
# Pure-function tests (no tokenizer).
# ---------------------------------------------------------------------------


def test_mode_table_matches_official_config():
    # Verbatim from the model repo's config.py (captured in the issue refs).
    assert RESOLUTION_MODES == {
        "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
        "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    }
    # The module defaults ARE the gundam mode (byte-compat anchor).
    assert (BASE_SIZE, IMAGE_SIZE, CROP_MODE) == (1024, 640, True)


def _count(mode: str, width: int, height: int, **kw) -> int:
    cfg = RESOLUTION_MODES[mode]
    return count_image_tokens_for(
        image_width=width,
        image_height=height,
        base_size=cfg["base_size"],
        image_size=cfg["image_size"],
        cropping=cfg["crop_mode"],
        **kw,
    )


def test_no_crop_modes_are_size_independent():
    for mode in ("tiny", "small", "base", "large"):
        assert _count(mode, 100, 100) == _count(mode, 4000, 3000)


def test_gundam_small_image_equals_no_crop_arithmetic():
    # <= image_size on both dims: never cropped, regardless of crop_mode.
    assert _count("gundam", 640, 480) == count_image_tokens_for(
        image_width=640,
        image_height=480,
        base_size=1024,
        image_size=640,
        cropping=False,
    )


def test_gundam_large_image_counts_tiles():
    assert _count("gundam", 1700, 2200) > _count("gundam", 100, 100)


def test_max_crops_bounds_the_tile_count():
    big = _count("gundam", 4000, 4000, max_crops=9)
    small = _count("gundam", 4000, 4000, max_crops=2)
    assert big > small


# ---------------------------------------------------------------------------
# Tokenizer-backed tests (real checkpoint; run on the GPU host, CPU-only).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    if not os.path.isdir(DSOCR_PATH):
        pytest.skip("checkpoint unavailable")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(DSOCR_PATH)


def _images(sizes):
    from PIL import Image

    return [Image.new("RGB", s, color="white") for s in sizes]


def test_default_construction_is_gundam_byte_compat(tokenizer):
    default = DeepseekOCRProcessor(tokenizer=tokenizer)
    explicit = DeepseekOCRProcessor(tokenizer=tokenizer, image_mode="gundam")
    for attr in ("base_size", "image_size", "crop_mode", "min_crops", "max_crops"):
        assert getattr(default, attr) == getattr(explicit, attr)
    assert default.crop_mode is True and default.image_size == 640


def test_invalid_mode_and_crops_raise(tokenizer):
    with pytest.raises(ValueError, match="image_mode"):
        DeepseekOCRProcessor(tokenizer=tokenizer, image_mode="giant")
    with pytest.raises(ValueError, match="crop bounds"):
        DeepseekOCRProcessor(tokenizer=tokenizer, min_crops=5, max_crops=2)


def test_image_mode_authoritative_over_trio(tokenizer):
    # The processing info always forwards the constant trio; a named mode
    # must win over it.
    proc = DeepseekOCRProcessor(
        tokenizer=tokenizer,
        image_size=640,
        base_size=1024,
        crop_mode=True,
        image_mode="base",
    )
    assert (proc.base_size, proc.image_size, proc.crop_mode) == (1024, 1024, False)


@pytest.mark.parametrize("mode", [None, "base", "large", "tiny", "gundam"])
@pytest.mark.parametrize("size", [(100, 100), (1200, 800), (1000, 2200)])
def test_producer_counter_consistency(tokenizer, mode, size):
    kwargs = {} if mode is None else {"image_mode": mode}
    proc = DeepseekOCRProcessor(tokenizer=tokenizer, **kwargs)
    out = proc(prompt="<image>\nFree OCR. ", images=_images([size]))
    produced = int(out["num_image_tokens"][0])
    counted = proc.count_image_tokens(
        image_width=size[0], image_height=size[1], num_images=1
    )
    assert produced == counted


def test_multi_image_guard_disables_crops_consistently(tokenizer):
    proc = DeepseekOCRProcessor(tokenizer=tokenizer, disable_crop_for_multi_image=True)
    sizes = [(1200, 800), (1000, 2200)]
    out = proc(prompt="<image>a<image>b", images=_images(sizes))
    spatial = out["images_spatial_crop"]
    assert (spatial <= 1).all(), "multi-image request must not be tiled"
    for i, size in enumerate(sizes):
        produced = int(out["num_image_tokens"][i])
        counted = proc.count_image_tokens(
            image_width=size[0], image_height=size[1], num_images=len(sizes)
        )
        assert produced == counted


def test_multi_image_default_keeps_cropping_byte_compat(tokenizer):
    # DeepSeek-OCR's max_crops=6 is documented safe for multi-image; the
    # guard defaults OFF so existing behavior is unchanged.
    proc = DeepseekOCRProcessor(tokenizer=tokenizer)
    out = proc(prompt="<image>a<image>b", images=_images([(1200, 800)] * 2))
    assert (out["images_spatial_crop"] > 1).any()
