# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3 vision budgets must cover every image the real processor emits
with the checkpoint's settings."""

from types import SimpleNamespace

import pytest
from PIL import Image

from vllm.models.glm5next.nvidia.multimodal import Glm5NextProcessingInfo
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
from vllm.transformers_utils.processors.glm5next import (
    Glm5NextImageProcessor,
    Glm5NextVideoProcessor,
)

# GLM-5.3-Flash checkpoint: vision_config of config.json and the image
# processor of processor_config.json.
_VISION_CONFIG = {"patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2}
_IMAGE_PROCESSOR = {
    "patch_size": 14,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "patch_expand_factor": 1,
    "min_image_tokens": 16,
    "max_image_tokens": 8000,
}


class _Context:
    def get_hf_config(self, *args):
        return Glm5NextConfig(vision_config=_VISION_CONFIG)

    def get_merged_mm_kwargs(self, kwargs):
        return dict(kwargs)


class _Info(Glm5NextProcessingInfo):
    """Processing info on the checkpoint's processor settings, without
    loading the checkpoint's tokenizer."""

    def get_hf_processor(self, **kwargs: object):
        return SimpleNamespace(
            image_processor=Glm5NextImageProcessor(**_IMAGE_PROCESSOR),
            video_processor=Glm5NextVideoProcessor(),
        )


def _info() -> _Info:
    return _Info(_Context())  # type: ignore[arg-type]


def _processed_tokens(width: int, height: int) -> int:
    processor = Glm5NextImageProcessor(**_IMAGE_PROCESSOR)
    grid = processor(Image.new("RGB", (width, height)), return_tensors="pt")
    _, grid_h, grid_w = grid["image_grid_thw"][0].tolist()
    return grid_h * grid_w // _IMAGE_PROCESSOR["merge_size"] ** 2


@pytest.mark.parametrize(
    ("width", "height", "expected_tokens"),
    [
        (3840, 2160, 7973),  # 4K screenshot, 16:9
        (3840, 2880, 7931),  # phone photo, 4:3, after a 3840 px downscale
        (2160, 3840, 7973),  # portrait 4K
        (640, 480, 414),  # small image, below the ceiling
    ],
)
def test_image_encoder_budget_covers_processed_images(width, height, expected_tokens):
    info = _info()
    # The per-item maximum the encoder cache is sized from.
    budget = info.get_mm_max_tokens_per_item(seq_len=524288, mm_counts={"image": 1})

    tokens = _processed_tokens(width, height)

    assert tokens == expected_tokens
    assert tokens <= budget["image"]


def test_profiling_image_reaches_processor_ceiling():
    info = _info()
    size = info.get_image_size_with_most_features()

    assert info.get_max_image_tokens() == _IMAGE_PROCESSOR["max_image_tokens"]
    assert _processed_tokens(size.width, size.height) == info.get_max_image_tokens()
