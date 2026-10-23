# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU token-budget parity between GLM profiling and real preprocessing."""

from types import SimpleNamespace as NS

import pytest
import torch
from PIL import Image

from vllm.models.glm5next.nvidia.multimodal import Glm5NextProcessingInfo
from vllm.transformers_utils.processors.glm5next import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
)


def processing_info(overrides):
    image = Glm5NextImageProcessor(max_image_tokens=8000)
    video = Glm5NextVideoProcessor(max_image_tokens=30000)
    info = Glm5NextProcessingInfo.__new__(Glm5NextProcessingInfo)
    info.ctx = NS(get_merged_mm_kwargs=lambda value: {**overrides, **value})
    info._glm5_hf_processor = NS(image_processor=image, video_processor=video)
    info.get_hf_config = lambda: NS(
        vision_config=NS(
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )
    )
    return info, image, video


@pytest.mark.parametrize("limit", [None, 256])
@pytest.mark.parametrize("size", [(768, 512), (7, 9)])
def test_profile_matches_image_preprocessing(limit, size):
    overrides = {} if limit is None else {"max_image_tokens": limit}
    info, processor, _ = processing_info(overrides)
    actual = processor(Image.new("RGB", size), return_tensors="pt", **overrides)
    tokens = int(actual["image_grid_thw"].prod()) // 4
    assert tokens == info.get_num_image_tokens(
        image_width=size[0], image_height=size[1]
    )
    assert 16 <= tokens <= (limit or 8000)
    assert info._get_image_max_pixels() == (limit or 8000) * 1568
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("image_min,video_min", [(16, 64), (64, 16)])
def test_separate_minimum_budgets_match_small_inputs(image_min, video_min):
    info, image, video = processing_info(
        {
            "max_image_tokens": 1024,
            "images_kwargs": {"min_image_tokens": image_min},
            "videos_kwargs": {"min_image_tokens": video_min},
        }
    )
    image_data = image(
        Image.new("RGB", (7, 9)),
        min_image_tokens=image_min,
        max_image_tokens=1024,
        return_tensors="pt",
    )
    video_data = video(
        videos=torch.zeros(4, 3, 9, 7, dtype=torch.uint8),
        min_image_tokens=video_min,
        max_image_tokens=1024,
        do_sample_frames=False,
        return_tensors="pt",
    )
    assert (
        info.get_num_image_tokens(image_width=7, image_height=9)
        == int(image_data["image_grid_thw"].prod()) // 4
    )
    assert (
        info.get_num_video_tokens(image_width=7, image_height=9, num_frames=4)
        == int(video_data["video_grid_thw"].prod()) // 4
    )


@pytest.mark.parametrize("limit", [None, 256])
def test_profile_matches_video_preprocessing(limit):
    overrides = {} if limit is None else {"max_image_tokens": limit}
    info, _, processor = processing_info(overrides)
    frames = torch.zeros(4, 3, 512, 768, dtype=torch.uint8)
    actual = processor(
        videos=frames, return_tensors="pt", do_sample_frames=False, **overrides
    )
    tokens = int(actual["video_grid_thw"].prod()) // 4
    assert tokens == info.get_num_video_tokens(
        image_width=768, image_height=512, num_frames=4
    )
    assert tokens <= (limit or 30000)
    assert info._get_video_max_pixels() == (limit or 30000) * 1568
    assert not torch.cuda.is_initialized()


def test_capped_video_budget_terminates_profile_frame_search():
    info, _, _ = processing_info({"max_image_tokens": 256})
    assert info._get_max_video_frames(8192) == 512


@pytest.mark.parametrize(
    "overrides,image_limit,video_limit",
    [
        (
            {
                "images_kwargs": {"max_image_tokens": 128},
                "videos_kwargs": {"max_image_tokens": 1024},
            },
            128,
            1024,
        ),
        (
            {"max_image_tokens": 256, "videos_kwargs": {"max_image_tokens": 1024}},
            256,
            1024,
        ),
        (
            {"max_image_tokens": 1024, "images_kwargs": {"max_image_tokens": 256}},
            256,
            1024,
        ),
    ],
)
def test_separate_modality_budgets_match_real_patch_counts(
    overrides, image_limit, video_limit
):
    info, image, video = processing_info(overrides)
    image_data = image(
        Image.new("RGB", (768, 512)),
        max_image_tokens=image_limit,
        return_tensors="pt",
    )
    video_data = video(
        videos=torch.zeros(4, 3, 512, 768, dtype=torch.uint8),
        max_image_tokens=video_limit,
        do_sample_frames=False,
        return_tensors="pt",
    )
    assert (
        info.get_num_image_tokens(image_width=768, image_height=512)
        == int(image_data["image_grid_thw"].prod()) // 4
    )
    assert (
        info.get_num_video_tokens(image_width=768, image_height=512, num_frames=4)
        == int(video_data["video_grid_thw"].prod()) // 4
    )
    assert info._get_image_max_pixels() == image_limit * 1568
    assert info._get_video_max_pixels() == video_limit * 1568
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pixels": 401408},
        {"min_pixels": 1568},
        {"images_kwargs": {"max_pixels": 401408}},
        {"videos_kwargs": {"max_pixels": 401408}},
    ],
)
def test_legacy_pixel_budget_is_rejected_by_profile_and_processor(kwargs):
    info, _, _ = processing_info(kwargs)
    with pytest.raises(ValueError, match="min_image_tokens/max_image_tokens"):
        info._get_image_max_pixels()
    processor = Glm5NextProcessor.__new__(Glm5NextProcessor)
    with pytest.raises(ValueError, match="min_image_tokens/max_image_tokens"):
        processor(text="test", **kwargs)
