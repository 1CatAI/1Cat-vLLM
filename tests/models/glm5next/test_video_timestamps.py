# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU video sampling, field splitting, and prompt expansion stay aligned."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Glm4vProcessor, PreTrainedTokenizerFast
from transformers.video_utils import VideoMetadata

from vllm.model_executor.models import glm4_1v
from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
from vllm.models.glm5next.nvidia.multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
)
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.transformers_utils.processors.glm5next import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
)


def processor_and_info():
    vocab = {
        name: index
        for index, name in enumerate(
            [
                "[UNK]",
                "<|image|>",
                "<|video|>",
                "<|begin_of_image|>",
                "<|end_of_image|>",
                "<|begin_of_video|>",
                "<|end_of_video|>",
            ]
        )
    }
    special_tokens = list(vocab)[1:]
    for value in [*(f"{i / 10:.1f}" for i in range(201)), "seconds"]:
        vocab[value] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=special_tokens,
    )
    processor = Glm5NextProcessor(
        image_processor=Glm5NextImageProcessor(),
        video_processor=Glm5NextVideoProcessor(),
        tokenizer=tokenizer,
    )
    info = Glm5NextProcessingInfo.__new__(Glm5NextProcessingInfo)
    info.ctx = NS(get_merged_mm_kwargs=lambda kwargs: kwargs)
    info._glm5_hf_processor = processor
    info.get_tokenizer = lambda: tokenizer
    info.get_hf_config = lambda: NS(
        image_start_token_id=3,
        image_end_token_id=4,
        video_start_token_id=5,
        video_end_token_id=6,
        vision_config=NS(spatial_merge_size=2),
    )
    return processor, info


def video_input(total, fps, selected=None):
    indices = list(range(total)) if selected is None else selected
    frames = np.zeros((len(indices), 56, 56, 3), dtype=np.uint8)
    metadata = VideoMetadata(
        total_num_frames=total,
        fps=fps,
        duration=total / fps,
        frames_indices=indices,
    )
    raw_metadata = dict(
        total_num_frames=total,
        fps=fps,
        duration=total / fps,
        frames_indices=indices,
        do_sample_frames=selected is None,
    )
    return frames, metadata, raw_metadata


def assert_prompt_matches(processor, info, features, videos, raw_metadata):
    fields = MultiModalKwargsItems.from_hf_inputs(
        features, _create_qwen2vl_field_factory(2)(features)
    )
    mm_processor = Glm5NextMultiModalProcessor.__new__(Glm5NextMultiModalProcessor)
    mm_processor.info = info
    updates = mm_processor._get_prompt_updates(
        {"video": list(zip(videos, raw_metadata))}, {}, fields
    )
    video_update = next(update for update in updates if update.modality == "video")
    for index, grid in enumerate(features["video_grid_thw"]):
        detail = video_update.resolve(index).content
        full = detail.full
        temporal, height, width = grid.tolist()
        assert full[0] == 5 and full[-1] == 6
        assert full.count(3) == full.count(4) == temporal
        assert detail.is_embed(None, full).sum() == temporal * height * width // 4
        assert fields["video"][index]["timestamps"].data.ndim == 1
        assert len(fields["video"][index]["timestamps"].data) == temporal
        # Every image group has its timestamp after the end-of-image sentinel.
        expected = [5]
        embed_id = (
            processor.image_token_id
            if glm4_1v.TRANSFORMERS_WITH_GA
            else processor.video_token_id
        )
        for timestamp in features["timestamps"][index].tolist():
            expected += [3] + [embed_id] * (height * width // 4) + [4]
            expected += processor.tokenizer.encode(
                f"{timestamp:.1f} seconds", add_special_tokens=False
            )
        assert full == expected + [6]


@pytest.mark.parametrize(
    "total,fps,selected,expected",
    [
        (8, 8, None, [0]),
        (8, 2, None, [0, 1, 2, 3]),
        (16, 4, None, [0, 1, 2, 3]),
        (16, 4, [0, 2, 4, 6, 8, 10, 12, 15], [0, 1, 2, 3]),
        (9, 2, [0, 2, 4, 6, 8], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("return_metadata", [False, True])
def test_video_sampling_and_prompt_groups_match(
    total, fps, selected, expected, return_metadata
):
    processor, info = processor_and_info()
    frames, metadata, raw = video_input(total, fps, selected)
    features = processor(
        text="<|begin_of_video|><|video|><|end_of_video|>",
        videos=[frames],
        video_metadata=[metadata],
        return_tensors="pt",
        do_sample_frames=selected is None,
        max_image_tokens=256,
        return_metadata=return_metadata,
    )
    assert ("video_metadata" in features) is return_metadata
    assert features["timestamps"][0].tolist() == expected
    assert_prompt_matches(processor, info, features, [frames], [raw])
    assert not torch.cuda.is_initialized()


def test_different_length_video_batch_keeps_each_timestamp_vector():
    processor, info = processor_and_info()
    inputs = [video_input(8, 8), video_input(8, 2)]
    features = processor(
        text=["<|video|>", "<|video|>"],
        videos=[item[0] for item in inputs],
        video_metadata=[item[1] for item in inputs],
        do_sample_frames=True,
        return_tensors="pt",
        max_image_tokens=256,
    )
    assert [t.tolist() for t in features["timestamps"]] == [[0], [0, 1, 2, 3]]
    assert_prompt_matches(
        processor,
        info,
        features,
        [item[0] for item in inputs],
        [item[2] for item in inputs],
    )


@pytest.mark.parametrize(
    "fps,indices",
    [(4, list(range(8))), (4, [1, 2, 3, 4, 5]), (29.97, [1, 3, 7, 12, 20])],
)
def test_fractional_timestamps_match_official_glm5_prompt_tokens(fps, indices):
    reference_module = pytest.importorskip(
        "transformers.models.glm5_next.processing_glm5_next"
    )
    processor, info = processor_and_info()
    official = reference_module.Glm5NextProcessor(
        image_processor=processor.image_processor,
        video_processor=processor.video_processor,
        tokenizer=processor.tokenizer,
    )
    frames, metadata, raw = video_input(max(indices) + 1, fps, indices)
    features = processor(
        text="<|video|>",
        videos=[frames],
        video_metadata=[metadata],
        do_sample_frames=False,
        return_tensors="pt",
        return_metadata=True,
        max_image_tokens=256,
    )
    timestamps = features["timestamps"][0]
    assert timestamps.dtype == torch.float64
    assert timestamps.tolist() == metadata.timestamps[::2]
    assert any(value != int(value) for value in timestamps.tolist())
    expected_text = (
        "<|begin_of_video|>"
        + official.replace_video_token(features, 0)
        + "<|end_of_video|>"
    )
    expected_tokens = processor.tokenizer.encode(
        expected_text, add_special_tokens=False
    )
    assert 0 not in expected_tokens, "The fixture must distinguish fractional seconds"
    actual = info._construct_video_placeholder(
        frames, raw, features["video_grid_thw"][0], timestamps=timestamps
    )
    assert actual == expected_tokens


@pytest.mark.parametrize(
    "kind,processor_name,image_name,video_name",
    [
        ("glm4v", "Glm4vProcessor", "Glm4vImageProcessor", "Glm4vVideoProcessor"),
        ("glm46v", "Glm46VProcessor", "Glm46VImageProcessor", "Glm46VVideoProcessor"),
        ("glmga", "Glm46VProcessor", "GlmgaImageProcessor", "GlmgaVideoProcessor"),
    ],
)
def test_real_legacy_processors_do_not_activate_processed_timestamp_hook(
    kind, processor_name, image_name, video_name
):
    import transformers

    native, _ = processor_and_info()
    processor = getattr(transformers, processor_name)(
        image_processor=getattr(transformers, image_name)(),
        video_processor=getattr(transformers, video_name)(),
        tokenizer=native.tokenizer,
    )
    frames, metadata, _ = video_input(8, 2, list(range(8)))
    features = processor(
        text="<|begin_of_video|><|video|><|end_of_video|>",
        videos=[frames],
        video_metadata=[metadata],
        do_sample_frames=False,
        do_resize=False,
        return_tensors="pt",
        return_metadata=True,
    )
    assert "timestamps" not in features
    fields = MultiModalKwargsItems.from_hf_inputs(
        features, _create_qwen2vl_field_factory(2)(features)
    )
    assert "timestamps" not in fields["video"][0]


@pytest.mark.parametrize("shape", [(3,), (1, 1)])
def test_processed_timestamps_reject_wrong_grid_or_rank(shape):
    _, info = processor_and_info()
    with pytest.raises(ValueError, match="one value per temporal group"):
        info._construct_video_placeholder(
            np.zeros((8, 56, 56, 3)),
            {},
            torch.tensor([1, 4, 4]),
            timestamps=torch.zeros(shape, dtype=torch.int64),
        )


@pytest.mark.parametrize("kind", ["glm4v", "glmga", "glm46v"])
def test_models_without_processed_timestamps_keep_original_dispatch(kind):
    _, info = processor_and_info()
    original = info._glm5_hf_processor
    if kind == "glm4v":
        legacy = Glm4vProcessor.__new__(Glm4vProcessor)
        legacy.image_processor = original.image_processor
        legacy.video_token_id = original.video_token_id
        info._glm5_hf_processor = legacy
    info._is_glmga_model = lambda processor: kind == "glmga"
    calls = []
    for branch in ("glm4v", "glmga", "glm46v"):

        def timestamps(metadata, count, branch=branch):
            calls.append(branch)
            return [0, 1]

        setattr(info, "_get_video_second_idx_" + branch, timestamps)
    tokens = info._construct_video_placeholder(
        np.zeros((4, 56, 56, 3)), {}, torch.tensor([2, 4, 4])
    )
    assert calls == [kind]
    assert tokens.count(3) == tokens.count(4) == 2
