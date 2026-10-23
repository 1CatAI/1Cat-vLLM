# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the real GLM wrapper's PP ownership and GGUF coverage guard."""

import json
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import StageMissingLayer
from vllm.models.glm5next.nvidia import model as glm


@pytest.mark.parametrize("use_gguf_file", [False, True])
def test_processor_resolves_assets_beside_local_gguf(
    tmp_path, monkeypatch, use_gguf_file
):
    from transformers import AutoTokenizer

    from vllm.transformers_utils.processors.glm5next import Glm5NextProcessor

    config = {
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "patch_expand_factor": 1,
        "min_image_tokens": 16,
        "max_image_tokens": 8000,
    }
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Glm5NextProcessor",
                "image_processor": dict(
                    config, image_processor_type="Glm5NextImageProcessor"
                ),
                "video_processor": dict(
                    config,
                    video_processor_type="Glm5NextVideoProcessor",
                    max_image_tokens=240000,
                ),
            }
        )
    )
    model = tmp_path
    if use_gguf_file:
        model = tmp_path / "model-00001-of-00004.gguf"
        model.touch()
    tokenizer_calls = []
    sentinel = object()

    def tokenizer(path, **kwargs):
        tokenizer_calls.append(path)
        return sentinel

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer)

    class CapturedProcessor(Glm5NextProcessor):
        def __init__(self, **kwargs):
            self.captured = kwargs

    processor = CapturedProcessor.from_pretrained(str(model), local_files_only=True)
    assert tokenizer_calls == [str(tmp_path)]
    assert processor.captured["tokenizer"] is sentinel
    assert processor.captured["image_processor"].max_image_tokens == 8000
    assert processor.captured["video_processor"].max_image_tokens == 30000


def test_vision_is_owned_by_input_rank_and_gguf_loading_is_complete(monkeypatch):
    assert not torch.cuda.is_initialized()
    group = NS(is_first_rank=True)
    created = []

    class TinyVision(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))
            self.bias = nn.Parameter(torch.zeros(2))
            created.append(kwargs)

    class TinyLanguage(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.model = nn.Module()
            self.model.weight = nn.Parameter(torch.zeros(2, 2))
            self.make_empty_intermediate_tensors = lambda **kwargs: None

    monkeypatch.setattr(glm, "get_pp_group", lambda: group)
    monkeypatch.setattr(glm, "Glm5NextVisionTransformer", TinyVision)
    monkeypatch.setattr(glm, "init_vllm_registered_model", TinyLanguage)
    mm = NS(
        language_model_only=False,
        mm_encoder_only=False,
        mm_encoder_tp_mode="weights",
        is_multimodal_pruning_enabled=lambda: False,
        get_limit_per_prompt=lambda _: 1,
    )
    config = NS(
        text_config=NS(),
        vision_config=NS(rms_norm_eps=1e-5),
    )
    vconfig = NS(
        model_config=NS(hf_config=config, multimodal_config=mm, quantization="gguf")
    )
    first = glm.Glm5NextForConditionalGeneration(vllm_config=vconfig)
    assert first._tower_model_names == ["visual"]
    assert len(created) == 1
    assert created[0]["norm_eps"] == config.vision_config.rms_norm_eps
    assert created[0]["quant_config"] is None
    assert first.load_weights(
        [
            ("visual.weight", torch.ones(2, 2)),
            ("visual.bias", torch.ones(2)),
            ("language_model.model.weight", torch.ones(2, 2)),
        ]
    ) == {"visual.weight", "visual.bias", "language_model.model.weight"}
    with pytest.raises(ValueError, match="did not load GLM5Next vision.*visual.bias"):
        first.load_weights([("visual.weight", torch.ones(2, 2))])
    # HF's outer model prefix remains accepted by the inherited mapper.
    assert first.load_weights(
        [
            ("model.visual.weight", torch.ones(2, 2)),
            ("model.visual.bias", torch.ones(2)),
        ]
    ) == {"visual.weight", "visual.bias"}

    group.is_first_rank = False
    later = glm.Glm5NextForConditionalGeneration(vllm_config=vconfig)
    assert isinstance(later.visual, StageMissingLayer)
    assert later._tower_model_names == [] and len(created) == 1
    assert not list(later.visual.named_parameters())
    assert later.load_weights([("visual.weight", torch.ones(2, 2))]) == set()

    group.is_first_rank = True
    mm.language_model_only = True
    text = glm.Glm5NextForConditionalGeneration(vllm_config=vconfig)
    assert isinstance(text.visual, StageMissingLayer)
    assert text._tower_model_names == [] and len(created) == 1
    assert text.load_weights([("language_model.model.weight", torch.ones(2, 2))]) == {
        "language_model.model.weight"
    }

    # An encoder-only instance still owns the tower; its text modules are skipped.
    group.is_first_rank = False
    mm.language_model_only = False
    mm.mm_encoder_only = True
    encoder = glm.Glm5NextForConditionalGeneration(vllm_config=vconfig)
    assert isinstance(encoder.visual, TinyVision) and len(created) == 2
    assert isinstance(encoder.language_model, StageMissingLayer)
    assert not torch.cuda.is_initialized()
