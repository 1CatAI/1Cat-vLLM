# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression of actual multimodal wrapper construction and loading."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import model as original


def test_multimodal_stage_construction_and_loading():
    source = (
        Path(__file__).resolve().parents[3] / "vllm/models/qwen4_exp/nvidia/model.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpForConditionalGeneration"
    )
    node.decorator_list = []
    scope = dict(original.__dict__)
    pp = {"first": True, "end": 8}
    offload = {"active": False}
    created = []

    class FakeVision(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))
            self.bias = nn.Parameter(torch.ones(2))
            created.append(kwargs["prefix"])

    class FakeLanguage(nn.Module):
        def __init__(self, *, vllm_config, **kwargs):
            super().__init__()
            self.vllm_config = vllm_config
            self.model = NS(
                start_layer=0 if pp["first"] else 8,
                end_layer=pp["end"],
                layers=[],
                _kv_cache_dtype="auto",
            )
            self.make_empty_intermediate_tensors = lambda **kw: None
            self.hidden_probe = object()

        def embed_input_ids(self, input_ids):
            return torch.ones(len(input_ids), 4)

        def get_mtp_target_hidden_states(self):
            return self.hidden_probe

        def load_weights(self, weights):
            self.loaded_for_test = {name for name, _ in weights}
            return self.loaded_for_test

    scope.update(
        Qwen3_VisionTransformer=FakeVision,
        Qwen4ExpForCausalLM=FakeLanguage,
        get_pp_group=lambda: NS(is_first_rank=pp["first"]),
        is_offload_process=lambda: offload["active"],
        cached_tokenizer_from_config=lambda _: object(),
        _finalize_qsa_e4m3_scale_load=lambda *a, **kw: None,
        _maybe_enable_sm70_peer_paths=lambda *a, **kw: None,
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<candidate>", "exec"), scope
    )
    cls = scope["Qwen4ExpForConditionalGeneration"]
    cls.set_moe_parameters = lambda *args: None
    mm = NS(
        language_model_only=False,
        mm_encoder_only=False,
        mm_encoder_tp_mode="weights",
        video_pruning_rate=0,
        is_multimodal_pruning_enabled=lambda: False,
        get_limit_per_prompt=lambda _: 1,
    )
    config = NS(
        vision_config=NS(deepstack_visual_indexes=[5, 11, 17], out_hidden_size=4),
        text_config=NS(rms_norm_eps=1e-6, hidden_size=4),
    )
    vconfig = NS(
        model_config=NS(hf_config=config, multimodal_config=mm, quantization="gguf"),
        quant_config=None,
        scheduler_config=NS(max_num_batched_tokens=16),
        kernel_config=NS(qsa_auto_e4m3_active=False),
    )
    first = cls(vllm_config=vconfig, prefix="")
    assert created == ["visual"] and first.use_deepstack
    assert len(first.deepstack_input_embeds) == 3
    assert first.get_language_model() is first.language_model
    assert first.get_mtp_target_hidden_states() is first.language_model.hidden_probe
    first.load_weights(
        [
            ("model.visual.weight", torch.ones(2, 2)),
            ("model.visual.bias", torch.ones(2)),
        ]
    )
    missing_rejected = False
    try:
        first.load_weights([("model.visual.weight", torch.ones(2, 2))])
    except ValueError as exc:
        missing_rejected = "visual.bias" in str(exc)
    assert missing_rejected

    pp["first"] = False
    later = cls(vllm_config=vconfig, prefix="")
    assert created == ["visual"]
    assert isinstance(later.visual, original.StageMissingLayer)
    assert not later.use_deepstack and later._tower_model_names == []
    assert (
        later._tokenizer is not None
        and later.get_language_model() is later.language_model
    )
    assert later.get_mtp_target_hidden_states() is later.language_model.hidden_probe
    assert not list(later.visual.named_parameters())
    later.load_weights(
        [
            ("model.visual.weight", torch.ones(2, 2)),
            ("model.visual.bias", torch.ones(2)),
        ]
    )

    pp.update(first=True, end=8)
    offload["active"] = True
    ple_worker = cls(vllm_config=vconfig, prefix="")
    assert isinstance(ple_worker.visual, original.StageMissingLayer)
    assert created == ["visual"] and not ple_worker.use_deepstack
    assert ple_worker._tower_model_names == []
    ple_name = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    loaded = ple_worker.load_weights([(ple_name, torch.ones(2, 2))])
    assert loaded == {ple_name}
    assert ple_worker.language_model.loaded_for_test == {
        ple_name.removeprefix("language_model.")
    }
    offload["active"] = False

    pp.update(first=True, end=2)
    split_rejected = False
    try:
        cls(vllm_config=vconfig, prefix="")
    except AssertionError as exc:
        split_rejected = "deepstack" in str(exc)
    assert split_rejected

    mm.language_model_only = True
    text = cls(vllm_config=vconfig, prefix="")
    assert isinstance(text.visual, original.StageMissingLayer)
    assert not text.use_deepstack
    text.load_weights([])
    assert not torch.cuda.is_initialized()
