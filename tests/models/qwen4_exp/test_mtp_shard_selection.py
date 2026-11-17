# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select split MTP shards while preserving the existing tensor skip policy."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.config import LoadConfig
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("layout", ["split_scale", "compact", "ordinary"])
@pytest.mark.parametrize("share_io", [False, True])
def test_mtp_shards_and_tensor_filtering(
    tmp_path, monkeypatch, indexed, layout, share_io
):
    w13 = "mtp.layers.0.mlp.experts.gate_up_proj"
    w2 = "mtp.layers.0.mlp.experts.down_proj"
    scale = "mtp.layers.0.self_attn.k_scale"
    target = "model.language_model.layers.0.mlp.gate.weight"
    embedding = "model.language_model.embed_tokens.weight"
    head = "lm_head.weight"
    body = {
        w13: torch.arange(16).reshape(1, 4, 4).to(torch.bfloat16),
        w2: torch.arange(8).reshape(1, 4, 2).to(torch.bfloat16),
    }
    scales = {scale: torch.tensor(0.25)}
    shards = {
        "model-target.safetensors": {target: torch.ones(4, 4)},
    }
    io = {embedding: torch.ones(3, 4), head: torch.full((3, 4), 2.0)}
    if layout == "compact":
        shards["model-bf16-mtp.safetensors"] = body | scales | io
    elif layout == "split_scale":
        shards["model-body.safetensors"] = body | io
        shards["model-bf16-scales.safetensors"] = scales
    else:
        shards["model-body.safetensors"] = body | scales | io
    weight_map: dict[str, str] = {}
    for filename, tensors in shards.items():
        save_file(tensors, str(tmp_path / filename))
        weight_map.update(dict.fromkeys(tensors, filename))
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )

    # Bypass model construction, retaining its real pattern and skip methods.
    model = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
    torch.nn.Module.__init__(model)
    model.share_target_io_weights = share_io
    loader = DefaultModelLoader(
        LoadConfig(
            load_format="safetensors",
            use_tqdm_on_load=False,
            safetensors_load_strategy="lazy",
        )
    )
    reads: list[str] = []

    class TrackedFile:
        def __init__(self, *args, **kwargs):
            self.file = safe_open(*args, **kwargs)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def keys(self):
            return self.file.keys()

        def get_tensor(self, name):
            reads.append(name)
            return self.file.get_tensor(name)

    monkeypatch.setattr(weight_utils, "safe_open", TrackedFile)
    actual = dict(
        loader.get_all_weights(
            SimpleNamespace(model=str(tmp_path), revision=None), model
        )
    )
    expected = body | scales | ({} if share_io else io)
    assert set(actual) == set(expected)
    assert set(reads) == set(expected)
    assert target not in reads
    for name, tensor in expected.items():
        assert actual[name].dtype == tensor.dtype
        assert torch.equal(actual[name], tensor)
