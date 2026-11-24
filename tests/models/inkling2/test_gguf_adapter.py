# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Inkling2 GGUF adapter reproduces the official checkpoint (CPU only).

INKLING2_GGUF_Q8 / INKLING2_GGUF_Q4: Unsloth Inkling-2-Mini GGUFs;
INKLING2_OFFICIAL: the `thinkingmachines/Inkling-2-Mini` directory. Each test
is skipped when its files are not set.
"""

import os

import pytest
import torch

gguf = pytest.importorskip("gguf")
safetensors = pytest.importorskip("safetensors")

from vllm.model_executor.model_loader.gguf_adapters.inkling2 import (  # noqa: E402
    Inkling2Adapter,
)
from vllm.models.inkling2.gguf_config import inkling2_config_from_gguf  # noqa: E402

_OFFICIAL = os.environ.get("INKLING2_OFFICIAL", "")
_Q8 = os.environ.get("INKLING2_GGUF_Q8", "")
_Q4 = os.environ.get("INKLING2_GGUF_Q4", "")
_LAYERS = (0, 5, 11, 20, 23)  # sliding, full, looped full, BF16/Q8_0 experts, last


def _adapter(path):
    reader = gguf.GGUFReader(path)
    meta = {k: f.contents() for k, f in reader.fields.items()
            if k.startswith(("inkling2.", "general.architecture"))}
    config = inkling2_config_from_gguf(meta).text_config
    tensors = {t.name: t for t in reader.tensors}
    adapter = Inkling2Adapter(config)
    return adapter, tensors, adapter.build_name_map(tensors)


def _selected(name):
    if ".layers." not in name:
        return True
    return int(name.split(".layers.")[1].split(".")[0]) in _LAYERS


def _official():
    return safetensors.safe_open(
        os.path.join(_OFFICIAL, "model-00001-of-00001.safetensors"), "pt"
    )


def _mxfp8_dequant(w, s):
    return w.float() * torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=-1)


needs_q8 = pytest.mark.skipif(
    not (os.path.isdir(_OFFICIAL) and os.path.exists(_Q8)),
    reason="set INKLING2_OFFICIAL and INKLING2_GGUF_Q8",
)
needs_q4 = pytest.mark.skipif(
    not (os.path.isdir(_OFFICIAL) and os.path.exists(_Q4)),
    reason="set INKLING2_OFFICIAL and INKLING2_GGUF_Q4",
)


@needs_q8
def test_names_cover_the_official_text_model():
    adapter, tensors, name_map = _adapter(_Q8)
    with _official() as f:
        official = {k for k in f.keys() if not k.startswith("model.visual.")}
    # expert global scales ("*_exps.scale") are emitted with their bank
    yielded = {n for n in name_map.values() if "_exps." not in n}
    expert_extras = {
        f"{n}{s}"
        for n in yielded
        if n.endswith(("experts.w13_weight", "experts.w2_weight"))
        for s in ("_scale", "_amax", "_outer_scale_denominator")
    }
    produced = yielded | expert_extras
    # every official text tensor is produced; extras only for non-NVFP4 banks
    assert official - produced == set()
    extra = produced - official
    assert all(".mlp.experts.w" in n for n in extra), sorted(extra)[:5]


@needs_q8
def test_q8_matches_official_values():
    adapter, tensors, name_map = _adapter(_Q8)
    sub = {r: n for r, n in name_map.items() if _selected(n)}
    with _official() as f:
        keys = set(f.keys())
        for name, w in adapter.weights(tensors, sub, torch.float16):
            assert name in keys or ".mlp.experts.w" in name, name
            if name.endswith(("w13_weight", "w2_weight")) and w.is_floating_point():
                # BF16 banks of the layers the official checkpoint holds as MXFP8
                ref = _mxfp8_dequant(f.get_tensor(name), f.get_tensor(name + "_scale"))
                assert w.shape == ref.shape, name
                torch.testing.assert_close(w.float(), ref, rtol=0, atol=0)
                continue
            if name not in keys:
                continue  # *_scale of a BF16 bank (absent in the official MXFP8 path)
            ref = f.get_tensor(name)
            assert w.shape == ref.shape, (name, w.shape, ref.shape)
            if name.endswith("_amax"):
                denom = f.get_tensor(name.replace("_amax", "_outer_scale_denominator"))
                torch.testing.assert_close(w.float(), ref.float() / denom.float(),
                                           rtol=1e-6, atol=0)
            elif name.endswith("_outer_scale_denominator"):
                continue  # checked together with amax
            elif ref.dtype in (torch.uint8, torch.float8_e4m3fn):
                assert w.dtype == ref.dtype, name
                assert torch.equal(w.view(torch.uint8), ref.view(torch.uint8)), name
            else:
                torch.testing.assert_close(w.float(), ref.float(), rtol=0, atol=0,
                                           msg=name)


@needs_q4
def test_q4_dense_within_q8_0_error():
    adapter, tensors, name_map = _adapter(_Q4)
    sub = {r: n for r, n in name_map.items()
           if _selected(n) and n.endswith(("wq_du.weight", "wo_ud.weight"))}
    with _official() as f:
        for name, w in adapter.weights(tensors, sub, torch.float16):
            ref = f.get_tensor(name).float()
            assert w.shape == ref.shape, name
            err = (w.float() - ref).abs().max() / ref.abs().max()
            assert err < 1e-2, (name, float(err))
