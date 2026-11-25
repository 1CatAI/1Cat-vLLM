# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GGUF-derived Inkling2 config matches the official checkpoint's (CPU only).

INKLING2_GGUF_Q4: an Unsloth Inkling-2-Mini GGUF; INKLING2_OFFICIAL: the
`thinkingmachines/Inkling-2-Mini` directory. Skipped when unset.
"""

import json
import math
import os

import pytest

from vllm.models.inkling2.gguf_config import _visit_layers, inkling2_config_from_gguf

_GGUF = os.environ.get("INKLING2_GGUF_Q4", "")
_OFFICIAL = os.environ.get("INKLING2_OFFICIAL", "")


def test_visit_layers():
    order = _visit_layers(48, 6, 18, 3)
    assert order[:6] == list(range(6))
    assert order[6:18] == order[18:30] == order[30:42] == list(range(6, 18))
    assert order[42:] == list(range(18, 24))
    with pytest.raises(ValueError):
        _visit_layers(10, 6, 18, 3)  # core longer than the model


def _gguf_metadata(path):
    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(path)
    meta = {}
    for key, field in reader.fields.items():
        if key.startswith(
            (
                "inkling2.",
                "general.architecture",
                "tokenizer.ggml.eos",
                "tokenizer.ggml.padding",
            )
        ):
            meta[key] = field.contents()
    return meta


@pytest.mark.skipif(
    not (os.path.exists(_GGUF) and os.path.isdir(_OFFICIAL)),
    reason="set INKLING2_GGUF_Q4 and INKLING2_OFFICIAL",
)
def test_matches_official_config():
    config = inkling2_config_from_gguf(_gguf_metadata(_GGUF))
    with open(os.path.join(_OFFICIAL, "config.json")) as f:
        official = json.load(f)
    assert config.architectures == official["architectures"]
    assert config.model_type == official["model_type"]
    assert config.vision_config is None
    text = config.text_config
    for key, want in official["text_config"].items():
        if key in ("model_type", "torch_dtype", "q_bias", "o_bias"):
            continue
        if key == "vocab_size":
            assert text.padded_vocab_size == want
            continue
        got = getattr(text, key)
        if isinstance(want, float):
            # GGUF stores these as float32
            assert math.isclose(got, want, rel_tol=1e-6), key
        else:
            assert got == want, key
