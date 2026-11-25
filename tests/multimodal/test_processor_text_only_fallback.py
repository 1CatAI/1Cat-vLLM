# SPDX-License-Identifier: Apache-2.0
"""`_apply_hf_processor_text_only` must not fail when the HF processor
rejects placeholder tokens without multimodal items (transformers>=5)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.multimodal.processing.processor import BaseMultiModalProcessor


def _fake_processor(text_mm_side_effect, encoded=(1, 2, 3)):
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(encoded)
    proc = SimpleNamespace(
        info=SimpleNamespace(get_tokenizer=lambda: tokenizer),
        _apply_hf_processor_text_mm=MagicMock(side_effect=text_mm_side_effect),
    )
    return proc, tokenizer


def test_uses_hf_processor_when_it_accepts_text():
    proc, tokenizer = _fake_processor(lambda **kw: ([7, 8, 9], {}, False))
    ids = BaseMultiModalProcessor._apply_hf_processor_text_only(
        proc, "<|vision_start|><|image_pad|><|vision_end|>", {"truncation": False}
    )
    assert ids == [7, 8, 9]
    tokenizer.encode.assert_not_called()


def test_falls_back_to_tokenizer_when_hf_processor_rejects_placeholders():
    err = ValueError("Found 1 <|image_pad|> tokens in the text but no images were passed.")
    proc, tokenizer = _fake_processor(err, encoded=(151652, 151655, 151653))
    ids = BaseMultiModalProcessor._apply_hf_processor_text_only(
        proc,
        "<|vision_start|><|image_pad|><|vision_end|>",
        {"truncation": False, "return_tensors": "pt"},
    )
    assert ids == [151652, 151655, 151653]
    # only tokenizer-level kwargs are forwarded
    tokenizer.encode.assert_called_once_with(
        "<|vision_start|><|image_pad|><|vision_end|>", truncation=False
    )


def test_other_errors_still_propagate():
    proc, _ = _fake_processor(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        BaseMultiModalProcessor._apply_hf_processor_text_only(proc, "hello", {})
