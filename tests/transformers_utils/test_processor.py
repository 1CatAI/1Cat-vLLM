# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

from transformers.processing_utils import ProcessingKwargs
from typing_extensions import Unpack

from vllm.transformers_utils.processor import (
    get_processor_kwargs_keys,
    get_processor_kwargs_type,
)


class _FakeProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore
    pass


def _assert_has_all_expected(keys: set[str]) -> None:
    # text
    for k in ("text_pair", "text_target", "text_pair_target"):
        assert k in keys
    # image
    for k in ("do_convert_rgb", "do_resize"):
        assert k in keys
    # audio
    for k in (
        "fps",
        "do_sample_frames",
        "input_data_format",
        "default_to_square",
    ):
        assert k in keys
    # audio
    for k in ("padding", "return_attention_mask"):
        assert k in keys


# Path 1: __call__ method has kwargs: Unpack[*ProcessorKwargs]
class _ProcWithUnpack:
    def __call__(self, *args, **kwargs: Unpack[_FakeProcessorKwargs]):  # type: ignore
        return None


def test_get_processor_kwargs_from_processor_unpack_path_returns_full_union():
    proc = _ProcWithUnpack()
    keys = get_processor_kwargs_keys(get_processor_kwargs_type(proc))
    _assert_has_all_expected(keys)


# ---- Path 2: No Unpack, fallback to scanning *ProcessorKwargs in module ----


class _ProcWithoutUnpack:
    def __call__(self, *args, **kwargs):
        return None


def test_get_processor_kwargs_from_processor_module_scan_returns_full_union():
    # ensure the module scanned by fallback is this test module
    module_name = _ProcWithoutUnpack.__module__
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "_FakeProcessorKwargs")

    proc = _ProcWithoutUnpack()
    keys = get_processor_kwargs_keys(get_processor_kwargs_type(proc))
    _assert_has_all_expected(keys)


class _FakeProcessor:
    attributes = ["image_processor", "tokenizer"]
    calls: list[dict] = []
    tokenizer: object = None

    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        cls.calls.append(kwargs)
        if kwargs.get("fail_without_tokenizer") and "tokenizer" not in kwargs:
            raise OSError("no tokenizer files")
        processor = cls()
        processor.tokenizer = kwargs.get("tokenizer", "loaded-from-path")
        return processor


def test_processor_build_attaches_given_tokenizer_without_passing_it():
    from vllm.transformers_utils.processor import _from_pretrained_with_tokenizer

    _FakeProcessor.calls = []
    tokenizer = object()
    processor = _from_pretrained_with_tokenizer(
        _FakeProcessor, "path", revision="main", tokenizer=tokenizer
    )
    assert processor.tokenizer is tokenizer
    assert "tokenizer" not in _FakeProcessor.calls[0]  # not deep-copied by HF


def test_processor_build_falls_back_to_passing_tokenizer():
    from vllm.transformers_utils.processor import _from_pretrained_with_tokenizer

    _FakeProcessor.calls = []
    tokenizer = object()
    processor = _from_pretrained_with_tokenizer(
        _FakeProcessor, "path", tokenizer=tokenizer, fail_without_tokenizer=True
    )
    assert processor.tokenizer is tokenizer
    assert len(_FakeProcessor.calls) == 2
