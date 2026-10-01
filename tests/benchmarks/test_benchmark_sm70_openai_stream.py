# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from pathlib import Path

import pytest

_path = (
    Path(__file__).resolve().parents[2] / "benchmarks/benchmark_sm70_openai_stream.py"
)
_spec = importlib.util.spec_from_file_location("sm70_openai_stream", _path)
assert _spec is not None and _spec.loader is not None
stream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stream)


def test_speculative_chunks_are_counted_by_tokens():
    result = stream._latencies([(1.0, 1), (1.02, 4), (1.04, 4)])
    assert result["tpot_mean_ms"] == pytest.approx(5.0)
    assert result["chunk_interval_mean_ms"] == pytest.approx(20.0)
    assert result["tpot_p50_ms"] == 0


def test_prompt_can_fill_32k(monkeypatch):
    responses = iter([[1, 2, 6, 7], [1, 2, 3, 4, 5, 6, 7]])
    monkeypatch.setattr(stream, "_tokenize", lambda *args: next(responses))
    prompt = stream._build_prompt_ids("http://unused", "model", 32768)
    assert len(prompt) == 32768
    assert prompt[:5] == [1, 2, 3, 4, 5]
    assert prompt[-2:] == [6, 7]


def test_tokenization_preserves_chat_generation_prompt(monkeypatch):
    import io
    from contextlib import contextmanager

    @contextmanager
    def post(url, payload):
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["add_generation_prompt"] is True
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        yield io.StringIO('{"tokens": [1, 2]}')

    monkeypatch.setattr(stream, "_post_json", post)
    assert stream._tokenize("http://unused", "model", "hello") == [1, 2]


def test_one_token_has_no_decode_interval():
    assert stream._latencies([(1.0, 1)])["tpot_mean_ms"] is None
