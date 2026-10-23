# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/benchmark_dflash2_concurrent_rounds.py"
)
_SPEC = importlib.util.spec_from_file_location("concurrent_rounds", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def cohort(times, index=0):
    return {
        "request_index": index,
        "events": [(timestamp, [1, 2, 3]) for timestamp in times],
        "ttft_ms": 5000,
        "usage": {"completion_tokens": len(times) * 3},
    }


def test_common_window_excludes_prefill_and_partial_intervals():
    # The slower prefill determines the start. Neither early decode from the
    # first request nor the partially intersecting interval counts as a round.
    result = _MODULE.common_window(
        [cohort([0, 1, 2, 3, 4, 5, 6]), cohort([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], 1)],
        trim=1,
    )
    assert result["begin_wall_s"] == 2.5
    assert result["end_wall_s"] == 5
    assert [r["intervals"] for r in result["requests"]] == [2, 2]
    assert all(r["stream_interval_ms_mean"] == 1000 for r in result["requests"])
    assert all(r["tokens_per_stream_chunk"] == 3 for r in result["requests"])


def test_common_window_rejects_nonoverlapping_requests():
    with pytest.raises(ValueError, match="no common"):
        _MODULE.common_window(
            [cohort([0, 1, 2, 3, 4]), cohort([10, 11, 12, 13, 14], 1)], trim=1
        )


def test_profile_stops_when_a_stream_fails(monkeypatch, tmp_path):
    calls = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, url):
            calls.append(url.rsplit("/", 1)[-1])
            return SimpleNamespace(raise_for_status=lambda: None)

    async def failed_request(client, args, prompt, index, barrier, progress):
        await barrier.wait()
        progress(index, 1)
        raise RuntimeError("stream failed")

    monkeypatch.setattr(_MODULE.httpx, "AsyncClient", lambda **_kwargs: Client())
    monkeypatch.setattr(_MODULE, "request", failed_request)
    fixture = tmp_path / "inputs.json"
    fixture.write_text(json.dumps({"8192": ["prompt"]}))
    args = SimpleNamespace(
        inputs=fixture,
        input_tokens=8192,
        concurrency=1,
        base_url="https://benchmark.example",
        profile_after_chunks=1,
    )
    with pytest.raises(RuntimeError, match="stream failed"):
        asyncio.run(_MODULE.run(args))
    assert calls == ["start_profile", "stop_profile"]
