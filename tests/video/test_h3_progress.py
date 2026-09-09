# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Progress is request-local and never consumes a worker's terminal reply."""

import multiprocessing
import threading
import time

import pytest
from fastapi.testclient import TestClient

from vllm.media.progress import report, reporting, update_metadata
from vllm.model_executor.models.minimax_h3.config import H3Config
from vllm.video.engine import H3Engine
from vllm.video.server import create_app


def test_progress_scope_restores_after_errors():
    events: list[dict] = []
    with pytest.raises(ValueError), reporting(events.append):
        report("encoding")
        with reporting(None):
            report("invisible")
        report("denoising", completed=1, total=4)
        raise ValueError("failed inference")
    report("not part of the request")
    assert events == [
        {"stage": "encoding"},
        {"stage": "denoising", "completed": 1, "total": 4},
    ]


def test_receive_consumes_all_progress_before_terminal_reply():
    parent, child = multiprocessing.Pipe()
    engine = object.__new__(H3Engine)
    engine.connections = [parent]
    engine.workers = []
    events: list[dict] = []
    try:
        child.send({"event": "progress", "stage": "encoding"})
        child.send(
            {"event": "progress", "stage": "denoising", "completed": 1, "total": 4}
        )
        child.send({"rank": 0, "video": "output.mp4"})
        assert engine._receive_all(timeout=2, on_progress=events.append) == [
            {"rank": 0, "video": "output.mp4"}
        ]
        assert len(events) == 2
        assert events[-1]["total"] == 4
    finally:
        parent.close()
        child.close()


def test_stage_clock_is_preserved_within_stage_and_counts_are_not_percentages():
    record: dict = {}
    update_metadata(record, {"stage": "denoising", "completed": 1, "total": 4}, 10)
    update_metadata(record, {"stage": "denoising", "completed": 2, "total": 4}, 15)
    assert record["stage_started_at"] == 10
    assert record["updated_at"] == 15
    assert record["denoise_progress"] == {"completed": 2, "total": 4}
    assert "progress" not in record
    update_metadata(record, {"stage": "decoding"}, 20)
    assert record["stage_started_at"] == 20
    assert record["stage_progress"] is None


def test_api_reports_progress_before_output_exists(tmp_path):
    at_step = threading.Event()
    finish = threading.Event()

    class Engine:
        _closed = False

        def generate(self, request, output, *, on_progress):
            on_progress({"stage": "denoising", "completed": 1, "total": 4})
            at_step.set()
            assert finish.wait(5)
            on_progress({"stage": "decoding"})
            output.mkdir(parents=True)
            (output / "video.mp4").write_bytes(b"video")
            return {"ranks": []}

        def close(self):
            self._closed = True

    app = create_app(H3Config(), tmp_path, engine_factory=lambda _: Engine())
    with TestClient(app) as client:
        try:
            identity = client.post(
                "/v1/videos", json={"prompt": "A running cat"}
            ).json()["id"]
            assert at_step.wait(5)
            snapshot = client.get(f"/v1/videos/{identity}").json()
            assert snapshot["status"] == "in_progress"
            assert snapshot["stage"] == "denoising"
            assert snapshot["denoise_progress"] == {"completed": 1, "total": 4}
            assert client.get(f"/v1/videos/{identity}/content").status_code == 409
        finally:
            finish.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snapshot = client.get(f"/v1/videos/{identity}").json()
            if snapshot["status"] == "completed":
                break
            time.sleep(0.01)
        assert snapshot["stage"] == "completed"
