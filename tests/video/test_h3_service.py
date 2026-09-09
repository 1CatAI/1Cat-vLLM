# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vllm.model_executor.models.minimax_h3.config import H3Config, H3Request
from vllm.video.server import create_app


class RecordingEngine:
    requests: list[H3Request] = []
    _closed = False

    def __init__(self, config):
        self.config = config

    def generate(self, request, output):
        self.requests.append(request)
        Path(output).mkdir(parents=True)
        (Path(output) / "video.mp4").write_bytes(b"test video")
        return {"ranks": []}

    def close(self):
        self._closed = True


def test_job_submission_completion_download_and_keyframes(tmp_path):
    from PIL import Image

    image = tmp_path / "last.png"
    Image.new("RGB", (256, 256)).save(image)
    app = create_app(H3Config(), tmp_path, engine_factory=RecordingEngine)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        submitted = client.post(
            "/v1/videos", json={"image": [str(image)], "keyframe_indices": [-1]}
        )
        assert submitted.status_code == 202
        identity = submitted.json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = client.get(f"/v1/videos/{identity}").json()
            if status["status"] == "completed":
                break
            time.sleep(0.01)
        assert status["status"] == "completed"
        assert RecordingEngine.requests[-1].sampling.extra_args["frame_indices"] == [-1]
        assert client.get(f"/v1/videos/{identity}/content").content == b"test video"
        assert "onecat_video_completed_total 1" in client.get("/metrics").text
        assert client.get("/v1/videos/missing").status_code == 404


def test_invalid_request_is_rejected_before_queueing(tmp_path):
    app = create_app(H3Config(), tmp_path, engine_factory=RecordingEngine)
    with TestClient(app) as client:
        assert client.post("/v1/videos", json={"width": 1345}).status_code == 422
        assert client.post("/v1/videos", json={"prompt": ""}).status_code == 422
        assert client.post("/v1/videos", json={"unexpected": True}).status_code == 422
        assert (
            client.post(
                "/v1/videos", json={"image": [str(tmp_path / "missing.png")]}
            ).status_code
            == 422
        )
        assert (
            client.post("/v1/videos", json={"keyframe_indices": [-1]}).status_code
            == 422
        )
        assert client.get("/health").status_code == 200


def test_export_preserves_native_mono_audio_and_video_contract(tmp_path):
    import numpy as np
    import torch

    from vllm.video.media import export_video
    from vllm.video.quality import inspect_video

    # Moving, non-black RGB frames and a finite 32-kHz waveform.
    video = torch.randint(32, 224, (1, 24, 64, 96, 3), dtype=torch.uint8)
    waveform = np.sin(2 * np.pi * 440 * np.arange(32000) / 32000) * 0.1
    audio = torch.from_numpy(waveform.astype(np.float32))[None, None]
    result = export_video(video, audio, tmp_path)
    report = inspect_video(
        result["video"], expected_frames=24, expected_width=96, expected_height=64
    )
    assert report["automatic_passed"]
    assert report["human_review"]["status"] == "pending"


@pytest.mark.parametrize("indices", [[0], [-1], [0, -1]])
def test_fl2va_preflight_accepts_first_last_and_both(indices):
    from PIL import Image

    from vllm.model_executor.models.minimax_h3.config import H3SamplingParams
    from vllm.model_executor.models.minimax_h3.validation import validate_request

    request = H3Request(
        sampling=H3SamplingParams(extra_args={"frame_indices": indices}),
        media={"image": [Image.new("RGB", (256, 256)) for _ in indices]},
    )
    validate_request(H3Config(), request)


def test_reference_contract_errors_are_rejected_before_worker_dispatch():
    from PIL import Image

    from vllm.model_executor.models.minimax_h3.config import (
        H3InputError,
        H3SamplingParams,
    )
    from vllm.model_executor.models.minimax_h3.validation import validate_request

    with pytest.raises(H3InputError, match="at least one"):
        validate_request(H3Config(partition="ref2va"), H3Request())
    image = Image.new("RGB", (256, 256))
    with pytest.raises(H3InputError, match="at most 9"):
        validate_request(
            H3Config(partition="ref2va"), H3Request(media={"image": [image] * 10})
        )
    with pytest.raises(H3InputError, match="frame_indices"):
        validate_request(
            H3Config(),
            H3Request(
                media={"image": [image]},
                sampling=H3SamplingParams(extra_args={"frame_indices": [1]}),
            ),
        )
