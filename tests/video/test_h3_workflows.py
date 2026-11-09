# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official H3 media combinations reach the serial API without GPU work."""

import subprocess

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vllm.model_executor.models.minimax_h3.config import H3Config
from vllm.video.server import create_app


class RecordingEngine:
    _closed = False

    def __init__(self, config):
        pass

    def generate(self, request, output):
        return {"task": request.sampling.extra_args.get("task")}

    def close(self):
        self._closed = True


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("h3-workflows")
    image = root / "image.png"
    video = root / "video.mp4"
    audio = root / "audio.wav"
    Image.new("RGB", (256, 256), (160, 40, 10)).save(image)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=256x256:rate=24:duration=2.5",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000:duration=2",
            str(audio),
        ],
        check=True,
    )
    return {"image": str(image), "video": str(video), "audio": str(audio)}


@pytest.mark.parametrize(
    "task,kinds,indices",
    [
        ("t2va", (), None),
        ("fl2va", ("image",), [0]),
        ("fl2va", ("image",), [-1]),
        ("fl2va", ("image", "image"), [0, -1]),
        ("ref2va", ("image",), None),
        ("ref2va", ("image", "audio"), None),
        ("ref2va", ("video",), None),
        ("ref2va", ("image", "video", "audio"), None),
    ],
)
def test_official_workflow_requests(task, kinds, indices, media, tmp_path):
    partition = "ref2va" if task == "ref2va" else "fl2va"
    app = create_app(
        H3Config(partition=partition), tmp_path, engine_factory=RecordingEngine
    )
    body = {"task": task}
    for key in kinds:
        body.setdefault(key, []).append(media[key])
    if indices:
        body["keyframe_indices"] = indices
    if "video" in kinds:
        body["reference_video_start_times"] = [0.5]
    with TestClient(app) as client:
        response = client.post("/v1/videos", json=body)
        assert response.status_code == 202, response.text
        request = response.json()["request"]
        assert request["sampling"]["extra_args"]["task"] == task
        if "video" in kinds:
            assert request["sampling"]["extra_args"]["start_time_seconds"] == [0.5]
        assert request["sampling"]["num_inference_steps"] == 50


def test_invalid_reference_does_not_enter_worker_or_break_health(media, tmp_path):
    app = create_app(
        H3Config(partition="ref2va"), tmp_path, engine_factory=RecordingEngine
    )
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"invalid media")
    with TestClient(app) as client:
        cases = [
            {"audio": [media["audio"]]},  # Officially unsupported audio-only.
            {"video": [str(corrupt)]},
            {"image": [media["image"]], "audio": [str(corrupt)]},
            {"video": [media["video"]], "reference_video_start_times": [1.0]},
            {"video": [media["video"]], "reference_video_start_times": [0.0, 0.0]},
            {"image": [media["image"]], "reference_video_start_times": [0.0]},
            {"task": "t2va"},
        ]
        for body in cases:
            response = client.post("/v1/videos", json=body)
            assert response.status_code == 422, response.text
        assert client.get("/health").status_code == 200
        assert "onecat_video_failed_total 0" in client.get("/metrics").text


def test_turbo_http_uses_adapter_defaults_and_rejects_mismatched_steps(tmp_path):
    from test_h3_lora import _manifest

    config = H3Config(lora_path=str(_manifest(tmp_path)))
    app = create_app(config, tmp_path / "outputs", engine_factory=RecordingEngine)
    with TestClient(app) as client:
        response = client.post("/v1/videos", json={"task": "t2va"})
        assert response.status_code == 202
        sampling = response.json()["request"]["sampling"]
        assert sampling["num_inference_steps"] == 5
        assert sampling["extra_args"] == {
            "task": "t2va",
            "flow_shift": 6,
            "audio_flow_shift": 3,
        }
        for override in (
            {"num_inference_steps": 4},
            {"flow_shift": 12},
            {"audio_flow_shift": 6},
        ):
            assert client.post("/v1/videos", json=override).status_code == 422
        response = client.post("/v1/videos", json={"lora_scale": 0})
        assert response.json()["request"]["sampling"]["num_inference_steps"] == 50


def test_cli_exposes_workflow_and_turbo_controls():
    import argparse

    from vllm.entrypoints.cli.video import VideoSubcommand

    parser = argparse.ArgumentParser()
    VideoSubcommand().subparser_init(parser.add_subparsers())
    args = parser.parse_args(
        [
            "video",
            "generate",
            "--task",
            "ref2va",
            "--partition",
            "ref2va",
            "--image",
            "image.png",
            "--video",
            "video.mp4",
            "--audio",
            "audio.wav",
            "--reference-video-start-times",
            "0.5",
            "--lora-path",
            "adapter.safetensors",
            "--flow-shift",
            "6",
            "--audio-flow-shift",
            "3",
            "--lora-scale",
            "1",
        ]
    )
    assert args.task == args.partition == "ref2va"
    assert args.reference_video_start_times == [0.5]
    assert args.num_inference_steps is None
