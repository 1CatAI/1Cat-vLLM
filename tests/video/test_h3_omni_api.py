# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni wire requests use native jobs with bounded, request-owned media."""

import json
import time
from pathlib import Path

import httpx
import pybase64 as base64
import pytest
import test_h3_workflows
from fastapi.testclient import TestClient

from vllm.model_executor.models.minimax_h3.config import H3Config
from vllm.video.server import create_app

media = test_h3_workflows.media


class RecordingEngine:
    def __init__(self, config):
        self.requests = []
        self._closed = False

    def generate(self, request, output):
        for paths in request.media.values():
            assert all(Path(path).is_file() for path in paths)
        self.requests.append(request)
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "video.mp4").write_bytes(str(request.sampling.seed).encode())
        return {"ranks": []}

    def close(self):
        self._closed = True


def wait_job(client, identity):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = client.get(f"/v1/videos/{identity}").json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.01)
    pytest.fail("video job did not complete")


@pytest.mark.parametrize(
    "task,kinds,indices",
    [
        ("t2va", [], None),
        ("fl2va", ["image"], [0]),
        ("fl2va", ["image"], [-1]),
        ("fl2va", ["image", "image"], [0, -1]),
        ("ref2va", ["image"], None),
        ("ref2va", ["image", "image"], None),
        ("ref2va", ["image", "audio"], None),
        ("ref2va", ["video", "video"], None),
        ("ref2va", ["image", "video"], None),
        ("ref2va", ["video", "audio"], None),
        ("ref2va", ["image", "video", "audio"], None),
    ],
)
def test_official_multipart_workflow_matrix(task, kinds, indices, media, tmp_path):
    config = H3Config(partition="ref2va" if task == "ref2va" else "fl2va")
    engine = RecordingEngine(config)
    app = create_app(config, tmp_path, engine_factory=lambda _: engine)
    extra = {"task": task, "duration": 4.4, "audio_flow_shift": 3}
    if indices is not None:
        extra["frame_indices"] = indices
    if "video" in kinds:
        extra["start_time_seconds"] = [0.5] * kinds.count("video")
    mime = {"image": "image/png", "video": "video/mp4", "audio": "audio/wav"}
    files = [
        (
            "input_references",
            ("../../source", Path(media[kind]).read_bytes(), mime[kind]),
        )
        for kind in kinds
    ]
    data = {
        "model": config.model,
        "prompt": "A subject moves through the scene.",
        "fps": "24",
        "num_inference_steps": "50",
        "flow_shift": "12",
        "guidance_scale": "1.0",
        "true_cfg_scale": "1.0",
        "vae_use_tiling": "False",
        "vae_use_slicing": "False",
        "extra_params": json.dumps(extra),
    }
    with TestClient(app) as client:
        submitted = client.post("/v1/videos", data=data, files=files)
        assert submitted.status_code == 202, submitted.text
        identity = submitted.json()["id"]
        assert wait_job(client, identity)["status"] == "completed"
        assert engine.requests[-1].sampling.extra_args["task"] == task
        assert engine.requests[-1].sampling.extra_args["duration_seconds"] == 4.4
        assert {
            key: len(value) for key, value in engine.requests[-1].media.items()
        } == {kind: kinds.count(kind) for kind in set(kinds)}
        assert client.get(f"/v1/videos/{identity}/content").content == b"42"
        assert client.delete(f"/v1/videos/{identity}").json()["deleted"]
        assert not (tmp_path / identity).exists()
    assert not list((tmp_path / ".inputs").glob("*")) if kinds else True
    assert all(Path(path).exists() for path in media.values())


def test_multi_outputs_seeds_downloads_listing_and_delete(tmp_path):
    engine = RecordingEngine(H3Config())
    app = create_app(H3Config(), tmp_path, engine_factory=lambda _: engine)
    with TestClient(app) as client:
        assert client.get("/v1/models").json()["data"][0]["id"] == H3Config().model
        response = client.post(
            "/v1/videos", json={"seed": 7, "num_outputs_per_prompt": 3}
        )
        identity = response.json()["id"]
        job = wait_job(client, identity)
        assert job["status"] == "completed"
        assert [output["seed"] for output in job["outputs"]] == [7, 8, 9]
        for index, output in enumerate(job["outputs"]):
            assert client.get(output["url"]).content == str(7 + index).encode()
        assert (
            client.get(f"/v1/videos/{identity}/content?output_index=3").status_code
            == 404
        )
        assert len(client.get("/v1/videos").json()["data"]) == 1
        assert client.get("/v1/videos", params={"after": identity}).json()["data"] == []
        assert client.delete(f"/v1/videos/{identity}").status_code == 200
        assert client.get(f"/v1/videos/{identity}").status_code == 404
        assert client.get("/v1/videos").json()["data"] == []


def test_sync_returns_first_output_and_cleans_transient_job(tmp_path):
    engine = RecordingEngine(H3Config())
    app = create_app(H3Config(), tmp_path, engine_factory=lambda _: engine)
    with TestClient(app) as client:
        response = client.post(
            "/v1/videos/sync",
            data={"prompt": "A boat", "seed": "31", "num_outputs_per_prompt": "2"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "video/mp4"
        assert response.content == b"31"
        assert [request.sampling.seed for request in engine.requests] == [31, 32]
        assert client.get("/v1/videos").json()["data"] == []
        assert not list(tmp_path.glob("video_*"))


def test_typed_data_urls_and_http_references(media, tmp_path, monkeypatch):
    original_client = httpx.AsyncClient
    calls = []

    def download(request):
        calls.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=Path(media["image"]).read_bytes(),
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(download), **kwargs
        ),
    )
    config = H3Config(partition="ref2va")
    engine = RecordingEngine(config)
    app = create_app(config, tmp_path, engine_factory=lambda _: engine)
    audio = (
        "data:audio/wav;base64,"
        + base64.b64encode(Path(media["audio"]).read_bytes()).decode()
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/videos/sync",
            json={
                "model": config.model,
                "image_reference": {"image_url": "https://assets.example/ref.png"},
                "audio_reference": [{"audio_url": audio}],
                "extra_params": {"task": "ref2va"},
            },
        )
        assert response.status_code == 200, response.text
        assert calls == ["https://assets.example/ref.png"]
        assert set(engine.requests[-1].media) == {"image", "audio"}
        assert not list((tmp_path / ".inputs").glob("*"))


@pytest.mark.parametrize(
    "body",
    [
        {"extra_params": {"duration": 4.4}, "duration": 5.0},
        {"extra_params": {"unknown": True}},
        {"guidance_scale": 7.5},
        {"num_outputs_per_prompt": 11},
        {"size": "invalid"},
        {"width": 768},
        {"model": "not-served"},
        {"image_reference": {"image_url": "data:image/png;base64,broken"}},
        {"audio_reference": {"image_url": "data:image/png;base64,YQ=="}},
    ],
)
def test_invalid_wire_requests_do_not_reach_engine(body, tmp_path):
    engine = RecordingEngine(H3Config())
    with TestClient(
        create_app(H3Config(), tmp_path, engine_factory=lambda _: engine)
    ) as client:
        response = client.post("/v1/videos", json=body)
        assert response.status_code == 422, response.text
        assert not engine.requests
        assert client.get("/health").status_code == 200
        assert not list((tmp_path / ".inputs").glob("*"))


def test_uploaded_reference_is_removed_after_validation_error(media, tmp_path):
    app = create_app(H3Config(), tmp_path, engine_factory=RecordingEngine)
    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            files={
                "input_reference": (
                    "image.png",
                    Path(media["image"]).read_bytes(),
                    "image/png",
                )
            },
            data={"extra_params": '{"task":"t2va"}'},
        )
        assert response.status_code == 422
        assert not list((tmp_path / ".inputs").glob("*"))


def test_lora_request_and_explicit_base_request(tmp_path):
    from test_h3_lora import _manifest

    config = H3Config(lora_path=str(_manifest(tmp_path)))
    engine = RecordingEngine(config)
    app = create_app(config, tmp_path / "outputs", engine_factory=lambda _: engine)
    with TestClient(app) as client:
        data = {
            "model": config.model,
            "num_inference_steps": "5",
            "flow_shift": "6",
            "extra_params": '{"task":"t2va","audio_flow_shift":3}',
            "lora": json.dumps(
                {
                    "local_path": config.lora_path,
                    "name": "turbo",
                    "scale": 1,
                    "int_id": 1,
                }
            ),
        }
        assert client.post("/v1/videos/sync", data=data).status_code == 200
        assert engine.requests[-1].sampling.lora_scale == 1
        assert (
            client.post("/v1/videos/sync", data={"model": config.model}).status_code
            == 200
        )
        assert engine.requests[-1].sampling.lora_scale == 1
        assert engine.requests[-1].sampling.num_inference_steps == 5
        assert (
            client.post(
                "/v1/videos/sync", data={"model": config.model, "lora_scale": "0"}
            ).status_code
            == 200
        )
        assert engine.requests[-1].sampling.lora_scale == 0
        assert engine.requests[-1].sampling.num_inference_steps == 50


def test_frontend_openapi_describes_json_uploads_and_video_response(tmp_path):
    with TestClient(
        create_app(H3Config(), tmp_path, engine_factory=RecordingEngine)
    ) as client:
        spec = client.get("/openapi.json").json()
        content = spec["paths"]["/v1/videos"]["post"]["requestBody"]["content"]
        assert set(content) == {"application/json", "multipart/form-data"}
        schemas = spec["components"]["schemas"]
        assert "RemoteLoRA" in schemas
        assert "video_reference" in schemas["VideoRequest"]["properties"]
        assert "input_references" in schemas["VideoForm"]["properties"]
        sync = spec["paths"]["/v1/videos/sync"]["post"]["responses"]["200"]
        assert "video/mp4" in sync["content"]
