# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from vllm.image.config import ImageConfig, ImageRequest
from vllm.image.server import create_app


class RecordingEngine:
    def __init__(self, config):
        self.config = config
        self.requests = []
        self.release = threading.Event()
        self.entered = threading.Event()

    def generate(self, request, output_dir, *, on_progress):
        self.requests.append(request)
        on_progress({"stage": "denoising", "completed": 1, "total": 8})
        self.entered.set()
        assert self.release.wait(5)
        output_dir.mkdir(exist_ok=True, parents=True)
        Image.new("RGB", (request.width, request.height), "orange").save(
            output_dir / "image.png"
        )
        return {"width": request.width, "height": request.height}

    def close(self):
        pass


def wait(client, identity):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = client.get(f"/v1/images/jobs/{identity}").json()
        if record["status"] in {"completed", "failed", "cancelled"}:
            return record
        time.sleep(0.01)
    pytest.fail("Image task did not settle")


def test_async_counts_cancel_queue_keep_running_result_and_restart(tmp_path):
    engine = RecordingEngine(ImageConfig(str(tmp_path)))
    app = create_app(engine.config, tmp_path, engine_factory=lambda _: engine)
    with TestClient(app) as client:
        try:
            first = client.post(
                "/v1/images/jobs",
                json={"prompt": "A cat"},
                headers={"Idempotency-Key": "same-key"},
            ).json()
            identity = first["id"]
            assert engine.entered.wait(5)
            again = client.post(
                "/v1/images/jobs",
                json={"prompt": "A cat"},
                headers={"Idempotency-Key": "same-key"},
            ).json()
            assert again["id"] == identity
            assert again["denoise_progress"] == {"completed": 1, "total": 8}
            assert client.delete(f"/v1/images/jobs/{identity}").status_code == 409
            queued = client.post("/v1/images/jobs", json={"prompt": "A dog"}).json()
            assert (
                client.delete(f"/v1/images/jobs/{queued['id']}").json()["status"]
                == "cancelled"
            )
        finally:
            engine.release.set()
        assert wait(client, identity)["status"] == "completed"
        assert (
            client.delete(f"/v1/images/jobs/{identity}").json()["status"] == "completed"
        )
        assert (
            client.get(f"/v1/images/jobs/{identity}/content").headers["content-type"]
            == "image/png"
        )
        assert len(engine.requests) == 1
    with TestClient(
        create_app(engine.config, tmp_path, engine_factory=lambda _: engine)
    ) as client:
        assert client.get(f"/v1/images/jobs/{identity}").json()["status"] == "completed"
        assert client.get(f"/v1/images/jobs/{identity}/content").status_code == 200
        replay = client.post(
            "/v1/images/jobs",
            json={"prompt": "A cat"},
            headers={"Idempotency-Key": "same-key"},
        )
        assert replay.json()["id"] == identity
        assert len(engine.requests) == 1


def test_invalid_model_and_dimensions_do_not_allocate_a_task(tmp_path):
    engine = RecordingEngine(ImageConfig(str(tmp_path)))
    with TestClient(
        create_app(engine.config, tmp_path, engine_factory=lambda _: engine)
    ) as client:
        for body in (
            {"prompt": " "},
            {"prompt": "cat", "model": "different"},
            {"prompt": "cat", "size": "1x1"},
            {"prompt": "cat", "width": 257},
            {"prompt": "cat", "size": "1024x768", "height": 1024},
        ):
            assert client.post("/v1/images/jobs", json=body).status_code == 422
    assert not engine.requests


def test_sync_compatibility_and_size_alias(tmp_path):
    engine = RecordingEngine(ImageConfig(str(tmp_path)))
    engine.release.set()
    with TestClient(
        create_app(engine.config, tmp_path, engine_factory=lambda _: engine)
    ) as client:
        response = client.post(
            "/v1/images/generations",
            json={"prompt": "A cat", "size": "512x512", "response_format": "b64_json"},
        )
        assert response.status_code == 200
        assert response.json()["data"][0]["b64_json"].startswith("iVBOR")
        assert engine.requests[0].width == 512


def test_bad_saved_record_does_not_block_other_image_jobs(tmp_path):
    broken = tmp_path / ("image_" + "a" * 32) / "job.json"
    broken.parent.mkdir()
    broken.write_text('{"unfinished":')
    engine = RecordingEngine(ImageConfig(str(tmp_path)))
    engine.release.set()
    with TestClient(
        create_app(engine.config, tmp_path, engine_factory=lambda _: engine)
    ) as client:
        assert client.get("/health").status_code == 200
        identity = client.post("/v1/images/jobs", json={"prompt": "cat"}).json()["id"]
        assert wait(client, identity)["status"] == "completed"
        assert broken.read_text() == '{"unfinished":'


def test_terminal_disk_failure_releases_waiter_and_does_not_kill_queue(
    tmp_path, monkeypatch
):
    engine = RecordingEngine(ImageConfig(str(tmp_path)))
    with TestClient(
        create_app(engine.config, tmp_path, engine_factory=lambda _: engine)
    ) as client:
        identity = client.post("/v1/images/jobs", json={"prompt": "cat"}).json()["id"]
        assert engine.entered.wait(5)
        replace = Path.replace

        def full(path, target):
            if path.name == "job.json.tmp":
                raise OSError("No space left on device")
            return replace(path, target)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", full)
            engine.release.set()
            failed = wait(client, identity)
            assert failed["status"] == "failed"
            assert failed["error"]["code"] == "persistence_failed"
        following = client.post("/v1/images/jobs", json={"prompt": "dog"}).json()["id"]
        assert wait(client, following)["status"] == "completed"


@pytest.mark.parametrize(
    "size", ["0x0", "4096x4096", "1024", "one x two", "256x256x256"]
)
def test_size_alias_is_revalidated(size):
    with pytest.raises(ValidationError):
        ImageRequest(prompt="cat", size=size)
