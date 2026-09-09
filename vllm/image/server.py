# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native image jobs with persisted terminal state and synchronous compatibility."""

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pybase64 as base64
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from vllm.media.progress import update_metadata

from .config import RECIPE_VERSION, ImageConfig, ImageRequest


def create_app(config: ImageConfig, output_dir: str | Path, *, engine_factory=None):
    from .engine import ImageEngine

    root = Path(output_dir).absolute()
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=32)
    jobs: dict[str, dict] = {}
    done: dict[str, asyncio.Event] = {}
    state: dict[str, Any] = {"ready": False, "closing": False, "engine": None}

    def save(record):
        folder = root / record["id"]
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / "job.json.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False))
        temporary.replace(folder / "job.json")

    def get(identity):
        if (
            not identity.startswith("image_")
            or len(identity) != 38
            or not all(c in "0123456789abcdef" for c in identity[6:])
        ):
            raise HTTPException(404, "Image job not found")
        if identity not in jobs:
            try:
                jobs[identity] = json.loads((root / identity / "job.json").read_text())
            except (OSError, ValueError) as exc:
                raise HTTPException(404, "Image job not found") from exc
        return jobs[identity]

    async def process():
        while True:
            identity = await queue.get()
            if identity is None:
                queue.task_done()
                return
            record = jobs[identity]
            if record["status"] != "queued":
                queue.task_done()
                continue
            record.update(status="in_progress", started_at=time.time())
            loop = asyncio.get_running_loop()

            def apply(event, record=record):
                if record["status"] == "in_progress":
                    update_metadata(record, event, time.time())
                    save(record)

            def callback(event, loop=loop, apply=apply):
                loop.call_soon_threadsafe(apply, event)

            try:
                result = await asyncio.to_thread(
                    state["engine"].generate,
                    ImageRequest.model_validate(record["request"]),
                    root / identity,
                    on_progress=callback,
                )
                if not (root / identity / "image.png").is_file():
                    raise RuntimeError("Native engine returned without an image")
                update_metadata(record, {"stage": "completed"}, time.time())
                record.update(
                    status="completed",
                    completed_at=time.time(),
                    result=result,
                    content_url=f"/v1/images/jobs/{identity}/content",
                )
            except Exception as exc:
                update_metadata(record, {"stage": "failed"}, time.time())
                record.update(
                    status="failed",
                    completed_at=time.time(),
                    error={
                        "code": "generation_failed",
                        "message": str(exc),
                    },
                )
            finally:
                save(record)
                done[identity].set()
                queue.task_done()

    @asynccontextmanager
    async def lifespan(app):
        root.mkdir(parents=True, exist_ok=True)
        # A process restart cannot replay a submitted request without consent.
        for path in root.glob("image_*/job.json"):
            record = json.loads(path.read_text())
            if record["status"] in {"queued", "in_progress"}:
                update_metadata(record, {"stage": "failed"}, time.time())
                record.update(
                    status="failed",
                    error={
                        "code": "interrupted",
                        "message": "Native image service restarted; "
                        "retry creates a new job",
                    },
                )
                save(record)
        state["engine"] = await asyncio.to_thread(engine_factory or ImageEngine, config)
        state["ready"] = True
        task = asyncio.create_task(process())
        try:
            yield
        finally:
            state.update(ready=False, closing=True)
            while not queue.empty():
                identity = queue.get_nowait()
                queue.task_done()
                if identity:
                    record = jobs[identity]
                    record.update(
                        status="cancelled", stage="cancelled", updated_at=time.time()
                    )
                    save(record)
                    done[identity].set()
            await queue.put(None)
            await task  # Do not release the GPU lease while its worker still runs.
            await asyncio.to_thread(state["engine"].close)

    app = FastAPI(title="1Cat native image API", lifespan=lifespan)

    @app.get("/health")
    async def health():
        if not state["ready"]:
            raise HTTPException(503, "Native image engine unavailable")
        return {
            "status": "ok",
            "kind": "image",
            "checkpoint": config.checkpoint,
            "recipe_version": RECIPE_VERSION,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": config.checkpoint, "object": "model", "owned_by": "1Cat"}],
        }

    def submit(body: ImageRequest, request_key: str | None):
        if not state["ready"]:
            raise HTTPException(503, "Native image engine unavailable")
        if body.model and body.model not in {config.checkpoint, config.model}:
            raise HTTPException(422, "Requested checkpoint is not loaded")
        if request_key and len(request_key) > 128:
            raise HTTPException(422, "Idempotency key is too long")
        body_hash = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
        if request_key:
            for record in jobs.values():
                if record.get("request_key") == request_key:
                    if record["request_hash"] != body_hash:
                        raise HTTPException(
                            409, "Idempotency key already used with different input"
                        )
                    return record
        if queue.full():
            raise HTTPException(429, "Native image queue is full")
        identity = "image_" + uuid.uuid4().hex
        now = time.time()
        record = {
            "id": identity,
            "object": "image.job",
            "model": config.checkpoint,
            "status": "queued",
            "stage": "queued",
            "created_at": now,
            "updated_at": now,
            "stage_started_at": now,
            "stage_progress": None,
            "denoise_progress": None,
            "request": body.model_dump(),
            "request_key": request_key,
            "request_hash": body_hash,
        }
        save(record)
        jobs[identity] = record
        done[identity] = asyncio.Event()
        queue.put_nowait(identity)
        return record

    @app.post("/v1/images/jobs", status_code=202)
    async def create(
        body: ImageRequest, idempotency_key: str | None = Header(default=None)
    ):
        return submit(body, idempotency_key)

    @app.get("/v1/images/jobs/{identity}")
    async def status(identity: str):
        return get(identity)

    @app.get("/v1/images/jobs/{identity}/content")
    async def content(identity: str):
        record = get(identity)
        if record["status"] != "completed":
            raise HTTPException(409, "Image is not ready")
        path = root / identity / "image.png"
        if not path.is_file():
            raise HTTPException(410, "Image output is missing")
        return FileResponse(path, media_type="image/png", filename=identity + ".png")

    @app.delete("/v1/images/jobs/{identity}")
    async def cancel(identity: str):
        record = get(identity)
        if record["status"] == "in_progress":
            raise HTTPException(
                409,
                "Running image generation cannot be interrupted; "
                "its output will be saved",
            )
        if record["status"] == "queued":
            update_metadata(record, {"stage": "cancelled"}, time.time())
            record["status"] = "cancelled"
            save(record)
            done[identity].set()
        return record

    @app.post("/v1/images/generations")
    async def compatible(body: ImageRequest):
        record = submit(body, None)
        await done[record["id"]].wait()
        if record["status"] != "completed":
            raise HTTPException(
                500, record.get("error", {"message": "Image job cancelled"})
            )
        if body.response_format == "b64_json":
            encoded = await asyncio.to_thread(
                (root / record["id"] / "image.png").read_bytes
            )
            output = {"b64_json": base64.b64encode(encoded).decode()}
        else:
            output = {"url": record["content_url"]}
        return {"created": record["created_at"], "data": [output]}

    return app
