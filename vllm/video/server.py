# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native H3 video jobs with Omni-compatible media and task APIs."""

from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask

from vllm.media.progress import update_metadata
from vllm.model_executor.models.minimax_h3.config import H3Config, H3Request

from .protocol import VideoRequest as VideoRequest
from .protocol import parse_video_request


@dataclass
class VideoJob:
    metadata: dict[str, Any]
    request: H3Request
    input_dir: Path
    output_dir: Path
    output_count: int = 1
    visible: bool = True
    done: asyncio.Event = field(default_factory=asyncio.Event)


def create_app(config: H3Config, output_dir: str | Path, *, engine_factory=None):
    from .engine import H3Engine

    factory = engine_factory or H3Engine
    root = Path(output_dir).absolute()
    jobs: dict[str, VideoJob] = {}
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=32)
    state: dict[str, Any] = {
        "engine": None,
        "ready": False,
        "closing": False,
        "completed": 0,
        "failed": 0,
    }

    async def remove_job(identity: str):
        job = jobs.pop(identity, None)
        if job:
            await asyncio.to_thread(shutil.rmtree, job.input_dir, True)
            await asyncio.to_thread(shutil.rmtree, job.output_dir, True)

    async def process():
        while True:
            identity = await queue.get()
            job = jobs.get(identity)
            if job is None or job.metadata["status"] == "cancelled":
                queue.task_done()
                continue
            metadata = job.metadata
            metadata.update(status="in_progress", started_at=time.time())
            loop = asyncio.get_running_loop()

            def apply_progress(event, metadata=metadata):
                if metadata["status"] == "in_progress":
                    update_metadata(metadata, event, time.time())

            def on_progress(event, loop=loop, apply_progress=apply_progress):
                loop.call_soon_threadsafe(apply_progress, event)

            try:
                outputs = []
                for index in range(job.output_count):
                    request = replace(
                        job.request,
                        sampling=replace(
                            job.request.sampling, seed=job.request.sampling.seed + index
                        ),
                    )
                    directory = (
                        job.output_dir
                        if index == 0
                        else job.output_dir / f"output_{index}"
                    )
                    result = await asyncio.to_thread(
                        state["engine"].generate,
                        request,
                        directory,
                        on_progress=on_progress,
                    )
                    outputs.append(
                        {
                            "index": index,
                            "seed": request.sampling.seed,
                            "url": (
                                f"/v1/videos/{identity}/content?output_index={index}"
                            ),
                            "result": result,
                        }
                    )
                    metadata["progress"] = round(100 * (index + 1) / job.output_count)
                update_metadata(metadata, {"stage": "completed"}, time.time())
                metadata.update(
                    status="completed",
                    completed_at=time.time(),
                    result=outputs[0]["result"],
                    outputs=outputs,
                    content_url=f"/v1/videos/{identity}/content",
                )
                state["completed"] += 1
            except Exception as exc:
                update_metadata(metadata, {"stage": "failed"}, time.time())
                metadata.update(
                    status="failed",
                    error={"code": "generation_failed", "message": str(exc)},
                    completed_at=time.time(),
                )
                state["failed"] += 1
                state["ready"] = not state["engine"]._closed
            finally:
                job.done.set()
                # Shutdown closes workers before removing their active inputs.
                if not state["closing"]:
                    await asyncio.to_thread(shutil.rmtree, job.input_dir, True)
                queue.task_done()

    @asynccontextmanager
    async def lifespan(app):
        root.mkdir(parents=True, exist_ok=True)
        state["engine"] = await asyncio.to_thread(factory, config)
        state["ready"] = True
        task = asyncio.create_task(process())
        try:
            yield
        finally:
            state.update(ready=False, closing=True)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await asyncio.to_thread(state["engine"].close)
            for job in jobs.values():
                await asyncio.to_thread(shutil.rmtree, job.input_dir, True)

    app = FastAPI(title="1Cat H3 video API", lifespan=lifespan)

    @app.get("/health")
    async def health():
        if not state["ready"]:
            raise HTTPException(503, "video engine unavailable")
        return {"status": "ok", "partition": config.partition}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "1Cat",
                }
            ],
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return (
            f"onecat_video_ready {int(state['ready'])}\n"
            f"onecat_video_queue_depth {queue.qsize()}\n"
            f"onecat_video_completed_total {state['completed']}\n"
            f"onecat_video_failed_total {state['failed']}\n"
        )

    async def submit(request: Request, *, visible: bool) -> tuple[str, VideoJob]:
        if not state["ready"]:
            raise HTTPException(503, "video engine unavailable")
        identity = "video_" + uuid.uuid4().hex
        input_dir = root / ".inputs" / identity
        admitted = False
        try:
            body = await parse_video_request(request, input_dir)
            generation = await asyncio.to_thread(body.request, config)
            from vllm.model_executor.models.minimax_h3.validation import (
                validate_request,
            )

            await asyncio.to_thread(validate_request, config, generation)
            if not state["ready"]:
                raise HTTPException(503, "video engine unavailable")
            if queue.full():
                raise HTTPException(429, "video queue is full")
            job = VideoJob(
                metadata={
                    "id": identity,
                    "object": "video",
                    "model": config.model,
                    "status": "queued",
                    "progress": 0,
                    "stage": "queued",
                    "stage_started_at": time.time(),
                    "updated_at": time.time(),
                    "stage_progress": None,
                    "denoise_progress": None,
                    "created_at": int(time.time()),
                    "partition": config.partition,
                    "size": f"{generation.sampling.width}x{generation.sampling.height}",
                    "seconds": str(
                        generation.sampling.extra_args.get(
                            "duration_seconds", generation.sampling.num_frames / 24
                        )
                    ),
                    "request": asdict(generation),
                },
                request=generation,
                input_dir=input_dir,
                output_dir=root / identity,
                output_count=body.num_outputs_per_prompt,
                visible=visible,
            )
            jobs[identity] = job
            queue.put_nowait(identity)
            admitted = True
            return identity, job
        except ValidationError as exc:
            raise HTTPException(
                422, exc.errors(include_context=False, include_input=False)
            ) from exc
        except (ValueError, OSError, httpx.HTTPError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            if not admitted:
                await asyncio.to_thread(shutil.rmtree, input_dir, True)

    request_docs = {
        "requestBody": {
            "required": True,
            "content": {
                kind: {"schema": {"$ref": "#/components/schemas/" + schema}}
                for kind, schema in (
                    ("application/json", "VideoRequest"),
                    ("multipart/form-data", "VideoForm"),
                )
            },
        }
    }

    @app.post("/v1/videos", status_code=202, openapi_extra=request_docs)
    async def create_video(request: Request):
        _, job = await submit(request, visible=True)
        return job.metadata

    @app.post(
        "/v1/videos/sync",
        openapi_extra=request_docs,
        response_class=FileResponse,
        responses={
            200: {
                "content": {
                    "video/mp4": {"schema": {"type": "string", "format": "binary"}}
                }
            }
        },
    )
    async def sync_video(request: Request):
        identity, job = await submit(request, visible=False)
        await job.done.wait()
        if job.metadata["status"] != "completed":
            await remove_job(identity)
            raise HTTPException(500, job.metadata.get("error", "generation stopped"))
        return FileResponse(
            job.output_dir / "video.mp4",
            media_type="video/mp4",
            filename=identity + ".mp4",
            background=BackgroundTask(remove_job, identity),
        )

    @app.get("/v1/videos")
    async def list_videos(
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = None,
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
    ):
        records = [job.metadata for job in jobs.values() if job.visible]
        if order == "desc":
            records.reverse()
        if after is not None:
            indices = [i for i, record in enumerate(records) if record["id"] == after]
            if not indices:
                raise HTTPException(404, "video cursor not found")
            records = records[indices[0] + 1 :]
        data = records[:limit]
        return {
            "object": "list",
            "data": data,
            "has_more": len(records) > limit,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }

    def find_job(identity: str) -> VideoJob:
        job = jobs.get(identity)
        if job is None or not job.visible:
            raise HTTPException(404, "video job not found")
        return job

    @app.get("/v1/videos/{identity}")
    async def get_video(identity: str):
        return find_job(identity).metadata

    @app.get("/v1/videos/{identity}/content")
    async def download_video(identity: str, output_index: int = Query(default=0, ge=0)):
        job = find_job(identity)
        if job.metadata["status"] != "completed":
            raise HTTPException(409, "video is not completed")
        if output_index >= job.output_count:
            raise HTTPException(404, "video output not found")
        directory = (
            job.output_dir
            if output_index == 0
            else job.output_dir / f"output_{output_index}"
        )
        return FileResponse(
            directory / "video.mp4",
            media_type="video/mp4",
            filename=f"{identity}_{output_index}.mp4",
        )

    @app.post("/v1/videos/{identity}/cancel")
    async def cancel_video(identity: str):
        job = find_job(identity)
        if job.metadata["status"] == "in_progress":
            raise HTTPException(
                409, "Running generation cannot be interrupted; output will be retained"
            )
        if job.metadata["status"] == "queued":
            update_metadata(job.metadata, {"stage": "cancelled"}, time.time())
            job.metadata.update(status="cancelled", completed_at=time.time())
            job.done.set()
        # A late cancellation never deletes a completed result.
        return job.metadata

    @app.delete("/v1/videos/{identity}")
    async def delete_video(identity: str):
        job = find_job(identity)
        if job.metadata["status"] == "in_progress":
            raise HTTPException(
                409, "cannot delete a running video; wait for completion"
            )
        await remove_job(identity)
        return {"id": identity, "object": "video.deleted", "deleted": True}

    original_openapi = app.openapi

    def openapi():
        document = original_openapi()
        components = document.setdefault("components", {}).setdefault("schemas", {})
        schema = VideoRequest.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        components.update(schema.pop("$defs", {}))
        for kind in ("image", "video", "audio"):
            item = {
                "type": "object",
                "required": [kind + "_url"],
                "properties": {
                    kind + "_url": {
                        "type": "string",
                        "description": "HTTP(S) or data URL",
                    }
                },
            }
            schema["properties"][kind + "_reference"] = {
                "anyOf": [item, {"type": "array", "items": item}]
            }
        components["VideoRequest"] = schema
        form = {
            **schema,
            "title": "VideoForm",
            "properties": dict(schema["properties"]),
        }
        for field_name in (
            "extra_params",
            "lora",
            "image_reference",
            "video_reference",
            "audio_reference",
            "keyframe_indices",
            "reference_video_start_times",
        ):
            form["properties"][field_name] = {
                "type": "string",
                "description": "JSON-encoded value",
            }
        binary = {"type": "string", "format": "binary"}
        form["properties"].update(
            {
                "input_reference": binary,
                "input_references": {"type": "array", "items": binary},
            }
        )
        components["VideoForm"] = form
        return document

    app.openapi = openapi
    return app


def serve(config, *, host, port, output_dir):
    import uvicorn

    uvicorn.run(create_app(config, output_dir), host=host, port=port)
