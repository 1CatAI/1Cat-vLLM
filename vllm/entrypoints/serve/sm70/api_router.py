# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/v1/sm70/acceleration")
async def acceleration(request: Request):
    config = getattr(request.app.state, "vllm_config", None)
    report = getattr(config, "sm70_acceleration_report", {})
    return JSONResponse(report)


def attach_router(app):
    app.include_router(router)
