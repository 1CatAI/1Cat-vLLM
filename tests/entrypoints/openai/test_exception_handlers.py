# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rejected requests are answered without re-raising into the ASGI server."""

from argparse import Namespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.server_utils import register_exception_handlers
from vllm.exceptions import VLLMNotFoundError, VLLMValidationError


def _client() -> TestClient:
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False)
    register_exception_handlers(app)

    @app.get("/raise/{kind}")
    async def fail(kind: str):
        if kind == "validation":
            raise VLLMValidationError("prompt is longer than the maximum length")
        if kind == "not_found":
            raise VLLMNotFoundError("unknown model")
        raise RuntimeError("server bug")

    # A re-raised exception surfaces here exactly as uvicorn would log it.
    return TestClient(app, raise_server_exceptions=True)


@pytest.mark.parametrize(("kind", "status"), [("validation", 400), ("not_found", 404)])
def test_client_errors_are_answered_without_a_traceback(kind, status):
    response = _client().get(f"/raise/{kind}")
    assert response.status_code == status
    assert response.json()["error"]["code"] == status


def test_server_errors_still_reach_the_server_log():
    with pytest.raises(RuntimeError, match="server bug"):
        _client().get("/raise/bug")
