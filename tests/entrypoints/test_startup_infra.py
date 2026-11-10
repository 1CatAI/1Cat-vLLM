# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

import pytest
from fastapi.testclient import TestClient

from vllm import envs
from vllm.entrypoints.openai.api_server import build_app
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.logger import _StartupLogHandler
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture(autouse=True)
def cleanup(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_STARTUP_LOG_FILE", raising=False)
    yield
    if _StartupLogHandler._installed is not None:
        _StartupLogHandler._installed.close()
    envs.disable_envs_cache()


def test_startup_log_is_opt_in():
    _StartupLogHandler.install()
    assert _StartupLogHandler._installed is None


def test_startup_log_appends_and_installs_once(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "startup.log"
    path.write_text("previous start\n")
    monkeypatch.setenv("VLLM_STARTUP_LOG_FILE", "startup.log")
    _StartupLogHandler.install()
    handler = _StartupLogHandler._installed
    _StartupLogHandler.install()
    assert _StartupLogHandler._installed is handler
    for name in ("vllm", "uvicorn.error"):
        assert logging.getLogger(name).handlers.count(handler) == 1
    handler.handle(logging.makeLogRecord({"msg": "new start", "name": "vllm"}))
    assert path.read_text() == "previous start\nnew start\n"


def test_uvicorn_readiness_detaches_both_loggers(monkeypatch, tmp_path):
    path = tmp_path / "startup.log"
    monkeypatch.setenv("VLLM_STARTUP_LOG_FILE", str(path))
    _StartupLogHandler.install()
    handler = _StartupLogHandler._installed
    handler.handle(
        logging.makeLogRecord({"msg": "Application startup complete.", "name": "vllm"})
    )
    assert _StartupLogHandler._installed is handler
    handler.handle(
        logging.makeLogRecord(
            {"msg": "Application startup complete.", "name": "uvicorn.error"}
        )
    )
    assert _StartupLogHandler._installed is None
    assert handler.stream is None
    assert all(handler not in logging.getLogger(n).handlers for n in handler._loggers)


def test_unwritable_log_does_not_abort_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_STARTUP_LOG_FILE", str(tmp_path))
    _StartupLogHandler.install()
    assert _StartupLogHandler._installed is None


def test_head_probe_does_not_mask_missing_routes():
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args([])
    app = build_app(args, supported_tasks=("generate",))
    client = TestClient(app)
    assert client.head("/").status_code == 200
    assert client.head("/this-route-does-not-exist").status_code == 404
