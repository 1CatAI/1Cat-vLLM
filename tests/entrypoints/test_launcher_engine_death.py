# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An API server stopped by engine death must fail (non-zero exit), never stop
cleanly; a normal stop stays clean."""

import asyncio
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from vllm.entrypoints import launcher
from vllm.v1.engine.exceptions import EngineDeadError


class _Engine:
    def __init__(self, dead: bool) -> None:
        self.errored = dead
        self.is_running = not dead
        self.vllm_config = SimpleNamespace(shutdown_timeout=0)

    def shutdown(self, timeout=None) -> None:
        pass


def _app(engine: _Engine) -> FastAPI:
    app = FastAPI()
    app.state.engine_client = engine
    return app


async def _serve(app: FastAPI, stop_from_outside: bool):
    if stop_from_outside:

        async def stop() -> None:
            while getattr(app.state, "server", None) is None:
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.2)
            app.state.server.should_exit = True

        asyncio.get_running_loop().create_task(stop())
    return await launcher.serve_http(app, sock=None, host="127.0.0.1", port=0, log_level="warning")


def test_engine_death_shutdown_raises(caplog) -> None:
    async def main() -> None:
        shutdown = await _serve(_app(_Engine(dead=True)), stop_from_outside=False)   # watchdog stops it
        await shutdown

    with pytest.raises(EngineDeadError):
        asyncio.run(main())


def test_terminate_if_errored_marks_engine_death() -> None:
    server = SimpleNamespace(should_exit=False)
    launcher.terminate_if_errored(server, _Engine(dead=False))
    assert not server.should_exit and not getattr(server, launcher.ENGINE_DEAD_ATTR, False)
    launcher.terminate_if_errored(server, _Engine(dead=True))
    assert server.should_exit and getattr(server, launcher.ENGINE_DEAD_ATTR)


def test_clean_stop_stays_clean() -> None:
    async def main() -> None:
        shutdown = await _serve(_app(_Engine(dead=False)), stop_from_outside=True)
        await shutdown

    asyncio.run(main())


def test_process_exit_code_non_zero_on_engine_death() -> None:
    code = textwrap.dedent(
        """
        import asyncio, sys
        sys.path.insert(0, %r)
        from tests.entrypoints.test_launcher_engine_death import _Engine, _app, _serve

        async def main():
            await (await _serve(_app(_Engine(dead=True)), stop_from_outside=False))

        asyncio.run(main())
        """
    ) % (str(Path(__file__).resolve().parents[2]),)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0, proc.stderr[-2000:]
    out = proc.stdout + proc.stderr          # vLLM's logger writes to stdout
    assert "vLLM engine died" in out and "EngineDeadError" in proc.stderr
