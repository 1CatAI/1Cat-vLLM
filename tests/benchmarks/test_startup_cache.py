# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

SERVER = """
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        drift = sys.argv[2] == 'drift'
        cached = os.environ['VLLM_DISABLE_COMPILE_CACHE'] == '0'
        answer = 'changed' if drift and cached else 'same'
        choice = {'message': {'content': answer}, 'finish_reason': 'stop'}
        choice['message']['tool_calls'] = [{'id': str(os.getpid()),
            'type': 'function', 'function': {'name': 'test', 'arguments': '{}'}}]
        body = json.dumps({'choices': [choice]}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)
if os.environ['VLLM_DISABLE_COMPILE_CACHE'] == '0':
    print('Directly load AOT compilation', flush=True)
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""


@pytest.mark.parametrize("behavior", ["stable", "drift"])
def test_restart_matrix_records_evidence_and_rejects_output_drift(tmp_path, behavior):
    script = (
        Path(__file__).resolve().parents[2] / "benchmarks/benchmark_startup_cache.py"
    )
    server = tmp_path / "server.py"
    server.write_text(SERVER)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"model": "fixture", "temperature": 0}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    output = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output),
            "--port",
            str(port),
            "--request",
            str(request),
            "--timeout",
            "10",
            "--",
            sys.executable,
            str(server),
            str(port),
            behavior,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    records = json.loads((output / "results.json").read_text())
    assert records[0]["status"] == "passed"
    if behavior == "stable":
        assert result.returncode == 0, result.stderr
        assert [r["phase"] for r in records] == ["uncached", "cache-fill", "cache-hit"]
        assert records[-1]["aot_load_messages"] == 1
        assert all(r["ready_seconds"] > 0 for r in records)
    else:
        assert result.returncode != 0
        assert len(records) == 2
        assert records[-1]["error"] == "Output drift between startup phases"
    # Even a failed quality gate must stop the subprocess it owns.
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0
