# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure fresh server processes with compilation caching off, filled and reused.

Run on reserved GPUs. The command must serve on the supplied loopback port.
Request files contain OpenAI chat-completion JSON, including a model name and
explicit sampling settings. Use deterministic requests for the equality gate;
run the model's normal sampling and quality checks separately before promotion.
"""

import argparse
import copy
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx


def response_fingerprint(choices):
    stable = copy.deepcopy(choices)
    for choice in stable:
        for call in choice.get("message", {}).get("tool_calls") or []:
            # Tool call IDs identify a request, not the model's generated answer.
            call.pop("id", None)
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def wait_ready(process, client, timeout):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}")
        try:
            if client.get("/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError("Server readiness deadline exceeded")


def stop_process(process):
    # Only the process group created by this benchmark is terminated.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--request", type=Path, action="append", required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply the server command after --")
    if args.output_dir.exists():
        parser.error("Use a new output directory; previous evidence is never removed")
    requests = [json.loads(path.read_text()) for path in args.request]
    if any(body.get("temperature") != 0 or body.get("stream") for body in requests):
        parser.error("Equality checks require non-streaming temperature=0 requests")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    args.output_dir.mkdir(parents=True)
    env = dict(os.environ)
    for variable, folder in (
        ("VLLM_CACHE_ROOT", "vllm"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TORCH_EXTENSIONS_DIR", "extensions"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        path = args.output_dir.resolve() / "cache" / folder
        path.mkdir(parents=True)
        env[variable] = str(path)
    results = []
    destination = args.output_dir / "results.json"
    phases = (("uncached", "1"), ("cache-fill", "0"), ("cache-hit", "0"))
    for phase, disabled in phases:
        env["VLLM_DISABLE_COMPILE_CACHE"] = disabled
        logfile = args.output_dir / f"{phase}.log"
        row = {"phase": phase, "started_at": time.time(), "requests": []}
        results.append(row)
        destination.write_text(json.dumps(results, indent=2) + "\n")
        with logfile.open("w") as output:
            start = time.monotonic()
            process = subprocess.Popen(
                command,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                with httpx.Client(
                    base_url=f"http://127.0.0.1:{args.port}",
                    timeout=args.timeout,
                    trust_env=False,
                ) as client:
                    wait_ready(process, client, args.timeout)
                    row["ready_seconds"] = time.monotonic() - start
                    print(f"{phase}: ready in {row['ready_seconds']:.2f}s", flush=True)
                    for body in requests:
                        response = client.post("/v1/chat/completions", json=body)
                        response.raise_for_status()
                        data = response.json()
                        choices = data["choices"]
                        if any(c["finish_reason"] == "length" for c in choices):
                            raise RuntimeError("Quality request was truncated")
                        row["requests"].append(
                            {
                                "sha256": response_fingerprint(choices),
                                "response": data,
                            }
                        )
                    if len(results) > 1:
                        expected = [r["sha256"] for r in results[0]["requests"]]
                        actual = [r["sha256"] for r in row["requests"]]
                        if actual != expected:
                            raise RuntimeError("Output drift between startup phases")
                    log = logfile.read_text(errors="replace")
                    row["aot_load_messages"] = log.count("Directly load AOT")
                    row["aot_load_failures"] = log.count(
                        "Compiling model again due to a load failure"
                    )
                    if phase == "cache-hit" and (
                        not row["aot_load_messages"] or row["aot_load_failures"]
                    ):
                        raise RuntimeError(
                            "Warm launch did not fully reuse the AOT cache"
                        )
                    row["status"] = "passed"
            except BaseException as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                raise
            finally:
                row["elapsed_seconds"] = time.monotonic() - start
                stop_process(process)
                destination.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
