"""Engine lifecycle: launch the per-model shell script, wait for ready, teardown.

The launch scripts (~/launch_<model>.sh) are user-local files outside this repo;
the fleet runner treats them as a contract. See tests/fleet/README.md.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

READY_MARKER = "Application startup complete"
# Only Traceback and explicit CUDA aborts are fatal — vLLM's logger uses
# "ERROR" liberally for transient/recovered conditions (HF-hub fallback misses,
# retries, etc.) that don't actually fail startup. The 'died unexpectedly'
# message means the executor is shutting down because a worker process exited
# without warning (e.g. silent OOM during torch.compile under disk pressure).
FAIL_MARKERS = (
    "Traceback",
    "CUDA error",
    "Aborted (core dumped)",
    "died unexpectedly, shutting down executor",
)
KV_CACHE_LINE = "GPU KV cache size:"

VLLM_PROC_PATTERN = "vllm serve|EngineCore|Worker_TP"


class EngineLaunchError(RuntimeError):
    pass


def _gpu_used_mib() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [int(line.strip()) for line in out.splitlines() if line.strip()]


def assert_gpus_clean(threshold_mib: int = 200) -> None:
    """Refuse to launch if any GPU is holding non-trivial memory.

    Threshold is generous (200 MiB) — driver/CUDA context can sit at ~10s of MiB
    even with no workload. We're checking for forgotten engine processes, not
    library overhead.
    """
    used = _gpu_used_mib()
    busy = [(i, m) for i, m in enumerate(used) if m > threshold_mib]
    if busy:
        raise EngineLaunchError(
            f"GPUs not clean before launch (>{threshold_mib} MiB used): "
            + ", ".join(f"GPU{i}={m}MiB" for i, m in busy)
            + "\nLikely orphan vLLM workers — run: pkill -9 -f 'vllm serve|EngineCore|Worker_TP'"
        )


def kill_running_engines() -> None:
    subprocess.run(["pkill", "-9", "-f", VLLM_PROC_PATTERN], check=False)
    # pkill returns 1 when nothing matched — that's fine.
    deadline = time.time() + 30
    while time.time() < deadline:
        r = subprocess.run(
            ["pgrep", "-f", VLLM_PROC_PATTERN], capture_output=True, text=True
        )
        if r.returncode != 0:  # no matches
            time.sleep(2)  # let CUDA contexts drain
            return
        time.sleep(1)
    raise EngineLaunchError("vLLM processes still running 30s after SIGKILL")


def launch(launch_script: str, log_path: str) -> subprocess.Popen:
    """Start the launch script in the background, redirecting all output to log_path."""
    script = os.path.expanduser(launch_script)
    if not Path(script).is_file():
        raise EngineLaunchError(f"Launch script not found: {script}")
    log = open(log_path, "wb", buffering=0)
    # start_new_session=True puts the launcher in its own process group so we
    # can clean up any descendants the script forks (EngineCore, workers).
    return subprocess.Popen(
        ["bash", script],
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_for_ready(log_path: str, timeout_s: int) -> None:
    """Tail the launch log until 'Application startup complete' appears or we hit failure."""
    deadline = time.time() + timeout_s
    seen = 0
    while time.time() < deadline:
        try:
            text = Path(log_path).read_text(errors="ignore")
        except FileNotFoundError:
            time.sleep(0.5)
            continue
        if READY_MARKER in text[seen:]:
            return
        for marker in FAIL_MARKERS:
            if marker in text[seen:]:
                tail = "\n".join(text.splitlines()[-30:])
                raise EngineLaunchError(
                    f"Launch failed (marker={marker!r}):\n{tail}"
                )
        seen = len(text)
        time.sleep(2)
    tail = "\n".join(Path(log_path).read_text(errors="ignore").splitlines()[-30:])
    raise EngineLaunchError(
        f"Engine did not become ready in {timeout_s}s. Tail:\n{tail}"
    )


def kv_cache_size_tokens(log_path: str) -> int | None:
    """Extract 'GPU KV cache size: N tokens' from the launch log, if present."""
    try:
        text = Path(log_path).read_text(errors="ignore")
    except FileNotFoundError:
        return None
    for line in text.splitlines():
        if KV_CACHE_LINE in line:
            # ... "GPU KV cache size: 339,472 tokens"
            try:
                after = line.split(KV_CACHE_LINE, 1)[1].strip()
                num = after.split("tokens", 1)[0].strip().replace(",", "")
                return int(num)
            except (IndexError, ValueError):
                return None
    return None


@contextmanager
def running_engine(launch_script: str, log_path: str, ready_timeout_s: int):
    """Launch + wait_for_ready + (on exit) kill. Yields nothing — callers hit HTTP."""
    assert_gpus_clean()
    Path(log_path).write_bytes(b"")  # truncate
    proc = launch(launch_script, log_path)
    try:
        wait_for_ready(log_path, ready_timeout_s)
        yield
    finally:
        # SIGTERM the launcher's process group, then escalate.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        # Belt-and-braces: kill any stragglers (workers can outlive the launcher).
        kill_running_engines()
