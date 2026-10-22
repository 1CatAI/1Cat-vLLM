# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NCCL PP lifecycle regressions with bounded process-tree cleanup.

Default: two GPUs, D=2. Set VLLM_TEST_PP_NCCL_WORLD_SIZE=7 to also exercise D=4.
The worker uses small tensors and does not download/load a language model.
"""

import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest
import torch


def _run_process_group(
    command: list[str],
    *,
    timeout: float,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> int:
    """Bound the entire torchrun tree, including children stuck in NCCL."""
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        finally:
            # Also clean up on assertion failure/KeyboardInterrupt and after a
            # launcher exits while leaving workers behind. The group is unique
            # because this launcher starts a new session.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


@pytest.mark.cpu_test
@pytest.mark.skipif(sys.platform != "linux", reason="POSIX process-group cleanup")
def test_pp_nccl_launcher_enforces_timeout(tmp_path: Path) -> None:
    # Exercise the same timeout path without requiring CUDA or waiting for a
    # deliberate GPU hang. The detached descendant must not write its marker.
    marker = tmp_path / "orphan-survived"
    started = tmp_path / "descendant-started"
    child = (
        "import pathlib,time; "
        f"pathlib.Path({str(started)!r}).write_text('started'); "
        "time.sleep(2); "
        f"pathlib.Path({str(marker)!r}).write_text('orphan')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _run_process_group(
            [sys.executable, "-c", parent],
            timeout=1,
            log_path=tmp_path / "timeout.log",
        )
    # Wait in a separate process so this assertion does not depend on a sleep
    # mock. The grace period is longer than the descendant's marker delay.
    assert started.exists(), "descendant never started; timeout test was not exercised"
    subprocess.run([sys.executable, "-c", "import time; time.sleep(2.1)"], check=True)
    assert not marker.exists(), "timeout left a descendant process alive"


@pytest.mark.distributed
@pytest.mark.skipif(sys.platform != "linux", reason="NCCL and POSIX process cleanup")
def test_pp_nccl_lifecycle(tmp_path: Path) -> None:
    world_size = int(os.environ.get("VLLM_TEST_PP_NCCL_WORLD_SIZE", "2"))
    if world_size < 2:
        pytest.fail("VLLM_TEST_PP_NCCL_WORLD_SIZE must be at least two")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"requires {world_size} visible CUDA GPUs")
    if not torch.distributed.is_nccl_available():
        pytest.skip("requires NCCL")
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native"
    env.pop("PYTORCH_ALLOC_CONF", None)
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    # A torchrun parent must start a fresh rendezvous, even when pytest itself
    # was started by a distributed harness.
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        env.pop(name, None)
    worker = Path(__file__).with_name("pp_nccl_lifecycle_worker.py")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        "--max-restarts=0",
        str(worker),
    ]
    log_path = tmp_path / "pp-nccl-lifecycle.log"
    timeout = float(os.environ.get("VLLM_TEST_PP_NCCL_TIMEOUT", "240"))
    try:
        returncode = _run_process_group(
            command, timeout=timeout, log_path=log_path, env=env
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"PP NCCL timed out after {timeout}s:\n{log_path.read_text()}")
    output = log_path.read_text()
    assert returncode == 0, output
    prefix = "PP_NCCL_LIFECYCLE_OK "
    reports = [
        json.loads(line.split(prefix, 1)[1])
        for line in output.splitlines()
        if prefix in line
    ]
    assert {report["rank"] for report in reports} == set(range(world_size)), output
    expected_cases = 6 if world_size >= 4 else 3
    assert all(len(report["cases"]) == expected_cases for report in reports), output
    # Keep the per-rank counters visible under pytest -s and in captured logs.
    print("\n".join(json.dumps(report) for report in reports))
