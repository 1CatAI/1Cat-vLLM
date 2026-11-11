# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_70 compile check of the V4.1 attention PREP / POST fusions (CPU, no GPU).

Runs benchmarks/deepseek_v41/small_fuse_sm70_check.py in a clean process
(no TRITON_INTERPRET, no visible GPU): the fused kernels compile for sm_70,
their FP32 add-reductions use the replaced kernels' layouts and ptxas reports
no spills.

    python tests/models/test_deepseek_v41_small_fuse_sm70.py   # or pytest
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "benchmarks" / "deepseek_v41" / "small_fuse_sm70_check.py"


def test_small_fuse_sm70_compile():
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    run = subprocess.run(
        [sys.executable, str(CHECK)],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    if run.returncode == 2:
        pytest.skip(run.stdout.strip())
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    report = json.loads(run.stdout)
    assert report["ok"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
