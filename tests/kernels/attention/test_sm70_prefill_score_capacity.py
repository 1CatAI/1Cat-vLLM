# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score capacity is a startup setting: use a fresh process for each override."""

import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.accelerator.is_available()
    or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)

_SCRIPT = """
import json
import torch
from vllm.vllm_flash_attn import flash_attn_interface
q = torch.randn(1, 8192, 6, 256, device="cuda", dtype=torch.float16)
k = torch.randn(1, 16384, 1, 256, device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
out = torch.empty_like(q)
before = torch.accelerator.memory_allocated()
op = torch.ops._vllm_fa2_C.sm70_d256_gqa_architecture_q8192_fwd
try:
    op(q, k, v, out, 0.0625, True)
except RuntimeError as error:
    print(json.dumps({"error": str(error)}))
else:
    allocated = torch.accelerator.memory_allocated() - before
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(q, k, v, out, 0.0625, True)
    graph.replay()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    print(json.dumps({"allocated": allocated, "finite": bool(out.isfinite().all())}))
"""


@pytest.mark.parametrize("block", [None, 8192, 16384, 24576, 0, 8193, "invalid"])
def test_score_capacity_and_graph_replay(block):
    env = dict(os.environ)
    if block is None:
        env.pop("VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS", None)
    else:
        env["VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS"] = str(block)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    record = json.loads(result.stdout.strip().splitlines()[-1])
    if block not in (None, 8192, 16384, 24576):
        assert "VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS" in record["error"]
        return
    assert record["finite"]
    score_bytes = (8192 if block is None else block) * 8192 * 6 * 2
    # Remaining fixed scratch is ~408 MiB. Allow CUDA/cuBLAS initialization
    # overhead without allowing an ignored small-capacity override to pass.
    assert score_bytes <= record["allocated"] < score_bytes + 512 * 1024**2
