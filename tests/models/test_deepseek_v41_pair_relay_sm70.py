# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_70 compile check of the V4.1 TP4 prefill pair relay (CPU, no GPU).

In a clean process (no TRITON_INTERPRET, no visible GPU), the row-split
activation kernel compiles for sm_70 with eight FP16 stores per element pair
when the relay is off and six when it skips one destination, on every rank;
the peer pull compiles to a volatile load and a store.

    python tests/models/test_deepseek_v41_pair_relay_sm70.py   # or pytest
"""

import os
import subprocess
import sys

import pytest

CHECK = r"""
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton
from vllm.model_executor.layers.quantization.gguf_shared_prefill import (
    _v41_prefill_activation_pair_pull,
    _v41_prefill_activation_rows_p,
)

width = 576
target = GPUTarget("cuda", 70, 32)
common = dict(EXPERTS=32, WIDTH=width, FULL=4 * width, BLOCK=256, PROGRAMS=1024)
signature = {
    "projection": "*fp16", "probabilities": "*fp32", "order": "*i64",
    "out0": "u64", "out1": "u64", "out2": "u64", "out3": "u64", "bounds": "*i32",
}
for rank in range(4):
    for skip, stores in ((-1, 8), (rank ^ 3, 6)):
        source = ASTSource(
            _v41_prefill_activation_rows_p,
            signature=signature,
            constexprs=common | dict(COLUMN=rank * width, SKIP_DEST=skip),
        )
        options = {"enable_fp_fusion": False}
        ptx = triton.compile(source, target=target, options=options).asm["ptx"]
        assert ".target sm_70" in ptx
        assert ptx.count("st.global.b16") == stores, (rank, skip)
    source = ASTSource(
        _v41_prefill_activation_pair_pull,
        signature={"source": "u64", "destination": "u64", "bounds": "*i32"},
        constexprs=common | dict(COLUMN=(rank ^ 3) * width),
    )
    ptx = triton.compile(source, target=target).asm["ptx"]
    assert "ld.volatile.global.b16" in ptx and "st.global.b16" in ptx
print("ok")
"""


def test_pair_relay_sm70_compile():
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    run = subprocess.run(
        [sys.executable, "-c", CHECK],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    assert run.stdout.strip().endswith("ok")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
