# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys

import pytest

from vllm.env_override import _default_nccl_graph_register

_EXPANDABLE = "expandable_segments:True"


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"PYTORCH_CUDA_ALLOC_CONF": _EXPANDABLE}, "0"),
        ({"PYTORCH_ALLOC_CONF": _EXPANDABLE}, "0"),
        # Other options, including bracketed lists, around a spaced token.
        (
            {
                "PYTORCH_CUDA_ALLOC_CONF": (
                    "max_split_size_mb:512,"
                    "roundup_power2_divisions:[256:1,512:2],"
                    " expandable_segments : True"
                )
            },
            "0",
        ),
        # PyTorch reads the CUDA-specific variable whenever it is present.
        ({"PYTORCH_CUDA_ALLOC_CONF": "", "PYTORCH_ALLOC_CONF": _EXPANDABLE}, None),
        ({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}, None),
        ({}, None),
        # An explicit choice is kept either way.
        ({"PYTORCH_CUDA_ALLOC_CONF": _EXPANDABLE, "NCCL_GRAPH_REGISTER": "1"}, "1"),
    ],
)
def test_nccl_graph_register_default(environ, expected):
    environ = dict(environ)
    _default_nccl_graph_register(environ)
    assert environ.get("NCCL_GRAPH_REGISTER") == expected


def test_import_applies_the_default():
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "NCCL_GRAPH_REGISTER")
    }
    env["PYTORCH_CUDA_ALLOC_CONF"] = _EXPANDABLE
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, vllm; print(os.environ.get('NCCL_GRAPH_REGISTER'))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "0"
