# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise import order in a clean interpreter, before model registration."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "statement",
    [
        "from vllm._aiter_ops import rocm_aiter_ops",
        (
            "from vllm.v1.attention.backends.mla.indexer "
            "import DeepseekV32IndexerMetadata"
        ),
        (
            "from vllm.model_executor.layers.sparse_attn_indexer_kpool "
            "import SparseAttnIndexerKpool, _get_kpool_ops; "
            "assert callable(_get_kpool_ops().kpool_compress_and_write_cache)"
        ),
        # This is the import chain used by CLI EngineArgs.create_engine_config
        # before any worker/model has been constructed.
        (
            "from vllm.engine.arg_utils import EngineArgs; "
            "from vllm.model_executor.model_loader.weight_utils import get_quant_config"
        ),
    ],
)
def test_sparse_provider_import_before_model_registration(statement):
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", statement],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root)},
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
