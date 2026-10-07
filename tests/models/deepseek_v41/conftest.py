# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures for the DeepSeek-V4.1 port tests (owner: L-CORE, PORT_DESIGN §2.2).

Every lane's ``test_<lane>_*.py`` may use these. Weight-reading tests MUST go through
``verified_shard`` (LANE_RULES rule 6): a shard is read only after verify_shard.sh prints OK.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

CHECKPOINT_DIR = Path(os.environ.get("DS41_CHECKPOINT_DIR", "/mnt/nvme2/models/DeepSeek-V4.1-Flash"))
VERIFY_SHARD = Path(os.environ.get("DS41_VERIFY_SHARD", "/mnt/nvme2/models/_dl-logs/verify_shard.sh"))
PACKAGE_DIR = Path(__file__).resolve().parents[3] / "vllm" / "models" / "deepseek_v41"


def _sm70_available() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    return all(torch.cuda.get_device_capability(i)[0] == 7 for i in range(torch.cuda.device_count()))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "sm70: needs visible CUDA devices that are all compute capability 7.x")
    config.addinivalue_line("markers", "weights: reads official DeepSeek-V4.1 checkpoint shards (verified only)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    sm70 = None
    for item in items:
        if item.get_closest_marker("sm70") is None:
            continue
        if sm70 is None:
            sm70 = _sm70_available()
        if not sm70:
            item.add_marker(pytest.mark.skip(reason="needs SM70 (V100) CUDA devices"))


@pytest.fixture(scope="session")
def ds41_package_dir() -> Path:
    """Absolute path of ``vllm/models/deepseek_v41`` in the tree under test."""
    assert PACKAGE_DIR.is_dir(), f"package dir missing: {PACKAGE_DIR}"
    return PACKAGE_DIR


@pytest.fixture(scope="session")
def ds41_checkpoint_dir() -> Path:
    if not (CHECKPOINT_DIR / "config.json").is_file():
        pytest.skip(f"DeepSeek-V4.1 checkpoint config not found under {CHECKPOINT_DIR}")
    return CHECKPOINT_DIR


@pytest.fixture(scope="session")
def ds41_config_json(ds41_checkpoint_dir: Path) -> dict[str, Any]:
    """The raw official config.json (nested ``text_config``)."""
    with open(ds41_checkpoint_dir / "config.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def ds41_text_config(ds41_config_json: dict[str, Any]) -> SimpleNamespace:
    """``text_config`` flattened over the top level, as an attribute namespace (stand-in for hf_config in
    pure-python tests; the real flattening lives in vllm/transformers_utils/configs/deepseek_v41.py)."""
    merged = {k: v for k, v in ds41_config_json.items() if k != "text_config"}
    merged.update(ds41_config_json.get("text_config") or {})
    return SimpleNamespace(**merged)


@pytest.fixture(scope="session")
def verified_shard(ds41_checkpoint_dir: Path):
    """``verified_shard("model-00003-of-00048.safetensors") -> Path``; fails loudly unless verify_shard.sh says OK."""
    def _verify(basename: str) -> Path:
        if not VERIFY_SHARD.is_file():
            pytest.fail(f"shard verifier missing: {VERIFY_SHARD}")
        proc = subprocess.run(["bash", str(VERIFY_SHARD), basename], capture_output=True, text=True, timeout=600)
        out = proc.stdout.strip()
        if proc.returncode != 0 or not out.startswith("OK"):
            pytest.fail(f"shard {basename} not verified (rc={proc.returncode}): {out} {proc.stderr.strip()}")
        path = ds41_checkpoint_dir / basename
        if not path.exists():
            alt = Path("/home/mermiges/ds41-engram") / basename
            if not alt.exists():
                pytest.fail(f"verified shard {basename} not found under {ds41_checkpoint_dir} or {alt.parent}")
            path = alt
        return path

    return _verify


@pytest.fixture
def ds41_dist_single():
    """World size 1 (TP 1, PP 1) on gloo so vLLM parallel layers can be built in-process on CPU or one GPU.
    Function-scoped: the root conftest's autouse ``cleanup_fixture`` tears the groups down after every test."""
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import model_parallel_is_initialized

    if not model_parallel_is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                         distributed_init_method=f"tcp://127.0.0.1:{port}")
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    yield
