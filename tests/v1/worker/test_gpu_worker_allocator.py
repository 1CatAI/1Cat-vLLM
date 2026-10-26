# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm.v1.worker import gpu_worker

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "legacy,unified,scoped,restored",
    [
        (None, None, "max_split_size_mb:20", ""),
        (
            None,
            "max_split_size_mb:512,garbage_collection_threshold:0.8",
            "max_split_size_mb:20,garbage_collection_threshold:0.8",
            "max_split_size_mb:512,garbage_collection_threshold:0.8",
        ),
        ("", "max_split_size_mb:512", "max_split_size_mb:20", ""),
        (
            "roundup_power2_divisions:[256:1,512:2,>:4], expandable_segments:True",
            "garbage_collection_threshold:0.6",
            (
                "roundup_power2_divisions:[256:1,512:2,>:4], "
                "expandable_segments:True,max_split_size_mb:20"
            ),
            "roundup_power2_divisions:[256:1,512:2,>:4], expandable_segments:True",
        ),
        (
            "garbage_collection_threshold:0.8, max_split_size_mb : 512",
            None,
            "garbage_collection_threshold:0.8,max_split_size_mb:20",
            "garbage_collection_threshold:0.8, max_split_size_mb : 512",
        ),
    ],
)
@pytest.mark.parametrize("fail", [False, True])
def test_load_model_preserves_allocator_config(
    monkeypatch, legacy, unified, scoped, restored, fail
):
    for name, value in [
        ("PYTORCH_CUDA_ALLOC_CONF", legacy),
        ("PYTORCH_ALLOC_CONF", unified),
    ]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: True)
    settings: list[str] = []
    monkeypatch.setattr(
        gpu_worker.torch._C, "_accelerator_setAllocatorSettings", settings.append
    )
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    try:
        with worker._scoped_allocator_max_split(20):
            assert settings == [scoped]
            if fail:
                raise RuntimeError("loading failed")
    except RuntimeError as error:
        assert fail and str(error) == "loading failed"
    assert settings == [scoped, restored]


@pytest.mark.parametrize("scope", [None, "1", "0"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize(
    "conf_name", ["PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "both"]
)
@pytest.mark.parametrize("expandable", [False, True], ids=["fixed", "expandable"])
def test_load_model_allocator_state(scope, fail, conf_name, expandable):
    """Observe the native allocator, isolated from the test process's settings."""
    if not gpu_worker.current_platform.is_cuda():
        pytest.skip("requires the CUDA caching allocator")

    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    for name in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
        "VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE",
    ):
        env.pop(name, None)
    config = (
        "backend:native,max_split_size_mb:512,"
        "roundup_power2_divisions:2,garbage_collection_threshold:0.8,"
        f"expandable_segments:{expandable}"
    )
    if conf_name == "both":
        env["PYTORCH_CUDA_ALLOC_CONF"] = config
        env["PYTORCH_ALLOC_CONF"] = f"expandable_segments:{not expandable}"
    else:
        env[conf_name] = config
    if scope is not None:
        env["VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE"] = scope

    # A fresh interpreter is required: allocator configuration is process-global
    # and restoring only the environment would leak runtime state to other tests.
    code = """
import os
import sys
from types import SimpleNamespace

import torch

if not torch.cuda.is_available():
    sys.exit(77)
if not hasattr(torch._C, "_accelerator_setAllocatorSettings"):
    sys.exit(77)

from vllm.v1.worker.gpu_worker import Worker

MiB = 1024 * 1024
fail = sys.argv[1] == "True"
scoped = sys.argv[2] == "True"

def allocator_settings():
    # memory_stats()["max_split_size"] is initialized with the device allocator
    # and does not track runtime setter changes. Snapshot metadata reads the
    # current native allocator configuration instead.
    return torch.cuda.memory._snapshot()["allocator_settings"]

def split_limit():
    return allocator_settings()["max_split_size"]

# Initialize the CUDA allocator before entering Worker.load_model.
initial = torch.empty(1, device="cuda")
original_settings = allocator_settings()
assert split_limit() == 512 * MiB

class AllocatingRunner:
    def load_model(self, *, load_dummy_weights=False):
        assert split_limit() == (20 if scoped else 512) * MiB
        settings = allocator_settings()
        assert settings["garbage_collection_threshold"] == 0.8
        assert (
            settings["roundup_power2_divisions"]
            == original_settings["roundup_power2_divisions"]
        )
        # Check a preserved option through allocation behavior, not setter calls:
        # two power-of-two divisions round 1100 bytes up to 1536, not 2048.
        before = torch.cuda.memory_allocated()
        self.weight = torch.empty(1100, dtype=torch.uint8, device="cuda")
        assert torch.cuda.memory_allocated() - before == 1536
        if fail:
            raise RuntimeError("loading failed")

worker = Worker.__new__(Worker)
worker.vllm_config = SimpleNamespace(
    model_config=SimpleNamespace(enable_cumem_allocator=False),
    weight_transfer_config=None,
)
worker._ple_offload_enabled = False
worker.model_runner = AllocatingRunner()
try:
    worker.load_model()
except RuntimeError as error:
    assert fail and str(error) == "loading failed"
else:
    assert not fail
assert split_limit() == 512 * MiB
assert allocator_settings() == original_settings

# The same helper used by SM70 warmup must still scope with the load opt-out.
with worker._scoped_allocator_max_split(20):
    assert split_limit() == 20 * MiB
assert split_limit() == 512 * MiB
assert allocator_settings() == original_settings
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(fail),
            str(scope == "1" or (scope is None and not expandable)),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 77:
        pytest.skip("requires CUDA and the runtime allocator-settings API")
    assert result.returncode == 0, result.stdout + result.stderr
