# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import errno
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest
import regex as re
import torch
from torch import nn

from vllm.model_executor.models.minimax_h3.config import H3Config, H3InputError
from vllm.model_executor.models.minimax_h3.residency import (
    MMapHostWeights,
    PinnedModuleStager,
)


def aliased_module():
    module = nn.Module()
    values = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    module.weight = nn.Parameter(values)
    module.register_buffer("view", values[3:7, 1:5])
    module.register_buffer("raw", values.view(torch.uint8))
    module.register_buffer("empty", torch.empty((0, 3)))
    return module


def test_mapped_weights_keep_bytes_views_and_restore(tmp_path):
    module = aliased_module()
    expected = {name: value.clone() for name, value in module.state_dict().items()}
    backing = MMapHostWeights(tmp_path)
    PinnedModuleStager.map_cpu_weights(module, backing)
    filename = module.weight.untyped_storage().filename
    assert filename and not Path(filename).exists()
    assert not list(tmp_path.iterdir())
    assert not module.weight.is_pinned()
    assert module.view.stride() == (8, 1) and module.view.storage_offset() == 25
    assert (
        module.weight.untyped_storage().data_ptr()
        == module.raw.untyped_storage().data_ptr()
    )
    reserved = backing.bytes_reserved
    stager = object.__new__(PinnedModuleStager)
    stager._groups = stager._snapshot_groups(
        (module,), pin_memory=True, host_backing=backing
    )
    assert backing.bytes_reserved == reserved  # Existing mappings are reused.
    module.weight.data = torch.zeros_like(module.weight)
    stager._restore_masters()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_map_before_fused_loading_preserves_buffers(tmp_path):
    module = nn.Module()
    module.weight = nn.Parameter(torch.empty(12, 8, dtype=torch.float16))
    module.register_buffer("scale", torch.tensor([0.125, 3.0], dtype=torch.float32))
    PinnedModuleStager.map_cpu_weights(
        module, MMapHostWeights(tmp_path), preserve_parameters=False
    )
    with torch.no_grad():
        for part in range(3):
            module.weight[part * 4 : (part + 1) * 4].copy_(torch.full((4, 8), part + 1))
    torch.testing.assert_close(module.scale, torch.tensor([0.125, 3.0]), rtol=0, atol=0)
    expected = (
        torch.arange(1, 4, dtype=torch.float16)
        .repeat_interleave(4)[:, None]
        .expand(12, 8)
    )
    torch.testing.assert_close(module.weight, expected, rtol=0, atol=0)


def test_disk_reservation_failure_preserves_original_storage(monkeypatch, tmp_path):
    module = nn.Linear(8, 4)
    before = module.weight.detach().clone()

    def full_disk(*args):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(os, "posix_fallocate", full_disk)
    with pytest.raises(RuntimeError, match="reserve disk storage"):
        PinnedModuleStager.map_cpu_weights(module, MMapHostWeights(tmp_path))
    torch.testing.assert_close(module.weight, before, rtol=0, atol=0)
    assert not list(tmp_path.iterdir())


def test_invalid_host_memory_policy_is_rejected():
    with pytest.raises(H3InputError, match="host memory mode"):
        H3Config(host_memory_mode="invalid")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an explicitly selected idle GPU"
)
def test_disk_backed_gpu_roundtrip_keeps_storage_aliases(tmp_path):
    module = aliased_module()
    expected = module.view.clone()
    stager = PinnedModuleStager(
        module, torch.device("cuda"), host_backing=MMapHostWeights(tmp_path)
    )
    for _ in range(3):
        stager.load()
        assert module.weight.is_cuda
        assert (
            module.weight.untyped_storage().data_ptr()
            == module.raw.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(module.view.cpu(), expected, rtol=0, atol=0)
        stager.offload()
        assert module.weight.device.type == "cpu" and not module.weight.is_pinned()
        assert module.weight.untyped_storage().filename
        torch.testing.assert_close(module.view, expected, rtol=0, atol=0)


@pytest.mark.parametrize("extensions", [True, False])
def test_source_wheel_reuse_keeps_h3_native_operators(
    tmp_path, monkeypatch, extensions
):
    # Execute only the extraction method, avoiding setup() and build/network work.
    source = Path(__file__).resolve().parents[2] / "setup.py"
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "extract_precompiled_and_patch_package"
    )
    function.decorator_list = []
    scope: dict[str, Any] = {"os": os, "re": re, "shutil": shutil, "Path": Path}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        scope,
    )
    wheel = tmp_path / "native.whl"
    names = [
        f"_h3_{kind}_C.cpython-312-x86_64-linux-gnu.so"
        for kind in ("w8a16", "flashinfer", "flashattn")
    ]
    names.append("_sm70_exact_reduce_C.cpython-312-x86_64-linux-gnu.so")
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in names:
            archive.writestr("vllm/" + name, b"native-test-bytes")
    monkeypatch.chdir(tmp_path)
    result = scope["extract_precompiled_and_patch_package"](
        str(wheel), None, extract_extensions=extensions, extract_rust_frontend=False
    )
    assert result == ({"vllm": names} if extensions else {})
    for name in names:
        path = tmp_path / "vllm" / name
        assert path.exists() == extensions
        if extensions:
            assert path.read_bytes() == b"native-test-bytes"
