# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
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


@pytest.mark.parametrize("mode", ["generate", "serve"])
def test_pageable_weight_masters_are_an_explicit_deployment_option(mode):
    from vllm.entrypoints.cli.video import VideoSubcommand

    parser = argparse.ArgumentParser()
    VideoSubcommand().subparser_init(parser.add_subparsers())
    assert parser.parse_args(["video", mode]).host_weight_pin_memory
    assert not parser.parse_args(
        ["video", mode, "--disable-host-weight-pinning"]
    ).host_weight_pin_memory
    assert not H3Config(host_weight_pin_memory=False).host_weight_pin_memory
    with pytest.raises(H3InputError, match="boolean"):
        H3Config(host_weight_pin_memory="false")


@pytest.mark.parametrize("pin_memory", [True, False])
@pytest.mark.parametrize("kind", ["MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE"])
def test_both_vae_stagers_honor_the_host_policy(monkeypatch, pin_memory, kind):
    from vllm.model_executor.models.minimax_h3 import vae

    remote = nn.Module()
    remote.model = nn.Linear(2, 2)
    expected = remote.model.weight.clone()
    calls = []
    monkeypatch.setattr(
        vae, "_load_component_config", lambda path: {"sample_rate": 44100}
    )
    monkeypatch.setattr(vae, "_load_remote_component", lambda *args: remote)
    monkeypatch.setattr(
        vae,
        "PinnedModuleStager",
        lambda module, device, **kwargs: calls.append((module, device, kwargs)),
    )
    wrapper = getattr(vae, kind)(
        "test",
        device=torch.device("cuda"),
        load_device=torch.device("cpu"),
        pin_memory=pin_memory,
    )
    assert calls == [(remote, torch.device("cuda"), {"pin_memory": pin_memory})]
    assert wrapper.model.weight.dtype == torch.float32
    torch.testing.assert_close(wrapper.model.weight, expected, rtol=0, atol=0)


def _alias_module():
    raw = torch.arange(257, dtype=torch.float32)
    module = nn.Module()
    module.weight = nn.Parameter(raw[8:104].reshape(8, 12).t())
    module.register_buffer("strided", raw[31:67:2])
    module.register_buffer("byte_alias", raw.view(torch.uint8)[9:63])
    module.other = nn.Parameter(
        torch.arange(77, dtype=torch.float16).reshape(7, 11).t()
    )
    module.register_buffer("empty", torch.empty(0))
    return module


def _cpu_snapshot(module):
    from vllm.model_executor.models.minimax_h3.residency import PinnedModuleStager

    stager = PinnedModuleStager.__new__(PinnedModuleStager)
    stager._groups = stager._snapshot_groups((module,), pin_memory=False)
    stager.loaded = False
    return stager


def test_shared_snapshot_preserves_offsets_aliases_and_private_writes(tmp_path):
    left, right = _alias_module(), _alias_module()
    a, b = _cpu_snapshot(left), _cpu_snapshot(right)
    before = {name: value.clone() for name, value in left.state_dict().items()}
    directory = tmp_path / "snapshot"
    a._write_shared_groups(directory)
    a._read_shared_groups(directory)
    b._read_shared_groups(directory)
    for module in (left, right):
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        assert module.weight.stride() == (1, 12)
        assert module.strided.stride() == (2,)
        assert (
            module.weight.untyped_storage().data_ptr()
            == module.strided.untyped_storage().data_ptr()
        )
    with torch.no_grad():
        right.weight.add_(7)
    torch.testing.assert_close(left.weight, before["weight"], atol=0, rtol=0)
    # A private mapping must not modify a later reader or the shared snapshot.
    third = _alias_module()
    _cpu_snapshot(third)._read_shared_groups(directory)
    torch.testing.assert_close(third.weight, before["weight"], atol=0, rtol=0)
    a._restore_masters()
    for name, value in left.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("corrupt", ["layout", "weights", "file"])
def test_shared_snapshot_rejects_different_replicas(tmp_path, corrupt):
    left, right = _alias_module(), _alias_module()
    a = _cpu_snapshot(left)
    directory = tmp_path / "snapshot"
    a._write_shared_groups(directory)
    if corrupt == "layout":
        right.weight = nn.Parameter(right.weight.t())
    elif corrupt == "weights":
        with torch.no_grad():
            right.other.add_(1)
    else:
        data = directory / "weights.bin"
        with data.open("r+b") as f:
            f.seek(257)
            f.write(b"\xff")
    with pytest.raises(ValueError, match="shared component"):
        _cpu_snapshot(right)._read_shared_groups(directory)


def test_shared_vae_configuration_requires_pageable_storage():
    with pytest.raises(H3InputError, match="pageable"):
        H3Config(share_host_vae_weights=True)
    assert H3Config(
        share_host_vae_weights=True, host_weight_pin_memory=False
    ).share_host_vae_weights
    assert not H3Config().share_host_vae_weights


def test_engine_cleans_only_its_shared_directory(tmp_path):
    import tempfile

    from vllm.video.engine import H3Engine

    untouched = tmp_path / "other"
    untouched.mkdir()
    engine = H3Engine.__new__(H3Engine)
    engine._closed = False
    engine.workers = []
    engine.connections = []
    engine._gpu_lease = None
    engine._shared_weights = tempfile.TemporaryDirectory(dir=tmp_path, prefix="owned-")
    directory = engine._shared_weights.name
    engine.close()
    from pathlib import Path

    assert not Path(directory).exists()
    assert untouched.is_dir()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a leased GPU")
def test_gpu_shared_snapshot_roundtrips_all_storage_views(tmp_path):
    from vllm.model_executor.models.minimax_h3.residency import PinnedModuleStager

    model = _alias_module()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    stager = PinnedModuleStager(model, torch.device("cuda"), pin_memory=False)
    stager.share_cpu_storage(tmp_path / "shared")
    for _ in range(3):
        stager.load()
        for name, value in model.state_dict().items():
            assert value.device.type == "cuda"
            torch.testing.assert_close(value.cpu(), before[name], atol=0, rtol=0)
        stager.offload()
        for name, value in model.state_dict().items():
            assert value.device.type == "cpu"
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)


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
