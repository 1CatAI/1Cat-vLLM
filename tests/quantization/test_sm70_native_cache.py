# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.warmup import sm70_native_cache as cache

pytestmark = pytest.mark.cpu_test


def _config(tune=None):
    kernel = KernelConfig()
    kernel.sm70_fp8.native.fp8_dense_tune_max_m = tune
    kernel.sm70_fp8.native.resolve("fp8")
    return SimpleNamespace(kernel_config=kernel)


@pytest.fixture
def native_tables(monkeypatch):
    imported = []
    exported: dict[tuple, bytes] = {}

    class Native:
        def __init__(self, values=()):
            self.values = values

        def sm70_gemm_export_cache(self, device, path):
            payload = exported[self.values]
            Path(path).write_bytes(payload)
            return 1

        def sm70_gemm_import_cache(self, device, path):
            imported.append((self.values, Path(path).read_bytes()))
            return 1

    monkeypatch.setattr(cache, "NativeBindings", Native)
    return exported, imported


def test_partition_roundtrip_preserves_legacy_file_and_only_imports_matching_policy(
    tmp_path, native_tables
):
    exported, imported = native_tables
    first = _config(8)
    second = _config(16)
    one = first.kernel_config.sm70_fp8.native.values
    two = second.kernel_config.sm70_fp8.native.values
    path = str(tmp_path / "lut.bin")
    exported.update({(): b"old-runtime-table", one: b"policy-eight"})
    with set_current_vllm_config(first):
        assert cache.export_cache(None, path) == 2
    assert Path(path).read_bytes() == b"old-runtime-table"
    assert cache.packed_cache_bytes(Path(path)).startswith(cache._MAGIC)
    with set_current_vllm_config(second):
        assert cache.import_cache(None, path) == 1
    assert imported == [((), b"old-runtime-table")]
    assert all(policy != two for policy, payload in imported)
    imported.clear()
    with set_current_vllm_config(first):
        assert cache.import_cache(None, path) == 2
    assert imported == [((), b"old-runtime-table"), (one, b"policy-eight")]


def test_untagged_legacy_seed_does_not_cross_typed_override(tmp_path, native_tables):
    exported, imported = native_tables
    path = str(tmp_path / "legacy.bin")
    Path(path).write_bytes(b"historic-table")
    with set_current_vllm_config(_config(8)):
        assert cache.import_cache(None, path) == 1
    assert imported == [((), b"historic-table")]
    imported.clear()
    legacy = _config()
    with set_current_vllm_config(legacy):
        assert cache.import_cache(None, path) == 2
    assert imported[-1] == (
        legacy.kernel_config.sm70_fp8.native.values,
        b"historic-table",
    )


def test_diagnostic_change_reuses_same_partition(tmp_path, native_tables, monkeypatch):
    exported, imported = native_tables
    first = _config(8)
    monkeypatch.setenv("TM_GEMM_TRACE", "1")
    second = _config(8)
    one, two = (cfg.kernel_config.sm70_fp8.native.values for cfg in (first, second))
    assert one != two
    exported.update({(): b"old", one: b"same-calculation"})
    path = str(tmp_path / "lut.bin")
    with set_current_vllm_config(first):
        cache.export_cache(None, path)
    # Exercise the distributed packed-byte import, without a sidecar filename.
    packed = str(tmp_path / "received.bin")
    Path(packed).write_bytes(cache.packed_cache_bytes(Path(path)))
    with set_current_vllm_config(second):
        cache.import_cache(None, packed)
    assert imported[-1] == (two, b"same-calculation")
