# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib

import vllm
from vllm.video.benchmark import source_provenance


def test_provenance_tracks_shared_operators_outside_model_directory(
    tmp_path, monkeypatch
):
    package = tmp_path / "vllm"
    layers = package / "model_executor/layers"
    layers.mkdir(parents=True)
    monkeypatch.setattr(vllm, "__file__", str(package / "__init__.py"))
    contents = {
        "sm70_diffusion.py": b"gemm revision 1",
        "sm70_attention.py": b"dense attention revision 1",
        "sm70_sparse_attention.py": b"sparse attention revision 1",
    }
    for name, content in contents.items():
        (layers / name).write_bytes(content)
    before = source_provenance()["sources_sha256"]
    for name, content in contents.items():
        assert (
            before[f"model_executor/layers/{name}"]
            == hashlib.sha256(content).hexdigest()
        )

    changed = "model_executor/layers/sm70_attention.py"
    (package / changed).write_bytes(b"dense attention revision 2")
    after = source_provenance()["sources_sha256"]
    assert after[changed] != before[changed]
    assert {k: v for k, v in before.items() if k != changed} == {
        k: v for k, v in after.items() if k != changed
    }


def test_provenance_tracks_loaded_generic_sm70_binary(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from vllm.video.metrics import loaded_kernel_provenance

    binary = tmp_path / "exact_reduce.so"
    binary.write_bytes(b"generic collective binary")
    monkeypatch.setitem(
        sys.modules, "onecat_sm70_exact_reduce", SimpleNamespace(__file__=str(binary))
    )
    result = loaded_kernel_provenance()
    assert result[str(binary)] == hashlib.sha256(binary.read_bytes()).hexdigest()
