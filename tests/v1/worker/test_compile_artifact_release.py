# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

cache = pytest.importorskip("torch.compiler._cache")
codecache = pytest.importorskip("torch._inductor.codecache")


def test_release_compile_artifact_records_drops_loaded_bytes():
    from vllm.v1.worker.gpu_worker import Worker

    manager = cache.CacheArtifactManager
    manager.clear()
    manager.record_artifact(
        codecache.InductorCacheArtifact.type(), "key-a", b"x" * 4096
    )
    manager.record_artifact(codecache.InductorCacheArtifact.type(), "key-b", b"y" * 10)
    assert Worker._release_compile_artifact_records() == 4106
    assert not manager._seen_artifacts
    assert not manager.need_serialize()
    assert Worker._release_compile_artifact_records() == 0
