# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
            "roundup_power2_divisions:[256:1,512:2,>:4], "
            "expandable_segments:True,max_split_size_mb:20",
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
