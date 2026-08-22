# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.device_communicators import custom_all_reduce

pytestmark = pytest.mark.skip_global_cleanup


def _patch_allocator_settings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    settings: list[str] = []
    monkeypatch.setattr(custom_all_reduce.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        custom_all_reduce.torch.cuda.memory,
        "_set_allocator_settings",
        settings.append,
    )
    return settings


def test_cuda_ipc_capture_temporarily_disables_expandable_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:512,expandable_segments:True",
    )
    settings = _patch_allocator_settings(monkeypatch)

    with custom_all_reduce._disable_expandable_segments_for_cuda_ipc(True):
        assert settings == ["expandable_segments:False"]

    assert settings == ["expandable_segments:False", "expandable_segments:True"]


def test_cuda_ipc_capture_restores_allocator_after_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    settings = _patch_allocator_settings(monkeypatch)

    with (
        pytest.raises(RuntimeError, match="capture failed"),
        custom_all_reduce._disable_expandable_segments_for_cuda_ipc(True),
    ):
        raise RuntimeError("capture failed")

    assert settings == ["expandable_segments:False", "expandable_segments:True"]


def test_custom_allreduce_registers_graph_buffers_before_allocator_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    settings = _patch_allocator_settings(monkeypatch)
    communicator = object.__new__(custom_all_reduce.CustomAllreduce)
    communicator.disabled = False
    communicator._IS_CAPTURING = False
    communicator._ptr = None
    registration_state: list[tuple[list[str], bool]] = []

    def register_graph_buffers() -> None:
        registration_state.append((list(settings), communicator._IS_CAPTURING))

    monkeypatch.setattr(
        communicator,
        "register_graph_buffers",
        register_graph_buffers,
    )

    with communicator.capture():
        assert communicator._IS_CAPTURING
        assert settings == ["expandable_segments:False"]

    assert registration_state == [(["expandable_segments:False"], False)]
    assert settings == ["expandable_segments:False", "expandable_segments:True"]


@pytest.mark.parametrize(
    ("active", "allocator_conf"),
    [
        (False, "expandable_segments:True"),
        (True, "expandable_segments:False"),
        (True, ""),
    ],
)
def test_cuda_ipc_capture_allocator_guard_is_otherwise_a_noop(
    monkeypatch: pytest.MonkeyPatch,
    active: bool,
    allocator_conf: str,
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", allocator_conf)
    settings = _patch_allocator_settings(monkeypatch)

    with custom_all_reduce._disable_expandable_segments_for_cuda_ipc(active):
        pass

    assert settings == []
