# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import regex as re

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
    allocator_conf = (
        "max_split_size_mb:512,garbage_collection_threshold:0.8,"
        "expandable_segments:True"
    )
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF",
        allocator_conf,
    )
    settings = _patch_allocator_settings(monkeypatch)

    with custom_all_reduce._disable_expandable_segments_for_cuda_ipc(True):
        assert settings == [allocator_conf.replace("True", "False")]

    assert settings == [allocator_conf.replace("True", "False"), allocator_conf]


def test_cuda_ipc_capture_preserves_allocator_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator_conf = (
        "max_split_size_mb:512,roundup_power2_divisions:4,"
        "garbage_collection_threshold:0.8,expandable_segments:True"
    )
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", allocator_conf)
    monkeypatch.setattr(custom_all_reduce.current_platform, "is_cuda", lambda: True)
    allocator_state: dict[str, int | float | bool] = {
        "max_split_size_mb": 512,
        "roundup_power2_divisions": 4,
        "garbage_collection_threshold": 0.8,
        "expandable_segments": True,
    }

    def set_allocator_settings(conf: str) -> None:
        # Model PyTorch's parseArgs behavior: these options reset whenever
        # they are omitted from a runtime settings update.
        allocator_state.update(
            max_split_size_mb=-1,
            roundup_power2_divisions=0,
            garbage_collection_threshold=0.0,
        )
        for key in allocator_state:
            match = re.search(rf"(?:^|,){key}:([^,]+)", conf)
            if match is None:
                continue
            value = match.group(1)
            if key == "garbage_collection_threshold":
                allocator_state[key] = float(value)
            elif key == "expandable_segments":
                allocator_state[key] = value == "True"
            else:
                allocator_state[key] = int(value)

    monkeypatch.setattr(
        custom_all_reduce.torch.cuda.memory,
        "_set_allocator_settings",
        set_allocator_settings,
    )

    with custom_all_reduce._disable_expandable_segments_for_cuda_ipc(True):
        assert allocator_state == {
            "max_split_size_mb": 512,
            "roundup_power2_divisions": 4,
            "garbage_collection_threshold": 0.8,
            "expandable_segments": False,
        }

    assert allocator_state == {
        "max_split_size_mb": 512,
        "roundup_power2_divisions": 4,
        "garbage_collection_threshold": 0.8,
        "expandable_segments": True,
    }


def test_cuda_ipc_capture_supports_unified_allocator_conf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    allocator_conf = "max_split_size_mb:512,expandable_segments:True"
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", allocator_conf)
    settings = _patch_allocator_settings(monkeypatch)

    with custom_all_reduce._disable_expandable_segments_for_cuda_ipc(True):
        pass

    assert settings == [allocator_conf.replace("True", "False"), allocator_conf]


@pytest.mark.parametrize("legacy_conf", ["", "expandable_segments:False"])
def test_legacy_allocator_conf_takes_precedence_over_unified_conf(
    monkeypatch: pytest.MonkeyPatch,
    legacy_conf: str,
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", legacy_conf)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    settings = _patch_allocator_settings(monkeypatch)

    with custom_all_reduce._disable_expandable_segments_for_cuda_ipc(True):
        pass

    assert settings == []


def test_enabled_legacy_allocator_conf_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_conf = "max_split_size_mb:512,expandable_segments:True"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", legacy_conf)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    settings = _patch_allocator_settings(monkeypatch)

    with custom_all_reduce._disable_expandable_segments_for_cuda_ipc(True):
        pass

    assert settings == [legacy_conf.replace("True", "False"), legacy_conf]


def test_cuda_ipc_capture_preserves_bracketed_allocator_syntax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator_conf = (
        "roundup_power2_divisions:[256:1,512:2,>:4], expandable_segments : True"
    )
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", allocator_conf)
    settings = _patch_allocator_settings(monkeypatch)

    with custom_all_reduce._disable_expandable_segments_for_cuda_ipc(True):
        pass

    assert settings == [
        allocator_conf.replace("True", "False"),
        allocator_conf,
    ]


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
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", allocator_conf)
    settings = _patch_allocator_settings(monkeypatch)

    with custom_all_reduce._disable_expandable_segments_for_cuda_ipc(active):
        pass

    assert settings == []
