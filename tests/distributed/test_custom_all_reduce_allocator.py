# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

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


def test_capture_enables_ipc_guard_only_when_expandable_segments_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = object.__new__(custom_all_reduce.CustomAllreduce)
    communicator.disabled = False
    communicator._IS_CAPTURING = False
    guard_states: list[bool] = []
    monkeypatch.setattr(communicator, "register_graph_buffers", lambda: None)
    _patch_allocator_settings(monkeypatch)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

    for allocator_conf in ("expandable_segments:True", ""):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", allocator_conf)
        with communicator.capture():
            guard_states.append(communicator._capture_ipc_guard)

    assert guard_states == [True, False]


def test_safe_allocation_disables_expandable_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator_conf = "max_split_size_mb:512,expandable_segments:True"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", allocator_conf)
    settings = _patch_allocator_settings(monkeypatch)
    monkeypatch.setattr(
        custom_all_reduce.torch.cuda, "is_current_stream_capturing", lambda: False
    )

    with custom_all_reduce.custom_ar_ipc_safe_allocation():
        assert settings == [allocator_conf.replace("True", "False")]

    assert settings == [allocator_conf.replace("True", "False"), allocator_conf]


@pytest.mark.parametrize(
    ("allocator_conf", "capturing"),
    [("expandable_segments:False", False), ("expandable_segments:True", True)],
)
def test_safe_allocation_is_otherwise_a_noop(
    monkeypatch: pytest.MonkeyPatch,
    allocator_conf: str,
    capturing: bool,
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", allocator_conf)
    settings = _patch_allocator_settings(monkeypatch)
    monkeypatch.setattr(
        custom_all_reduce.torch.cuda,
        "is_current_stream_capturing",
        lambda: capturing,
    )

    with custom_all_reduce.custom_ar_ipc_safe_allocation():
        pass

    assert settings == []


def _capturing_communicator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ipc_guard: bool,
    exportable,
) -> tuple[custom_all_reduce.CustomAllreduce, list[tuple]]:
    communicator = object.__new__(custom_all_reduce.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.world_size = 4
    communicator.fully_connected = True
    communicator._IS_CAPTURING = True
    communicator._capture_ipc_guard = ipc_guard
    communicator._ipc_guard_copies = 0
    communicator._ipc_guard_fallbacks = 0
    calls: list[tuple] = []
    monkeypatch.setattr(
        custom_all_reduce.torch.cuda, "is_current_stream_capturing", lambda: True
    )
    monkeypatch.setattr(custom_all_reduce, "_is_legacy_cuda_ipc_capable", exportable)
    monkeypatch.setattr(communicator, "should_custom_ar", lambda inp: True)

    def all_reduce(inp, *, out=None, registered=False):
        calls.append(("all_reduce", inp, registered))
        return inp

    def all_reduce_sum2(inp_a, inp_b, *, out=None):
        calls.append(("all_reduce_sum2", inp_a, inp_b))
        return inp_a

    def top1_argmax(inp, *, out=None, registered=False):
        calls.append(("top1_argmax", inp, registered))
        return inp

    monkeypatch.setattr(communicator, "all_reduce", all_reduce)
    monkeypatch.setattr(communicator, "all_reduce_sum2", all_reduce_sum2)
    monkeypatch.setattr(communicator, "top1_argmax", top1_argmax)
    return communicator, calls


def test_capture_without_ipc_guard_registers_inputs_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_probe(ptr: int) -> bool:
        raise AssertionError("an unguarded capture must not query IPC capability")

    communicator, calls = _capturing_communicator(
        monkeypatch, ipc_guard=False, exportable=no_probe
    )
    torch = custom_all_reduce.torch
    inp = torch.arange(16, dtype=torch.float16)
    pair = torch.zeros(2, dtype=torch.float32)

    communicator.custom_all_reduce(inp)
    communicator.custom_all_reduce_sum2(inp, inp.clone())
    communicator.custom_top1_argmax(pair)

    assert calls[0] == ("all_reduce", inp, True)
    assert calls[1][0] == "all_reduce_sum2" and calls[1][1] is inp
    assert calls[2] == ("top1_argmax", pair, True)


@pytest.mark.parametrize("capable", [True, None])
def test_ipc_guard_keeps_exportable_or_unknown_inputs_without_a_copy(
    monkeypatch: pytest.MonkeyPatch,
    capable: bool | None,
) -> None:
    communicator, calls = _capturing_communicator(
        monkeypatch, ipc_guard=True, exportable=lambda ptr: capable
    )
    torch = custom_all_reduce.torch
    inp = torch.arange(16, dtype=torch.float16)
    other = inp.clone()

    communicator.custom_all_reduce(inp)
    communicator.custom_all_reduce_sum2(inp, other)

    assert calls[0] == ("all_reduce", inp, True)
    assert calls[1][1] is inp and calls[1][2] is other
    assert communicator._ipc_guard_copies == 0
    assert communicator._ipc_guard_fallbacks == 0


def test_ipc_guard_copies_unexportable_inputs_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = custom_all_reduce.torch
    inp = torch.arange(16, dtype=torch.float16)
    resident = torch.arange(16, 32, dtype=torch.float16)
    vmm_ptrs = {inp.data_ptr(), resident.data_ptr()}
    communicator, calls = _capturing_communicator(
        monkeypatch, ipc_guard=True, exportable=lambda ptr: ptr not in vmm_ptrs
    )
    shared = torch.ones(16, dtype=torch.float16)

    communicator.custom_all_reduce(inp)
    communicator.custom_all_reduce_sum2(shared, resident)
    communicator.custom_top1_argmax(torch.zeros(2, dtype=torch.float32))

    name, staged, registered = calls[0]
    assert name == "all_reduce" and registered
    assert staged.data_ptr() not in vmm_ptrs and torch.equal(staged, inp)
    assert staged.stride() == inp.stride()
    name, sum_a, sum_b = calls[1]
    assert name == "all_reduce_sum2" and sum_a is shared
    assert sum_b.data_ptr() not in vmm_ptrs and torch.equal(sum_b, resident)
    assert calls[2][0] == "top1_argmax" and calls[2][2]
    assert communicator._ipc_guard_copies == 2
    assert communicator._ipc_guard_fallbacks == 0


def test_ipc_guard_falls_back_when_no_exportable_copy_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator, calls = _capturing_communicator(
        monkeypatch, ipc_guard=True, exportable=lambda ptr: False
    )
    torch = custom_all_reduce.torch
    inp = torch.arange(16, dtype=torch.float16)
    pair = torch.zeros(2, dtype=torch.float32)

    communicator.custom_all_reduce(inp)
    assert communicator.custom_all_reduce_sum2(inp, inp.clone()) is None
    communicator.custom_top1_argmax(pair)

    # Staging buffer registration (all_reduce/top1) or the NCCL add path (sum2).
    assert calls == [("all_reduce", inp, False), ("top1_argmax", pair, False)]
    assert communicator._ipc_guard_copies == 0
    assert communicator._ipc_guard_fallbacks == 3


def test_ipc_guard_counts_are_logged_and_reset_before_ipc_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = object.__new__(custom_all_reduce.CustomAllreduce)
    communicator.disabled = False
    communicator._capture_ipc_guard = True
    communicator._ipc_guard_copies = 3
    communicator._ipc_guard_fallbacks = 1
    communicator._ptr = 0
    messages: list[tuple] = []
    monkeypatch.setattr(
        custom_all_reduce.logger,
        "info",
        lambda msg, *args: messages.append((msg, args)),
    )

    def failed_export(_ptr):
        raise RuntimeError("cudaIpcGetMemHandle failed")

    communicator._native = SimpleNamespace(get_graph_buffer_ipc_meta=failed_export)

    with pytest.raises(RuntimeError, match="cudaIpcGetMemHandle"):
        communicator.register_graph_buffers()

    assert messages and messages[0][1] == (3, 1)
    assert communicator._ipc_guard_copies == 0
    assert communicator._ipc_guard_fallbacks == 0


def test_ipc_probe_failure_is_reported_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_driver(*args, **kwargs):
        raise OSError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(custom_all_reduce, "_LEGACY_CUDA_IPC_PROBE", None)
    monkeypatch.setattr(custom_all_reduce, "_LEGACY_CUDA_IPC_PROBE_FAILED", False)
    monkeypatch.setattr(custom_all_reduce, "_LegacyCudaIpcProbe", missing_driver)

    assert custom_all_reduce._is_legacy_cuda_ipc_capable(0x1000) is None
    assert custom_all_reduce._LEGACY_CUDA_IPC_PROBE_FAILED
    assert custom_all_reduce._is_legacy_cuda_ipc_capable(0x2000) is None
