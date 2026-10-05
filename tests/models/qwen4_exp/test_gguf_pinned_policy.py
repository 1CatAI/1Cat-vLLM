# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels.ple.gguf_pinned import pinned_table_capability


def capability(**changes):
    params = dict(
        tables=[dict(type=20, k=160, n=320001536, bytes=28800138240)],
        enabled=True,
        sm70=True,
        fp16=True,
        tp_size=4,
        local_ranks=4,
        max_seqs=4,
        local_workers=True,
        dual_compile_full_graphs=True,
        disk_cascade=False,
        available_bytes=160 << 30,
        reserve_bytes=47 << 30,
    )
    params.update(changes)
    return pinned_table_capability(**params)


def test_all_four_shards_fit_without_double_booking_host_memory():
    status = capability()
    assert status["enabled"]
    assert status["packed_bytes_per_rank"] == 7200034560
    assert not capability(available_bytes=60 << 30)["enabled"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"enabled": False}, "disabled_by_kernel_config"),
        ({"sm70": False}, "requires_sm70_fp16_output"),
        ({"max_seqs": 8}, "decode_capacity_has_no_calibration"),
        ({"local_workers": False}, "requires_local_tp4_workers"),
        ({"disk_cascade": True}, "disk_cascade_placement_takes_precedence"),
        (
            {"dual_compile_full_graphs": False},
            "requires_v2_dual_compile_full_graphs",
        ),
        ({"explicit_host_bytes": 1 << 30}, "explicit_host_budget_too_small"),
        ({"available_bytes": None}, "host_capacity_unavailable"),
    ],
)
def test_unqualified_route_reports_reason(change, reason):
    assert capability(**change)["reason"] == reason


def test_graph_hash_changes_with_pinned_topology_but_not_status_report():
    config = KernelConfig()
    before = config.compute_hash()
    config.ple_pinned_decoders["table"] = {"reason": "no_rows"}
    assert config.compute_hash() == before
    config.ple_pinned_decode_active = True
    assert config.compute_hash() != before


@pytest.mark.parametrize("decode", [False, True])
def test_pinned_layer_uses_local_rows_only_during_decode(monkeypatch, decode):
    from types import SimpleNamespace

    from vllm.model_executor.layers import ple_offload_layer as module

    marker = object()
    calls = []

    def local(*args, **kwargs):
        calls.append("local")
        return marker

    def remote(*args):
        calls.append("remote")
        return marker

    layer = SimpleNamespace(
        _is_cpu_offloaded=True,
        _pinned_decode=True,
        forward_impl=local,
        wait_offloaded_output=remote,
    )
    monkeypatch.setattr(module.envs, "VLLM_SM70_QWEN38_HYBRID_PLE", False)
    monkeypatch.setattr(module, "use_sm70_decode_graph_semantics", lambda: decode)
    result = module.PleOffloadLayer.forward(layer, None, SimpleNamespace(shape=(512,)))
    assert result is marker
    assert calls == ["local" if decode else "remote"]


@pytest.mark.parametrize(
    "local_model,pinned", [(False, True), (True, False), (True, True)]
)
def test_connector_only_skips_requests_for_owned_local_decode(
    monkeypatch, local_model, pinned
):
    from types import SimpleNamespace

    from vllm.v1.ple_offload import connector as module

    calls = []
    connector = SimpleNamespace(
        _all_pinned_decode=pinned,
        _launch=lambda *a: calls.append(a),
    )
    monkeypatch.setattr(module.envs, "VLLM_SM70_QWEN38_HYBRID_PLE", False)
    module.PleOffloadConnector.prepare_forward(
        connector, 4, 20, dummy_run=False, use_local_model=local_model
    )
    assert calls == ([] if local_model and pinned else [(4, 20)])


def test_pinned_first_touch_affinity_restored_after_loader_error(monkeypatch):
    from types import SimpleNamespace

    import torch

    from vllm import platforms
    from vllm.model_executor.kernels.ple import gguf_pinned as module

    calls = []
    monkeypatch.setattr(module, "pinned_decode_active", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(get_device_numa_node=lambda i: 1)
    )
    monkeypatch.setattr(module.Path, "read_text", lambda p: "24-25,72-73")
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: {0, 24, 72})
    monkeypatch.setattr(
        module.os, "sched_setaffinity", lambda _, cpus: calls.append(cpus)
    )
    with (
        pytest.raises(ValueError, match="loader error"),
        module.pinned_table_load_affinity(),
    ):
        assert calls == [{24, 72}]
        raise ValueError("loader error")
    assert calls == [{24, 72}, {0, 24, 72}]


def test_disabled_pinned_load_keeps_existing_affinity(monkeypatch):
    from vllm.model_executor.kernels.ple import gguf_pinned as module

    monkeypatch.setattr(module, "pinned_decode_active", lambda: False)
    calls = []
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: calls.append(True))
    with module.pinned_table_load_affinity():
        pass
    assert calls == []
