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
