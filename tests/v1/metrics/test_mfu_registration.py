# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MFU registration and observation follow the existing metrics switch."""

from functools import partial
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, Counter

from vllm.v1.metrics.perf import PerfMetricsProm, PerfStats


def _metrics(enabled, registry):
    class Isolated(PerfMetricsProm):
        _counter_cls = staticmethod(partial(Counter, registry=registry))

    config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_mfu_metrics=enabled)
    )
    return Isolated(config, ["engine"], {0: ["0"], 1: ["1"]})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("has_stats", [False, True])
def test_mfu_registration_and_observation(enabled, has_stats):
    registry = CollectorRegistry()
    metrics = _metrics(enabled, registry)
    assert len(list(registry.collect())) == (3 if enabled else 0)

    stats = PerfStats(11, 13, 17) if has_stats else PerfStats()
    metrics.observe(stats, engine_idx=1)
    for name, amount in (
        ("estimated_flops_per_gpu", 11),
        ("estimated_read_bytes_per_gpu", 13),
        ("estimated_write_bytes_per_gpu", 17),
    ):
        sample = f"vllm:{name}_total"
        value = registry.get_sample_value(sample, {"engine": "1"})
        assert value == ((amount if has_stats else 0) if enabled else None)
        assert registry.get_sample_value(sample, {"engine": "0"}) == (
            0 if enabled else None
        )


def test_disabled_instance_does_not_reserve_metric_names():
    registry = CollectorRegistry()
    _metrics(False, registry)
    _metrics(True, registry)
    assert len(list(registry.collect())) == 3
