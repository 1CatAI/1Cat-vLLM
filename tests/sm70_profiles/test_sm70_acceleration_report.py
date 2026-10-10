# SPDX-License-Identifier: Apache-2.0
"""The kernel-policy report must survive set-valued fields on resolved policies."""

from dataclasses import dataclass, field

from vllm.sm70_profiles.acceleration import linear_policy_report


@dataclass
class _Filters:
    layers: frozenset = field(default_factory=lambda: frozenset((0, 1)))
    shapes: frozenset | None = None
    probes: tuple = (1, 2)


@dataclass
class _Policy:
    enabled: bool = True
    filters: _Filters = field(default_factory=_Filters)


@dataclass
class _KernelConfig:
    sm70_dump: _Policy = field(default_factory=_Policy)
    other: int = 3


def test_linear_policy_report_serializes_sets():
    report = linear_policy_report(_KernelConfig())
    assert set(report) == {"sm70_dump"}
    cfg = report["sm70_dump"]["configuration"]
    assert cfg["filters"]["layers"] == [0, 1]
    assert cfg["filters"]["shapes"] is None
    assert cfg["filters"]["probes"] == [1, 2]
    assert cfg["enabled"] is True
