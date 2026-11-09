# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered providers preserve explicit payload precedence and lazy reads."""

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.flash_v100.spec.features import (
    FEATURES,
    DDTreeFeature,
    DFlash2Feature,
    MTPFeature,
    prepare_verification,
)
from vllm.v1.spec_decode.attention_features import SpecFeatureRegistry

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("method", [None, "dflash", "dflash_ddtree", "mtp"])
@pytest.mark.parametrize("tree_verify", [False, True])
@pytest.mark.parametrize("has_prepared", [False, True])
def test_method_providers_preserve_payload_priority_and_lazy_capacity_reads(
    method, tree_verify, has_prepared
):
    events: list[object] = []
    prepared = object() if has_prepared else None
    attn = object()

    class Common:
        @property
        def max_seq_len(self):
            events.append("capacity")
            assert not tree_verify and prepared is None
            return 31

    common = Common()
    state = SimpleNamespace(
        feature=FEATURES.for_method(method),
        _attach_prepared_dflash2_smallq_metadata=(
            lambda *args, **kwargs: events.append(("prepared", args, kwargs))
        ),
        _update_smallq_decode_metadata=(
            lambda *args, **kwargs: events.append(("linear", args, kwargs))
        ),
    )
    prepare_verification(state, attn, common, tree_verify, prepared)
    if tree_verify:
        assert events == []
    elif prepared is not None:
        assert events == [("prepared", (attn, prepared), {})]
    else:
        assert events == [
            "capacity",
            ("linear", (attn, common), {"workspace_seq_capacity_cap": 31}),
        ]


def test_registry_selects_three_distinct_providers_and_retains_fallback():
    for method, expected in (
        ("dflash", DFlash2Feature),
        ("dflash_ddtree", DDTreeFeature),
        ("mtp", MTPFeature),
        (None, MTPFeature),
        ("external-method", MTPFeature),
    ):
        first = FEATURES.for_method(method)
        assert type(first) is expected
        assert FEATURES.for_method(method) is not first


def test_registration_copies_inputs_and_supports_external_provider():
    class External:
        def prepare(self, *args):
            return args

    registrations = {"external": External}
    registry = SpecFeatureRegistry(registrations, fallback=MTPFeature)
    registrations.clear()
    assert registry.for_method("external").prepare(1, 2) == (1, 2)
    with pytest.raises(TypeError):
        registry.providers["new"] = External
