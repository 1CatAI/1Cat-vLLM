# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission remains lazy; preparation and observations survive a declined run."""

import pytest

from vllm.v1.attention.backends.flash_v100.plan import routing

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("prepare", [False, True])
def test_selection_retains_attempt_order_and_falsey_results(monkeypatch, prepare):
    events = []
    context = {"prepared": False}
    monkeypatch.setattr(
        routing.accounting, "_record_route", lambda name: events.append(name)
    )

    class Prepare:
        def admit(self, context):
            return prepare

        def run(self, context, record):
            events.append("gather")
            context["prepared"] = True
            record("decode_dense_cache")
            return None

    class Complete:
        def admit(self, context):
            assert context["prepared"] == prepare
            return True

        def run(self, context, record):
            events.append("compute")
            record("decode_scalar_paged")
            return 0

    def choices():
        yield Prepare()
        yield Complete()
        raise AssertionError("selection must stop after a non-None result")

    assert routing.execute(context, choices()) == 0
    assert events == (
        ["gather", "decode_dense_cache", "compute", "decode_scalar_paged"]
        if prepare
        else ["compute", "decode_scalar_paged"]
    )


def test_partial_selection_can_leave_all_rows_for_sequence_dispatch():
    from vllm.v1.attention.backends.flash_v100.plan.routing import execute, try_execute

    class Decline:
        def admit(self, request):
            return True

        def run(self, request, record):
            request.append("prepared")
            return None

    prepared: list[str] = []
    assert try_execute(prepared, (Decline(),)) is None
    assert prepared == ["prepared"]
    with pytest.raises(RuntimeError, match="No attention candidate completed"):
        execute(prepared, (Decline(),))
    assert prepared == ["prepared", "prepared"]
