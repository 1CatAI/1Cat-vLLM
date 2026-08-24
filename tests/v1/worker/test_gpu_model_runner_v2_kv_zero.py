# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import torch

from vllm.v1.worker.gpu import model_runner as mrv2


class FakeKVBlockZeroer:
    def __init__(self, device: torch.device, pin_memory: bool):
        self.device = device
        self.pin_memory = pin_memory
        self.init_kwargs: dict[str, Any] | None = None

    def init_meta(self, **kwargs: Any) -> None:
        kwargs["attn_groups_iter"] = list(kwargs["attn_groups_iter"])
        self.init_kwargs = kwargs


def test_v2_initializes_kv_zeroer_from_all_attention_groups(monkeypatch) -> None:
    monkeypatch.setattr(mrv2, "KVBlockZeroer", FakeKVBlockZeroer)
    monkeypatch.setattr(mrv2, "is_pin_memory_available", lambda: False)

    groups = [[object()], [object(), object()]]
    static_forward_context = object()
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.attn_groups = groups
    runner._kernel_block_sizes = [16, 32]
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.compilation_config = SimpleNamespace(
        static_forward_context=static_forward_context
    )

    runner._init_kv_zero_meta()

    zeroer = runner._kv_block_zeroer
    assert isinstance(zeroer, FakeKVBlockZeroer)
    assert zeroer.device == runner.device
    assert zeroer.pin_memory is False
    assert zeroer.init_kwargs == {
        "attn_groups_iter": [*groups[0], *groups[1]],
        "kernel_block_sizes": [16, 32],
        "cache_dtype": "auto",
        "runner_only_attn_layers": set(),
        "static_forward_context": static_forward_context,
    }


def test_v2_clears_new_cache_blocks_before_zero_token_return() -> None:
    events: list[object] = []
    empty_output = object()

    def no_forward(_: Any) -> object:
        events.append("no_forward")
        return empty_output

    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.finish_requests = lambda _: events.append("finish")
    runner.free_states = lambda _: events.append("free")
    runner.add_requests = lambda _: events.append("add")
    runner.update_requests = lambda _: events.append("update")
    runner.block_tables = SimpleNamespace(
        apply_staged_writes=lambda: events.append("apply")
    )
    runner._zero_block_ids = lambda block_ids: events.append(("zero", block_ids))
    runner.kv_connector = SimpleNamespace(no_forward=no_forward)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=0,
        new_block_ids_to_zero=[3, 7],
    )

    output = runner.execute_model(scheduler_output)

    assert output is empty_output
    assert events == [
        "finish",
        "free",
        "add",
        "update",
        "apply",
        ("zero", [3, 7]),
        "no_forward",
    ]
