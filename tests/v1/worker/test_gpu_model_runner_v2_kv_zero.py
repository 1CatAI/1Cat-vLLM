# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch

from vllm import envs
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator


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
    runner.device = torch.device("cpu")
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


def test_v2_warms_qwen38_mtp_moe_prefill_and_decode_shapes(monkeypatch) -> None:
    zeroer = mock.Mock()
    zeroer.warmup_kernel.return_value = True
    draft_model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_experts_per_tok=10,
            moe_intermediate_size=640,
        ),
        get_num_experts=lambda: 512,
        get_hidden_size=lambda: 2560,
    )
    speculator = EagleSpeculator.__new__(EagleSpeculator)
    speculator.method = "mtp"
    speculator.device = torch.device("cuda")
    speculator.draft_model_config = draft_model_config
    speculator.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1]),
    )
    speculator.max_num_tokens = 8192
    speculator.num_speculative_steps = 0
    speculator.max_num_reqs = 1

    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner._auxiliary_warmup_enabled = True
    runner._kv_block_zeroer = zeroer
    runner.speculator = speculator
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner._dummy_run = mock.Mock()

    with (
        mock.patch.object(
            mrv2.current_platform, "is_device_capability", return_value=True
        ),
        mock.patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "_warmup_sm70_qwen_gdn_causal_conv1d",
            return_value=False,
        ),
        mock.patch.object(envs, "VLLM_SM70_AUX_KERNEL_WARMUP", True),
        mock.patch.object(envs, "VLLM_SM70_MTP_MOE_TUNED_CONFIG", True),
        mock.patch.object(torch.accelerator, "synchronize"),
    ):
        runner._warmup_sm70_aux_kernels()

    zeroer.warmup_kernel.assert_called_once_with()
    assert runner._dummy_run.call_args_list == [
        mock.call(16),
        mock.call(1, uniform_decode=True),
    ]
