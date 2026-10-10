# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retained MTP adapters preserve current runtime owner contracts on CPU."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.config.sm70_draft import Sm70DraftConfig
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.worker.gpu import sm70_runner_ops
from vllm.v1.worker.gpu.model_states import sm70_mtp_metadata
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode import sm70_greedy_verify
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle
from vllm.v1.worker.gpu.spec_decode.eagle.prefill_moe_rows import (
    DraftMoERowSelector,
    PadRowOps,
)
from vllm.v1.worker.gpu.spec_decode.target_sampling import ComputedTargetLogits


def _output():
    return SamplerOutput(
        sampled_token_ids=torch.tensor([[3, 4]]),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=torch.tensor([2]),
    )


@pytest.mark.parametrize("completed", ["sampled", "logits", "non_gather"])
def test_completed_target_work_preempts_retained_projection(monkeypatch, completed):
    output = _output() if completed == "sampled" else None
    logits = (
        None
        if completed == "sampled"
        else ComputedTargetLogits(
            torch.empty((2, 8)) if completed == "logits" else None
        )
    )
    fallback = Mock(side_effect=AssertionError("completed projection must be reused"))
    monkeypatch.setattr(sm70_runner_ops, "try_target_sample", fallback)
    model = NS(get_top_tokens=Mock(side_effect=AssertionError("duplicate projection")))
    actual = sm70_greedy_verify.maybe_sample_greedy(
        model,
        None,
        NS(synthetic_conditional_rates=None),
        True,
        True,
        torch.empty((2, 4)),
        NS(num_draft_tokens=1),
        None,
        output,
        logits,
        speculator=object(),
    )
    assert actual is output
    model.get_top_tokens.assert_not_called()
    fallback.assert_not_called()


@pytest.mark.parametrize("upstream_enabled", [False, True])
def test_upstream_greedy_verifier_precedes_retained_fallback(
    monkeypatch, upstream_enabled
):
    output = _output()
    fallback = Mock(return_value=output)
    monkeypatch.setattr(sm70_runner_ops, "try_target_sample", fallback)
    verify = Mock(return_value=(output.sampled_token_ids, output.num_sampled))
    monkeypatch.setattr(sm70_greedy_verify, "greedy_verify", verify)
    model = NS(get_top_tokens=Mock(return_value=torch.tensor([3, 4])))
    sampler = NS(can_use_sm70_greedy_token_fastpath=lambda _: True)
    reject = NS(
        synthetic_conditional_rates=None, sampler=sampler, num_speculative_steps=1
    )
    batch = NS(
        num_draft_tokens=1,
        input_ids=torch.tensor([1, 3]),
        logits_indices=torch.tensor([0, 1]),
        cu_num_logits=torch.tensor([0, 2]),
    )
    actual = sm70_greedy_verify.maybe_sample_greedy(
        model,
        sampler,
        reject,
        True,
        upstream_enabled,
        torch.empty((2, 4)),
        batch,
        None,
        None,
        None,
        speculator=object(),
    )
    assert torch.equal(actual.sampled_token_ids, output.sampled_token_ids)
    assert verify.call_count == int(upstream_enabled)
    assert model.get_top_tokens.call_count == int(upstream_enabled)
    assert fallback.call_count == int(not upstream_enabled)


@pytest.mark.parametrize("fail_multistep", [False, True])
def test_padding_covers_both_graph_captures_and_restores_scope(fail_multistep):
    def mask(ids, num_valid):
        ids[int(num_valid[0]) :] = -1

    router = NS(
        select_experts=lambda: (torch.ones(3, 1), torch.tensor([[1], [2], [3]]))
    )
    layer = NS(
        runner=NS(
            router=router,
            _quant_method=NS(is_monolithic=False),
            _forward_impl=lambda: None,
        )
    )
    selector = DraftMoERowSelector([layer], pad_ops=PadRowOps(mask=mask, note=Mock()))
    observed = []

    class Capture:
        def __init__(self, phase):
            self.phase = phase
            self.graphs = {}

        def capture(self, *args, **kwargs):
            _, ids = router.select_experts()
            observed.append((self.phase, ids.tolist()))
            if fail_multistep and self.phase == "multistep":
                raise RuntimeError("capture failure")

    spec = object.__new__(eagle.EagleSpeculator)
    skip_topk = Mock()
    compact_topk = Mock()
    spec.model = NS(
        model=NS(set_skip_topk=skip_topk, compact_topk_indices=compact_topk)
    )
    spec.share_mtp_topk_indices = True
    spec.num_speculative_steps = 4
    spec.last_token_indices = torch.tensor([3])
    spec.prefill_moe_rows = selector
    spec.vllm_config = NS(
        kernel_config=KernelConfig(sm70_draft=Sm70DraftConfig(units=("d2a",)))
    )
    spec.max_num_reqs = 8
    spec.input_buffers = NS(query_start_loc=torch.tensor([0] * 8 + [2]))
    spec.prefill_cudagraph_manager = Capture("prefill")
    spec.decode_cudagraph_manager = Capture("decode")
    spec.multistep_cudagraph_manager = Capture("multistep")
    spec.model_state = spec.block_tables = spec.attn_groups = spec.kv_cache_config = (
        None
    )
    if fail_multistep:
        with pytest.raises(RuntimeError, match="capture failure"):
            spec.capture({})
    else:
        spec.capture({})
    assert observed == [
        ("prefill", [[1], [2], [3]]),
        ("decode", [[1], [2], [-1]]),
        ("multistep", [[1], [2], [-1]]),
    ]
    assert selector._num_valid is None
    assert skip_topk.call_args.args == (False,)
    compact_topk.assert_called_once()
    assert torch.equal(compact_topk.call_args.args[0], torch.tensor([0]))


def test_shortconv_provider_captures_policy_and_owns_each_engine_descriptor(
    monkeypatch,
):
    monkeypatch.setattr(
        sm70_mtp_metadata.current_platform, "is_device_capability", lambda _: True
    )
    monkeypatch.setattr(sm70_mtp_metadata._sc_meta, "enabled", lambda: True)
    cfg = NS(kernel_config=KernelConfig(), speculative_config=NS(method="mtp"))
    first = cfg.kernel_config.shortconv_metadata_provider(
        cfg, torch.device("cuda"), "align"
    )
    second = cfg.kernel_config.shortconv_metadata_provider(
        cfg, torch.device("cuda"), "align"
    )
    monkeypatch.setattr(sm70_mtp_metadata._sc_meta, "enabled", lambda: False)
    disabled = cfg.kernel_config.shortconv_metadata_provider(
        cfg, torch.device("cuda"), "align"
    )
    assert first.enabled and second.enabled and not disabled.enabled
    builder = object.__new__(PleShortConvAttentionMetadataBuilder)
    groups = [[NS(get_metadata_builder=lambda _: builder)]]
    descriptors = [object(), object(), object()]
    prepare = Mock(side_effect=[({id(builder): object()}, d) for d in descriptors])
    monkeypatch.setattr(
        sm70_mtp_metadata._sc_meta, "prepare_ple_shortconv_group_metadata", prepare
    )
    indices = torch.tensor([7, 9])
    mapping = torch.tensor([1, 0])
    args = (
        NS(query_start_loc=torch.tensor([0, 5, 10])),
        (torch.empty(2, 5),),
        groups,
        torch.tensor([0, 5, 10]),
        torch.tensor([4, 4]),
        torch.tensor([1, 2]),
        2,
        10,
    )
    for provider in (first, second, first):
        result = provider.prepare(
            *args, state_start_indices=indices, req_index_mapping=mapping
        )
        assert set(result) == {id(builder)}
    calls = prepare.call_args_list
    assert [c.kwargs["descriptor"] for c in calls] == [None, None, descriptors[0]]
    assert first._ple_shortconv_descriptor is descriptors[2]
    assert second._ple_shortconv_descriptor is descriptors[1]
    for c in calls:
        assert c.kwargs["state_start_indices"] is indices
        assert c.kwargs["req_index_mapping"] is mapping


@pytest.mark.parametrize("reverse", [False, True])
def test_retained_logits_use_each_captured_collective_policy(monkeypatch, reverse):
    from vllm.config.execution_policy import CommunicationPolicy
    from vllm.model_executor.layers import logits_processor as module

    monkeypatch.setenv("VLLM_SM70_TOP1_CUSTOM_AR", "1")
    policies = [CommunicationPolicy(top1_custom_ar=value) for value in (False, True)]
    for policy in policies:
        policy.resolve()
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 4)
    processors = []
    for policy in reversed(policies) if reverse else policies:
        monkeypatch.setattr(module, "communication_policy", lambda p=policy: p)
        processors.append(module.LogitsProcessor(16))
    monkeypatch.setenv("VLLM_SM70_TOP1_CUSTOM_AR", "0")
    monkeypatch.setattr(module.current_platform, "is_device_capability", lambda _: True)
    answer = torch.tensor([9])
    collective = Mock(return_value=answer)
    monkeypatch.setattr(
        module,
        "get_tp_group",
        lambda: NS(device_communicator=NS(ca_comm=NS(custom_top1_argmax=collective))),
    )
    pair = torch.tensor([[1.0, 9.0]])
    for processor in processors:
        result = processor._maybe_custom_top1_argmax(pair)
        assert result is (
            answer if processor._communication_policy.top1_custom_ar else None
        )
    collective.assert_called_once()
