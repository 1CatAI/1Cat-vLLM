# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref

import pytest
import torch

from vllm.config import DeviceConfig, KernelConfig, VllmConfig, set_current_vllm_config
from vllm.config.execution_policy import LayerExecutionPolicy
from vllm.model_executor.kernels.lm_head import sm70 as provider
from vllm.model_executor.kernels.norm import sm70 as norm_provider
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    maybe_prepare_sm70_lm_head_top1,
)
from vllm.model_executor.models.shared_weights import (
    DFLASH_DRAFT_WEIGHTS,
    EAGLE_DRAFT_WEIGHTS,
    LEGACY_DRAFT_WEIGHTS,
    share_embeddings,
    share_lm_head,
)

pytestmark = pytest.mark.skip_global_cleanup


def bare_embedding():
    layer = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    torch.nn.Module.__init__(layer)
    layer.weight = torch.nn.Parameter(torch.ones((32, 16), dtype=torch.float16))
    layer.prefix = "model.lm_head"
    layer.shard_indices = provider.VocabShard()
    return layer


def config(dense):
    return VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        kernel_config=KernelConfig(
            layer_execution=LayerExecutionPolicy(
                lm_head_dense=dense,
                lm_head_top1=False,
                lm_head_top1_tc=False,
            )
        ),
    )


def test_single_buffer_owner_and_weight_rebind_release(monkeypatch):
    monkeypatch.setattr(provider, "NativeBindings", lambda values: provider.sm70_ops)
    monkeypatch.setattr(provider, "_is_sm70_lm_head_fastpath_eligible", lambda _: True)
    monkeypatch.setattr(torch.ops._C, "sm70_f16_prepare", object(), raising=False)
    calls = []

    def prepare(weight):
        calls.append(weakref.ref(weight))
        return torch.empty_like(weight), torch.tensor([16])

    monkeypatch.setattr(provider.sm70_ops, "sm70_f16_prepare", prepare)
    with set_current_vllm_config(config(True)):
        layer = bare_embedding()
        assert maybe_prepare_sm70_lm_head_top1(layer)
        assert maybe_prepare_sm70_lm_head_top1(layer)
    assert len(calls) == 1
    state = layer._sm70_lm_head_state
    assert state.weight is layer.weight
    assert layer._sm70_f16_tm_weight is state._sm70_f16_tm_weight
    assert list(layer.state_dict()) == ["weight"]
    assert len(list(layer.buffers())) == 1
    old_weight = weakref.ref(layer.weight)
    old_pack = weakref.ref(state._sm70_f16_tm_weight)
    del state
    layer.weight = torch.nn.Parameter(torch.zeros_like(layer.weight))
    gc.collect()
    assert old_weight() is None
    assert old_pack() is None
    assert not hasattr(layer, "_sm70_lm_head_state")
    assert not hasattr(layer, "_sm70_f16_prepared")


@pytest.mark.parametrize("reverse", [False, True])
def test_layer_initialization_borrows_own_engine_policy(monkeypatch, reverse):
    configs = [config(False), config(True)]
    if reverse:
        configs.reverse()
    layers = []
    for cfg in configs:
        with set_current_vllm_config(cfg):
            layer = bare_embedding()
            maybe_prepare_sm70_lm_head_top1(layer)  # CPU rejection keeps the owner.
            layers.append(layer)
    monkeypatch.setenv("VLLM_SM70_ENABLE_LM_HEAD_FASTPATH", "1")
    for layer, cfg in zip(layers, configs):
        assert layer._sm70_lm_head_state.policy is cfg.kernel_config.layer_execution
    assert layers[0]._sm70_lm_head_state is not layers[1]._sm70_lm_head_state
    assert (
        layers[0]._sm70_lm_head_state.policy.lm_head_dense
        != layers[1]._sm70_lm_head_state.policy.lm_head_dense
    )


@pytest.mark.parametrize("chosen", [0, 1, 2, None])
def test_gemma_stages_keep_order_and_stop_after_selected_provider(monkeypatch, chosen):
    seen = []

    def gate(stage):
        def check(*args):
            seen.append(stage)
            return chosen == stage

        return check

    result = (torch.tensor([1]), torch.tensor([2]))
    monkeypatch.setattr(norm_provider, "_use_sm70_dflash2_fixed_gemma_rms", gate(0))
    monkeypatch.setattr(norm_provider, "_use_sm70_dflash2_gemma_fused_add_rms", gate(1))
    monkeypatch.setattr(norm_provider, "use_long_prefill_fused", gate(2))
    monkeypatch.setattr(
        norm_provider, "_sm70_dflash2_fixed_gemma_rms_norm", lambda *a: result
    )
    monkeypatch.setattr(
        torch.ops.vllm, "sm70_dflash2_gemma_fused_add_rms_norm", lambda *a: result
    )
    monkeypatch.setattr(
        torch.ops.vllm, "sm70_gemma_long_prefill_fused_add_rms_norm", lambda *a: result
    )
    x = torch.ones((2, 16))
    actual = norm_provider.maybe_gemma_norm(
        x,
        x,
        x[0],
        1e-6,
        policy=None,
        dflash=None,
        graph=None,
    )
    assert actual is (None if chosen is None else result)
    assert seen == list(range(3 if chosen is None else chosen + 1))


@pytest.mark.parametrize(
    "contract", [LEGACY_DRAFT_WEIGHTS, EAGLE_DRAFT_WEIGHTS, DFLASH_DRAFT_WEIGHTS]
)
@pytest.mark.parametrize("pp", [1, 2])
def test_shared_embedding_contract_releases_only_eligible_draft(contract, pp):
    target, draft = torch.nn.Module(), torch.nn.Module()
    target.model, draft.model = torch.nn.Module(), torch.nn.Module()
    target.model.embed_tokens = torch.nn.Embedding(8, 4)
    draft.model.embed_tokens = torch.nn.Embedding(8, 4)
    original = weakref.ref(draft.model.embed_tokens)
    shared = share_embeddings(draft, target, contract, pp_size=pp)
    expected = pp == 1 or contract is DFLASH_DRAFT_WEIGHTS
    assert shared == expected
    if expected:
        assert draft.model.embed_tokens is target.model.embed_tokens
        assert original() is None
    else:
        assert draft.model.embed_tokens is original()


def test_mtp_head_and_buffer_have_one_target_owner():
    target, draft = torch.nn.Module(), torch.nn.Module()
    target.model, draft.model = torch.nn.Module(), torch.nn.Module()
    target.lm_head = torch.nn.Linear(4, 8, bias=False)
    target.model.register_buffer("topk_indices_buffer", torch.arange(4))
    draft.lm_head = torch.nn.Linear(4, 8, bias=False)
    layer = torch.nn.Module()
    layer.shared_head = torch.nn.Module()
    layer.shared_head.head = draft.lm_head
    draft.model.layers = torch.nn.ModuleList([layer])
    previous = weakref.ref(draft.lm_head)
    assert share_lm_head(draft, target, target, LEGACY_DRAFT_WEIGHTS)
    assert draft.lm_head is layer.shared_head.head is target.lm_head
    assert draft.model.topk_indices_buffer is target.model.topk_indices_buffer
    assert previous() is None


@pytest.mark.parametrize(
    "cached,handled,capable",
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_greedy_verification_respects_existing_result(cached, handled, capable):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm.v1.worker.gpu.spec_decode.sm70_greedy_verify import maybe_sample_greedy

    model = Mock()
    batch = SimpleNamespace(num_draft_tokens=4)
    previous = object() if handled else None
    output = maybe_sample_greedy(
        model,
        None,
        None,
        capable,
        True,
        torch.zeros(5, 4),
        batch,
        None,
        previous,
        object() if cached else None,
    )
    assert output is previous
    model.get_top_tokens.assert_not_called()
    model.compute_logits.assert_not_called()


def test_preparation_does_not_require_native_abi_on_rejected_device(monkeypatch):
    from unittest.mock import Mock

    from vllm.config.sm70_runtime import RuntimeTraceConfig
    from vllm.model_executor.kernels.linear import sm70_dense
    from vllm.model_executor.kernels.linear.sm70_dense import (
        DenseLinearState,
        prepare_dense,
    )

    native = Mock(
        side_effect=AssertionError("rejected CPU path must not bind CUDA ABI")
    )
    monkeypatch.setattr(provider, "NativeBindings", native)
    monkeypatch.setattr(sm70_dense, "NativeBindings", native)
    with set_current_vllm_config(config(True)):
        layer = bare_embedding()
        assert not maybe_prepare_sm70_lm_head_top1(layer)
        state = DenseLinearState(
            layer.weight,
            prefix="down_proj",
            policy=LayerExecutionPolicy(dense_f16=True),
            trace=RuntimeTraceConfig(),
        )
        prepare_dense(
            state, force_enable=True, input_parallel=True, suffix_allowed=True
        )
    native.assert_not_called()
