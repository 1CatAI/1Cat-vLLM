# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.model_executor.layers.logits_processor as logits_processor_module
import vllm.model_executor.models.qwen3_dflash as dflash_model
import vllm.model_executor.models.qwen3_dflash2 as dflash2_model
import vllm.v1.attention.backends.flash_attn_v100 as flash_v100
import vllm.v1.worker.gpu.attn_utils as attn_utils
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_speculator
import vllm.v1.worker.gpu.spec_decode.dflash.utils as dflash_utils
from vllm import envs
from vllm.config.speculative import (
    SpeculativeConfig,
    _get_dflash2_checkpoint_draft_tokens,
)
from vllm.config.vllm import (
    _SM70_DFLASH2_VERIFIER_DEFAULTS,
    _SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS,
    _apply_sm70_dflash2_verifier_defaults,
    _configure_sm70_glm5_dflash_tp4_pp2_acceptance_path,
    _configure_sm70_glm5_dflash_tp4_push_allreduce,
    _configure_sm70_glm5_dflash_tp8_pp1_verifier_path,
    _is_sm70_dflash2_verifier_contract,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _is_dflash2_spec_config,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    _sm70_dflash2_dense_order_topk,
    _sm70_dflash2_rerank_output_buffers,
    _sm70_dflash2_use_dense_order,
)
from vllm.model_executor.models.dflash_sm70 import (
    DFLASH_SM70_GATE_UP_INPUT_SCALE,
    DFLASH_SM70_WIDE_OUTPUT_SCALE,
    DFlashSM70RMSNorm,
    dflash_layered_rms_norm_sm70,
    dflash_scale_output_sm70,
    dflash_silu_and_mul_sm70,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    _dflash_layer_causal,
)
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
    _grouped_conv,
    _score_edges,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
    _parse_alignment_steps,
    _supports_sparse_sampling_contract,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    DFlash2Speculator,
    _requires_sm70_tail,
    _selector_walk_kernel,
)


@pytest.mark.parametrize(
    ("has_own_embed", "has_own_head", "shared_embed", "shared_head"),
    [
        (False, False, True, True),
        (True, True, False, False),
    ],
)
def test_dflash_final_pp_stage_accepts_valid_shared_weight_contract(
    monkeypatch,
    has_own_embed,
    has_own_head,
    shared_embed,
    shared_head,
):
    monkeypatch.setattr(
        dflash_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    model = SimpleNamespace(
        has_own_embed_tokens=has_own_embed,
        has_own_lm_head=has_own_head,
    )

    dflash_utils._validate_dflash_shared_weights(model, shared_embed, shared_head)


@pytest.mark.parametrize(
    ("shared_embed", "shared_head", "message"),
    [
        (False, True, "no embedding"),
        (True, False, "no lm_head"),
    ],
)
def test_dflash_final_pp_stage_rejects_missing_shared_weight(
    monkeypatch, shared_embed, shared_head, message
):
    monkeypatch.setattr(
        dflash_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    model = SimpleNamespace(has_own_embed_tokens=False, has_own_lm_head=False)

    with pytest.raises(RuntimeError, match=message):
        dflash_utils._validate_dflash_shared_weights(model, shared_embed, shared_head)


@pytest.mark.parametrize("loaded", [False, True])
def test_dflash_final_pp_stage_validates_replicated_embedding_load(monkeypatch, loaded):
    monkeypatch.setattr(
        dflash_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    embed = SimpleNamespace(
        _dflash_pp_replica_expected=True,
        _dflash_pp_replica_loaded=loaded,
    )
    model = SimpleNamespace(
        has_own_embed_tokens=False,
        has_own_lm_head=False,
        model=SimpleNamespace(embed_tokens=embed),
    )

    if loaded:
        dflash_utils._validate_dflash_shared_weights(model, True, True)
    else:
        with pytest.raises(RuntimeError, match="embedding replica.*not loaded"):
            dflash_utils._validate_dflash_shared_weights(model, True, True)


def _dflash_attention_contract(causal=False, sliding_window=(2047, 0)):
    impl = object.__new__(FlashAttnV100Impl)
    impl.sliding_window = sliding_window
    layer = SimpleNamespace(
        is_dflash_draft_attn=True,
        dflash_expected_causal=False,
        dflash_expected_sliding_window=2048,
        dflash_rope_is_neox_style=True,
        layer_name="draft.layers.0.attn",
    )
    metadata = SimpleNamespace(causal=causal)
    return impl, layer, metadata


def test_flash_v100_accepts_glm53_dflash_attention_contract():
    impl, layer, metadata = _dflash_attention_contract()

    impl._validate_dflash_attention_contract(layer, metadata)


@pytest.mark.parametrize(
    ("causal", "sliding_window", "message"),
    [
        (True, (2047, 0), "causality mismatch"),
        (False, (1023, 0), "sliding-window mismatch"),
    ],
)
def test_flash_v100_rejects_glm53_dflash_attention_contract_mismatch(
    causal, sliding_window, message
):
    impl, layer, metadata = _dflash_attention_contract(causal, sliding_window)

    with pytest.raises(RuntimeError, match=message):
        impl._validate_dflash_attention_contract(layer, metadata)


def _sm70_dflash2_verifier_contract_args():
    model_config = SimpleNamespace(
        architectures=("Qwen3_5ForConditionalGeneration",),
        quantization="compressed-tensors",
        dtype=torch.float16,
        model_arch_config=SimpleNamespace(
            quantization_config={
                "format": "mixed-precision",
                "config_groups": {
                    "nvfp4": {"format": "nvfp4-pack-quantized"},
                    "fp8": {"format": "float-quantized"},
                },
            }
        ),
        hf_text_config=SimpleNamespace(
            hidden_size=5120,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=256,
        ),
    )
    speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=7,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 16})
        ),
    )
    parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        tensor_parallel_size=4,
        enable_dbo=False,
        ubatch_size=0,
    )
    return model_config, speculative_config, parallel_config


def test_dflash2_checkpoint_draft_tokens_follow_block_contract():
    hf_config = SimpleNamespace(dflash_config={"block_size": 8, "selector_top_k": 16})
    assert _get_dflash2_checkpoint_draft_tokens(hf_config) == 7

    hf_config.dflash_config["selector_top_k"] = 0
    assert _get_dflash2_checkpoint_draft_tokens(hf_config) is None

    hf_config.dflash_config = {"block_size": 1, "selector_top_k": 16}
    assert _get_dflash2_checkpoint_draft_tokens(hf_config) is None


def test_sm70_dflash2_verifier_contract_is_narrow():
    args = _sm70_dflash2_verifier_contract_args()
    assert _is_sm70_dflash2_verifier_contract(*args)

    for config_index, attribute, incompatible_value in (
        (0, "dtype", torch.bfloat16),
        (1, "num_speculative_tokens", 5),
        (2, "pipeline_parallel_size", 2),
        (2, "enable_dbo", True),
        (2, "ubatch_size", 2),
    ):
        incompatible_args = _sm70_dflash2_verifier_contract_args()
        setattr(incompatible_args[config_index], attribute, incompatible_value)
        assert not _is_sm70_dflash2_verifier_contract(*incompatible_args)

    incompatible_args = _sm70_dflash2_verifier_contract_args()
    incompatible_args[1].draft_model_config.hf_config.dflash_config[
        "selector_top_k"
    ] = 8
    assert not _is_sm70_dflash2_verifier_contract(*incompatible_args)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4, 8])
@pytest.mark.parametrize("quantization", [None, "fp8", "compressed-tensors"])
def test_sm70_dflash2_verifier_contract_is_tp_and_quantization_independent(
    tensor_parallel_size, quantization
):
    args = _sm70_dflash2_verifier_contract_args()
    args[0].quantization = quantization
    args[2].tensor_parallel_size = tensor_parallel_size
    assert _is_sm70_dflash2_verifier_contract(*args)


def test_sm70_dflash2_verifier_defaults_preserve_overrides(monkeypatch):
    for name in _SM70_DFLASH2_VERIFIER_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    overridden_name = "VLLM_SM70_DFLASH2_QPN8_RERANK"
    monkeypatch.setenv(overridden_name, "0")

    applied = _apply_sm70_dflash2_verifier_defaults()

    assert overridden_name not in applied
    assert os.environ[overridden_name] == "0"
    for name, expected_value in _SM70_DFLASH2_VERIFIER_DEFAULTS.items():
        if name != overridden_name:
            assert name in applied
            assert os.environ[name] == expected_value


def test_dflash2_gdn_fastpaths_are_default_off(monkeypatch):
    names = (
        "VLLM_SM70_DFLASH2_QPN8_RERANK",
        "VLLM_SM70_DFLASH2_QPN8_ALLOW_CANDIDATE_ORDER",
        "VLLM_SM70_DFLASH2_VERIFY_FASTPATH",
        "VLLM_SM70_DFLASH2_FUSED_GDN_METADATA",
        "VLLM_SM70_DFLASH2_GDN_METADATA_SHADOW",
        "VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY",
        "VLLM_SM70_DFLASH2_FUSED_GDN_NORM",
        "VLLM_SM70_DFLASH2_FUSED_GDN_SPLIT",
        "VLLM_SM70_DFLASH2_FUSED_SMALLQ_METADATA",
        "VLLM_SM70_DFLASH2_GROUPED_SMALLQ_METADATA",
        "VLLM_SM70_DFLASH2_FUSED_QKV_PACK",
        "VLLM_SM70_DFLASH2_FUSED_GEMMA_RMS",
        "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION",
        "VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    try:
        assert not any(getattr(envs, name) for name in names)
    finally:
        envs.disable_envs_cache()


def test_dflash2_grouped_verify_is_default_on_with_rollback(monkeypatch):
    name = "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY"
    monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    try:
        assert getattr(envs, name)
        monkeypatch.setenv(name, "0")
        envs.disable_envs_cache()
        assert not getattr(envs, name)
    finally:
        envs.disable_envs_cache()


def test_sm70_tp4_push_allreduce_is_default_on_with_rollback(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_TP4_PUSH_ALLREDUCE", raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_TP4_PUSH_ALLREDUCE
        monkeypatch.setenv("VLLM_SM70_TP4_PUSH_ALLREDUCE", "0")
        envs.disable_envs_cache()
        assert not envs.VLLM_SM70_TP4_PUSH_ALLREDUCE
    finally:
        envs.disable_envs_cache()


def test_glm5_dflash_tp4_push_allreduce_is_quality_safe_by_default(monkeypatch):
    name = "VLLM_SM70_TP4_PUSH_ALLREDUCE"
    monkeypatch.delenv(name, raising=False)
    _configure_sm70_glm5_dflash_tp4_push_allreduce(
        SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next_text")),
        SimpleNamespace(method="dflash"),
        SimpleNamespace(tensor_parallel_size=4),
        is_sm70=True,
    )

    envs.disable_envs_cache()
    try:
        assert os.environ[name] == "0"
        assert not envs.VLLM_SM70_TP4_PUSH_ALLREDUCE
    finally:
        envs.disable_envs_cache()


def test_sm70_tp4_push_allreduce_mtp5_is_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5", raising=False)
    envs.disable_envs_cache()
    try:
        assert not envs.VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5
        monkeypatch.setenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5", "1")
        envs.disable_envs_cache()
        assert envs.VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5
    finally:
        envs.disable_envs_cache()


def test_sm70_tp4_push_allreduce_qwen38_batch_defaults_on(monkeypatch):
    name = "VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH"
    monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH
        monkeypatch.setenv(name, "0")
        envs.disable_envs_cache()
        assert not envs.VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH
    finally:
        envs.disable_envs_cache()


def test_sm70_tp4_push_allreduce_sum2_m1_is_default_on_with_rollback(monkeypatch):
    name = "VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1"
    monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    try:
        assert getattr(envs, name)
        monkeypatch.setenv(name, "0")
        envs.disable_envs_cache()
        assert not getattr(envs, name)
    finally:
        envs.disable_envs_cache()


@pytest.mark.parametrize(
    ("model_type", "method", "tp_size", "is_sm70"),
    [
        ("qwen3", "dflash", 4, True),
        ("glm5_next_text", "draft_model", 4, True),
        ("glm5_next_text", "dflash", 2, True),
        ("glm5_next_text", "dflash", 4, False),
    ],
)
def test_glm5_dflash_tp4_policy_does_not_change_other_routes(
    monkeypatch, model_type, method, tp_size, is_sm70
):
    name = "VLLM_SM70_TP4_PUSH_ALLREDUCE"
    monkeypatch.delenv(name, raising=False)
    _configure_sm70_glm5_dflash_tp4_push_allreduce(
        SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
        SimpleNamespace(method=method),
        SimpleNamespace(tensor_parallel_size=tp_size),
        is_sm70=is_sm70,
    )

    assert name not in os.environ


def test_glm5_dflash_tp4_push_allreduce_preserves_explicit_override(monkeypatch):
    name = "VLLM_SM70_TP4_PUSH_ALLREDUCE"
    monkeypatch.setenv(name, "1")
    _configure_sm70_glm5_dflash_tp4_push_allreduce(
        SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next_text")),
        SimpleNamespace(method="dflash"),
        SimpleNamespace(tensor_parallel_size=4),
        is_sm70=True,
    )

    assert os.environ[name] == "1"


def _glm5_dflash_tp8_verifier_config():
    return (
        SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm5_next_text"),
            quantization="modelopt_fp4",
            dtype=torch.float16,
        ),
        SimpleNamespace(
            method="dflash",
            draft_sample_method="probabilistic",
            num_speculative_tokens=7,
        ),
        SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            enable_dbo=False,
            ubatch_size=0,
        ),
    )


def test_glm5_dflash_tp8_pp1_auto_selects_verifier_path(monkeypatch):
    for name in _SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS:
        monkeypatch.delenv(name, raising=False)

    selected = _configure_sm70_glm5_dflash_tp8_pp1_verifier_path(
        *_glm5_dflash_tp8_verifier_config(), is_sm70=True
    )

    assert selected
    assert {
        name: os.environ.get(name) for name in _SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS
    } == _SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "qwen3"),
        ("quantization", None),
        ("dtype", torch.bfloat16),
        ("method", "draft_model"),
        ("draft_sample_method", "greedy"),
        ("num_speculative_tokens", 6),
        ("tensor_parallel_size", 4),
        ("pipeline_parallel_size", 2),
        ("enable_dbo", True),
        ("ubatch_size", 2),
        ("is_sm70", False),
    ],
)
def test_glm5_dflash_tp8_verifier_policy_does_not_change_other_routes(
    monkeypatch, field, value
):
    for name in _SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS:
        monkeypatch.delenv(name, raising=False)

    model_config, speculative_config, parallel_config = (
        _glm5_dflash_tp8_verifier_config()
    )
    is_sm70 = True
    if field == "model_type":
        model_config.hf_text_config.model_type = value
    elif field == "is_sm70":
        is_sm70 = value
    elif hasattr(model_config, field):
        setattr(model_config, field, value)
    elif hasattr(speculative_config, field):
        setattr(speculative_config, field, value)
    else:
        setattr(parallel_config, field, value)

    selected = _configure_sm70_glm5_dflash_tp8_pp1_verifier_path(
        model_config,
        speculative_config,
        parallel_config,
        is_sm70=is_sm70,
    )

    assert not selected
    assert not any(name in os.environ for name in _SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS)


def test_glm5_dflash_tp8_verifier_policy_preserves_explicit_override(monkeypatch):
    overridden_name = "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION"
    monkeypatch.setenv(overridden_name, "1")
    for name in _SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS:
        if name != overridden_name:
            monkeypatch.delenv(name, raising=False)

    selected = _configure_sm70_glm5_dflash_tp8_pp1_verifier_path(
        *_glm5_dflash_tp8_verifier_config(), is_sm70=True
    )

    assert selected
    assert os.environ[overridden_name] == "1"
    for name, value in _SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS.items():
        if name != overridden_name:
            assert os.environ[name] == value


def _glm5_dflash_acceptance_config():
    return (
        SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="glm5_next_text", num_hidden_layers=45
            ),
            quantization="modelopt_fp4",
        ),
        SimpleNamespace(
            method="dflash",
            draft_sample_method="probabilistic",
            num_speculative_tokens=7,
        ),
        SimpleNamespace(tensor_parallel_size=4, pipeline_parallel_size=2),
    )


def test_glm5_dflash_tp4_pp2_auto_selects_quality_path(monkeypatch):
    expected = {
        "VLLM_PP_LAYER_PARTITION": "24,21",
        "VLLM_SM70_DFLASH2_PROPOSAL_TEMPERATURE_SCALE": "0.8",
        "VLLM_SM70_DFLASH2_PROPOSAL_TOP_P": "0.95",
    }
    for name in expected:
        monkeypatch.delenv(name, raising=False)

    _configure_sm70_glm5_dflash_tp4_pp2_acceptance_path(
        *_glm5_dflash_acceptance_config(), is_sm70=True
    )

    assert {name: os.environ.get(name) for name in expected} == expected


@pytest.mark.parametrize("num_layers", [32, 46, 70, None])
def test_glm5_partition_does_not_override_other_layer_counts(monkeypatch, num_layers):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    model, spec, parallel = _glm5_dflash_acceptance_config()
    model.hf_text_config.num_hidden_layers = num_layers
    _configure_sm70_glm5_dflash_tp4_pp2_acceptance_path(
        model, spec, parallel, is_sm70=True
    )
    assert "VLLM_PP_LAYER_PARTITION" not in os.environ


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "qwen3"),
        ("quantization", None),
        ("method", "draft_model"),
        ("draft_sample_method", "greedy"),
        ("num_speculative_tokens", 6),
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 1),
        ("is_sm70", False),
    ],
)
def test_glm5_dflash_acceptance_policy_does_not_change_other_routes(
    monkeypatch, field, value
):
    names = (
        "VLLM_PP_LAYER_PARTITION",
        "VLLM_SM70_DFLASH2_PROPOSAL_TEMPERATURE_SCALE",
        "VLLM_SM70_DFLASH2_PROPOSAL_TOP_P",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    model_config, speculative_config, parallel_config = _glm5_dflash_acceptance_config()
    is_sm70 = True
    if field == "model_type":
        model_config.hf_text_config.model_type = value
    elif field == "is_sm70":
        is_sm70 = value
    elif hasattr(model_config, field):
        setattr(model_config, field, value)
    elif hasattr(speculative_config, field):
        setattr(speculative_config, field, value)
    else:
        setattr(parallel_config, field, value)

    _configure_sm70_glm5_dflash_tp4_pp2_acceptance_path(
        model_config,
        speculative_config,
        parallel_config,
        is_sm70=is_sm70,
    )

    assert not any(name in os.environ for name in names)


def test_glm5_dflash_acceptance_policy_preserves_explicit_overrides(monkeypatch):
    overrides = {
        "VLLM_PP_LAYER_PARTITION": "24,21",
        "VLLM_SM70_DFLASH2_PROPOSAL_TEMPERATURE_SCALE": "1.0",
        "VLLM_SM70_DFLASH2_PROPOSAL_TOP_P": "1.0",
    }
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)

    _configure_sm70_glm5_dflash_tp4_pp2_acceptance_path(
        *_glm5_dflash_acceptance_config(), is_sm70=True
    )

    assert {name: os.environ[name] for name in overrides} == overrides


def test_glm5_dflash_acceptance_policy_preserves_materialize_diagnostic(
    monkeypatch,
):
    name = "VLLM_GLM53_PP_MHC_MATERIALIZE"
    monkeypatch.setenv(name, "1")

    _configure_sm70_glm5_dflash_tp4_pp2_acceptance_path(
        *_glm5_dflash_acceptance_config(), is_sm70=True
    )

    assert os.environ[name] == "1"


def test_sm70_dflash2_bf16_emulation_has_explicit_ab_switch(monkeypatch):
    config = SimpleNamespace(dtype=torch.bfloat16)
    monkeypatch.setattr(dflash2_model.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        dflash2_model.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )
    monkeypatch.delenv("VLLM_SM70_DFLASH2_BF16_EMULATION", raising=False)
    assert dflash2_model._use_sm70_bf16_emulation(config)

    monkeypatch.setenv("VLLM_SM70_DFLASH2_BF16_EMULATION", "0")
    assert not dflash2_model._use_sm70_bf16_emulation(config)


def _bare_dflash2_model() -> DFlash2Qwen3Model:
    model = DFlash2Qwen3Model.__new__(DFlash2Qwen3Model)
    torch.nn.Module.__init__(model)
    model.quant_config = None
    return model


def test_dflash2_local_argmax_delegates_to_vocab_parallel_processor():
    model = DFlash2Qwen3ForCausalLM.__new__(DFlash2Qwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.lm_head = Mock()
    model.logits_processor = Mock()
    hidden_states = torch.randn(3, 8)
    expected = torch.tensor([7, 11, 13])
    model.logits_processor.get_top_tokens.return_value = expected

    actual = model.get_top_tokens(hidden_states)

    assert actual is expected
    model.logits_processor.get_top_tokens.assert_called_once_with(
        model.lm_head, hidden_states
    )


def test_dflash_sliding_kv_spec_uses_draft_attention_contract_with_mla_target():
    attn = Attention.__new__(Attention)
    torch.nn.Module.__init__(attn)
    attn.attn_type = AttentionType.DECODER
    attn.kv_cache_dtype = "auto"
    attn.kv_cache_torch_dtype = torch.float16
    attn.sliding_window = 2048
    attn.num_kv_heads = 1
    attn.head_size = 256
    attn.head_size_v = 256
    attn.is_dflash_draft_attn = True
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        attention_config=SimpleNamespace(prefix_anchored_decode_window=None),
        model_config=SimpleNamespace(use_mla=True),
    )

    spec = attn.get_kv_cache_spec(config)

    assert isinstance(spec, SlidingWindowSpec)
    assert spec.sliding_window == 2048


@pytest.mark.parametrize("draft_style", [False, True])
def test_dflash_loader_preserves_draft_rope_layout(monkeypatch, draft_style):
    draft_hf_config = SimpleNamespace(
        is_neox_style=draft_style,
        is_causal=False,
        num_hidden_layers=1,
    )
    draft_model_config = SimpleNamespace(hf_config=draft_hf_config)
    speculative_config = SimpleNamespace(
        draft_model_config=draft_model_config,
        kv_cache_dtype=None,
        attention_backend="FLASH_ATTN_V100",
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        attention_config=SimpleNamespace(use_non_causal=False, backend=None),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    draft_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=None),
        has_own_embed_tokens=True,
        has_own_lm_head=True,
        lm_head=None,
    )
    target_model = SimpleNamespace(
        get_language_model=lambda: SimpleNamespace(
            config=SimpleNamespace(is_neox_style=not draft_style)
        )
    )

    def fake_replace(value, **updates):
        values = vars(value).copy()
        values.update(updates)
        return SimpleNamespace(**values)

    monkeypatch.setattr(dflash_utils, "replace", fake_replace)
    monkeypatch.setattr(dflash_utils, "get_model", lambda **_kwargs: draft_model)
    monkeypatch.setattr(dflash_utils, "get_target_lm_head", lambda *_args: None)
    monkeypatch.setattr(
        dflash_utils, "_validate_dflash_shared_weights", lambda *_args: None
    )
    monkeypatch.setattr(dflash_model, "dflash_has_any_non_causal", lambda _config: True)
    monkeypatch.setattr(
        "vllm.compilation.backends.set_model_tag", lambda _tag: nullcontext()
    )

    dflash_utils.load_dflash_model(target_model, vllm_config)

    assert draft_hf_config.is_neox_style is draft_style


def _fake_rms_norm(
    output: torch.Tensor,
    input_: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    normalized = input_.float() * torch.rsqrt(
        input_.float().pow(2).mean(dim=-1, keepdim=True) + epsilon
    )
    if weight.ndim == 2:
        weight = weight.view(weight.shape[0], *([1] * (input_.ndim - 2)), -1)
    output.copy_((normalized * weight.float()).to(output.dtype))


def _bare_context_k_model() -> DFlashQwen3Model:
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    torch.nn.Module.__init__(model)
    model._rms_norm_eps = 1e-6
    model._k_norm_weights = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.5, 1.5, 0.75, 1.25],
            [1.5, 0.5, 1.25, 0.75],
        ],
        dtype=torch.float32,
    )
    model._batched_k_norm_runtime_verified = None
    return model


def test_context_k_norm_trusts_verified_batched_runtime(monkeypatch):
    model = _bare_context_k_model()
    all_k = torch.randn(3, 2, 1, 4)
    calls = []

    def grouped_runtime(output, input_, weight, epsilon):
        calls.append(weight.ndim)
        _fake_rms_norm(output, input_, weight, epsilon)

    monkeypatch.setattr(dflash_model.ops, "rms_norm", grouped_runtime)
    first = model._normalize_context_k(all_k)
    assert model._batched_k_norm_runtime_verified is True
    assert calls == [2, 1, 1, 1, 2]

    calls.clear()
    second = model._normalize_context_k(all_k)
    assert calls == [2]
    assert torch.equal(second, first)


def test_context_k_norm_falls_back_for_stale_stable_binary(monkeypatch):
    model = _bare_context_k_model()
    # An all-zero profiling input would make the stale grouped output appear
    # correct unless the capability probe uses its own nonzero fixture.
    all_k = torch.zeros(3, 2, 1, 4)
    calls = []

    def stale_runtime(output, input_, weight, epsilon):
        calls.append(weight.ndim)
        # Historical stable-ABI binaries silently used row zero for a 2-D
        # weight tensor. One-dimensional calls remained correct.
        effective_weight = weight[0] if weight.ndim == 2 else weight
        _fake_rms_norm(output, input_, effective_weight, epsilon)

    monkeypatch.setattr(dflash_model.ops, "rms_norm", stale_runtime)
    actual = model._normalize_context_k(all_k)
    expected = torch.empty_like(all_k)
    for layer_idx in range(all_k.shape[0]):
        _fake_rms_norm(
            expected[layer_idx],
            all_k[layer_idx],
            model._k_norm_weights[layer_idx],
            model._rms_norm_eps,
        )

    assert model._batched_k_norm_runtime_verified is False
    assert calls == [2, 1, 1, 1, 1, 1, 1]
    assert torch.equal(actual, expected)

    calls.clear()
    repeated = model._normalize_context_k(all_k)
    assert calls == [1, 1, 1]
    assert torch.equal(repeated, expected)


def test_sm70_layered_context_k_norm_uses_each_weight_row():
    input_ = torch.tensor(
        [
            [[[1.0, 2.0, 3.0, 4.0]], [[4.0, 3.0, 2.0, 1.0]]],
            [[[2.0, 1.0, 4.0, 3.0]], [[3.0, 4.0, 1.0, 2.0]]],
        ],
        dtype=torch.float16,
    )
    weight = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [0.5, 1.5, 0.75, 1.25]],
        dtype=torch.float16,
    )

    actual = dflash_layered_rms_norm_sm70(input_, weight, 1e-6)
    expected = torch.empty_like(input_)
    for layer_idx in range(input_.shape[0]):
        _fake_rms_norm(expected[layer_idx], input_[layer_idx], weight[layer_idx], 1e-6)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("input_size", "output_size"), [(25600, 5120), (20480, 4096)])
def test_sm70_tp4_shards_only_compatible_dflash2_context_projection(
    monkeypatch, input_size, output_size
):
    model = _bare_dflash2_model()
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        model_config=SimpleNamespace(dtype=torch.float16),
    )
    created = {}

    def fake_column_parallel(**kwargs):
        created.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        dflash2_model.envs,
        "VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC",
        True,
    )
    monkeypatch.setattr(dflash2_model.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        dflash2_model.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )
    monkeypatch.setattr(dflash2_model, "ColumnParallelLinear", fake_column_parallel)

    projection = model._make_context_projection(
        vllm_config=config,
        input_size=input_size,
        output_size=output_size,
        prefix="model.fc",
    )

    assert created["gather_output"] is True
    assert created["input_size"] == input_size
    assert created["output_size"] == output_size
    assert projection._sm70_f16_force_enable is True
    assert projection._sm70_f16_max_m == 64


@pytest.mark.parametrize(
    ("enabled", "tp_size", "input_size", "output_size"),
    [
        (False, 4, 25600, 5120),
        (True, 2, 25600, 5120),
        (True, 4, 5120, 5120),
        (True, 4, 25600, 4096),
    ],
)
def test_sharded_context_projection_falls_back_outside_exact_contract(
    monkeypatch, enabled, tp_size, input_size, output_size
):
    model = _bare_dflash2_model()
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        model_config=SimpleNamespace(dtype=torch.float16),
    )
    sentinel = object()
    monkeypatch.setattr(
        dflash2_model.envs,
        "VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC",
        enabled,
    )
    monkeypatch.setattr(dflash2_model.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        dflash2_model.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )
    monkeypatch.setattr(
        DFlashQwen3Model,
        "_make_context_projection",
        lambda *_args, **_kwargs: sentinel,
    )

    projection = model._make_context_projection(
        vllm_config=config,
        input_size=input_size,
        output_size=output_size,
        prefix="model.fc",
    )

    assert projection is sentinel


@pytest.mark.parametrize("block_size", [4, 6, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _stub_base(monkeypatch: pytest.MonkeyPatch, draft_logits):
    def init_base(self, _vllm_config, device):
        self.device = device
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 16})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 8
        self.num_speculative_steps = 7
        self.draft_block = 7
        self.vocab_size = 31
        self.draft_tokens = torch.empty((2, 7), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits
        self._draft_logits_init = (
            None if draft_logits is None else (torch.float32, -float("inf"))
        )

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_without_proposal_logits(monkeypatch):
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    assert speculator.draft_logits is None


def test_selector_default_path_does_not_allocate_sparse_score_cache(monkeypatch):
    allocated = torch.full((2, 7, 31), -float("inf"), dtype=torch.float32)
    _stub_base(monkeypatch, allocated)
    monkeypatch.setattr(envs, "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION", False)
    monkeypatch.setattr(envs, "VLLM_SPEC_DUMP_ALIGNMENT", False)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    assert speculator.draft_logits is allocated
    assert torch.isneginf(speculator.draft_logits).all()
    assert speculator.get_sparse_draft_logits() is None
    assert speculator.get_selector_alignment_shadow() is None


def test_selector_opt_in_allocates_sparse_score_cache(monkeypatch):
    allocated = torch.full((2, 7, 31), -float("inf"), dtype=torch.float32)
    _stub_base(monkeypatch, allocated)
    monkeypatch.setattr(envs, "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION", True)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    sparse_logits = speculator.get_sparse_draft_logits()
    assert sparse_logits is not None
    candidate_ids, candidate_scores = sparse_logits
    assert candidate_ids.shape == (2, 7, 16)
    assert candidate_scores.shape == (2, 7, 16)
    assert candidate_scores.dtype is torch.float32


def test_selector_alignment_shadow_is_explicit_and_keeps_full_lattice(monkeypatch):
    allocated = torch.full((2, 7, 31), -float("inf"), dtype=torch.float32)
    _stub_base(monkeypatch, allocated)
    monkeypatch.setattr(envs, "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION", True)
    monkeypatch.setattr(envs, "VLLM_SPEC_DUMP_ALIGNMENT", True)

    speculator = DFlash2Speculator(None, torch.device("cpu"))
    shadow = speculator.get_selector_alignment_shadow()

    assert shadow is not None
    candidate_ids, unary_logits, lattice_scores = shadow
    assert candidate_ids.shape == (2, 7, 16)
    assert unary_logits.shape == (2, 7, 16)
    assert unary_logits.dtype is torch.float32
    assert lattice_scores.shape == (2, 7, 16, 16)
    assert lattice_scores.dtype is torch.float32


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("1,3-5", {1, 3, 4, 5}),
        ("5-3", set()),
        ("bad", set()),
        ("-1", set()),
    ],
)
def test_selector_alignment_step_filter(raw, expected):
    assert _parse_alignment_steps(raw) == expected


def test_selector_uses_checkpoint_top16_and_fp32_proposal_cache(monkeypatch):
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)
    assert speculator.selector_top_k == 16
    assert dtype is torch.float32
    assert fill == float("-inf")


def test_probabilistic_dense_fallback_is_allocated_after_initialization(monkeypatch):
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    speculator._draft_logits_init = (torch.float32, -float("inf"))

    assert speculator.draft_logits is None
    speculator._allocate_draft_logits()

    assert speculator.draft_logits is not None
    assert speculator.draft_logits.shape == (2, 7, 31)
    assert speculator.draft_logits.dtype is torch.float32
    assert torch.isneginf(speculator.draft_logits).all()


def _sparse_sampling_contract_fixture():
    idx = np.array([0], dtype=np.int32)
    sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.array([1.0], dtype=np.float32)),
        top_k=SimpleNamespace(np=np.array([20], dtype=np.int32)),
        top_p=SimpleNamespace(np=np.array([0.95], dtype=np.float32)),
        min_p=SimpleNamespace(np=np.array([0.0], dtype=np.float32)),
        max_num_logprobs=Mock(return_value=-1),
    )
    sampler = SimpleNamespace(
        sampling_states=sampling_states,
        penalties_state=SimpleNamespace(use_penalty=np.array([False])),
        logit_bias_state=SimpleNamespace(use_logit_bias=np.array([False])),
        bad_words_state=SimpleNamespace(
            num_bad_words=SimpleNamespace(np=np.array([0], dtype=np.int32))
        ),
        logprob_token_ids_state=SimpleNamespace(max_num_token_ids=Mock(return_value=0)),
        compute_nans=False,
    )
    rejection_sampler = SimpleNamespace(
        rejection_sample_method="standard",
        sampler=sampler,
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        is_prefilling_np=np.array([False]),
        idx_mapping_np=idx,
    )
    return rejection_sampler, input_batch


def test_sparse_target_rejection_accepts_official_sampling_contract():
    rejection_sampler, input_batch = _sparse_sampling_contract_fixture()
    assert _supports_sparse_sampling_contract(rejection_sampler, input_batch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.0),
        ("top_k", 16),
        ("top_p", 0.0),
        ("min_p", 0.05),
        ("penalty", True),
        ("logit_bias", True),
        ("bad_words", 1),
        ("logprobs", 1),
        ("custom_logprobs", 1),
        ("compute_nans", True),
    ],
)
def test_sparse_target_rejection_falls_back_for_unsupported_sampling(
    field: str,
    value: float | int | bool,
):
    rejection_sampler, input_batch = _sparse_sampling_contract_fixture()
    sampler = rejection_sampler.sampler
    if field in {"temperature", "top_k", "top_p", "min_p"}:
        getattr(sampler.sampling_states, field).np[0] = value
    elif field == "penalty":
        sampler.penalties_state.use_penalty[0] = value
    elif field == "logit_bias":
        sampler.logit_bias_state.use_logit_bias[0] = value
    elif field == "bad_words":
        sampler.bad_words_state.num_bad_words.np[0] = value
    elif field == "logprobs":
        sampler.sampling_states.max_num_logprobs.return_value = value
    elif field == "custom_logprobs":
        sampler.logprob_token_ids_state.max_num_token_ids.return_value = value
    else:
        sampler.compute_nans = value

    assert not _supports_sparse_sampling_contract(rejection_sampler, input_batch)


def test_probabilistic_selector_caches_temperature_applied_scores():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the DFlash2 selector kernel")

    device = torch.device("cuda")
    num_steps, top_k = 2, 4
    scores = torch.tensor(
        [
            [
                [
                    [0.0, 0.5, 1.0, 1.5],
                    [2.0, 2.5, 3.0, 3.5],
                    [4.0, 4.5, 5.0, 5.5],
                    [6.0, 6.5, 7.0, 7.5],
                ],
                [
                    [8.0, 8.5, 9.0, 9.5],
                    [10.0, 10.5, 11.0, 11.5],
                    [12.0, 12.5, 13.0, 13.5],
                    [14.0, 14.5, 15.0, 15.5],
                ],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    candidates = torch.arange(top_k, dtype=torch.int64, device=device).repeat(
        1, num_steps, 1
    )
    sample_pos = torch.tensor([10, 11], dtype=torch.int64, device=device)
    req_state = torch.zeros(num_steps, dtype=torch.int32, device=device)
    temperature = torch.tensor([0.5], dtype=torch.float32, device=device)
    seeds = torch.tensor([123], dtype=torch.int64, device=device)
    tokens = torch.full((num_steps,), -1, dtype=torch.int64, device=device)
    realized = torch.full(
        (1, num_steps, top_k),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    path_state = torch.empty(1, dtype=torch.int32, device=device)

    _selector_walk_kernel[(1,)](
        scores,
        candidates,
        sample_pos,
        req_state,
        temperature,
        seeds,
        tokens,
        realized,
        path_state,
        num_steps=num_steps,
        walk_steps=num_steps,
        top_k=top_k,
        BLOCK_K=top_k,
        SAMPLE_PROBABILISTIC=True,
        USE_FP64=False,
        PROPOSAL_TEMPERATURE_SCALE=1.0,
        PROPOSAL_TOP_P=1.0,
        num_warps=1,
    )

    first_index = int(tokens[0].item())
    torch.testing.assert_close(realized[0, 0], scores[0, 0, 0] / temperature[0])
    torch.testing.assert_close(
        realized[0, 1], scores[0, 1, first_index] / temperature[0]
    )


def test_probabilistic_selector_applies_proposal_calibration():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the DFlash2 selector kernel")

    device = torch.device("cuda")
    top_k = 4
    scores = torch.tensor(
        [[[[4.0, 3.0, 2.0, 1.0]] * top_k]],
        dtype=torch.float32,
        device=device,
    )
    candidates = torch.arange(top_k, dtype=torch.int64, device=device).view(1, 1, top_k)
    realized = torch.full(
        (1, 1, top_k), float("nan"), dtype=torch.float32, device=device
    )
    tokens = torch.full((1,), -1, dtype=torch.int64, device=device)

    _selector_walk_kernel[(1,)](
        scores,
        candidates,
        torch.tensor([10], dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.float32, device=device),
        torch.tensor([123], dtype=torch.int64, device=device),
        tokens,
        realized,
        torch.empty(1, dtype=torch.int32, device=device),
        num_steps=1,
        walk_steps=1,
        top_k=top_k,
        BLOCK_K=top_k,
        SAMPLE_PROBABILISTIC=True,
        USE_FP64=False,
        PROPOSAL_TEMPERATURE_SCALE=0.8,
        PROPOSAL_TOP_P=0.8,
        num_warps=1,
    )

    expected = scores[0, 0, 0] / 0.8
    torch.testing.assert_close(realized[0, 0, :2], expected[:2])
    assert torch.isneginf(realized[0, 0, 2:]).all()


def test_probabilistic_cache_respects_column_stride():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Gumbel kernel")

    device = torch.device("cuda")
    vocab_size = 17
    padded_vocab_size = 23
    logits = torch.arange(vocab_size, dtype=torch.float32, device=device)[None]
    storage = torch.full(
        (1, 2, padded_vocab_size),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    cache = storage[:, :, :vocab_size]

    gumbel_sample(
        logits,
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
        temperature=torch.tensor([1.0], dtype=torch.float32, device=device),
        seed=torch.tensor([123], dtype=torch.int64, device=device),
        pos=torch.tensor([7], dtype=torch.int64, device=device),
        apply_temperature=True,
        is_drafting=True,
        output_processed_logits=cache,
        output_processed_logits_col=torch.tensor(1, device=device),
    )

    assert torch.isnan(cache[0, 0]).all()
    torch.testing.assert_close(cache[0, 1], logits[0])
    assert torch.isnan(storage[:, :, vocab_size:]).all()


def test_probabilistic_cache_keeps_ids_and_scores_in_request_slot_order(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the DFlash2 cache kernel")

    device = torch.device("cuda")
    dense_cache = torch.zeros((2, 7, 31), dtype=torch.float32, device=device)
    _stub_base(monkeypatch, dense_cache)
    monkeypatch.setattr(envs, "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION", True)
    speculator = DFlash2Speculator(None, device)
    speculator.sample_idx_mapping = torch.tensor(
        [1] * 7 + [0] * 7,
        dtype=torch.int32,
        device=device,
    )
    candidate_ids = torch.stack(
        (
            torch.arange(16, dtype=torch.int64, device=device).repeat(7, 1),
            torch.arange(15, 31, dtype=torch.int64, device=device).repeat(7, 1),
        )
    )
    selector_scores = torch.arange(
        2 * 7 * 16,
        dtype=torch.float32,
        device=device,
    ).view(2, 7, 16)
    speculator._selector_scores.copy_(selector_scores)

    speculator._cache_draft_logits(candidate_ids, num_sample=14)
    sparse_logits = speculator.get_sparse_draft_logits()
    assert sparse_logits is not None
    cached_ids, cached_scores = sparse_logits

    assert torch.equal(cached_ids[1], candidate_ids[0])
    assert torch.equal(cached_ids[0], candidate_ids[1])
    assert torch.equal(cached_scores[1], selector_scores[0])
    assert torch.equal(cached_scores[0], selector_scores[1])
    assert torch.equal(dense_cache[1].gather(1, candidate_ids[0]), selector_scores[0])
    assert torch.equal(dense_cache[0].gather(1, candidate_ids[1]), selector_scores[1])


def test_dflash2_selector_contract_dispatches_to_mrv2(monkeypatch):
    monkeypatch.setattr(DFlash2Speculator, "__init__", lambda self, *_args: None)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(dflash_config={"selector_top_k": 16})
            ),
        )
    )
    assert isinstance(init_speculator(config, torch.device("cpu")), DFlash2Speculator)


def test_dflash_without_selector_stays_on_official_mrv2_speculator(monkeypatch):
    monkeypatch.setattr(DFlashSpeculator, "__init__", lambda self, *_args: None)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(dflash_config={})
            ),
        )
    )
    assert isinstance(init_speculator(config, torch.device("cpu")), DFlashSpeculator)


@pytest.mark.parametrize(
    ("method", "selector_top_k", "expected"),
    [
        ("dflash", 16, True),
        ("dflash", 0, False),
        ("dflash_ddtree", 16, False),
        ("mtp", 16, False),
        ("eagle3", 16, False),
    ],
)
def test_fused_gdn_verify_config_uses_selector_engine_contract(
    method: str,
    selector_top_k: int,
    expected: bool,
):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    dflash_config={"selector_top_k": selector_top_k}
                )
            ),
        )
    )

    assert _is_dflash2_spec_config(config) is expected


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("dflash", False),
        ("dflash_ddtree", True),
        ("eagle3", True),
        ("mtp", True),
    ],
)
def test_only_mrv2_dflash_skips_eagle_prefix_block_drop(method, expected):
    config = SimpleNamespace(
        method=method,
        use_eagle=lambda: True,
        use_dflash=lambda: method == "dflash",
    )
    assert SpeculativeConfig.use_eagle_kv_cache(config) is expected


@pytest.mark.parametrize("method", ["eagle3", "mtp"])
def test_non_dflash_speculators_keep_eagle_dispatch(monkeypatch, method):
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

    monkeypatch.setattr(EagleSpeculator, "__init__", lambda self, *_args: None)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            use_eagle=lambda: True,
        )
    )
    assert isinstance(init_speculator(config, torch.device("cpu")), EagleSpeculator)


def test_top_level_noncausal_override_wins_over_sliding_layer_default():
    config = SimpleNamespace(
        is_causal=False,
        dflash_config={},
        layer_types=["sliding_attention"],
    )
    assert _dflash_layer_causal(config, 0) is False


def test_aux_hidden_states_follow_loaded_draft_projection_dtype():
    fc = torch.nn.Linear(10, 2, bias=False, dtype=torch.float16)
    fc.input_size = 10
    model = SimpleNamespace(use_aux_hidden_state=True, fc=fc)
    outer = SimpleNamespace(model=model)
    hidden_states = torch.randn(3, 10, dtype=torch.float32)

    output = DFlashQwen3ForCausalLM.combine_hidden_states(outer, hidden_states)

    assert output.dtype is torch.float16


@pytest.mark.parametrize("mamba_page_size_padded", [None, 16 * 512])
def test_fp16_draft_cache_grows_padded_fp8_hybrid_pages(
    mamba_page_size_padded: int | None,
):
    block_size = 16
    target_page_size = 16 * 512
    specs = {
        "target.attn": FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.float8_e5m2,
        ),
        "target.mamba": MambaSpec(
            block_size=block_size,
            shapes=((target_page_size,),),
            dtypes=(torch.uint8,),
            page_size_padded=mamba_page_size_padded,
        ),
        "draft.attn": FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.float16,
        ),
    }

    unified = unify_kv_cache_spec_page_size(specs)

    expected_page_size = specs["draft.attn"].page_size_bytes
    assert {spec.page_size_bytes for spec in unified.values()} == {expected_page_size}
    assert unified["target.attn"].block_size == 2 * block_size
    assert unified["target.mamba"].block_size == 2 * block_size
    assert unified["target.mamba"].page_size_padded == expected_page_size


def test_flashinfer_topk_is_capability_gated_on_sm70(monkeypatch):
    dflash2_model._flashinfer_topk.cache_clear()
    monkeypatch.setattr(dflash2_model.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        dflash2_model.current_platform,
        "has_device_capability",
        lambda capability: capability <= 70,
    )
    monkeypatch.setattr(dflash2_model, "has_flashinfer", lambda: True)
    assert dflash2_model._flashinfer_topk() is None
    dflash2_model._flashinfer_topk.cache_clear()


def test_target_topk_uses_reranked_local_candidates_without_dense_logits(monkeypatch):
    dense_apply = Mock(side_effect=AssertionError("dense logits must not run"))
    values = torch.linspace(5.0, -5.0, 20, dtype=torch.float16).reshape(1, 20)
    ids = torch.arange(100, 120, dtype=torch.int64).reshape(1, 20)
    lm_head = SimpleNamespace(
        quant_method=SimpleNamespace(apply=dense_apply),
        weight=torch.empty((62080, 1), dtype=torch.float16),
        maybe_get_sm70_dflash2_top20=lambda hidden, top_k, bias: (values, ids),
    )
    processor = LogitsProcessor(vocab_size=248320, scale=0.5, soft_cap=3.0)
    monkeypatch.setattr(
        logits_processor_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    actual_ids, actual_values = processor.get_topk_tokens_and_logits(
        lm_head,
        torch.empty((1, 1), dtype=torch.float16),
        20,
    )

    expected_values = (torch.tanh(values / 3.0) * 3.0 * 0.5).float()
    assert torch.equal(actual_ids, ids)
    torch.testing.assert_close(actual_values, expected_values)
    dense_apply.assert_not_called()


def test_draft_candidates_use_reranked_lm_head_without_dense_logits(monkeypatch):
    dense_apply = Mock(side_effect=AssertionError("dense logits must not run"))
    values = torch.linspace(4.0, -4.0, 16, dtype=torch.float16).reshape(1, 16)
    ids = torch.arange(200, 216, dtype=torch.int64).reshape(1, 16)
    candidate_path = Mock(return_value=(values, ids))
    lm_head = SimpleNamespace(
        quant_method=UnquantizedEmbeddingMethod(),
        maybe_get_sm70_dflash2_top20=candidate_path,
    )
    lm_head.quant_method.apply = dense_apply
    model = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=16))
    dflash2 = SimpleNamespace(
        lm_head=lm_head,
        model=model,
        output_multiplier=1.0,
        final_logit_softcapping=None,
    )
    hidden_states = torch.empty((1, 8), dtype=torch.float16)
    monkeypatch.setattr(
        dflash2_model,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    actual_ids, actual_values = DFlash2Qwen3ForCausalLM.compute_candidates(
        dflash2, hidden_states
    )

    candidate_path.assert_called_once_with(hidden_states, 16)
    dense_apply.assert_not_called()
    assert torch.equal(actual_ids, ids)
    assert torch.equal(actual_values, values.float())


def test_lm_head_candidate_interface_falls_back_when_rerank_is_disabled(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SM70_DFLASH2_QPN8_RERANK", False)
    layer = SimpleNamespace()

    assert (
        VocabParallelEmbedding.maybe_get_sm70_dflash2_top20(
            layer,
            torch.empty((1, 8)),
            20,
        )
        is None
    )


@pytest.mark.parametrize("selector_k", [16, 20])
@pytest.mark.parametrize("num_rows", [1, 7, 8])
def test_qpn8_rerank_output_buffers_are_contiguous(selector_k, num_rows):
    layer = SimpleNamespace()
    for top_k in (16, 20):
        setattr(
            layer,
            f"_sm70_dflash2_rerank_values_{top_k}",
            torch.empty((8, top_k), dtype=torch.float16),
        )
        setattr(
            layer,
            f"_sm70_dflash2_rerank_positions_{top_k}",
            torch.empty((8, top_k), dtype=torch.int64),
        )
        setattr(
            layer,
            f"_sm70_dflash2_rerank_ids_{top_k}",
            torch.empty((8, top_k), dtype=torch.int64),
        )

    buffers = _sm70_dflash2_rerank_output_buffers(layer, num_rows, selector_k)

    assert all(buffer.shape == (num_rows, selector_k) for buffer in buffers)
    assert all(buffer.is_contiguous() for buffer in buffers)


def test_qpn8_rerank_restores_dense_vocab_tie_order():
    vocab_start = 96
    candidate_ids = torch.tensor(
        [[9, 1, 7, 3, 12, 4], [18, 2, 15, 5, 11, 8]], dtype=torch.int64
    )
    candidate_logits = torch.tensor(
        [[5, 5, 5, 5, 4, 3], [7, 7, 7, 7, 6, 5]], dtype=torch.float16
    )
    sparse_logits = torch.empty((2, 24), dtype=torch.float16)
    actual_values = torch.empty((2, 4), dtype=torch.float16)
    actual_ids = torch.empty((2, 4), dtype=torch.int64)

    _sm70_dflash2_dense_order_topk(
        sparse_logits,
        candidate_ids,
        candidate_logits,
        actual_values,
        actual_ids,
        4,
        vocab_start,
    )

    reference = torch.full_like(sparse_logits, -float("inf"))
    reference.scatter_(1, candidate_ids, candidate_logits)
    expected_values, expected_ids = torch.topk(reference, 4, dim=-1, sorted=True)
    assert torch.equal(actual_values, expected_values)
    assert torch.equal(actual_ids, expected_ids + vocab_start)


def test_qpn8_candidate_order_requires_explicit_experimental_opt_in(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER", False)
    monkeypatch.setattr(
        envs,
        "VLLM_SM70_DFLASH2_QPN8_ALLOW_CANDIDATE_ORDER",
        False,
    )
    assert _sm70_dflash2_use_dense_order()

    monkeypatch.setattr(
        envs,
        "VLLM_SM70_DFLASH2_QPN8_ALLOW_CANDIDATE_ORDER",
        True,
    )
    assert not _sm70_dflash2_use_dense_order()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("selector_k", [16, 20])
@pytest.mark.parametrize("num_rows", [1, 7, 8])
def test_sm70_f16_rerank_topk_matches_composite_key_contract(
    selector_k: int, num_rows: int
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("the exact rerank top-k kernel is SM70-only")
    required_ops = (
        "sm70_f16_rerank_keys_out",
        "sm70_f16_rerank_topk_out",
    )
    if any(not hasattr(torch.ops._C, name) for name in required_ops):
        pytest.skip("the exact rerank top-k operators are not built")

    from vllm import _sm70_ops

    generator = torch.Generator().manual_seed(20260824)
    local_vocab = 62080
    candidate_ids = torch.stack(
        [torch.randperm(local_vocab, generator=generator)[:64] for _ in range(num_rows)]
    ).to(device="cuda", dtype=torch.int64)
    torch.manual_seed(20260824)
    candidate_logits = torch.randn(
        (num_rows, 64), dtype=torch.float16, device="cuda"
    ).clamp_(-3, 3)
    # Force more equal maxima than either requested top-k so ID-ascending tie
    # precedence is part of every comparison, not just an incidental edge.
    candidate_logits[:, :24] = 4
    if num_rows > 1:
        candidate_logits[1, 24:28] = torch.tensor(
            [-float("inf"), -0.0, 0.0, float("inf")],
            dtype=torch.float16,
            device="cuda",
        )

    actual_values = torch.empty(
        (num_rows, selector_k), dtype=torch.float16, device="cuda"
    )
    actual_ids = torch.empty((num_rows, selector_k), dtype=torch.int64, device="cuda")
    vocab_start = 186240
    _sm70_ops.sm70_f16_rerank_topk_out(
        actual_values,
        actual_ids,
        candidate_logits,
        candidate_ids,
        vocab_start,
    )

    keys = torch.empty_like(candidate_ids)
    _sm70_ops.sm70_f16_rerank_keys_out(keys, candidate_logits, candidate_ids)
    _, positions = torch.topk(keys, selector_k, dim=-1, sorted=True)
    expected_values = candidate_logits.gather(1, positions)
    expected_ids = candidate_ids.gather(1, positions).add_(vocab_start)
    torch.accelerator.synchronize()

    assert torch.equal(
        actual_values.view(torch.int16), expected_values.view(torch.int16)
    )
    assert torch.equal(actual_ids, expected_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_rows", [1, 7, 8])
def test_sm70_dflash2_exact_rerank_matches_gathered_bmm(num_rows):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("the exact rerank kernel is SM70-only")
    if not hasattr(torch.ops._C, "sm70_f16_indexed_rerank_out"):
        pytest.skip("the exact TurboMind rerank op is not built")

    from vllm import _sm70_ops

    torch.manual_seed(20260824)
    device = torch.device("cuda")
    candidates, vocab, hidden_size = 64, 256, 5120
    hidden = torch.randn((num_rows, hidden_size), dtype=torch.float16, device=device)
    weight = torch.randn((vocab, hidden_size), dtype=torch.float16, device=device)
    candidate_ids = torch.randint(
        0, vocab, (num_rows, candidates), dtype=torch.int64, device=device
    )
    actual = torch.empty((num_rows, candidates), dtype=torch.float16, device=device)
    selected_raw = torch.empty(
        (num_rows * candidates, hidden_size), dtype=torch.float16, device=device
    )
    selected_packed = torch.empty_like(selected_raw)
    expanded = torch.empty(
        (num_rows, num_rows * candidates), dtype=torch.float16, device=device
    )
    partials = torch.empty(
        (num_rows, num_rows * candidates), dtype=torch.float32, device=device
    )
    barriers = torch.zeros(64, dtype=torch.int32, device=device)

    _sm70_ops.sm70_f16_indexed_rerank_out(
        actual,
        hidden,
        weight,
        candidate_ids,
        selected_raw,
        selected_packed,
        expanded,
        partials,
        barriers,
        128,
        10,
    )
    gathered = weight.index_select(0, candidate_ids.reshape(-1))
    expected = torch.bmm(
        gathered.view(num_rows, candidates, hidden_size),
        hidden.unsqueeze(-1),
    ).squeeze(-1)

    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.002)


@pytest.mark.parametrize(
    ("capability", "num_steps", "expected"),
    [((7, 0), 7, True), ((7, 0), 1, False), ((8, 0), 7, False)],
)
def test_selector_tail_split_is_sm70_only(monkeypatch, capability, num_steps, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    assert _requires_sm70_tail(torch.device("cuda:0"), num_steps) is expected


def test_noncausal_draft_cannot_enter_flash_v100_small_query_fast_path():
    impl = SimpleNamespace(
        use_flash_v100_decode=True,
        smallq_decode_max_query_len=8,
        smallq_decode_max_model_len=4096,
    )
    metadata = SimpleNamespace(
        causal=False,
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        max_model_len=128,
    )
    assert not FlashAttnV100Impl._small_query_decode_enabled(impl, metadata)


def test_dflash_attention_builders_receive_the_draft_model_config(monkeypatch):
    def fake_replace(config, **updates):
        return SimpleNamespace(source=config, **updates)

    monkeypatch.setattr(dflash_speculator, "replace", fake_replace)
    target_model_config = object()
    draft_model_config = object()
    attention_config = object()
    target_parallel_config = object()
    draft_parallel_config = object()
    speculator = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=target_model_config,
            attention_config=attention_config,
            parallel_config=target_parallel_config,
        ),
        speculative_config=SimpleNamespace(draft_parallel_config=draft_parallel_config),
        draft_model_config=draft_model_config,
        requires_non_causal=True,
    )

    config = DFlashSpeculator.attn_vllm_config.fget(speculator)

    assert config.model_config is draft_model_config
    assert config.parallel_config is draft_parallel_config
    assert config.attention_config.source is attention_config
    assert config.attention_config.use_non_causal is True


def test_dflash_disables_aot_schedule_only_for_sliding_draft_groups(monkeypatch):
    sliding_builder = SimpleNamespace(
        aot_schedule=True,
        kv_cache_spec=SimpleNamespace(sliding_window=2048),
    )
    full_builder = SimpleNamespace(
        aot_schedule=True,
        kv_cache_spec=SimpleNamespace(sliding_window=None),
    )
    groups = [
        [SimpleNamespace(get_metadata_builder=lambda: sliding_builder)],
        [SimpleNamespace(get_metadata_builder=lambda: full_builder)],
    ]

    def fake_base_set_attn(self, *_args):
        self.attn_groups = groups

    monkeypatch.setattr(
        dflash_speculator.DraftModelSpeculator,
        "set_attn",
        fake_base_set_attn,
    )
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.max_num_tokens = 8
    speculator.device = torch.device("cpu")
    speculator.requires_non_causal = True
    speculator.model = SimpleNamespace()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])

    speculator.set_attn(None, kv_cache_config, None, None, [])

    assert sliding_builder.aot_schedule is False
    assert full_builder.aot_schedule is True


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (np.array([True], dtype=np.bool_), True),
        (np.array([True, True], dtype=np.bool_), True),
        (np.array([True, False], dtype=np.bool_), False),
        (np.array([False], dtype=np.bool_), False),
        (None, False),
    ],
)
def test_dflash_context_only_prefill_requires_every_live_request(mask, expected):
    batch = SimpleNamespace(
        num_reqs=1 if mask is None else mask.size,
        is_incomplete_prefilling_np=mask,
    )
    assert dflash_speculator._is_context_only_prefill(batch) is expected


def test_dflash_intermediate_prefill_materializes_context_without_query(monkeypatch):
    prepare_inputs = Mock()
    monkeypatch.setattr(dflash_speculator, "prepare_dflash_inputs", prepare_inputs)

    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.max_model_len = 256
    speculator.num_query_per_req = 8
    speculator.num_speculative_steps = 7
    speculator.draft_block = 7
    speculator.max_num_reqs = 1
    speculator.max_num_tokens = 8
    speculator.hidden_states = torch.zeros(8, 3)
    speculator.draft_tokens = torch.zeros(1, 7, dtype=torch.int64)
    speculator.input_buffers = object()
    speculator.context_positions = torch.zeros(8, dtype=torch.int64)
    speculator.sample_indices = torch.zeros(7, dtype=torch.int64)
    speculator.sample_pos = torch.zeros(7, dtype=torch.int64)
    speculator.sample_idx_mapping = torch.zeros(7, dtype=torch.int32)
    speculator.temperature = torch.ones(1)
    speculator.seeds = torch.zeros(1, dtype=torch.int64)
    speculator.parallel_drafting_token_id = 0
    speculator.sample_from_anchor = False
    speculator.draft_kv_cache_group_id = 0
    speculator.draft_kv_cache_group_ids = [0]
    speculator.block_tables = SimpleNamespace(
        slot_mappings=[torch.zeros(8, dtype=torch.int64)],
        input_block_tables=[torch.zeros(1, 8, dtype=torch.int32)],
        kernel_block_sizes=[1],
        cp_rank=0,
        cp_size=1,
        cp_interleave=1,
    )
    speculator._context_slot_mappings = torch.zeros(1, 8, dtype=torch.int64)
    speculator._query_slot_mappings = torch.zeros(1, 8, dtype=torch.int64)
    speculator._layer_group_idx = None
    speculator._context_only_prefill_logged = False
    speculator._prepare_ngram_assist = Mock(
        side_effect=AssertionError("ngram lookup must not run mid-prefill")
    )
    speculator.model = SimpleNamespace(
        precompute_and_store_context_kv=Mock(),
    )

    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=4,
        seq_lens_cpu_upper_bound=torch.tensor([4], dtype=torch.int32),
        is_incomplete_prefilling_np=np.array([True], dtype=np.bool_),
    )
    output = speculator.propose(
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.arange(12, dtype=torch.float32).view(4, 3),
        aux_hidden_states=None,
        num_sampled=torch.zeros(1, dtype=torch.int32),
        num_rejected=torch.zeros(1, dtype=torch.int32),
        last_sampled=torch.zeros(1, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(1, dtype=torch.int64),
        temperature=torch.ones(1),
        seeds=torch.zeros(1, dtype=torch.int64),
    )

    prepare_inputs.assert_called_once()
    speculator.model.precompute_and_store_context_kv.assert_called_once()
    torch.testing.assert_close(
        speculator.hidden_states[:4],
        torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    assert output.tolist() == [[-1] * 7]


def test_noncausal_dflash_capture_binds_paged_prefix_attention(monkeypatch):
    monkeypatch.setattr(flash_v100, "_is_cuda_graph_capturing", lambda _query: True)
    output = torch.empty(8, 1, 1)
    paged_prefix = Mock(return_value=output)
    impl = SimpleNamespace(
        _supports_flash_v100_path=lambda: True,
        _validate_dflash_attention_contract=lambda _layer, _metadata: None,
        _layer_debug_info=lambda _layer: {
            "layer_name": "draft",
            "is_dflash_draft_attn": True,
        },
        use_triton_prefill=False,
        use_decode_scalar_paged=True,
        use_decode_paged_prefill=False,
        use_flash_v100_prefill_paged=True,
        _small_query_decode_enabled=lambda _metadata: False,
        _flash_v100_prefill_with_prefix=paged_prefix,
    )
    metadata = SimpleNamespace(
        max_query_len=8,
        max_seq_len=1024,
        num_actual_tokens=8,
        causal=False,
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 8], dtype=torch.int32),
        # Capture-time CPU metadata looks like no-prefix prefill, while the
        # persistent device metadata is updated before replay.
        seq_lens=torch.tensor([17], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
    )
    query = torch.empty(8, 1, 1)
    layer = SimpleNamespace(is_dflash_draft_attn=True)

    result = FlashAttnV100Impl.forward(
        impl,
        layer,
        query,
        query,
        query,
        torch.empty(1),
        metadata,
        output,
    )

    assert result is output
    paged_prefix.assert_called_once()


def test_draft_attention_causality_is_resolved_per_kv_group(monkeypatch):
    observed_causality = []

    class CapturedCommonAttentionMetadata:
        def __init__(self, **kwargs):
            observed_causality.append(kwargs["causal"])

    monkeypatch.setattr(
        attn_utils,
        "CommonAttentionMetadata",
        CapturedCommonAttentionMetadata,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(), SimpleNamespace()]
    )
    attn_utils.build_attn_metadata(
        attn_groups=[[], []],
        num_reqs=1,
        num_tokens=8,
        query_start_loc_gpu=torch.tensor([0, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 8], dtype=torch.int32),
        max_query_len=8,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        max_seq_len=8,
        block_tables=[
            torch.zeros((1, 1), dtype=torch.int32),
            torch.zeros((1, 1), dtype=torch.int32),
        ],
        slot_mappings=torch.zeros((2, 8), dtype=torch.int64),
        kv_cache_config=kv_cache_config,
        causal={0: False, 1: True},
    )

    assert observed_causality == [False, True]


def test_sm70_rmsnorm_keeps_bf16_residual_in_fp32():
    norm = DFlashSM70RMSNorm(8, 1e-6, torch.float16)
    norm.weight.data.copy_(torch.linspace(0.5, 1.5, 8, dtype=torch.float16))
    x = torch.full((2, 8), 300.0, dtype=torch.float16)
    residual = torch.full((2, 8), 70000.0, dtype=torch.float32)

    output, residual_output = norm(x, residual)
    expected_residual = (
        (x.float() * DFLASH_SM70_WIDE_OUTPUT_SCALE + residual)
        .to(torch.bfloat16)
        .float()
    )

    assert residual_output.dtype is torch.float32
    assert torch.isfinite(output).all()
    assert residual_output.max() > torch.finfo(torch.float16).max
    torch.testing.assert_close(residual_output, expected_residual, rtol=0, atol=0)


def test_sm70_swiglu_uses_power_of_two_row_scale():
    gate_up = (
        torch.tensor(
            [
                [2000.0, -1500.0, 1000.0, 800.0, 1800.0, 900.0, -700.0, 600.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ],
            dtype=torch.float16,
        )
        / DFLASH_SM70_GATE_UP_INPUT_SCALE
    )
    transported, row_scales = dflash_silu_and_mul_sm70(gate_up)

    assert torch.all(row_scales >= 1)
    torch.testing.assert_close(
        torch.log2(row_scales),
        torch.log2(row_scales).round(),
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(transported).all()

    down = torch.ones((2, 8), dtype=torch.float16)
    restored = dflash_scale_output_sm70(down, row_scales)
    expected = (
        down.float() * (row_scales[:, None] / DFLASH_SM70_WIDE_OUTPUT_SCALE)
    ).half()
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
