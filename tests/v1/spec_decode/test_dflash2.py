# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.model_executor.layers.logits_processor as logits_processor_module
import vllm.model_executor.models.qwen3_dflash2 as dflash2_model
import vllm.v1.attention.backends.flash_attn_v100 as flash_v100
import vllm.v1.worker.gpu.attn_utils as attn_utils
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_speculator
from vllm import envs
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _is_dflash2_spec_config,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    _sm70_dflash2_dense_order_topk,
    _sm70_dflash2_rerank_output_buffers,
)
from vllm.model_executor.models.dflash_sm70 import (
    DFLASH_SM70_GATE_UP_INPUT_SCALE,
    DFLASH_SM70_WIDE_OUTPUT_SCALE,
    DFlashSM70RMSNorm,
    dflash_scale_output_sm70,
    dflash_silu_and_mul_sm70,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    _dflash_layer_causal,
)
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3Model,
    _grouped_conv,
    _score_edges,
)
from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
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


def test_dflash2_gdn_fastpaths_are_default_off(monkeypatch):
    names = (
        "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY",
        "VLLM_SM70_DFLASH2_QPN8_RERANK",
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


def _bare_dflash2_model() -> DFlash2Qwen3Model:
    model = DFlash2Qwen3Model.__new__(DFlash2Qwen3Model)
    torch.nn.Module.__init__(model)
    model.quant_config = None
    return model


def test_sm70_tp4_shards_only_compatible_dflash2_context_projection(monkeypatch):
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
        input_size=25600,
        output_size=5120,
        prefix="model.fc",
    )

    assert created["gather_output"] is True
    assert created["input_size"] == 25600
    assert created["output_size"] == 5120
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
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 16})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 8
        self.num_speculative_steps = 7
        self.vocab_size = 31
        self.draft_tokens = torch.empty((2, 7), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_without_proposal_logits(monkeypatch):
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    assert speculator.draft_logits is None


def test_selector_default_path_does_not_allocate_sparse_score_cache(monkeypatch):
    allocated = torch.zeros((2, 7, 31), dtype=torch.float32)
    _stub_base(monkeypatch, allocated)
    monkeypatch.setattr(envs, "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION", False)
    monkeypatch.setattr(envs, "VLLM_SPEC_DUMP_ALIGNMENT", False)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    assert speculator.draft_logits is allocated
    assert torch.isneginf(speculator.draft_logits).all()
    assert speculator.get_sparse_draft_logits() is None
    assert speculator.get_selector_alignment_shadow() is None


def test_selector_opt_in_allocates_sparse_score_cache(monkeypatch):
    allocated = torch.zeros((2, 7, 31), dtype=torch.float32)
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
    allocated = torch.zeros((2, 7, 31), dtype=torch.float32)
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
        num_warps=1,
    )

    first_index = int(tokens[0].item())
    torch.testing.assert_close(realized[0, 0], scores[0, 0, 0] / temperature[0])
    torch.testing.assert_close(
        realized[0, 1], scores[0, 1, first_index] / temperature[0]
    )


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
    speculator = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=target_model_config,
            attention_config=attention_config,
        ),
        draft_model_config=draft_model_config,
        requires_non_causal=True,
    )

    config = DFlashSpeculator.attn_vllm_config.fget(speculator)

    assert config.model_config is draft_model_config
    assert config.attention_config.source is attention_config
    assert config.attention_config.use_non_causal is True


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
