# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.models.qwen3_dflash2 as dflash2_model
import vllm.v1.attention.backends.flash_attn_v100 as flash_v100
import vllm.v1.worker.gpu.attn_utils as attn_utils
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_speculator
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    _maybe_sm70_dflash2_lm_head_top20,
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
    _dflash_layer_causal,
)
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3ForCausalLM,
    _grouped_conv,
    _score_edges,
)
from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    DFlash2Speculator,
    _requires_sm70_tail,
    _selector_walk_kernel,
)


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


def test_selector_initializes_probabilistic_cache_to_negative_infinity(monkeypatch):
    allocated = torch.zeros((2, 7, 31), dtype=torch.float32)
    _stub_base(monkeypatch, allocated)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    assert speculator.draft_logits is allocated
    assert torch.isneginf(speculator.draft_logits).all()


def test_selector_uses_checkpoint_top16_and_fp32_proposal_cache(monkeypatch):
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)
    assert speculator.selector_top_k == 16
    assert dtype is torch.float32
    assert fill == float("-inf")


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


def test_dflash2_architecture_dispatches_to_mrv2(monkeypatch):
    monkeypatch.setattr(DFlash2Speculator, "__init__", lambda self, *_args: None)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
        )
    )
    assert isinstance(init_speculator(config, torch.device("cpu")), DFlash2Speculator)


def test_dflash1_architecture_stays_on_official_mrv2_speculator(monkeypatch):
    monkeypatch.setattr(DFlashSpeculator, "__init__", lambda self, *_args: None)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            draft_model_config=SimpleNamespace(architectures=["DFlashDraftModel"]),
        )
    )
    assert isinstance(init_speculator(config, torch.device("cpu")), DFlashSpeculator)


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


def test_compute_candidates_trims_fused_local_top20_to_checkpoint_top16(monkeypatch):
    quant_method = UnquantizedEmbeddingMethod()
    dense_apply = Mock(side_effect=AssertionError("dense logits must not run"))
    monkeypatch.setattr(quant_method, "apply", dense_apply)
    values = torch.arange(40, dtype=torch.float32).reshape(2, 20)
    ids = torch.arange(100, 140, dtype=torch.int64).reshape(2, 20)
    lm_head = SimpleNamespace(
        quant_method=quant_method,
        maybe_get_sm70_dflash2_top20=lambda hidden, selector_k: (values, ids),
    )
    model = SimpleNamespace(
        lm_head=lm_head,
        model=SimpleNamespace(candidate_selector=SimpleNamespace(top_k=16)),
        output_multiplier=1.0,
        final_logit_softcapping=None,
    )
    monkeypatch.setattr(
        dflash2_model, "get_tensor_model_parallel_world_size", lambda: 1
    )

    actual_ids, actual_values = DFlash2Qwen3ForCausalLM.compute_candidates(
        model, torch.empty((2, 8))
    )

    expected_values, positions = torch.topk(values, 16, dim=-1)
    assert torch.equal(actual_ids, ids.gather(-1, positions))
    assert torch.equal(actual_values, expected_values)
    dense_apply.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 4, 5, 7, 8])
def test_sm70_fused_lm_head_top20_matches_tp4_global_top16(rows: int):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("the fused selector is SM70-only")
    if not hasattr(torch.ops._C, "sm70_f16_lm_head_top20_tc_workspace_out"):
        pytest.skip("the SM70 TurboMind top-20 op is not built")

    from vllm import _sm70_ops

    torch.manual_seed(44)
    device = torch.device("cuda")
    hidden_size, tp_size, local_vocab = 64, 4, 256
    selector_k, local_k, final_padding = 16, 20, 3
    hidden = torch.zeros(rows, hidden_size, dtype=torch.float16, device=device)
    hidden[torch.arange(rows), torch.arange(rows)] = 1
    weight = 0.01 * torch.randn(
        tp_size * local_vocab, hidden_size, dtype=torch.float16, device=device
    )
    valid_vocab = tp_size * local_vocab - final_padding
    for row in range(rows):
        controlled_ids = (
            torch.arange(selector_k, device=device) * 61 + row * 7
        ) % valid_vocab
        weight[controlled_ids, row] = torch.linspace(
            64, 49, selector_k, dtype=torch.float16, device=device
        )
        weight[controlled_ids[9], row] = weight[controlled_ids[8], row]
    dense_logits = torch.mm(hidden, weight.t())
    dense_logits[:, -final_padding:] = -float("inf")
    reference_local_values = []
    reference_local_ids = []
    for rank in range(tp_size):
        values, ids = torch.topk(
            dense_logits[:, rank * local_vocab : (rank + 1) * local_vocab],
            selector_k,
            dim=-1,
        )
        reference_local_values.append(values)
        reference_local_ids.append(ids + rank * local_vocab)
    reference_candidates = torch.cat(reference_local_values, dim=-1)
    reference_values, reference_positions = torch.topk(
        reference_candidates, selector_k, dim=-1
    )
    reference_ids = torch.cat(reference_local_ids, dim=-1).gather(
        -1, reference_positions
    )

    local_values = []
    local_ids = []
    prepared_weights = []
    for rank in range(tp_size):
        local_weight = weight[
            rank * local_vocab : (rank + 1) * local_vocab
        ].contiguous()
        prepared_weight, prepared_meta = _sm70_ops.sm70_f16_prepare(local_weight)
        # Keep source and prepared tensors alive so the test cannot recycle a
        # TensorImpl address through the process-wide prepared-weight cache.
        prepared_weights.append((local_weight, prepared_weight))
        values = torch.empty((rows, local_k), dtype=torch.float32, device=device)
        ids = torch.empty((rows, local_k), dtype=torch.int64, device=device)
        partial_values = torch.empty(
            (rows, 1, local_k), dtype=torch.float32, device=device
        )
        partial_ids = torch.empty((rows, 1, local_k), dtype=torch.int64, device=device)
        _sm70_ops.sm70_f16_lm_head_top20_tc_workspace_out(
            values,
            ids,
            partial_values,
            partial_ids,
            hidden,
            prepared_weight,
            int(prepared_meta[0].item()),
            rank * local_vocab,
            final_padding if rank == tp_size - 1 else 0,
        )
        sparse_logits = torch.full(
            (rows, local_vocab),
            -float("inf"),
            dtype=torch.float16,
            device=device,
        )
        sparse_logits.scatter_(
            1,
            ids - rank * local_vocab,
            values.to(torch.float16),
        )
        values, local_positions = torch.topk(sparse_logits, selector_k, dim=-1)
        local_values.append(values)
        local_ids.append(local_positions + rank * local_vocab)

    gathered_values = torch.cat(local_values, dim=-1)
    gathered_ids = torch.cat(local_ids, dim=-1)
    actual_values, positions = torch.topk(gathered_values, selector_k, dim=-1)
    actual_ids = gathered_ids.gather(-1, positions)
    torch.accelerator.synchronize()

    assert torch.equal(actual_ids, reference_ids)
    assert torch.equal(actual_values, reference_values)
    assert int(actual_ids.max()) < tp_size * local_vocab - final_padding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm70_fused_top20_restores_dense_topk_tie_order(monkeypatch):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("the fused selector is SM70-only")
    if not hasattr(torch.ops._C, "sm70_f16_lm_head_top20_tc_workspace_out"):
        pytest.skip("the SM70 TurboMind top-20 op is not built")

    device = torch.device("cuda")
    rows, hidden_size, local_vocab = 2, 16, 32
    vocab_start = 96
    candidate_ids = torch.tensor(
        [
            list(range(vocab_start + 19, vocab_start - 1, -1)),
            list(range(vocab_start + 3, vocab_start + 23)),
        ],
        dtype=torch.int64,
        device=device,
    )
    candidate_values = torch.tensor(
        [
            [20, 19, 18, 17, 16, 15, 14, 13, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],
            [
                30,
                29,
                28,
                27,
                26,
                25,
                24,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                16,
                15,
                14,
                13,
                12,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )
    layer = SimpleNamespace(
        weight=torch.empty(
            (local_vocab, hidden_size), dtype=torch.float16, device=device
        ),
        shard_indices=SimpleNamespace(
            org_vocab_start_index=vocab_start,
            num_org_vocab_padding=0,
        ),
        _sm70_f16_prepared=True,
        _sm70_f16_tm_weight=torch.empty(
            (local_vocab, hidden_size), dtype=torch.float16, device=device
        ),
        _sm70_f16_k_ld=hidden_size,
        _sm70_dflash2_top20_values=torch.empty(
            (8, 20), dtype=torch.float32, device=device
        ),
        _sm70_dflash2_top20_ids=torch.empty((8, 20), dtype=torch.int64, device=device),
        _sm70_dflash2_top20_values_fp16=torch.empty(
            (8, 20), dtype=torch.float16, device=device
        ),
        _sm70_dflash2_sparse_logits=torch.empty(
            (8, local_vocab), dtype=torch.float16, device=device
        ),
        _sm70_dflash2_local_values=torch.empty(
            (8, 16), dtype=torch.float16, device=device
        ),
        _sm70_dflash2_local_ids=torch.empty((8, 16), dtype=torch.int64, device=device),
        _sm70_dflash2_partial_values=torch.empty(
            (8, 1, 20), dtype=torch.float32, device=device
        ),
        _sm70_dflash2_partial_ids=torch.empty(
            (8, 1, 20), dtype=torch.int64, device=device
        ),
    )

    def fake_top20(values_out, ids_out, *_args):
        values_out.copy_(candidate_values)
        ids_out.copy_(candidate_ids)

    from vllm.model_executor.layers import vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding.envs,
        "VLLM_SM70_DFLASH2_FUSED_SELECTOR",
        True,
    )
    monkeypatch.setattr(
        vocab_parallel_embedding.sm70_ops,
        "sm70_f16_lm_head_top20_tc_workspace_out",
        fake_top20,
    )

    hidden = torch.zeros((rows, hidden_size), dtype=torch.float16, device=device)
    actual_values, actual_ids = _maybe_sm70_dflash2_lm_head_top20(layer, hidden, 16)
    dense_sparse = torch.full(
        (rows, local_vocab),
        -float("inf"),
        dtype=torch.float16,
        device=device,
    )
    dense_sparse.scatter_(
        1,
        candidate_ids - vocab_start,
        candidate_values.to(torch.float16),
    )
    expected_values, expected_ids = torch.topk(dense_sparse, 16, dim=-1)
    expected_ids += vocab_start

    assert torch.equal(actual_values, expected_values)
    assert torch.equal(actual_ids, expected_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm70_fused_top20_workspace_is_cudagraph_stable(monkeypatch):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("the fused selector is SM70-only")
    if not hasattr(torch.ops._C, "sm70_f16_lm_head_top20_tc_workspace_out"):
        pytest.skip("the graph-safe SM70 TurboMind top-20 op is not built")

    from vllm import _sm70_ops
    from vllm.model_executor.layers import vocab_parallel_embedding

    torch.manual_seed(45)
    device = torch.device("cuda")
    rows, hidden_size, local_vocab, selector_k = 7, 64, 256, 16
    vocab_start = 512
    weight = torch.randn(local_vocab, hidden_size, dtype=torch.float16, device=device)
    prepared_weight, prepared_meta = _sm70_ops.sm70_f16_prepare(weight)
    layer = SimpleNamespace(
        weight=weight,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=vocab_start,
            num_org_vocab_padding=0,
        ),
        _sm70_f16_prepared=True,
        _sm70_f16_tm_weight=prepared_weight,
        _sm70_f16_k_ld=int(prepared_meta[0].item()),
        _sm70_dflash2_top20_values=torch.empty(
            (8, 20), dtype=torch.float32, device=device
        ),
        _sm70_dflash2_top20_ids=torch.empty((8, 20), dtype=torch.int64, device=device),
        _sm70_dflash2_top20_values_fp16=torch.empty(
            (8, 20), dtype=torch.float16, device=device
        ),
        _sm70_dflash2_sparse_logits=torch.empty(
            (8, local_vocab), dtype=torch.float16, device=device
        ),
        _sm70_dflash2_local_values=torch.empty(
            (8, selector_k), dtype=torch.float16, device=device
        ),
        _sm70_dflash2_local_ids=torch.empty(
            (8, selector_k), dtype=torch.int64, device=device
        ),
        _sm70_dflash2_partial_values=torch.empty(
            (8, 1, 20), dtype=torch.float32, device=device
        ),
        _sm70_dflash2_partial_ids=torch.empty(
            (8, 1, 20), dtype=torch.int64, device=device
        ),
    )
    monkeypatch.setattr(
        vocab_parallel_embedding.envs,
        "VLLM_SM70_DFLASH2_FUSED_SELECTOR",
        True,
    )
    static_hidden = torch.randn(rows, hidden_size, dtype=torch.float16, device=device)

    values = layer._sm70_dflash2_top20_values[:rows]
    ids = layer._sm70_dflash2_top20_ids[:rows]
    values_fp16 = layer._sm70_dflash2_top20_values_fp16[:rows]
    sparse_logits = layer._sm70_dflash2_sparse_logits[:rows]
    local_values = layer._sm70_dflash2_local_values[:rows]
    local_ids = layer._sm70_dflash2_local_ids[:rows]

    def run_workspace_selector() -> None:
        _sm70_ops.sm70_f16_lm_head_top20_tc_workspace_out(
            values,
            ids,
            layer._sm70_dflash2_partial_values,
            layer._sm70_dflash2_partial_ids,
            static_hidden,
            prepared_weight,
            layer._sm70_f16_k_ld,
            vocab_start,
            0,
        )
        values_fp16.copy_(values)
        ids.sub_(vocab_start)
        sparse_logits.fill_(-float("inf"))
        sparse_logits.scatter_(1, ids, values_fp16)
        torch.topk(
            sparse_logits,
            selector_k,
            dim=-1,
            sorted=True,
            out=(local_values, local_ids),
        )
        local_ids.add_(vocab_start)

    # Warm all kernels before capture, then replay with different inputs. The
    # output must follow the replay input rather than capture-time workspace.
    run_workspace_selector()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_workspace_selector()

    replay_hidden = torch.randn_like(static_hidden)
    static_hidden.copy_(replay_hidden)
    graph.replay()
    torch.accelerator.synchronize()
    dense_logits = torch.mm(replay_hidden, weight.t())
    expected_values, expected_ids = torch.topk(dense_logits, selector_k, dim=-1)
    expected_ids += vocab_start

    assert torch.equal(local_values, expected_values)
    assert torch.equal(local_ids, expected_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compiling", [False, True])
def test_sm70_fused_selector_falls_back_for_compiled_or_cudagraph(
    monkeypatch, compiling
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("the fused selector is SM70-only")

    from vllm.model_executor.layers import vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding.envs,
        "VLLM_SM70_DFLASH2_FUSED_SELECTOR",
        True,
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: not compiling
    )
    layer = SimpleNamespace(_sm70_f16_prepared=True)
    hidden = torch.empty((7, 16), dtype=torch.float16, device="cuda")

    assert _maybe_sm70_dflash2_lm_head_top20(layer, hidden, 16) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm70_fused_selector_is_not_prepared_for_flash_v100_compiled_graph(
    monkeypatch,
):
    from vllm.model_executor.layers import vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding.envs,
        "VLLM_SM70_DFLASH2_FUSED_SELECTOR",
        True,
    )
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")
    layer = SimpleNamespace(_sm70_f16_prepared=True)
    hidden = torch.empty((7, 16), dtype=torch.float16, device="cuda")

    assert _maybe_sm70_dflash2_lm_head_top20(layer, hidden, 16) is None


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
