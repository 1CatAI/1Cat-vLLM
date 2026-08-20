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
from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
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
