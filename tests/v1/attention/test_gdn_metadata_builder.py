# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build().

SM70 Qwen MTP keeps ordinary non-spec decode rows on the decode fastpath
while separately routing speculative verification rows. This mirrors the
0.0.3 GDN state/update semantics and avoids moving decode-only rows through
the slower prefill path during active MTP.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as qwen_gdn
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _qwen_gdn_run_recurrent_core,
    _sm70_qwen_gdn_full_forward_enabled,
    _sm70_qwen_gdn_spec_core_enabled,
    qwen_gdn_attention_core_spec_commit,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
    build_gdn_spec_decode_state_contract,
    gdn_spec_metadata_tensors,
    get_registered_gdn_spec_metadata_tensors,
    prepare_dflash2_gdn_group_metadata,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.attn_utils import compute_common_gdn_attn_metadata
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


def test_sm70_qwen_gdn_warmup_uses_bound_forward_context(monkeypatch):
    calls: list[str] = []

    class BoundGDNLayer:
        def _warmup_sm70_causal_conv1d_real_state(self) -> bool:
            calls.append("bound")
            return True

    monkeypatch.setattr(
        qwen_gdn.envs, "VLLM_SM70_AUX_KERNEL_WARMUP", True, raising=False
    )
    monkeypatch.setattr(
        qwen_gdn.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )

    assert qwen_gdn._warmup_sm70_qwen_gdn_causal_conv1d(
        {"unrelated": object(), "model.layers.0.linear_attn": BoundGDNLayer()}
    )
    assert calls == ["bound"]


def test_disabled_gdn_core_dump_avoids_cuda_runtime_query(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_DUMP_GDN_CORE_DIR", raising=False)
    monkeypatch.delenv("VLLM_SM70_DUMP_GDN_GRAPH_BUFFERS", raising=False)
    monkeypatch.delenv("VLLM_SM70_DUMP_GDN_GRAPH_DIR", raising=False)

    def fail_debug_work(*_args, **_kwargs):
        raise AssertionError("disabled debug hook performed debug work")

    cuda_module = vars(qwen_gdn.torch)["cuda"]
    monkeypatch.setattr(cuda_module, "is_current_stream_capturing", fail_debug_work)
    monkeypatch.setattr(qwen_gdn, "_sm70_gdn_graph_buffer_copy", fail_debug_work)
    qwen_gdn._sm70_dump_gdn_core_tensor(
        "input_qkv",
        "layer.0",
        torch.zeros(1),
    )


def test_mamba_hybrid_passes_seq_lens_cpu_upper_bound(monkeypatch):
    """Flash-V100 metadata must not lazily copy seq_lens back from the GPU."""

    captured_kwargs = None

    def capture_build_attn_metadata(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return {}

    monkeypatch.setattr(
        mamba_hybrid,
        "build_attn_metadata",
        capture_build_attn_metadata,
    )
    state = object.__new__(MambaHybridModelState)
    state.max_model_len = 4096
    seq_lens_cpu_upper_bound = torch.tensor([130], dtype=torch.int32)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=8,
        num_reqs_after_padding=1,
        num_tokens_after_padding=8,
        query_start_loc_np=np.array([0, 8], dtype=np.int32),
        num_scheduled_tokens=np.array([8], dtype=np.int32),
        is_prefilling_np=np.array([False]),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        seq_lens=torch.tensor([123], dtype=torch.int32),
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        dcp_local_seq_lens=None,
        prefix_anchor_lens=None,
    )
    state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty((0, 8), dtype=torch.int32),
        attn_groups=[],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        for_capture=True,
    )

    assert captured_kwargs is not None
    assert captured_kwargs["seq_lens_cpu_upper_bound"] is seq_lens_cpu_upper_bound


@pytest.fixture
def local_gdn_model(tmp_path: Path) -> str:
    model_dir = tmp_path / "gdn-test-model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(
        """
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 1024,
  "intermediate_size": 4096,
  "num_hidden_layers": 1,
  "num_attention_heads": 4,
  "num_key_value_heads": 1,
  "head_dim": 256,
  "vocab_size": 32000,
  "max_position_embeddings": 2048,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "rope_theta": 10000.0
}
""",
        encoding="utf-8",
    )
    return str(model_dir)


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=1,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode stays separate from prefill
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=1,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests stay on the decode fastpath
    "multiple_decodes_kept_decode": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=1,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    model_name: str,
    num_speculative_tokens: int = 0,
    use_full_cuda_graph: bool = False,
    mamba_cache_mode: str = "none",
    max_cudagraph_capture_size: int = 4,
    device: torch.device = DEVICE,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(model_name=model_name, block_size=BLOCK_SIZE)
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    if use_full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
        vllm_config.compilation_config.max_cudagraph_capture_size = (
            max_cudagraph_capture_size
        )
    else:
        # Keep the helper's default contract explicit. The repository-wide
        # compilation default may select FULL graphs on CUDA platforms, which
        # would pad otherwise eager metadata and make these tests route-dependent.
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        num_speculative_blocks=num_speculative_tokens,
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=device,
    )


def _clear_envs_cache() -> None:
    cache_clear = getattr(
        getattr(qwen_gdn.envs, "__getattr__", None), "cache_clear", None
    )
    if cache_clear is not None:
        cache_clear()


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
    use_common_metadata: bool = False,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        num_decode_draft_tokens_cpu = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32, device="cpu"
        )
        num_accepted_tokens = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
        kwargs["num_decode_draft_tokens_cpu"] = num_decode_draft_tokens_cpu
        kwargs["num_accepted_tokens"] = num_accepted_tokens
        if use_common_metadata:
            kwargs["common_gdn_metadata"] = compute_common_gdn_attn_metadata(
                num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                query_start_loc=common.query_start_loc,
                query_start_loc_cpu=common.query_start_loc_cpu,
                num_spec_state_tokens=builder.num_spec_state_tokens,
                legacy_mixed_decode_routing=(
                    qwen_gdn.envs.VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING
                ),
            )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


def _assert_gdn_metadata_equal(
    actual: GDNAttentionMetadata,
    expected: GDNAttentionMetadata,
) -> None:
    scalar_fields = (
        "num_prefills",
        "num_prefill_tokens",
        "num_decodes",
        "num_decode_tokens",
        "num_spec_decodes",
        "num_spec_decode_tokens",
        "num_actual_tokens",
    )
    for field in scalar_fields:
        assert getattr(actual, field) == getattr(expected, field), field

    tensor_fields = (
        "has_initial_state",
        "spec_query_start_loc",
        "non_spec_query_start_loc",
        "spec_state_indices_tensor",
        "non_spec_state_indices_tensor",
        "spec_sequence_masks",
        "spec_token_indx",
        "non_spec_token_indx",
        "num_accepted_tokens",
        "spec_state_slot_selectors",
        "chunk_indices",
        "chunk_offsets",
        "batch_ptr",
        "token_chunk_offset_ptr",
    )
    for field in tensor_fields:
        actual_tensor = getattr(actual, field)
        expected_tensor = getattr(expected, field)
        assert (actual_tensor is None) == (expected_tensor is None), field
        if actual_tensor is not None:
            assert torch.equal(actual_tensor, expected_tensor), field


def _effective_spec_initial_state_slots(
    meta: GDNAttentionMetadata,
) -> list[int]:
    assert meta.spec_state_indices_tensor is not None
    assert meta.num_accepted_tokens is not None
    state_indices = meta.spec_state_indices_tensor
    selectors = (
        meta.spec_state_slot_selectors
        if meta.spec_state_slot_selectors is not None
        else meta.num_accepted_tokens
    )
    accepted_offsets = selectors.to(torch.long) - 1
    rows = torch.arange(state_indices.shape[0], dtype=torch.long, device=DEVICE)
    return state_indices[rows, accepted_offsets].tolist()


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(
    test_case: GDNBuildTestCase,
    local_gdn_model: str,
):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(local_gdn_model, test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


@pytest.mark.parametrize(
    "test_case",
    [
        case
        for case in GDN_BUILD_TEST_CASES.values()
        if case.num_decode_draft_tokens is not None
    ],
    ids=[
        name
        for name, case in GDN_BUILD_TEST_CASES.items()
        if case.num_decode_draft_tokens is not None
    ],
)
def test_common_gdn_metadata_matches_per_group_builder(
    test_case: GDNBuildTestCase,
    local_gdn_model: str,
):
    generic_builder = _create_gdn_builder(
        local_gdn_model, test_case.num_speculative_tokens
    )
    common_builder = _create_gdn_builder(
        local_gdn_model, test_case.num_speculative_tokens
    )
    batch = BatchSpec(
        seq_lens=test_case.seq_lens,
        query_lens=test_case.query_lens,
    )

    torch.manual_seed(0)
    expected = _build(
        generic_builder,
        batch,
        test_case.num_decode_draft_tokens,
    )
    torch.manual_seed(0)
    actual = _build(
        common_builder,
        batch,
        test_case.num_decode_draft_tokens,
        use_common_metadata=True,
    )

    _assert_gdn_metadata_equal(actual, expected)


def test_mixed_decode_stays_decode_fastpath(local_gdn_model):
    """A non-spec query_len=1 row should stay on the decode fastpath even
    when another row is doing speculative verification."""
    builder = _create_gdn_builder(local_gdn_model, num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_decodes == 1
    assert meta.num_prefills == 0
    assert meta.has_initial_state is None


def test_full_cuda_graph_decode_padding_uses_pad_slot(local_gdn_model):
    builder = _create_gdn_builder(local_gdn_model, use_full_cuda_graph=True)
    batch = BatchSpec(seq_lens=[10, 11, 0, 0], query_lens=[1, 1, 0, 0])
    common = create_common_attn_metadata(
        batch,
        BLOCK_SIZE,
        DEVICE,
        arange_block_indices=True,
    )
    block_table_tensor = common.block_table_tensor.clone()
    block_table_tensor[2:, :] = 0
    common = common.replace(
        block_table_tensor=block_table_tensor,
        num_actual_tokens=4,
    )

    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert meta.num_decodes == 4
    assert meta.num_decode_tokens == 4
    assert meta.non_spec_state_indices_tensor is not None
    assert meta.non_spec_state_indices_tensor.tolist() == [
        0,
        1,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
    assert meta.non_spec_query_start_loc is not None
    assert meta.non_spec_query_start_loc.tolist() == [0, 1, 2, 2, 2]


def test_full_cuda_graph_spec_replay_tail_uses_pad_slot(local_gdn_model):
    builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=2,
        use_full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[4097], query_lens=[3])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common = common.replace(
        block_table_tensor=torch.tensor(
            [[10, 11, 12, 13, 14]],
            dtype=torch.int32,
            device=DEVICE,
        ),
        num_actual_tokens=4,
    )

    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([3], dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor([2], dtype=torch.int32, device="cpu"),
    )

    assert meta.num_spec_decodes == 1
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.tolist() == [
        [10, 11, 12],
        [PAD_SLOT_ID] * 3,
        [PAD_SLOT_ID] * 3,
        [PAD_SLOT_ID] * 3,
    ]


def test_common_gdn_metadata_matches_full_graph_padding(local_gdn_model):
    generic_builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=2,
        use_full_cuda_graph=True,
    )
    common_builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=2,
        use_full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[4097], query_lens=[3])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        block_table_tensor=torch.tensor(
            [[10, 11, 12, 13, 14]],
            dtype=torch.int32,
            device=DEVICE,
        ),
        num_actual_tokens=4,
    )
    num_accepted_tokens = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    num_decode_draft_tokens_cpu = torch.tensor([2], dtype=torch.int32, device="cpu")

    expected = generic_builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )
    common_metadata = compute_common_gdn_attn_metadata(
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        query_start_loc=common.query_start_loc,
        query_start_loc_cpu=common.query_start_loc_cpu,
        num_spec_state_tokens=common_builder.num_spec_state_tokens,
        legacy_mixed_decode_routing=(
            qwen_gdn.envs.VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING
        ),
    )
    assert common_metadata is not None
    actual = common_builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        common_gdn_metadata=common_metadata,
    )

    _assert_gdn_metadata_equal(actual, expected)


def test_prepared_dflash2_metadata_belongs_to_exact_builder(local_gdn_model):
    builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=2,
        use_full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[4097], query_lens=[3])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        block_table_tensor=torch.tensor(
            [[10, 11, 12, 13]], dtype=torch.int32, device=DEVICE
        ),
        num_actual_tokens=4,
    )
    num_accepted_tokens = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    num_decode_draft_tokens_cpu = torch.tensor([2], dtype=torch.int32, device="cpu")
    common_metadata = compute_common_gdn_attn_metadata(
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        query_start_loc=common.query_start_loc,
        query_start_loc_cpu=common.query_start_loc_cpu,
        num_spec_state_tokens=builder.num_spec_state_tokens,
        legacy_mixed_decode_routing=False,
    )
    assert common_metadata is not None
    prepared = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        common_gdn_metadata=common_metadata,
    )

    actual = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        common_gdn_metadata=common_metadata,
        prepared_dflash2_metadata=prepared,
    )
    assert actual is prepared
    cached_contract = prepared._prepared_spec_metadata_tensors
    assert cached_contract is not None

    replayed = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        common_gdn_metadata=common_metadata,
        prepared_dflash2_metadata=prepared,
    )
    assert replayed is prepared
    assert prepared._prepared_spec_metadata_tensors is cached_contract

    other_builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=2,
        use_full_cuda_graph=True,
    )
    with pytest.raises(ValueError, match="does not belong"):
        other_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            common_gdn_metadata=common_metadata,
            prepared_dflash2_metadata=prepared,
        )


def test_dflash2_fused_gdn_group_metadata_replay(
    monkeypatch,
    local_gdn_model,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the fused GDN metadata kernel")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (7, 0):
        pytest.skip("the fused GDN metadata kernel is SM70-only")

    # The accepted DFlash2 verifier keeps the legacy spec-core operator off.
    # Fused metadata must therefore own its graph-stable buffers independently.
    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_SPEC_CORE_OP", "0")
    monkeypatch.setenv("VLLM_SM70_DFLASH2_FUSED_GDN_METADATA", "1")
    monkeypatch.setenv("VLLM_SM70_DFLASH2_GDN_METADATA_SHADOW", "1")
    qwen_gdn.envs.disable_envs_cache()

    builders = [
        _create_gdn_builder(
            local_gdn_model,
            num_speculative_tokens=7,
            use_full_cuda_graph=True,
            max_cudagraph_capture_size=16,
            device=device,
        )
        for _ in range(2)
    ]
    width = builders[0].num_spec_state_tokens + 1
    assert width == 8
    query_start_loc = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    common_metadata = compute_common_gdn_attn_metadata(
        num_decode_draft_tokens_cpu=torch.tensor([7, 7], dtype=torch.int32),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        num_spec_state_tokens=7,
        legacy_mixed_decode_routing=False,
    )
    assert common_metadata is not None

    block_tables = (
        torch.arange(10, 10 + 2 * width, dtype=torch.int32, device=device).view(
            2, width
        ),
        torch.arange(100, 100 + 2 * width, dtype=torch.int32, device=device).view(
            2, width
        ),
    )
    accepted = torch.tensor([3, 5], dtype=torch.int32, device=device)
    for builder in builders:
        builder.spec_state_indices_tensor.fill_(777)

    first_result = prepare_dflash2_gdn_group_metadata(
        builders_by_group=[(0, builders[0]), (1, builders[1])],
        block_tables=block_tables,
        common_gdn_metadata=common_metadata,
        num_accepted_tokens=accepted,
        num_actual_tokens=16,
        descriptor=None,
    )
    assert first_result is not None
    prepared, descriptor = first_result
    torch.accelerator.synchronize(device)

    for group_id, builder in enumerate(builders):
        state = prepared[id(builder)].spec_state_indices_tensor
        assert state is not None
        torch.testing.assert_close(state[:2], block_tables[group_id], rtol=0, atol=0)
        torch.testing.assert_close(
            state[2:], torch.full_like(state[2:], PAD_SLOT_ID), rtol=0, atol=0
        )
    first = prepared[id(builders[0])]
    assert first.spec_query_start_loc is not None
    assert first.spec_query_start_loc.tolist() == [0, 8, 16] + [16] * 14
    assert first.num_accepted_tokens is not None
    assert first.num_accepted_tokens.tolist() == [3, 5] + [1] * 14
    assert first.spec_state_slot_selectors is not None
    assert first.spec_state_slot_selectors.tolist() == [3, 5] + [1] * 14
    assert first.spec_sequence_masks is not None
    assert first.spec_sequence_masks.tolist() == [True, True] + [False] * 14

    # Poison every persistent output and replay with new source values. This
    # catches stale graph-tail data and accidental reuse of one group's table.
    for builder in builders:
        builder.spec_state_indices_tensor.fill_(888)
    common_buffers = builders[0]._ddtree_fast_common_buffers
    assert common_buffers is not None
    common_buffers.spec_sequence_masks.fill_(True)
    common_buffers.spec_query_start_loc.fill_(-1)
    common_buffers.num_accepted_tokens.fill_(-1)
    common_buffers.spec_state_slot_selectors.fill_(-1)
    block_tables[0].add_(1000)
    block_tables[1].add_(2000)
    accepted.copy_(torch.tensor([7, 2], dtype=torch.int32, device=device))

    second_result = prepare_dflash2_gdn_group_metadata(
        builders_by_group=[(0, builders[0]), (1, builders[1])],
        block_tables=block_tables,
        common_gdn_metadata=common_metadata,
        num_accepted_tokens=accepted,
        num_actual_tokens=16,
        descriptor=descriptor,
    )
    assert second_result is not None
    replayed, replay_descriptor = second_result
    assert replay_descriptor is descriptor
    assert replayed is prepared
    torch.accelerator.synchronize(device)
    for group_id, builder in enumerate(builders):
        state = replayed[id(builder)].spec_state_indices_tensor
        assert state is not None
        torch.testing.assert_close(state[:2], block_tables[group_id], rtol=0, atol=0)
        torch.testing.assert_close(
            state[2:], torch.full_like(state[2:], PAD_SLOT_ID), rtol=0, atol=0
        )
    assert replayed[id(builders[0])].num_accepted_tokens is not None
    assert replayed[id(builders[0])].num_accepted_tokens.tolist() == [7, 2] + [1] * 14

    # The pointer tables remain valid when the graph batch shape changes, but
    # the graph-buffer views do not. Ensure the cache key rebuilds their
    # lengths instead of returning the steady-state B2 metadata.
    single_common = compute_common_gdn_attn_metadata(
        num_decode_draft_tokens_cpu=torch.tensor([7], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 8], dtype=torch.int32),
        num_spec_state_tokens=7,
        legacy_mixed_decode_routing=False,
    )
    assert single_common is not None
    single_result = prepare_dflash2_gdn_group_metadata(
        builders_by_group=[(0, builders[0]), (1, builders[1])],
        block_tables=(block_tables[0][:1], block_tables[1][:1]),
        common_gdn_metadata=single_common,
        num_accepted_tokens=accepted[:1],
        num_actual_tokens=8,
        descriptor=descriptor,
    )
    assert single_result is not None
    single_prepared, single_descriptor = single_result
    assert single_descriptor is descriptor
    assert single_prepared is not replayed
    torch.accelerator.synchronize(device)
    single = single_prepared[id(builders[0])]
    assert single.spec_state_indices_tensor is not None
    assert single.spec_state_indices_tensor.shape == (8, 8)
    assert single.spec_query_start_loc is not None
    assert single.spec_query_start_loc.tolist() == [0, 8] + [8] * 7
    assert single.num_accepted_tokens is not None
    assert single.num_accepted_tokens.tolist() == [7] + [1] * 7


def test_dflash2_fused_gdn_group_metadata_align_replay(
    monkeypatch,
    local_gdn_model,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the fused GDN metadata kernel")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (7, 0):
        pytest.skip("the fused GDN metadata kernel is SM70-only")

    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_SPEC_CORE_OP", "0")
    monkeypatch.setenv("VLLM_SM70_DFLASH2_FUSED_GDN_METADATA", "1")
    monkeypatch.setenv("VLLM_SM70_DFLASH2_GDN_METADATA_SHADOW", "1")
    qwen_gdn.envs.disable_envs_cache()

    builders = [
        _create_gdn_builder(
            local_gdn_model,
            num_speculative_tokens=7,
            use_full_cuda_graph=True,
            mamba_cache_mode="align",
            max_cudagraph_capture_size=16,
            device=device,
        )
        for _ in range(2)
    ]
    width = builders[0].num_spec_state_tokens + 1
    query_start_loc = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    common_metadata = compute_common_gdn_attn_metadata(
        num_decode_draft_tokens_cpu=torch.tensor([7, 7], dtype=torch.int32),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        num_spec_state_tokens=7,
        legacy_mixed_decode_routing=False,
    )
    assert common_metadata is not None

    table_columns = 32
    block_tables = (
        torch.arange(
            10,
            10 + 2 * table_columns,
            dtype=torch.int32,
            device=device,
        ).view(2, table_columns),
        torch.arange(
            100,
            100 + 2 * table_columns,
            dtype=torch.int32,
            device=device,
        ).view(2, table_columns),
    )
    # Batch rows deliberately map to non-adjacent persistent request slots.
    req_index_mapping = torch.tensor([2, 0], dtype=torch.int32, device=device)
    state_start_indices = torch.tensor([3, 12, 7, 18], dtype=torch.int32, device=device)
    accepted = torch.tensor([3, 5], dtype=torch.int32, device=device)

    # Align metadata is incomplete without both persistent-state inputs.
    assert (
        prepare_dflash2_gdn_group_metadata(
            builders_by_group=[(0, builders[0]), (1, builders[1])],
            block_tables=block_tables,
            common_gdn_metadata=common_metadata,
            num_accepted_tokens=accepted,
            num_actual_tokens=16,
            descriptor=None,
            state_start_indices=state_start_indices,
        )
        is None
    )

    first_result = prepare_dflash2_gdn_group_metadata(
        builders_by_group=[(0, builders[0]), (1, builders[1])],
        block_tables=block_tables,
        common_gdn_metadata=common_metadata,
        num_accepted_tokens=accepted,
        num_actual_tokens=16,
        descriptor=None,
        state_start_indices=state_start_indices,
        req_index_mapping=req_index_mapping,
    )
    assert first_result is not None
    prepared, descriptor = first_result
    torch.accelerator.synchronize(device)

    starts = [7, 3]
    for group_id, builder in enumerate(builders):
        state = prepared[id(builder)].spec_state_indices_tensor
        assert state is not None
        expected = torch.stack(
            [
                block_tables[group_id][row, start : start + width]
                for row, start in enumerate(starts)
            ]
        )
        torch.testing.assert_close(state[:2], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            state[2:], torch.full_like(state[2:], PAD_SLOT_ID), rtol=0, atol=0
        )

    # Replay with the same persistent pointers but a different request mapping
    # and state columns. CUDA graph replay must observe the live values.
    for builder in builders:
        builder.spec_state_indices_tensor.fill_(888)
    # The model runner allocates this short mapping per batch. Its pointer is
    # intentionally allowed to change without rebuilding the group descriptor.
    req_index_mapping = torch.tensor([1, 3], dtype=torch.int32, device=device)
    state_start_indices.copy_(
        torch.tensor([2, 9, 5, 20], dtype=torch.int32, device=device)
    )
    second_result = prepare_dflash2_gdn_group_metadata(
        builders_by_group=[(0, builders[0]), (1, builders[1])],
        block_tables=block_tables,
        common_gdn_metadata=common_metadata,
        num_accepted_tokens=accepted,
        num_actual_tokens=16,
        descriptor=descriptor,
        state_start_indices=state_start_indices,
        req_index_mapping=req_index_mapping,
    )
    assert second_result is not None
    replayed, replay_descriptor = second_result
    assert replay_descriptor is descriptor
    assert replayed is prepared
    torch.accelerator.synchronize(device)

    starts = [9, 20]
    for group_id, builder in enumerate(builders):
        state = replayed[id(builder)].spec_state_indices_tensor
        assert state is not None
        expected = torch.stack(
            [
                block_tables[group_id][row, start : start + width]
                for row, start in enumerate(starts)
            ]
        )
        torch.testing.assert_close(state[:2], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            state[2:], torch.full_like(state[2:], PAD_SLOT_ID), rtol=0, atol=0
        )


def test_full_cuda_graph_capture_single_token_decode_is_not_spec(local_gdn_model):
    builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=4,
        use_full_cuda_graph=True,
        max_cudagraph_capture_size=8,
    )
    batch = BatchSpec(seq_lens=[4097], query_lens=[1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)

    meta = builder.build_for_cudagraph_capture(common)

    assert meta.num_decodes == 1
    assert meta.num_spec_decodes == 0


def test_full_cuda_graph_capture_keeps_spec_state_selector(local_gdn_model):
    builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=4,
        use_full_cuda_graph=True,
        max_cudagraph_capture_size=8,
    )
    batch = BatchSpec(seq_lens=[4097], query_lens=[5])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common = common.replace(
        block_table_tensor=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.int32, device=DEVICE
        )
    )

    meta = builder.build_for_cudagraph_capture(common)

    assert meta.num_spec_decodes == 1
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens[0].item() == 5
    assert meta.spec_state_slot_selectors is not None
    assert meta.spec_state_slot_selectors[0].item() == 5


def test_none_cache_non_spec_uses_legacy_slot0_by_default(local_gdn_model):
    builder = _create_gdn_builder(local_gdn_model, num_speculative_tokens=4)
    batch = BatchSpec(seq_lens=[4097], query_lens=[1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common = common.replace(
        block_table_tensor=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.int32, device=DEVICE
        )
    )

    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([3], dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor([-1], dtype=torch.int32, device="cpu"),
    )

    assert meta.spec_state_indices_tensor is None
    assert meta.non_spec_state_indices_tensor is not None
    assert meta.non_spec_state_indices_tensor.tolist() == [10]


def test_none_cache_mixed_non_spec_uses_legacy_slot0_by_default(local_gdn_model):
    builder = _create_gdn_builder(local_gdn_model, num_speculative_tokens=4)
    batch = BatchSpec(seq_lens=[4097, 4097], query_lens=[1, 5])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common = common.replace(
        block_table_tensor=torch.tensor(
            [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]],
            dtype=torch.int32,
            device=DEVICE,
        )
    )

    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([4, 1], dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor(
            [-1, 4], dtype=torch.int32, device="cpu"
        ),
    )

    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.tolist() == [[20, 21, 22, 23, 24]]
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [1]
    assert meta.non_spec_state_indices_tensor is not None
    assert meta.non_spec_state_indices_tensor.tolist() == [10]


def test_none_cache_spec_combination_uses_accepted_state_slot(local_gdn_model):
    builder = _create_gdn_builder(local_gdn_model, num_speculative_tokens=4)
    batch = BatchSpec(seq_lens=[4097], query_lens=[5])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common = common.replace(
        block_table_tensor=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.int32, device=DEVICE
        )
    )

    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([3], dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor([4], dtype=torch.int32, device="cpu"),
    )

    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.tolist() == [[10, 11, 12, 13, 14]]
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [3]
    assert meta.spec_state_slot_selectors is not None
    assert meta.spec_state_slot_selectors.tolist() == [3]
    assert _effective_spec_initial_state_slots(meta) == [12]


def test_spec_state_slot_selector_can_differ_from_accepted_count(local_gdn_model):
    builder = _create_gdn_builder(local_gdn_model, num_speculative_tokens=4)
    batch = BatchSpec(seq_lens=[4097], query_lens=[5])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common = common.replace(
        block_table_tensor=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.int32, device=DEVICE
        )
    )

    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        spec_state_slot_selectors=torch.tensor([4], dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor([4], dtype=torch.int32, device="cpu"),
    )

    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [2]
    assert meta.spec_state_slot_selectors is not None
    assert meta.spec_state_slot_selectors.tolist() == [4]
    assert _effective_spec_initial_state_slots(meta) == [13]


def test_align_cache_non_spec_uses_legacy_slot0_by_default(local_gdn_model):
    builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=4,
        mamba_cache_mode="align",
    )
    batch = BatchSpec(seq_lens=[33], query_lens=[1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common = common.replace(
        block_table_tensor=torch.tensor(
            [[0, 0, 50, 51, 52, 53, 54]], dtype=torch.int32, device=DEVICE
        )
    )

    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([3], dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor([-1], dtype=torch.int32, device="cpu"),
    )

    assert meta.spec_state_indices_tensor is None
    assert meta.non_spec_state_indices_tensor is not None
    assert meta.non_spec_state_indices_tensor.tolist() == [50]


def test_align_cache_spec_uses_current_state_ids_and_accepted_slot(
    local_gdn_model,
):
    builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=4,
        mamba_cache_mode="align",
    )
    batch = BatchSpec(seq_lens=[33], query_lens=[5])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common = common.replace(
        block_table_tensor=torch.tensor(
            [[0, 0, 10, 11, 12, 13, 14]], dtype=torch.int32, device=DEVICE
        )
    )

    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([4], dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor([4], dtype=torch.int32, device="cpu"),
        current_state_block_ids=torch.tensor(
            [[30, 31, 32, 33, 34]], dtype=torch.int32, device=DEVICE
        ),
    )

    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.tolist() == [[30, 31, 32, 33, 34]]
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [4]
    assert _effective_spec_initial_state_slots(meta) == [33]


def test_align_cache_mixed_non_spec_uses_legacy_current_state_slot0(
    local_gdn_model,
):
    builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=4,
        mamba_cache_mode="align",
    )
    batch = BatchSpec(seq_lens=[33, 33], query_lens=[1, 5])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common = common.replace(
        block_table_tensor=torch.tensor(
            [
                [0, 0, 10, 11, 12, 13, 14],
                [0, 0, 20, 21, 22, 23, 24],
            ],
            dtype=torch.int32,
            device=DEVICE,
        )
    )

    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([3, 2], dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor(
            [-1, 4], dtype=torch.int32, device="cpu"
        ),
        current_state_block_ids=torch.tensor(
            [[30, 31, 32, 33, 34], [40, 41, 42, 43, 44]],
            dtype=torch.int32,
            device=DEVICE,
        ),
    )

    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.tolist() == [[40, 41, 42, 43, 44]]
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [2]
    assert _effective_spec_initial_state_slots(meta) == [41]
    assert meta.non_spec_state_indices_tensor is not None
    assert meta.non_spec_state_indices_tensor.tolist() == [30]


def test_align_active_mtp_rollover_contract_near_block_boundary(
    local_gdn_model,
):
    builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=4,
        use_full_cuda_graph=True,
        mamba_cache_mode="align",
        max_cudagraph_capture_size=32,
    )
    accepted_counts = [1, 2, 3, 4, 5]
    batch = BatchSpec(
        # Small equivalent of positions around 2 * block_size. The production
        # failure was near 2 * 816; here block_size is 16, so the boundary is 32.
        seq_lens=[28, 31, 32, 33, 36],
        query_lens=[5, 5, 5, 5, 5],
    )
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common = common.replace(
        block_table_tensor=torch.tensor(
            [
                [0, 0, 10, 11, 12, 13, 14],
                [0, 0, 20, 21, 22, 23, 24],
                [0, 0, 30, 31, 32, 33, 34],
                [0, 0, 40, 41, 42, 43, 44],
                [0, 0, 50, 51, 52, 53, 54],
            ],
            dtype=torch.int32,
            device=DEVICE,
        )
    )
    current_state_block_ids = torch.tensor(
        [
            [100, 101, 102, 103, 104],
            [110, 111, 112, 113, 114],
            [120, 121, 122, 123, 124],
            [130, 131, 132, 133, 134],
            [140, 141, 142, 143, 144],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    spec_sequence_masks_cpu = torch.ones(5, dtype=torch.bool, device="cpu")
    num_accepted_tokens = torch.tensor(
        accepted_counts, dtype=torch.int32, device=DEVICE
    )

    expected_contract = build_gdn_spec_decode_state_contract(
        block_table_tensor=common.block_table_tensor,
        seq_lens=common.seq_lens,
        block_size=BLOCK_SIZE,
        num_spec=4,
        spec_sequence_masks_cpu=spec_sequence_masks_cpu,
        num_accepted_tokens=num_accepted_tokens,
        current_state_block_ids=current_state_block_ids,
        is_mamba_cache_all=False,
    )
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=torch.full(
            (5,), 4, dtype=torch.int32, device="cpu"
        ),
        current_state_block_ids=current_state_block_ids,
    )

    assert meta.num_spec_decodes == 5
    assert meta.num_spec_decode_tokens == 25
    assert meta.num_decodes == 0
    assert meta.num_prefills == 0
    assert meta.spec_state_indices_tensor is not None
    assert meta.num_accepted_tokens is not None
    assert meta.spec_sequence_masks is not None
    assert meta.spec_query_start_loc is not None
    assert meta.spec_state_indices_tensor[:5].tolist() == (
        expected_contract.spec_state_indices_tensor.tolist()
    )
    assert meta.num_accepted_tokens[:5].tolist() == (
        expected_contract.num_accepted_tokens.tolist()
    )
    assert _effective_spec_initial_state_slots(
        GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=5,
            num_spec_decode_tokens=25,
            num_actual_tokens=25,
            spec_state_indices_tensor=meta.spec_state_indices_tensor[:5],
            num_accepted_tokens=meta.num_accepted_tokens[:5],
        )
    ) == [100, 111, 122, 133, 144]

    # FULL graph replay rows after live requests must not touch real state.
    assert meta.spec_state_indices_tensor[5:].tolist() == [[PAD_SLOT_ID] * 5] * 20
    assert meta.num_accepted_tokens[5:].tolist() == [1] * 20
    assert meta.spec_sequence_masks[:5].tolist() == [True] * 5
    assert meta.spec_sequence_masks[5:].tolist() == [False] * 20
    assert meta.spec_query_start_loc[:6].tolist() == [0, 5, 10, 15, 20, 25]
    assert meta.spec_query_start_loc[6:].tolist() == [25] * 20

    (
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        spec_query_start_loc,
        spec_state_indices_tensor,
        spec_token_indx,
        non_spec_token_indx,
        spec_sequence_masks,
        accepted_tokens,
        spec_state_slot_selectors,
    ) = gdn_spec_metadata_tensors(meta, DEVICE)
    assert non_spec_query_start_loc.numel() == 0
    assert non_spec_state_indices_tensor.numel() == 0
    assert spec_query_start_loc.data_ptr() == meta.spec_query_start_loc.data_ptr()
    assert spec_state_indices_tensor.data_ptr() == (
        meta.spec_state_indices_tensor.data_ptr()
    )
    assert spec_token_indx.numel() == 25
    assert non_spec_token_indx.numel() == 0
    assert spec_sequence_masks.data_ptr() == meta.spec_sequence_masks.data_ptr()
    assert accepted_tokens.data_ptr() == meta.num_accepted_tokens.data_ptr()
    assert spec_state_slot_selectors.data_ptr() == (
        meta.spec_state_slot_selectors.data_ptr()
    )


def test_spec_core_placeholder_registry_includes_state_slot_selector(
    monkeypatch,
    local_gdn_model,
):
    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_SPEC_CORE_OP", "1")

    _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=4,
        use_full_cuda_graph=True,
    )
    tensors = get_registered_gdn_spec_metadata_tensors("layer.0", DEVICE)

    assert len(tensors) == 9
    assert tensors[7].numel() > 0
    assert tensors[8].numel() > 0


def test_gdn_state_contract_debug_assert_rejects_pad_accepted_slot(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_SM70_GDN_STATE_CONTRACT_ASSERT", "1")

    with pytest.raises(AssertionError, match="accepted slot points"):
        build_gdn_spec_decode_state_contract(
            block_table_tensor=torch.tensor(
                [[10, 11, 12, 13, PAD_SLOT_ID]],
                dtype=torch.int32,
                device=DEVICE,
            ),
            seq_lens=torch.tensor([36], dtype=torch.int32, device=DEVICE),
            block_size=BLOCK_SIZE,
            num_spec=4,
            spec_sequence_masks_cpu=torch.tensor(
                [True], dtype=torch.bool, device="cpu"
            ),
            num_accepted_tokens=torch.tensor([5], dtype=torch.int32, device=DEVICE),
            current_state_block_ids=None,
            is_mamba_cache_all=False,
        )


def test_spec_commit_pure_decode_consumes_padded_graph_rows_without_metadata_patch(
    monkeypatch,
    local_gdn_model,
):
    builder = _create_gdn_builder(
        local_gdn_model,
        num_speculative_tokens=4,
        use_full_cuda_graph=True,
        mamba_cache_mode="align",
        max_cudagraph_capture_size=32,
    )
    vllm_config = builder.vllm_config

    live_meta = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=10,
        num_actual_tokens=10,
        spec_query_start_loc=torch.tensor([0, 5, 10], dtype=torch.int32, device=DEVICE),
        spec_state_indices_tensor=torch.tensor(
            [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]],
            dtype=torch.int32,
            device=DEVICE,
        ),
        spec_sequence_masks=torch.tensor([True, True], device=DEVICE),
        spec_token_indx=torch.arange(10, dtype=torch.int32, device=DEVICE),
        non_spec_token_indx=torch.empty(0, dtype=torch.int32, device=DEVICE),
        num_accepted_tokens=torch.tensor([2, 3], dtype=torch.int32, device=DEVICE),
        spec_state_slot_selectors=torch.tensor(
            [4, 5], dtype=torch.int32, device=DEVICE
        ),
    )
    original_state_rows = live_meta.spec_state_indices_tensor
    original_query_start = live_meta.spec_query_start_loc
    original_num_spec_decodes = live_meta.num_spec_decodes

    padded_state_rows = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [PAD_SLOT_ID] * 5,
            [PAD_SLOT_ID] * 5,
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    padded_query_start = torch.tensor(
        [0, 5, 10, 10, 10], dtype=torch.int32, device=DEVICE
    )
    padded_masks = torch.tensor([True, True, False, False], device=DEVICE)
    padded_accepted = torch.tensor([2, 3, 1, 1], dtype=torch.int32, device=DEVICE)
    padded_selectors = torch.tensor([4, 5, 1, 1], dtype=torch.int32, device=DEVICE)
    spec_token_indx = torch.arange(10, dtype=torch.int32, device=DEVICE)
    non_spec_token_indx = torch.empty(0, dtype=torch.int32, device=DEVICE)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=DEVICE)

    seen: dict[str, object] = {}

    class FakeLayer:
        def __init__(self) -> None:
            self.conv1d = torch.nn.Identity()
            self.conv1d.weight = torch.empty((8, 1, 1), device=DEVICE)
            self.conv1d.bias = torch.empty(8, device=DEVICE)
            self.activation = "silu"
            self.A_log = torch.empty(1)
            self.dt_bias = torch.empty(1)

        def rearrange_mixed_qkv(
            self,
            mixed_qkv,
        ):
            seen["rearrange_shape"] = tuple(mixed_qkv.shape)
            q = mixed_qkv.reshape(1, mixed_qkv.shape[0], 1, mixed_qkv.shape[1])
            return q, q + 1, q + 2

        def _forward_core(self, **kwargs) -> None:
            raise AssertionError("pure spec_commit should not use metadata fallback")

    def fake_conv_update(
        x,
        conv_state,
        weight,
        bias,
        activation,
        *,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len,
        validate_data,
    ):
        del conv_state, weight, bias, activation, validate_data
        assert conv_state_indices.data_ptr() == padded_state_rows[:, 0].data_ptr()
        assert num_accepted_tokens is padded_selectors
        assert query_start_loc is padded_query_start
        assert max_query_len == padded_state_rows.shape[1]
        seen["conv"] = True
        return x + 1

    def fake_gating(a_log, a_tensor, b_tensor, dt_bias, *, beta_dtype=None):
        del a_log, dt_bias
        assert a_tensor is a
        assert b_tensor is b
        assert beta_dtype is torch.float32
        seen["gating"] = True
        return a_tensor + 2, b_tensor + 3

    def fake_recurrent(**kwargs):
        assert kwargs["cu_seqlens"] is padded_query_start
        assert kwargs["ssm_state_indices"] is padded_state_rows
        assert kwargs["num_accepted_tokens"] is padded_selectors
        assert kwargs["inplace_final_state"] is True
        seen["recurrent"] = True
        out = torch.full((1, mixed_qkv.shape[0], 1, 1), 7.0, device=DEVICE)
        return out, None

    monkeypatch.setattr(qwen_gdn, "causal_conv1d_update", fake_conv_update)
    monkeypatch.setattr(qwen_gdn, "fused_gdn_gating", fake_gating)
    monkeypatch.setattr(
        qwen_gdn,
        "fused_recurrent_gated_delta_rule",
        fake_recurrent,
    )

    vllm_config.compilation_config.static_forward_context = {"layer.0": FakeLayer()}
    mixed_qkv = torch.zeros((10, 8), dtype=torch.float32, device=DEVICE)
    b = torch.zeros((10, 1), dtype=torch.float32, device=DEVICE)
    a = torch.zeros((10, 1), dtype=torch.float32, device=DEVICE)
    core_attn_out = torch.empty((10, 1, 1), dtype=torch.float32, device=DEVICE)
    conv_state_cache = torch.empty((1, 1, 1), dtype=torch.float32, device=DEVICE)
    ssm_state_cache = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=DEVICE)

    with set_forward_context({"layer.0": live_meta}, vllm_config):
        out = qwen_gdn_attention_core_spec_commit(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_state_cache,
            ssm_state_cache,
            empty_i32,
            empty_i32,
            padded_query_start,
            padded_state_rows,
            spec_token_indx,
            non_spec_token_indx,
            padded_masks,
            padded_accepted,
            padded_selectors,
            "layer.0",
        )

    assert seen == {
        "conv": True,
        "rearrange_shape": (10, 8),
        "gating": True,
        "recurrent": True,
    }
    assert out is core_attn_out
    assert torch.all(core_attn_out == 7)
    assert live_meta.num_spec_decodes == original_num_spec_decodes
    assert live_meta.spec_state_indices_tensor is original_state_rows
    assert live_meta.spec_query_start_loc is original_query_start
    assert live_meta.num_accepted_tokens is not padded_accepted


def test_sm70_qwen_gdn_full_forward_auto_is_mtp_engine_scoped(
    monkeypatch,
    local_gdn_model,
):
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")
    monkeypatch.delenv("VLLM_SM70_QWEN_GDN_FULL_FORWARD", raising=False)
    monkeypatch.delenv("VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD", raising=False)

    regular_builder = _create_gdn_builder(local_gdn_model)
    regular_meta = _build(
        regular_builder,
        BatchSpec(seq_lens=[40], query_lens=[1]),
    )
    assert not _sm70_qwen_gdn_full_forward_enabled(
        "layer.0",
        force_enabled=False,
        disabled=False,
        auto_enabled=False,
    )
    with set_forward_context({"layer.0": regular_meta}, regular_builder.vllm_config):
        assert _sm70_qwen_gdn_full_forward_enabled(
            "layer.0",
            force_enabled=False,
            disabled=False,
            auto_enabled=True,
        )
    with set_forward_context(
        {"layer.0": regular_meta},
        regular_builder.vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        is_dummy_run=True,
    ):
        assert _sm70_qwen_gdn_full_forward_enabled(
            "layer.0",
            force_enabled=False,
            disabled=False,
            auto_enabled=True,
        )
    with set_forward_context(
        {"layer.0": regular_meta},
        regular_builder.vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        is_dummy_run=True,
    ):
        assert _sm70_qwen_gdn_full_forward_enabled(
            "layer.0",
            force_enabled=False,
            disabled=False,
            auto_enabled=True,
        )

    spec_builder = _create_gdn_builder(local_gdn_model, num_speculative_tokens=4)
    spec_meta = _build(
        spec_builder,
        BatchSpec(seq_lens=[40], query_lens=[5]),
        num_decode_draft_tokens=[4],
    )
    with set_forward_context({"layer.0": spec_meta}, spec_builder.vllm_config):
        assert _sm70_qwen_gdn_full_forward_enabled(
            "layer.0", force_enabled=False, disabled=False, auto_enabled=True
        )
        assert _sm70_qwen_gdn_spec_core_enabled("layer.0", auto_enabled=True)
        assert not _sm70_qwen_gdn_full_forward_enabled(
            "layer.0", force_enabled=False, disabled=False, auto_enabled=False
        )
        assert not _sm70_qwen_gdn_full_forward_enabled(
            "layer.0", force_enabled=False, disabled=True, auto_enabled=True
        )

    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_FULL_FORWARD", "1")
    with set_forward_context({"layer.0": regular_meta}, regular_builder.vllm_config):
        assert _sm70_qwen_gdn_full_forward_enabled(
            "layer.0",
            force_enabled=True,
            disabled=False,
            auto_enabled=False,
        )


@pytest.mark.parametrize(
    ("method", "num_speculative_tokens", "expected"),
    [
        ("mtp", 2, False),
        ("mtp", 3, True),
        ("mtp", 4, True),
        ("ngram", 4, False),
    ],
)
def test_sm70_qwen_gdn_blocks_003_spec_core_only_for_deep_native_mtp(
    monkeypatch,
    method,
    num_speculative_tokens,
    expected,
):
    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_003_SPEC_CORE_OP", "1")
    monkeypatch.delenv("VLLM_SM70_QWEN_GDN_003_SPEC_ALLOW_DEEP_MTP", raising=False)
    _clear_envs_cache()

    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            num_speculative_tokens=num_speculative_tokens,
        )
    )

    assert (
        qwen_gdn._sm70_qwen_gdn_block_003_spec_for_deep_native_mtp(vllm_config)
        is expected
    )


def test_sm70_qwen_gdn_003_spec_core_deep_mtp_override_is_diagnostic(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_003_SPEC_CORE_OP", "1")
    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_003_SPEC_ALLOW_DEEP_MTP", "1")
    _clear_envs_cache()

    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=4,
        )
    )

    assert not qwen_gdn._sm70_qwen_gdn_block_003_spec_for_deep_native_mtp(vllm_config)


def test_sm70_qwen_gdn_003_spec_core_route_has_priority(
    monkeypatch,
    local_gdn_model,
):
    spec_builder = _create_gdn_builder(local_gdn_model, num_speculative_tokens=4)
    spec_meta = _build(
        spec_builder,
        BatchSpec(seq_lens=[40], query_lens=[5]),
        num_decode_draft_tokens=[4],
    )

    class FakeSelf:
        auto_sm70_qwen_gdn_003_spec_core = True
        auto_sm70_qwen_gdn_spec_core = True
        auto_sm70_qwen_gdn_full_forward = True

    calls: list[str] = []

    def fake_003(
        mixed_qkv,
        b,
        a,
        core_attn_out,
        conv_state_cache,
        ssm_state_cache,
        layer_name,
    ):
        calls.append(str(layer_name))
        assert conv_state_cache.numel() == 0
        assert ssm_state_cache.numel() == 0
        return core_attn_out

    def fail_fallback(*args, **kwargs):
        raise AssertionError("003 route must not fall through to fallback op")

    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen_gdn_attention_core_003_spec",
        fake_003,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen_gdn_attention_core_spec_commit",
        fail_fallback,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen_gdn_attention_core_standard_spec",
        fail_fallback,
        raising=False,
    )

    mixed_qkv = torch.zeros(5, 8)
    b = torch.zeros(5, 2)
    a = torch.zeros(5, 2)
    core_attn_out = torch.zeros(5, 2, 4)
    empty_cache = torch.empty(0)

    with set_forward_context({"layer.0": spec_meta}, spec_builder.vllm_config):
        out = _qwen_gdn_run_recurrent_core(
            FakeSelf(),
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            layer_name="layer.0",
            conv_state_cache=empty_cache,
            ssm_state_cache=empty_cache,
        )

    assert out is core_attn_out
    assert calls == ["layer.0"]


def test_ddtree_depth_batches_accepts_verifier_prefix_row() -> None:
    parent_ids = torch.tensor([[-1, -1, 1, 1]], dtype=torch.int32)
    num_tree_tokens = torch.tensor([3], dtype=torch.int32)
    spec_query_start_loc = torch.tensor([0, 4], dtype=torch.int32)

    prefix_rows, batches = qwen_gdn._ddtree_depth_batches(
        parent_ids=parent_ids,
        num_tree_tokens_cpu=num_tree_tokens,
        spec_query_start_loc=spec_query_start_loc,
        num_spec_decodes=1,
    )

    assert prefix_rows == [(0, 0)]
    assert batches[1] == [(1, 0, 0, 1)]
    assert batches[2] == [(2, 0, 1, 2), (3, 0, 1, 3)]


def test_ddtree_depth_batches_keeps_legacy_no_prefix_rows() -> None:
    parent_ids = torch.tensor([[-1, -1, 1, 1]], dtype=torch.int32)
    num_tree_tokens = torch.tensor([3], dtype=torch.int32)
    spec_query_start_loc = torch.tensor([0, 3], dtype=torch.int32)

    prefix_rows, batches = qwen_gdn._ddtree_depth_batches(
        parent_ids=parent_ids,
        num_tree_tokens_cpu=num_tree_tokens,
        spec_query_start_loc=spec_query_start_loc,
        num_spec_decodes=1,
    )

    assert prefix_rows == []
    assert batches[1] == [(0, 0, 0, 1)]
    assert batches[2] == [(1, 0, 1, 2), (2, 0, 1, 3)]


def test_sm70_qwen_gdn_spec_commit_route_is_compile_stable(
    monkeypatch,
    local_gdn_model,
):
    regular_builder = _create_gdn_builder(local_gdn_model)
    regular_meta = _build(
        regular_builder,
        BatchSpec(seq_lens=[40], query_lens=[1]),
    )

    class FakeSelf:
        auto_sm70_qwen_gdn_003_spec_core = False
        auto_sm70_qwen_gdn_spec_core = True
        auto_sm70_qwen_gdn_full_forward = False

    calls: list[str] = []

    def fake_spec_commit(
        mixed_qkv,
        b,
        a,
        core_attn_out,
        conv_state_cache,
        ssm_state_cache,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        spec_query_start_loc,
        spec_state_indices_tensor,
        spec_token_indx,
        non_spec_token_indx,
        spec_sequence_masks,
        num_accepted_tokens,
        spec_state_slot_selectors,
        layer_name,
    ):
        del (
            mixed_qkv,
            b,
            a,
            conv_state_cache,
            ssm_state_cache,
            non_spec_query_start_loc,
            non_spec_state_indices_tensor,
            spec_query_start_loc,
            spec_state_indices_tensor,
            spec_token_indx,
            non_spec_token_indx,
            spec_sequence_masks,
            num_accepted_tokens,
            spec_state_slot_selectors,
        )
        calls.append(str(layer_name))
        return core_attn_out

    def fail_fallback(*args, **kwargs):
        raise AssertionError("SPEC_CORE_OP route must stay on spec_commit")

    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen_gdn_attention_core_spec_commit",
        fake_spec_commit,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen_gdn_attention_core_standard",
        fail_fallback,
        raising=False,
    )

    mixed_qkv = torch.zeros(1, 8)
    b = torch.zeros(1, 2)
    a = torch.zeros(1, 2)
    core_attn_out = torch.zeros(1, 2, 4)
    empty_cache = torch.empty(0)

    with set_forward_context({"layer.0": regular_meta}, regular_builder.vllm_config):
        out = _qwen_gdn_run_recurrent_core(
            FakeSelf(),
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            layer_name="layer.0",
            conv_state_cache=empty_cache,
            ssm_state_cache=empty_cache,
        )

    assert out is core_attn_out
    assert calls == ["layer.0"]
