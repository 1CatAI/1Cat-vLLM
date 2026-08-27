# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle_speculator


def test_qwen4_exp_mtp_v2_unpacks_logits_and_feedback_hidden_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator = eagle_speculator.EagleSpeculator.__new__(
        eagle_speculator.EagleSpeculator
    )
    speculator.device = torch.device("cpu")
    speculator.vllm_config = SimpleNamespace()
    speculator.supports_mm_inputs = False
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(3, dtype=torch.int64),
        positions=torch.arange(3, dtype=torch.int64),
    )
    speculator.hidden_states = torch.zeros(3, 16)

    logits_hidden = torch.ones(3, 4)
    feedback_hidden = torch.ones(3, 16)
    speculator.model = lambda **_kwargs: (logits_hidden, feedback_hidden)
    monkeypatch.setattr(
        eagle_speculator,
        "set_forward_context",
        lambda *_args, **_kwargs: nullcontext(),
    )

    actual_logits_hidden, actual_feedback_hidden = speculator.run_model(
        num_tokens=3,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_qwen4_exp_mtp_v2_uses_local_argmax_without_full_logits() -> None:
    speculator = eagle_speculator.EagleSpeculator.__new__(
        eagle_speculator.EagleSpeculator
    )
    speculator.use_local_argmax_reduction = True

    class DraftModel:
        def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
            assert hidden_states.shape == (2, 4)
            return torch.tensor([7, 11])

        def compute_logits(self, _hidden_states: torch.Tensor) -> torch.Tensor:
            raise AssertionError("local argmax must not materialize full logits")

    speculator.model = DraftModel()
    top_tokens = speculator._sample_draft(
        hidden_states=torch.zeros(2, 4),
        idx_mapping=torch.arange(2, dtype=torch.int32),
        pos=torch.arange(2),
        draft_step=torch.tensor(0),
        draft_logits=None,
    )

    torch.testing.assert_close(top_tokens, torch.tensor([7, 11]))


def test_qwen4_exp_mtp_v2_reuses_step_zero_qsa_indices() -> None:
    calls: list[tuple[str, object]] = []

    class MTPBackbone:
        def set_skip_topk(self, skip: bool) -> None:
            calls.append(("skip", skip))

        def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
            calls.append(("compact", row_indices.tolist()))

    speculator = eagle_speculator.EagleSpeculator.__new__(
        eagle_speculator.EagleSpeculator
    )
    speculator.share_mtp_topk_indices = True
    speculator.num_speculative_steps = 3
    speculator.last_token_indices = torch.tensor([4, 9, 12])
    speculator.model = SimpleNamespace(model=MTPBackbone())

    speculator._mtp_prefill_begin()
    speculator._mtp_prefill_end(num_reqs=2)
    speculator._mtp_decode_begin()
    speculator._mtp_decode_end()

    assert calls == [
        ("skip", False),
        ("compact", [4, 9]),
        ("skip", True),
        ("skip", False),
    ]


def test_qsa_circular_group_uses_one_block_and_custom_slot_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=False)
    runner.vllm_config = SimpleNamespace()
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cpu")

    circular_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )
    compressed_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["compressor_state"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8, kv_cache_specs={"compressor_state": circular_spec}
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["attention", "compressed"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={
                        "attention": attention_spec,
                        "compressed": compressed_spec,
                    },
                ),
            ),
        ],
    )

    monkeypatch.setattr(
        mrv2,
        "init_attn_backend",
        lambda *args: ([], SimpleNamespace(), [8, 16]),
    )
    captured: dict[str, object] = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(mrv2, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 16384]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an exact SM70 CUDA device",
)
def test_qsa_circular_group_emits_no_generic_slots_on_sm70() -> None:
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[8, 262144],
        max_num_reqs=1,
        max_num_batched_tokens=4,
        max_num_blocks_per_group=[1, 1],
        device=device,
        kernel_block_sizes=[8, 262144],
        slot_mapping_enabled=[False, True],
    )
    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([7], [12]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()

    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=device),
        positions=torch.tensor([153797, 165757], dtype=torch.int64, device=device),
        num_tokens_padded=2,
    )
    torch.accelerator.synchronize()

    assert slot_mappings[0].tolist() == [-1, -1]
    assert slot_mappings[1].tolist() == [
        12 * 262144 + 153797,
        12 * 262144 + 165757,
    ]
