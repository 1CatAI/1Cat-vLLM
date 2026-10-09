# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative metadata ownership preserves identities and graph storage."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100.metadata import (
    FlashAttnV100MetadataBuilder,
)
from vllm.v1.attention.backends.flash_v100.spec.metadata_contracts import (
    MetadataInputs,
    MetadataOps,
)
from vllm.v1.attention.backends.flash_v100.spec.metadata_state import SpecMetadataState
from vllm.v1.attention.backends.flash_v100.spec.smallq_metadata import (
    DFlash2SmallQPreparedMetadata,
)

pytestmark = pytest.mark.cpu_test


def _state(builder_id=73, base_build=None):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        model_config=SimpleNamespace(max_model_len=2048),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=8, cudagraph_capture_sizes=[8]
        ),
    )
    calls = []

    def callback(name):
        return lambda *args, **kwargs: calls.append((name, args, kwargs))

    state = SpecMetadataState(
        MetadataInputs(builder_id, config, torch.device("cpu"), 16, False),
        MetadataOps(
            base_build or callback("base"),
            callback("common"),
            callback("prefix"),
            callback("shape"),
            callback("partitions"),
        ),
        None,
    )
    return state, calls


def test_owner_keeps_draft_and_smallq_buffers_across_replays():
    state, _ = _state()
    common = SimpleNamespace(num_reqs=1)
    previous = None
    for value in (3, 7):
        attn = SimpleNamespace(
            block_table=torch.full((1, 2), value, dtype=torch.int32),
            seq_lens=torch.tensor([value], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        )
        state._stabilize_draft_graph_metadata(attn, common)
        assert state._ensure_smallq_decode_buffers(2, 1, attn.block_table)
        smallq = state.metadata_workspace.smallq
        addresses = tuple(
            tensor.data_ptr()
            for tensor in (
                attn.block_table,
                attn.seq_lens,
                attn.query_start_loc,
                smallq.block_table,
                smallq.seq_lens,
                smallq.query_start_loc,
            )
        )
        if previous is not None:
            assert previous == addresses
        previous = addresses
        assert attn.block_table.tolist() == [[value, value]]
        assert attn.seq_lens.tolist() == [value]


def test_prepared_metadata_checks_original_builder_identity():
    state, _ = _state()
    assert id(state) != state.inputs.builder_id
    state.metadata_workspace.smallq.ensure(8, 2, 2, 2, 1, state.device)
    prepared = DFlash2SmallQPreparedMetadata(73, 1, 2, 31, 64, 16)
    attn = SimpleNamespace()
    state._attach_prepared_dflash2_smallq_metadata(attn, prepared)
    assert attn.smallq_decode_block_table.data_ptr() == (
        state.metadata_workspace.smallq.block_table.data_ptr()
    )
    assert attn.smallq_decode_max_seq_len_hint == 31
    assert attn.smallq_decode_workspace_seq_capacity_hint == 64
    assert attn.smallq_decode_partition_size_hint == 16
    with pytest.raises(ValueError, match="another builder"):
        state._attach_prepared_dflash2_smallq_metadata(
            attn, replace(prepared, builder_id=id(state))
        )


@pytest.mark.parametrize("num_reqs,num_tokens", [(3, 2), (1, 9)])
def test_prepared_metadata_rejects_capacity_overflow_before_attachment(
    num_reqs, num_tokens
):
    state, _ = _state()
    state.metadata_workspace.smallq.ensure(8, 2, 2, 2, 1, state.device)
    sentinel = object()
    attn = SimpleNamespace(smallq_decode_block_table=sentinel)
    prepared = DFlash2SmallQPreparedMetadata(73, num_reqs, num_tokens, 31, 64)
    with pytest.raises(RuntimeError, match="captured capacity"):
        state._attach_prepared_dflash2_smallq_metadata(attn, prepared)
    assert attn.smallq_decode_block_table is sentinel


def test_builder_compatibility_updates_the_single_state_owner():
    instance = object.__new__(FlashAttnV100MetadataBuilder)
    state, _ = _state(id(instance))
    instance.spec_state = state
    instance._is_dflash_draft_model = True
    instance.block_size = 32
    instance.device = torch.device("cpu")
    assert state._is_dflash_draft_model and "_is_dflash_draft_model" not in vars(
        instance
    )
    assert instance.metadata_workspace is state.metadata_workspace
    assert state.block_size == instance.block_size == 32
    assert state.inputs.builder_id == id(instance)
    with pytest.raises(FrozenInstanceError):
        state.inputs.block_size = 64


def test_owner_propagates_base_build_failure_before_callbacks():
    error = RuntimeError("base build failed")

    def base(*args):
        raise error

    state, calls = _state(base_build=base)
    with pytest.raises(RuntimeError) as failure:
        state.build(0, SimpleNamespace())
    assert failure.value is error
    assert calls == []
