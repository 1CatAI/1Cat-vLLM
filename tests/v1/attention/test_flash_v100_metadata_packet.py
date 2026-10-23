# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request feature packets preserve metadata and tensor identities."""

import copy
import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import metadata, routing
from vllm.v1.attention.backends.flash_v100.spec import tree
from vllm.v1.attention.backends.flash_v100.spec.metadata_contracts import (
    METADATA_FIELDS,
    metadata_view,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

pytestmark = pytest.mark.cpu_test


def _metadata():
    value = object.__new__(TritonAttentionMetadata)
    value.block_table = torch.tensor([[3, 4]], dtype=torch.int32)
    value.query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    value.seq_lens = torch.tensor([7], dtype=torch.int32)
    return value


def test_adoption_keeps_object_and_tensor_identity_with_one_field_owner():
    original = _metadata()
    parents = torch.tensor([[-1, 0]], dtype=torch.int32)
    original.ddtree_parent_ids = parents
    original.smallq_decode_block_table = original.block_table
    addresses = [original.block_table.data_ptr(), original.query_start_loc.data_ptr()]
    adopted = metadata._as_flash_v100_metadata(original)
    assert adopted is original
    assert isinstance(adopted, TritonAttentionMetadata)
    assert type(adopted) is metadata.FlashAttnV100Metadata
    assert adopted.spec_state.ddtree_parent_ids is parents
    assert adopted.spec_state.smallq_decode_block_table is adopted.block_table
    assert metadata_view(adopted) is adopted.spec_state
    assert not METADATA_FIELDS.intersection(vars(adopted))
    assert addresses == [
        adopted.block_table.data_ptr(),
        adopted.query_start_loc.data_ptr(),
    ]
    assert metadata._as_flash_v100_metadata(adopted) is adopted


def test_legacy_reads_writes_and_deletes_reach_the_packet():
    value = metadata._as_flash_v100_metadata(_metadata())
    assert getattr(value, "smallq_decode_max_seq_len_hint", None) is None
    value.smallq_decode_max_seq_len_hint = 16
    assert value.spec_state.smallq_decode_max_seq_len_hint == 16
    value.spec_state.smallq_decode_max_seq_len_hint = 32
    assert value.smallq_decode_max_seq_len_hint == 32
    del value.smallq_decode_max_seq_len_hint
    assert not hasattr(value.spec_state, "smallq_decode_max_seq_len_hint")
    with pytest.raises(AttributeError):
        _ = value.smallq_decode_max_seq_len_hint
    value.causal = False
    assert vars(value)["causal"] is False


def test_shallow_copy_separates_packet_fields_and_shares_tensor_storage():
    original = metadata._as_flash_v100_metadata(_metadata())
    original.smallq_decode_block_table = original.block_table
    original.smallq_decode_max_seq_len_hint = 16
    cloned = copy.copy(original)
    assert cloned is not original and cloned.spec_state is not original.spec_state
    assert cloned.block_table is original.block_table
    assert cloned.smallq_decode_block_table is original.smallq_decode_block_table
    cloned.smallq_decode_max_seq_len_hint = 32
    assert original.smallq_decode_max_seq_len_hint == 16


def test_packet_does_not_retain_its_metadata_owner():
    value = metadata._as_flash_v100_metadata(_metadata())
    packet = value.spec_state
    reference = weakref.ref(value)
    del value
    assert reference() is None
    assert packet is not None


def test_tree_capture_writes_packet_and_keeps_authoritative_buffer_addresses(
    monkeypatch,
):
    value = metadata._as_flash_v100_metadata(_metadata())
    value.query_start_loc_cpu = torch.tensor([0, 3], dtype=torch.int32)
    value.seq_lens_cpu = torch.tensor([13], dtype=torch.int32)
    addresses = value.query_start_loc.data_ptr(), value.seq_lens.data_ptr()
    parents = torch.tensor([[-1, 0, 1]], dtype=torch.int32)
    counts = torch.tensor([3], dtype=torch.int32)
    monkeypatch.setattr(routing, "_is_cuda_graph_capturing", lambda _: True)
    tree.attach_metadata(
        None,
        value,
        ddtree_parent_ids=parents,
        ddtree_num_tree_tokens_cpu=counts,
    )
    assert value.spec_state.ddtree_parent_ids is parents
    assert value.spec_state.ddtree_num_tree_tokens_cpu is counts
    assert value.ddtree_seq_lens_restored_for_triton
    assert value.ddtree_query_start_loc_restored_for_triton
    assert value.query_start_loc.tolist() == [0, 3]
    assert value.seq_lens.tolist() == [13]
    assert addresses == (value.query_start_loc.data_ptr(), value.seq_lens.data_ptr())


def test_external_metadata_keeps_its_legacy_view():
    legacy = SimpleNamespace(smallq_decode_max_seq_len_hint=16)
    assert metadata_view(legacy) is legacy
    assert metadata._as_flash_v100_metadata(legacy) is legacy
    assert not hasattr(legacy, "spec_state")
