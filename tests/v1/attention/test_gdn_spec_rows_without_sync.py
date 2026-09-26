# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN spec-decode metadata selects speculative rows without host syncs when
they lead the batch, and exactly as boolean-mask indexing does otherwise."""

import pytest
import torch

from vllm.v1.attention.backends.gdn_attn import (
    _spec_sequence_masks_on_device,
    build_gdn_spec_decode_state_contract,
    gather_gdn_state_block_ids,
    select_gdn_state_block_ids,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GDN state selection runs on CUDA"
)

NUM_SPEC, BLOCK_SIZE, ROWS = 3, 16, 4
MASKS = {
    "all_spec": [True, True, True, True],
    "spec_then_padding": [True, True, False, False],
    "no_spec": [False, False, False, False],
    "interleaved": [False, True, False, True],
}
LEADING = ("all_spec", "spec_then_padding", "no_spec")


def _inputs(state_source: str):
    generator = torch.Generator().manual_seed(1600)

    def ints(low, high, shape):
        return torch.randint(low, high, shape, generator=generator, dtype=torch.int32)

    return dict(
        block_table=ints(1, 1000, (ROWS, 12)).cuda(),
        # (seq_len - 1) // 16 + NUM_SPEC stays inside the 12 table columns.
        seq_lens=ints(20, 120, (ROWS,)).cuda(),
        accepted=ints(1, NUM_SPEC + 2, (ROWS,)).cuda(),
        selectors=ints(1, NUM_SPEC + 2, (ROWS,)).cuda(),
        current=ints(1, 1000, (ROWS, 6)).cuda() if state_source == "current" else None,
    )


def _reference(inputs, mask_cpu, cache_all):
    """The contract computed by boolean-mask indexing, as before."""
    mask = mask_cpu.cuda()
    table, accepted = inputs["block_table"], inputs["accepted"]
    if inputs["current"] is not None:
        ids = inputs["current"][:, : NUM_SPEC + 1]
        spec = ids[mask]
        non_spec = select_gdn_state_block_ids(ids[~mask], accepted[~mask], NUM_SPEC)
    elif cache_all:
        seq_lens = inputs["seq_lens"]
        spec = gather_gdn_state_block_ids(
            table[mask], seq_lens[mask], BLOCK_SIZE, NUM_SPEC + 1
        )
        non_spec = gather_gdn_state_block_ids(
            table[~mask], seq_lens[~mask], BLOCK_SIZE, 1
        ).squeeze(1)
    else:
        spec = table[mask, : NUM_SPEC + 1]
        non_spec = select_gdn_state_block_ids(table[~mask], accepted[~mask], NUM_SPEC)
    return spec, non_spec, accepted[mask], inputs["selectors"][mask]


def _contract(inputs, mask_cpu, cache_all):
    return build_gdn_spec_decode_state_contract(
        block_table_tensor=inputs["block_table"],
        seq_lens=inputs["seq_lens"],
        block_size=BLOCK_SIZE,
        num_spec=NUM_SPEC,
        spec_sequence_masks_cpu=mask_cpu,
        num_accepted_tokens=inputs["accepted"],
        current_state_block_ids=inputs["current"],
        is_mamba_cache_all=cache_all,
        spec_state_slot_selectors=inputs["selectors"],
    )


@pytest.mark.parametrize("mask_name", list(MASKS))
@pytest.mark.parametrize("state_source", ["current", "cache_all", "block_table"])
def test_contract_matches_mask_indexing(mask_name: str, state_source: str):
    inputs = _inputs(state_source)
    mask_cpu = torch.tensor(MASKS[mask_name])
    cache_all = state_source == "cache_all"
    got = _contract(inputs, mask_cpu, cache_all)
    want = _reference(inputs, mask_cpu, cache_all)
    for actual, expected in zip(
        (
            got.spec_state_indices_tensor,
            got.non_spec_state_indices_tensor,
            got.num_accepted_tokens,
            got.spec_state_slot_selectors,
        ),
        want,
    ):
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        # At least as contiguous as the mask-indexed rows it replaces.
        assert actual.is_contiguous() or not expected.is_contiguous()
    # The selected rows are copies: writing them leaves the inputs untouched.
    before = inputs["accepted"].clone()
    got.num_accepted_tokens.fill_(-1)
    assert torch.equal(inputs["accepted"], before)


@pytest.mark.parametrize("mask_name", LEADING)
@pytest.mark.parametrize("state_source", ["current", "cache_all", "block_table"])
def test_leading_spec_rows_do_not_synchronize(mask_name: str, state_source: str):
    inputs = _inputs(state_source)
    mask_cpu = torch.tensor(MASKS[mask_name])
    torch.accelerator.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _contract(inputs, mask_cpu, state_source == "cache_all")
        device_mask = _spec_sequence_masks_on_device(mask_cpu, torch.device("cuda"))
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.equal(device_mask.cpu(), mask_cpu)


def test_device_mask_falls_back_to_upload_for_interleaved_rows():
    mask_cpu = torch.tensor(MASKS["interleaved"])
    device_mask = _spec_sequence_masks_on_device(mask_cpu, torch.device("cuda"))
    assert device_mask.dtype == torch.bool
    assert torch.equal(device_mask.cpu(), mask_cpu)
