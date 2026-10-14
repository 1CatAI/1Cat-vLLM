# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN speculative-decode state contract selects rows without host syncs.

The contract used to select rows with boolean masks on device tensors, which
runs nonzero() and blocks the host until the GPU catches up (five times per
call, three calls per MTP verify step). It now derives row indices from the
host-side mask. These tests pin the results to the boolean-mask formulation
and check on CUDA that building the contract no longer synchronizes.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.gdn_state import (
    build_state_contract as build_gdn_spec_decode_state_contract,
)
from vllm.v1.attention.ops.gdn_state import (
    gather_gdn_state_block_ids,
)
from vllm.v1.attention.ops.gdn_state import (
    select_state_block_ids as select_gdn_state_block_ids,
)

FIELDS = (
    "spec_state_indices_tensor",
    "non_spec_state_indices_tensor",
    "num_accepted_tokens",
    "spec_state_slot_selectors",
)
BRANCHES = ("current_state", "mamba_cache_all", "block_table")
BLOCK_SIZE = 16
TABLE_WIDTH = 8

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _boolean_mask_contract(
    *,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_spec: int,
    spec_sequence_masks_cpu: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    current_state_block_ids: torch.Tensor | None,
    is_mamba_cache_all: bool,
    spec_state_slot_selectors: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """The previous implementation: boolean-mask indexing on each tensor."""
    mask = spec_sequence_masks_cpu.to(block_table_tensor.device)
    if spec_state_slot_selectors is None:
        spec_state_slot_selectors = num_accepted_tokens
    if current_state_block_ids is not None:
        state_block_ids = current_state_block_ids[:, : num_spec + 1]
        spec_state = state_block_ids[mask]
        non_spec_state = select_gdn_state_block_ids(
            state_block_ids[~mask], num_accepted_tokens[~mask], num_spec
        )
    elif is_mamba_cache_all:
        spec_state = gather_gdn_state_block_ids(
            block_table_tensor[mask], seq_lens[mask], block_size, num_spec + 1
        )
        non_spec_state = gather_gdn_state_block_ids(
            block_table_tensor[~mask], seq_lens[~mask], block_size, 1
        ).squeeze(1)
    else:
        spec_state = block_table_tensor[mask, : num_spec + 1]
        non_spec_state = select_gdn_state_block_ids(
            block_table_tensor[~mask], num_accepted_tokens[~mask], num_spec
        )
    return {
        "spec_state_indices_tensor": spec_state,
        "non_spec_state_indices_tensor": non_spec_state,
        "num_accepted_tokens": num_accepted_tokens[mask],
        "spec_state_slot_selectors": spec_state_slot_selectors[mask],
    }


def _contract_inputs(
    seed: int,
    num_reqs: int,
    num_spec: int,
    branch: str,
    mask_kind: str,
    with_selectors: bool,
    device: str,
) -> dict:
    gen = torch.Generator().manual_seed(seed)

    def ints(low: int, high: int, shape: tuple[int, ...]) -> torch.Tensor:
        values = torch.randint(low, high, shape, generator=gen, dtype=torch.int32)
        return values.to(device)

    if mask_kind == "all":
        mask = torch.ones(num_reqs, dtype=torch.bool)
    elif mask_kind == "none":
        mask = torch.zeros(num_reqs, dtype=torch.bool)
    else:
        mask = torch.rand(num_reqs, generator=gen) < 0.6
    return dict(
        block_table_tensor=ints(1, 1000, (num_reqs, TABLE_WIDTH)),
        seq_lens=ints(1, BLOCK_SIZE * TABLE_WIDTH, (num_reqs,)),
        block_size=BLOCK_SIZE,
        num_spec=num_spec,
        spec_sequence_masks_cpu=mask,
        num_accepted_tokens=ints(1, num_spec + 2, (num_reqs,)),
        current_state_block_ids=(
            ints(1000, 2000, (num_reqs, num_spec + 3))
            if branch == "current_state"
            else None
        ),
        is_mamba_cache_all=branch == "mamba_cache_all",
        spec_state_slot_selectors=(
            ints(1, num_spec + 2, (num_reqs,)) if with_selectors else None
        ),
    )


def _assert_same(contract, expected: dict[str, torch.Tensor]) -> None:
    for field in FIELDS:
        actual = getattr(contract, field)
        reference = expected[field]
        assert actual.dtype == reference.dtype, field
        assert actual.device == reference.device, field
        assert actual.shape == reference.shape, field
        assert torch.equal(actual, reference), field


@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("num_spec", [1, 4])
@pytest.mark.parametrize("mask_kind", ["random", "all", "none"])
@pytest.mark.parametrize("num_reqs", [0, 1, 7, 64])
@pytest.mark.parametrize("with_selectors", [False, True])
def test_contract_matches_boolean_mask_reference(
    branch: str, num_spec: int, mask_kind: str, num_reqs: int, with_selectors: bool
):
    for seed in range(10):
        kwargs = _contract_inputs(
            seed, num_reqs, num_spec, branch, mask_kind, with_selectors, "cpu"
        )
        _assert_same(
            build_gdn_spec_decode_state_contract(**kwargs),
            _boolean_mask_contract(**kwargs),
        )


def test_contract_rejects_mask_length_mismatch():
    kwargs = _contract_inputs(0, 5, 4, "block_table", "random", False, "cpu")
    kwargs["spec_sequence_masks_cpu"] = torch.ones(4, dtype=torch.bool)
    with pytest.raises(IndexError):
        build_gdn_spec_decode_state_contract(**kwargs)


def test_group_metadata_legacy_alias_is_captured_with_typed_precedence(monkeypatch):
    from vllm.config.gdn_state import GdnStateConfig

    monkeypatch.setenv("VLLM_SM70_MTP4_SHARED_GDN_METADATA", "0")
    monkeypatch.setenv("VLLM_SM70_MTP4_FUSED_GDN_METADATA", "0")
    monkeypatch.setenv("ONECAT_MTP_GDN_GROUP_META", "yes")
    policy = GdnStateConfig()
    policy.resolve()
    assert policy.shared_mtp_metadata and policy.fused_mtp_metadata
    monkeypatch.setenv("ONECAT_MTP_GDN_GROUP_META", "0")
    policy.resolve()
    assert policy.shared_mtp_metadata and policy.fused_mtp_metadata
    other = GdnStateConfig()
    other.resolve()
    assert not other.shared_mtp_metadata and not other.fused_mtp_metadata
    monkeypatch.setenv("ONECAT_MTP_GDN_GROUP_META", "1")
    explicit = GdnStateConfig(shared_mtp_metadata=False, fused_mtp_metadata=False)
    explicit.resolve()
    assert not explicit.shared_mtp_metadata and not explicit.fused_mtp_metadata


def test_shortconv_builder_borrows_only_its_prepared_metadata():
    from vllm.v1.attention.backends.short_conv_attn import (
        PleShortConvAttentionMetadataBuilder,
    )

    builder = object.__new__(PleShortConvAttentionMetadataBuilder)
    other = object.__new__(PleShortConvAttentionMetadataBuilder)
    prepared = object()
    state = SimpleNamespace(
        num_accepted_tokens=torch.tensor([2, 3], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([4, 4], dtype=torch.int32),
        prepared_ple_shortconv_metadata={id(builder): prepared},
    )
    owned = builder.get_model_state_kwargs(state, 1)
    assert owned["prepared_shortconv"] is prepared
    assert torch.equal(owned["num_accepted_tokens"], torch.tensor([2]))
    assert "prepared_shortconv" not in other.get_model_state_kwargs(state, 1)


@requires_cuda
@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("with_selectors", [False, True])
def test_contract_does_not_synchronize_on_cuda(branch: str, with_selectors: bool):
    kwargs = _contract_inputs(3, 8, 4, branch, "random", with_selectors, "cuda")
    expected = _boolean_mask_contract(**kwargs)
    # The first call may initialize lazy CUDA state.
    build_gdn_spec_decode_state_contract(**kwargs)
    torch.accelerator.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        contract = build_gdn_spec_decode_state_contract(**kwargs)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    _assert_same(contract, expected)


@requires_cuda
def test_sync_debug_mode_catches_boolean_mask_indexing():
    # Guards the test above: the previous formulation trips sync debug mode.
    values = torch.arange(8, device="cuda")
    mask = torch.tensor([True, False] * 4, device="cuda")
    torch.accelerator.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            values[mask]
    finally:
        torch.cuda.set_sync_debug_mode("default")
