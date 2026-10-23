# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN tensor/state contracts independent of backend and runner implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.runtime_resources import current_runtime_resources, runtime_resources_for
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

if TYPE_CHECKING:
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

GDN_SPEC_METADATA_TENSORS = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass
class GdnStateResources:
    """One engine owns metadata references and shared capture buffers.

    Builders own each state-index tensor; registration borrows those tensors.
    Replacing a registration cannot change another engine with the same names.
    """

    metadata: dict[str, GDN_SPEC_METADATA_TENSORS] = field(default_factory=dict)
    common_buffers: dict[tuple, Any] = field(default_factory=dict)
    state_table_counts: dict[int, int] = field(default_factory=dict)
    diagnostic_suffix: str = ""

    def register(self, layer_names, tensors) -> None:
        for name in layer_names:
            self.metadata[name] = tensors

    def get(self, layer_name, device) -> GDN_SPEC_METADATA_TENSORS:
        tensors = self.metadata.get(layer_name)
        if tensors is None or tensors[0].device != device:
            return _empty_gdn_spec_metadata_tensors(device)
        return tensors


# Standalone callers of the historical API retain a separate compatibility
# owner. Engine execution never consults it, including missing registrations.
_legacy_state_resources = GdnStateResources()


def legacy_state_resources() -> GdnStateResources:
    """The historical standalone owner, independent of import-time contexts."""
    return _legacy_state_resources


def state_resources_for(config=None) -> GdnStateResources:
    resources = (
        runtime_resources_for(config)
        if config is not None
        else current_runtime_resources()
    )
    if resources is None:
        return _legacy_state_resources
    owner = resources.get("gdn_state")
    if owner is None:
        owner = GdnStateResources()
        owner.diagnostic_suffix = f"_engine{id(owner):x}"
        resources["gdn_state"] = owner
    return owner


def register_gdn_spec_metadata_tensors(layer_names, tensors, *, owner=None) -> None:
    (owner if owner is not None else state_resources_for()).register(
        layer_names, tensors
    )


def get_registered_gdn_spec_metadata_tensors(layer_name, device, *, owner=None):
    return (owner if owner is not None else state_resources_for()).get(
        layer_name, device
    )


@dataclass
class GDNSpecDecodeStateContract:
    spec_state_indices_tensor: torch.Tensor
    non_spec_state_indices_tensor: torch.Tensor | None
    num_accepted_tokens: torch.Tensor
    spec_state_slot_selectors: torch.Tensor


def _empty_gdn_spec_metadata_tensors(
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_bool = torch.empty(0, dtype=torch.bool, device=device)
    return (
        empty_i32,
        empty_i32,
        empty_i32,
        empty_i32,
        empty_i32,
        empty_i32,
        empty_bool,
        empty_i32,
        empty_i32,
    )


def gdn_spec_metadata_tensors(
    attn_metadata: GDNAttentionMetadata | None,
    device: torch.device,
) -> GDN_SPEC_METADATA_TENSORS:
    """Return graph-visible active-MTP metadata tensors for Qwen GDN ops."""
    if attn_metadata is None:
        return _empty_gdn_spec_metadata_tensors(device)

    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_bool = torch.empty(0, dtype=torch.bool, device=device)

    def _or_empty_i32(tensor: torch.Tensor | None) -> torch.Tensor:
        return tensor if tensor is not None else empty_i32

    return (
        _or_empty_i32(attn_metadata.non_spec_query_start_loc),
        _or_empty_i32(attn_metadata.non_spec_state_indices_tensor),
        _or_empty_i32(attn_metadata.spec_query_start_loc),
        _or_empty_i32(attn_metadata.spec_state_indices_tensor),
        _or_empty_i32(attn_metadata.spec_token_indx),
        _or_empty_i32(attn_metadata.non_spec_token_indx),
        (
            attn_metadata.spec_sequence_masks
            if attn_metadata.spec_sequence_masks is not None
            else empty_bool
        ),
        _or_empty_i32(attn_metadata.num_accepted_tokens),
        _or_empty_i32(
            attn_metadata.spec_state_slot_selectors
            if attn_metadata.spec_state_slot_selectors is not None
            else attn_metadata.num_accepted_tokens
        ),
    )


def gather_gdn_state_block_ids(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    width: int,
) -> torch.Tensor:
    current_block_idx = torch.clamp((seq_lens - 1) // block_size, min=0)
    offsets = torch.arange(width, device=block_table.device, dtype=torch.long)
    gather_indices = current_block_idx.to(torch.long).unsqueeze(1) + offsets
    gather_indices = torch.clamp(gather_indices, max=block_table.shape[1] - 1)
    return torch.gather(block_table, 1, gather_indices)


def select_state_block_ids(
    block_table: torch.Tensor,
    accepted_tokens: torch.Tensor | None,
    num_spec: int,
    *,
    legacy_slot0: bool = False,
) -> torch.Tensor:
    if legacy_slot0:
        return block_table[:, 0]
    if accepted_tokens is None:
        return block_table[:, 0]
    state_offsets = torch.clamp(
        accepted_tokens.to(device=block_table.device, dtype=torch.long) - 1,
        min=0,
        max=min(num_spec, block_table.shape[1] - 1),
    )
    row_indices = torch.arange(
        block_table.shape[0], device=block_table.device, dtype=torch.long
    )
    return block_table[row_indices, state_offsets]


def build_state_contract(
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
    legacy_slot0: bool = False,
    assert_contract: bool = False,
) -> GDNSpecDecodeStateContract:
    """Build the state-index/count contract consumed by active-MTP GDN.

    ``current_state_block_ids`` is authoritative for align-mode replay because
    it is materialized from the live ``mamba_state_idx`` after preprocess
    rollover. The accepted count historically also selected the committed
    speculative slot as ``num_accepted_tokens - 1`` in the recurrent kernels.
    DDTree can accept a non-linear tree path, so callers may pass
    ``spec_state_slot_selectors`` to select that slot independently.
    """
    assert spec_sequence_masks_cpu.dtype == torch.bool
    assert num_accepted_tokens is not None

    def _mask_for(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == spec_sequence_masks_cpu.device:
            return spec_sequence_masks_cpu
        return spec_sequence_masks_cpu.to(tensor.device, non_blocking=True)

    num_rows = spec_sequence_masks_cpu.numel()
    mask_row_indices: dict[tuple[bool, torch.device], torch.Tensor] = {}

    def _rows(tensor: torch.Tensor, speculative: bool) -> torch.Tensor:
        if spec_sequence_masks_cpu.device.type != "cpu":
            mask = _mask_for(tensor)
            return tensor[mask] if speculative else tensor[~mask]
        if tensor.shape[0] != num_rows:
            raise IndexError(
                f"mask of length {num_rows} does not match tensor rows "
                f"{tensor.shape[0]}"
            )
        key = (speculative, tensor.device)
        index = mask_row_indices.get(key)
        if index is None:
            mask = spec_sequence_masks_cpu if speculative else ~spec_sequence_masks_cpu
            index = mask.nonzero(as_tuple=True)[0].to(tensor.device, non_blocking=True)
            mask_row_indices[key] = index
        return tensor.index_select(0, index)

    if block_table_tensor.shape[0] != num_rows:
        raise IndexError(
            f"mask of length {num_rows} does not match tensor rows "
            f"{block_table_tensor.shape[0]}"
        )
    if spec_state_slot_selectors is None:
        spec_state_slot_selectors = num_accepted_tokens

    all_spec_rows = False
    if current_state_block_ids is not None:
        state_block_ids = current_state_block_ids[:, : num_spec + 1]
        spec_state_indices_tensor = _rows(state_block_ids, True)
        non_spec_source = _rows(state_block_ids, False)
        non_spec_state_indices_tensor = select_state_block_ids(
            non_spec_source,
            _rows(num_accepted_tokens, False),
            num_spec,
            legacy_slot0=legacy_slot0,
        )
    elif is_mamba_cache_all:
        spec_state_indices_tensor = gather_gdn_state_block_ids(
            _rows(block_table_tensor, True),
            _rows(seq_lens, True),
            block_size,
            num_spec + 1,
        )
        non_spec_state_indices_tensor = gather_gdn_state_block_ids(
            _rows(block_table_tensor, False),
            _rows(seq_lens, False),
            block_size,
            1,
        ).squeeze(1)
    else:
        all_spec_rows = bool(spec_sequence_masks_cpu.all().item())
        if all_spec_rows:
            # Preserve the independent, contiguous output of boolean indexing
            # without allocating row indices for the common pure-MTP batch.
            spec_state_indices_tensor = block_table_tensor[:, : num_spec + 1].clone()
            non_spec_state_indices_tensor = block_table_tensor.new_empty((0,))
        else:
            # The request classification is authoritative on CPU. GPU boolean
            # indexing invokes nonzero to discover its dynamic output shape and
            # synchronizes the host. Index selection keeps the same independently
            # allocated result without that device-side shape query.
            spec_rows_cpu = torch.nonzero(spec_sequence_masks_cpu).reshape(-1)
            non_spec_rows_cpu = torch.nonzero(~spec_sequence_masks_cpu).reshape(-1)
            row_indices: dict[tuple[torch.device, bool], torch.Tensor] = {}

            def _select_rows(tensor: torch.Tensor, speculative: bool) -> torch.Tensor:
                key = (tensor.device, speculative)
                indices = row_indices.get(key)
                if indices is None:
                    cpu_indices = spec_rows_cpu if speculative else non_spec_rows_cpu
                    indices = cpu_indices.to(tensor.device, non_blocking=True)
                    row_indices[key] = indices
                return torch.index_select(tensor, 0, indices)

            spec_state_indices_tensor = _select_rows(
                block_table_tensor[:, : num_spec + 1], True
            )
            non_spec_state_indices_tensor = select_state_block_ids(
                _select_rows(block_table_tensor, False),
                _select_rows(num_accepted_tokens, False),
                num_spec,
                legacy_slot0=legacy_slot0,
            )

    if current_state_block_ids is None and not is_mamba_cache_all:
        if all_spec_rows:
            spec_num_accepted_tokens = num_accepted_tokens.clone()
            spec_state_slot_selectors = spec_state_slot_selectors.clone()
        else:
            spec_num_accepted_tokens = _select_rows(num_accepted_tokens, True)
            spec_state_slot_selectors = _select_rows(spec_state_slot_selectors, True)
    else:
        spec_num_accepted_tokens = _rows(num_accepted_tokens, True)
        spec_state_slot_selectors = _rows(spec_state_slot_selectors, True)
    if assert_contract:
        if spec_num_accepted_tokens.numel() != spec_state_indices_tensor.shape[0]:
            raise AssertionError(
                "GDN spec state contract mismatch: accepted-token rows do "
                "not match spec state rows"
            )
        if spec_state_slot_selectors.numel() != spec_state_indices_tensor.shape[0]:
            raise AssertionError(
                "GDN spec state contract mismatch: state-selector rows do "
                "not match spec state rows"
            )
        invalid_accept = (spec_num_accepted_tokens < 1) | (
            spec_num_accepted_tokens > num_spec + 1
        )
        if torch.any(invalid_accept).item():
            raise AssertionError(
                "GDN spec state contract mismatch: num_accepted_tokens must "
                f"be in [1, {num_spec + 1}], got "
                f"{spec_num_accepted_tokens.detach().cpu().tolist()}"
            )
        invalid_selector = (spec_state_slot_selectors < 1) | (
            spec_state_slot_selectors > num_spec + 1
        )
        if torch.any(invalid_selector).item():
            raise AssertionError(
                "GDN spec state contract mismatch: spec_state_slot_selectors "
                f"must be in [1, {num_spec + 1}], got "
                f"{spec_state_slot_selectors.detach().cpu().tolist()}"
            )
        if spec_state_indices_tensor.numel() > 0:
            rows = torch.arange(
                spec_state_indices_tensor.shape[0],
                device=spec_state_indices_tensor.device,
                dtype=torch.long,
            )
            accepted_offsets = (
                spec_state_slot_selectors.to(
                    device=spec_state_indices_tensor.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                - 1
            )
            selected_state_slots = spec_state_indices_tensor[rows, accepted_offsets]
            if torch.any(selected_state_slots == PAD_SLOT_ID).item():
                raise AssertionError(
                    "GDN spec state contract mismatch: accepted slot points "
                    "to PAD_SLOT_ID"
                )
        if current_state_block_ids is not None:
            current_mask = _mask_for(current_state_block_ids)
            active_state_ids = current_state_block_ids[current_mask, : num_spec + 1]
            if torch.any(active_state_ids == PAD_SLOT_ID).item():
                raise AssertionError(
                    "GDN spec state contract mismatch: active align-mode "
                    "state ids contain PAD_SLOT_ID"
                )

    return GDNSpecDecodeStateContract(
        spec_state_indices_tensor=spec_state_indices_tensor,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_accepted_tokens=spec_num_accepted_tokens,
        spec_state_slot_selectors=spec_state_slot_selectors,
    )


class GdnMetadataOverride:
    """Borrow graph-visible tensors for one call and restore on every exit.

    Metadata builders retain ownership. Empty standard-call operands leave the
    prepared metadata intact; commit calls explicitly replace even empty rows.
    Nested borrowing restores the previous view without copying tensor state.
    """

    def __init__(self, metadata, *, skip_empty=False):
        self.metadata = metadata
        self.skip_empty = skip_empty
        self.previous = {}

    def set(self, name, value):
        if self.metadata is None or (
            self.skip_empty and isinstance(value, torch.Tensor) and value.numel() == 0
        ):
            return
        if name not in self.previous:
            self.previous[name] = getattr(self.metadata, name)
        setattr(self.metadata, name, value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name, value in self.previous.items():
            setattr(self.metadata, name, value)
        self.previous.clear()


@dataclass(frozen=True)
class CommonGDNSpecMetadata:
    """Batch-level GDN speculative metadata shared by every cache group."""

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    spec_query_start_loc: torch.Tensor
    non_spec_query_start_loc: torch.Tensor | None
    non_spec_query_start_loc_cpu: torch.Tensor | None
    spec_sequence_masks_cpu: torch.Tensor
    spec_sequence_masks: torch.Tensor
    spec_token_indx: torch.Tensor
    non_spec_token_indx: torch.Tensor


def compute_common_gdn_attn_metadata(
    *,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    num_spec_state_tokens: int,
    legacy_mixed_decode_routing: bool,
) -> CommonGDNSpecMetadata | None:
    """Compute GDN speculative batch metadata once instead of per group.

    State block IDs remain group-specific and are deliberately excluded. This
    is the dependency boundary needed to share request classification, token
    indices, and query offsets without changing recurrent-state ownership.
    """
    if num_spec_state_tokens <= 0 or num_decode_draft_tokens_cpu is None:
        return None

    num_reqs = query_start_loc_cpu.numel() - 1
    if num_decode_draft_tokens_cpu.ndim != 1:
        raise ValueError("num_decode_draft_tokens_cpu must be one-dimensional")
    if num_decode_draft_tokens_cpu.numel() != num_reqs:
        raise ValueError("num_decode_draft_tokens_cpu must align with query_start_loc")

    spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
    num_spec_decodes = int(spec_sequence_masks_cpu.sum().item())
    if num_spec_decodes == 0:
        return None
    num_spec_draft_tokens = int(
        num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()
    )
    if num_spec_draft_tokens == 0:
        return None

    spec_sequence_masks = spec_sequence_masks_cpu.to(
        query_start_loc.device, non_blocking=True
    )
    return prepare_gdn_token_metadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        query_lens_cpu=query_start_loc_cpu[1:] - query_start_loc_cpu[:-1],
        query_lens=None,
        spec_sequence_masks_cpu=spec_sequence_masks_cpu,
        spec_sequence_masks=spec_sequence_masks,
        num_spec_decodes=num_spec_decodes,
        num_spec_state_tokens=num_spec_state_tokens,
        legacy_mixed_decode_routing=legacy_mixed_decode_routing,
    )


def prepare_gdn_token_metadata(
    *,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    query_lens: torch.Tensor | None,
    spec_sequence_masks_cpu: torch.Tensor,
    spec_sequence_masks: torch.Tensor,
    num_spec_decodes: int,
    num_spec_state_tokens: int,
    legacy_mixed_decode_routing: bool,
) -> CommonGDNSpecMetadata:
    """Shared token ordering/ranges; state-index ownership stays with each group."""
    non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
    num_zero_len = int((non_spec_query_lens_cpu == 0).sum().item())

    if legacy_mixed_decode_routing:
        num_decodes = int((non_spec_query_lens_cpu == 1).sum().item())
        num_prefills = non_spec_query_lens_cpu.numel() - num_decodes - num_zero_len
        num_decode_tokens = num_decodes
        num_prefill_tokens = (
            int(non_spec_query_lens_cpu.sum().item()) - num_decode_tokens
        )
    else:
        num_decodes = 0
        num_prefills = non_spec_query_lens_cpu.numel() - num_zero_len
        num_decode_tokens = 0
        num_prefill_tokens = int(non_spec_query_lens_cpu.sum().item())

    num_spec_decode_tokens = (
        int(query_lens_cpu.sum().item()) - num_prefill_tokens - num_decode_tokens
    )
    if num_prefills == 0 and num_decodes == 0:
        spec_token_size = min(
            num_spec_decodes * (num_spec_state_tokens + 1),
            int(query_start_loc_cpu[-1].item()),
        )
        spec_token_indx = torch.arange(
            spec_token_size,
            dtype=torch.int32,
            device=query_start_loc.device,
        )
        non_spec_token_indx = torch.empty(
            0,
            dtype=torch.int32,
            device=query_start_loc.device,
        )
        spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
        non_spec_query_start_loc = None
        non_spec_query_start_loc_cpu = None
    else:
        if query_lens is None:
            query_lens = query_lens_cpu.to(query_start_loc.device, non_blocking=True)
        spec_token_masks = torch.repeat_interleave(
            spec_sequence_masks,
            query_lens,
            output_size=int(query_start_loc_cpu[-1].item()),
        )
        index = torch.argsort(spec_token_masks, stable=True)
        num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
        non_spec_token_indx = index[:num_non_spec_tokens]
        spec_token_indx = index[num_non_spec_tokens:]

        spec_query_start_loc = torch.zeros(
            num_spec_decodes + 1,
            dtype=torch.int32,
            device=query_start_loc.device,
        )
        torch.cumsum(
            query_lens[spec_sequence_masks],
            dim=0,
            out=spec_query_start_loc[1:],
        )
        non_spec_query_start_loc = torch.zeros(
            query_lens.numel() - num_spec_decodes + 1,
            dtype=torch.int32,
            device=query_start_loc.device,
        )
        torch.cumsum(
            query_lens[~spec_sequence_masks],
            dim=0,
            out=non_spec_query_start_loc[1:],
        )
        non_spec_query_start_loc_cpu = torch.zeros(
            query_lens_cpu.numel() - num_spec_decodes + 1,
            dtype=torch.int32,
            device="cpu",
        )
        torch.cumsum(
            query_lens_cpu[~spec_sequence_masks_cpu],
            dim=0,
            out=non_spec_query_start_loc_cpu[1:],
        )

    return CommonGDNSpecMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=num_spec_decode_tokens,
        spec_query_start_loc=spec_query_start_loc,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_query_start_loc_cpu=non_spec_query_start_loc_cpu,
        spec_sequence_masks_cpu=spec_sequence_masks_cpu,
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
    )
