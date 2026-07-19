# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

import json
import os
import time
from dataclasses import dataclass
from typing import Literal

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

logger = init_logger(__name__)

_SM70_GDN_STATE_TABLE_DUMP_COUNTS: dict[int, int] = {}

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
_GDN_SPEC_METADATA_TENSOR_REGISTRY: dict[str, GDN_SPEC_METADATA_TENSORS] = {}


@dataclass
class _GDNDdTreeFastCommonBuffers:
    spec_sequence_masks: torch.Tensor
    spec_token_indx: torch.Tensor
    non_spec_token_indx: torch.Tensor
    spec_query_start_loc: torch.Tensor
    num_accepted_tokens: torch.Tensor
    spec_state_slot_selectors: torch.Tensor
    initialized_key: tuple[int, int, int] | None = None
    updated_epoch: int | None = None
    token_index_initialized_size: int = 0


_GDN_DDTREE_FAST_COMMON_BUFFERS: dict[
    tuple[str, int | None, int, int], _GDNDdTreeFastCommonBuffers
] = {}


def _dflash_ddtree_gdn_shared_common_enabled() -> bool:
    return os.getenv(
        "VLLM_DFLASH_DDTREE_GDN_SHARED_COMMON", "1"
    ).strip().lower() not in ("0", "false", "no", "off", "")


def _get_ddtree_gdn_fast_common_buffers(
    device: torch.device,
    decode_cudagraph_max_bs: int,
    width: int,
) -> _GDNDdTreeFastCommonBuffers:
    key = (device.type, device.index, decode_cudagraph_max_bs, width)
    buffers = _GDN_DDTREE_FAST_COMMON_BUFFERS.get(key)
    if buffers is not None:
        return buffers
    spec_sequence_masks = torch.empty(
        (decode_cudagraph_max_bs,),
        dtype=torch.bool,
        device=device,
    )
    spec_token_indx = torch.empty(
        (decode_cudagraph_max_bs * width,),
        dtype=torch.int32,
        device=device,
    )
    non_spec_token_indx = torch.empty(
        (decode_cudagraph_max_bs * width,),
        dtype=torch.int32,
        device=device,
    )
    spec_query_start_loc = torch.empty(
        (decode_cudagraph_max_bs + 1,),
        dtype=torch.int32,
        device=device,
    )
    num_accepted_tokens = torch.empty(
        (decode_cudagraph_max_bs,),
        dtype=torch.int32,
        device=device,
    )
    spec_state_slot_selectors = torch.empty(
        (decode_cudagraph_max_bs,),
        dtype=torch.int32,
        device=device,
    )
    buffers = _GDNDdTreeFastCommonBuffers(
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        spec_query_start_loc=spec_query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        spec_state_slot_selectors=spec_state_slot_selectors,
    )
    _GDN_DDTREE_FAST_COMMON_BUFFERS[key] = buffers
    return buffers


def _ddtree_trace_path() -> str | None:
    return os.getenv("VLLM_DFLASH_DDTREE_TRACE_JSONL")


def _dflash_ddtree_metadata_profile_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_METADATA_PROFILE", "0") == "1"


def _dflash_ddtree_gdn_fast_build_enabled() -> bool:
    return (
        os.getenv("VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD", "0") == "1"
        and os.getenv("VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD", "0") != "1"
    )


def _dflash_ddtree_gdn_fast_build_cache_enabled() -> bool:
    return (
        os.getenv("VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD_CACHE", "0") == "1"
        and os.getenv("VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD_CACHE", "0") != "1"
    )


def _dflash_ddtree_gdn_fast_build_triton_enabled() -> bool:
    return os.getenv(
        "VLLM_DFLASH_DDTREE_GDN_FAST_BUILD_TRITON", "1"
    ).strip().lower() not in ("0", "false", "no", "off", "")


@triton.jit
def _ddtree_gdn_fast_metadata_kernel(
    state_src,
    spec_state,
    spec_sequence_masks,
    spec_query_start_loc,
    num_accepted_out,
    selector_out,
    num_accepted_src,
    selector_src,
    width: tl.constexpr,
    batch_size: tl.constexpr,
    query_len: tl.constexpr,
    tail_initialized: tl.constexpr,
    update_common: tl.constexpr,
    common_tail_initialized: tl.constexpr,
    state_block: tl.constexpr,
):
    state_offsets = tl.arange(0, state_block)
    state_mask = state_offsets < width
    tl.store(
        spec_state + state_offsets,
        tl.load(state_src + state_offsets, mask=state_mask, other=-1),
        mask=state_mask,
    )
    if update_common:
        tl.store(spec_sequence_masks, True)
        tl.store(spec_query_start_loc, 0)
        tl.store(num_accepted_out, tl.load(num_accepted_src))
        tl.store(selector_out, tl.load(selector_src))

    if not tail_initialized:
        total_state = batch_size * width
        tail_offsets = tl.arange(0, state_block)
        tail_mask = tail_offsets < (total_state - width)
        tl.store(spec_state + width + tail_offsets, -1, mask=tail_mask)

    if update_common and not common_tail_initialized:
        row_offsets = tl.arange(0, state_block)
        row_tail_mask = row_offsets < (batch_size - 1)
        tl.store(spec_sequence_masks + 1 + row_offsets, False, mask=row_tail_mask)
        tl.store(num_accepted_out + 1 + row_offsets, 1, mask=row_tail_mask)
        tl.store(selector_out + 1 + row_offsets, 1, mask=row_tail_mask)

        q_tail_mask = row_offsets < batch_size
        tl.store(spec_query_start_loc + 1 + row_offsets, query_len, mask=q_tail_mask)


def _trace_tensor(value: torch.Tensor | None) -> object:
    if value is None:
        return None
    return value.detach().cpu().tolist()


def _write_ddtree_trace_event(event: str, payload: dict[str, object]) -> None:
    trace_path = _ddtree_trace_path()
    if not trace_path:
        return
    record = {
        "event": event,
        "pid": os.getpid(),
        **payload,
    }
    try:
        with open(trace_path, "a", encoding="utf-8") as trace_file:
            json.dump(record, trace_file, ensure_ascii=True, sort_keys=True)
            trace_file.write("\n")
    except OSError:
        logger.exception("Failed to write DDTree trace event to %s", trace_path)


def _sm70_flashqla_original_prefill_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL")
    if raw is None:
        raw = os.getenv("FLASH_QLA_SM70_USE_ORIGINAL_TILELANG")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _parse_sm70_int_ranges(raw_ranges: str | None) -> set[int] | None:
    if not raw_ranges:
        return None
    values: set[int] = set()
    for raw_part in raw_ranges.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    return values


def _dump_dflash_state_table(payload: dict[str, object]) -> str:
    dump_path = f"/tmp/dflash_state_table_pid{os.getpid()}.pt"
    torch.save(payload, dump_path)
    return dump_path


def _dump_sm70_gdn_state_table(
    payload: dict[str, object],
    seq_lens: torch.Tensor,
    num_prefills: int,
    num_decodes: int,
) -> str | None:
    dump_dir = os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_DIR")
    if not dump_dir:
        return None

    seq_lens_cpu = seq_lens.detach().cpu()
    max_seq_len = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() else 0
    target_seqs = _parse_sm70_int_ranges(
        os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_SEQS")
    )
    if target_seqs is not None and max_seq_len not in target_seqs:
        return None
    start_seq = int(os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_START_SEQ", "0"))
    end_seq = int(os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_END_SEQ", "0"))
    if start_seq and max_seq_len < start_seq:
        return None
    if end_seq and max_seq_len > end_seq:
        return None

    pid = os.getpid()
    count = _SM70_GDN_STATE_TABLE_DUMP_COUNTS.get(pid, 0)
    max_dumps = int(os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_MAX_DUMPS", "32"))
    if count >= max_dumps:
        return None
    _SM70_GDN_STATE_TABLE_DUMP_COUNTS[pid] = count + 1

    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(
        dump_dir,
        "gdn_state_table"
        f"_pid{pid}"
        f"_dump{count:04d}"
        f"_seq{max_seq_len}"
        f"_p{num_prefills}"
        f"_d{num_decodes}.pt",
    )
    torch.save({**payload, "seq_lens_cpu_snapshot": seq_lens_cpu}, dump_path)
    return dump_path


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    spec_state_slot_selectors: torch.Tensor | None = None  # shape: [batch,]
    ddtree_parent_ids: torch.Tensor | None = None  # shape: [batch, tree_slots]
    ddtree_num_tree_tokens_cpu: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


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


def register_gdn_spec_metadata_tensors(
    layer_names: list[str],
    tensors: GDN_SPEC_METADATA_TENSORS,
) -> None:
    for layer_name in layer_names:
        _GDN_SPEC_METADATA_TENSOR_REGISTRY[layer_name] = tensors


def get_registered_gdn_spec_metadata_tensors(
    layer_name: str,
    device: torch.device,
) -> GDN_SPEC_METADATA_TENSORS:
    tensors = _GDN_SPEC_METADATA_TENSOR_REGISTRY.get(layer_name)
    if tensors is None:
        return _empty_gdn_spec_metadata_tensors(device)
    if tensors[0].device != device:
        return _empty_gdn_spec_metadata_tensors(device)
    return tensors


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


def select_gdn_state_block_ids(
    block_table: torch.Tensor,
    accepted_tokens: torch.Tensor | None,
    num_spec: int,
) -> torch.Tensor:
    if envs.VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0:
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


def build_gdn_spec_decode_state_contract(
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

    block_mask = _mask_for(block_table_tensor)
    seq_mask = _mask_for(seq_lens)
    accepted_mask = _mask_for(num_accepted_tokens)
    if spec_state_slot_selectors is None:
        spec_state_slot_selectors = num_accepted_tokens
    selector_mask = _mask_for(spec_state_slot_selectors)

    if current_state_block_ids is not None:
        current_mask = _mask_for(current_state_block_ids)
        state_block_ids = current_state_block_ids[:, : num_spec + 1]
        spec_state_indices_tensor = state_block_ids[current_mask]
        non_spec_source = state_block_ids[~current_mask]
        non_spec_state_indices_tensor = select_gdn_state_block_ids(
            non_spec_source,
            num_accepted_tokens[~accepted_mask],
            num_spec,
        )
    elif is_mamba_cache_all:
        spec_state_indices_tensor = gather_gdn_state_block_ids(
            block_table_tensor[block_mask],
            seq_lens[seq_mask],
            block_size,
            num_spec + 1,
        )
        non_spec_state_indices_tensor = gather_gdn_state_block_ids(
            block_table_tensor[~block_mask],
            seq_lens[~seq_mask],
            block_size,
            1,
        ).squeeze(1)
    else:
        spec_state_indices_tensor = block_table_tensor[block_mask, : num_spec + 1]
        non_spec_state_indices_tensor = select_gdn_state_block_ids(
            block_table_tensor[~block_mask],
            num_accepted_tokens[~accepted_mask],
            num_spec,
        )

    spec_num_accepted_tokens = num_accepted_tokens[accepted_mask]
    spec_state_slot_selectors = spec_state_slot_selectors[selector_mask]
    if os.getenv("VLLM_SM70_GDN_STATE_CONTRACT_ASSERT") == "1":
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


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal[
            "triton", "flashinfer", "cutedsl", "flashqla_sm70"
        ]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
            self.num_spec_state_tokens: int = (
                self.speculative_config.num_speculative_state_tokens()
            )
        else:
            self.num_spec = 0
            self.num_spec_state_tokens = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs
            * (self.num_spec_state_tokens + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec_state_tokens + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec_state_tokens + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec_state_tokens + 1),),
            dtype=torch.int32,
            device=device,
        )
        self._spec_token_indx_initialized_size = 0
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_state_slot_selectors: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self._ddtree_fast_common_buffers: _GDNDdTreeFastCommonBuffers | None = None
        self._ddtree_fast_tail_key: tuple[int, int, int] | None = None
        if (
            self.use_spec_decode
            and envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP
            and _dflash_ddtree_gdn_shared_common_enabled()
        ):
            self._ddtree_fast_common_buffers = _get_ddtree_gdn_fast_common_buffers(
                device,
                self.decode_cudagraph_max_bs,
                self.num_spec_state_tokens + 1,
            )
        if self.use_spec_decode and envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP:
            placeholder_rows = max(
                1, min(self.num_spec_state_tokens + 1, self.decode_cudagraph_max_bs)
            )
            common_buffers = self._ddtree_fast_common_buffers
            self.non_spec_query_start_loc[: placeholder_rows + 1].fill_(0)
            self.non_spec_state_indices_tensor[:placeholder_rows].fill_(PAD_SLOT_ID)
            spec_query_start_loc = (
                common_buffers.spec_query_start_loc
                if common_buffers is not None
                else self.spec_query_start_loc
            )
            spec_sequence_masks = (
                common_buffers.spec_sequence_masks
                if common_buffers is not None
                else self.spec_sequence_masks
            )
            spec_token_indx = (
                common_buffers.spec_token_indx
                if common_buffers is not None
                else self.spec_token_indx
            )
            non_spec_token_indx = (
                common_buffers.non_spec_token_indx
                if common_buffers is not None
                else self.non_spec_token_indx
            )
            num_accepted_tokens = (
                common_buffers.num_accepted_tokens
                if common_buffers is not None
                else self.num_accepted_tokens
            )
            spec_state_slot_selectors = (
                common_buffers.spec_state_slot_selectors
                if common_buffers is not None
                else self.spec_state_slot_selectors
            )
            spec_query_start_loc[: placeholder_rows + 1].fill_(0)
            self.spec_state_indices_tensor[:placeholder_rows].fill_(PAD_SLOT_ID)
            spec_sequence_masks[:placeholder_rows].fill_(False)
            spec_token_indx[:placeholder_rows].copy_(
                torch.arange(
                    placeholder_rows,
                    dtype=torch.int32,
                    device=device,
                )
            )
            if common_buffers is not None:
                common_buffers.token_index_initialized_size = max(
                    common_buffers.token_index_initialized_size,
                    placeholder_rows,
                )
            else:
                self._spec_token_indx_initialized_size = placeholder_rows
            non_spec_token_indx[:0].fill_(0)
            num_accepted_tokens[:placeholder_rows].fill_(1)
            spec_state_slot_selectors[:placeholder_rows].fill_(1)
            register_gdn_spec_metadata_tensors(
                self.layer_names,
                (
                    self.non_spec_query_start_loc[: placeholder_rows + 1],
                    self.non_spec_state_indices_tensor[:placeholder_rows],
                    spec_query_start_loc[: placeholder_rows + 1],
                    self.spec_state_indices_tensor[:placeholder_rows],
                    spec_token_indx[:placeholder_rows],
                    non_spec_token_indx[:0],
                    spec_sequence_masks[:placeholder_rows],
                    num_accepted_tokens[:placeholder_rows],
                    spec_state_slot_selectors[:placeholder_rows],
                ),
            )

    def _build_fast_pure_ddtree_full_graph(
        self,
        *,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None,
        spec_state_slot_selectors: torch.Tensor | None,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
        current_state_block_ids: torch.Tensor | None,
        ddtree_parent_ids: torch.Tensor | None,
        ddtree_num_tree_tokens_cpu: torch.Tensor | None,
        for_cudagraph_capture: bool,
        fast_build_epoch: int | None,
        metadata_profile: bool,
        metadata_profile_t0: float,
    ) -> GDNAttentionMetadata | None:
        def _miss(reason: str) -> GDNAttentionMetadata | None:
            if os.getenv("VLLM_DFLASH_DDTREE_FAST_BUILD_DEBUG", "0") != "1":
                return None
            count = getattr(self, "_ddtree_fast_build_miss_count", 0)
            if count >= 8:
                return None
            self._ddtree_fast_build_miss_count = count + 1
            logger.info(
                "DFLASH_DDTREE_METADATA_PROFILE gdn_build_fast_miss "
                "reason=%s full_graph=%s spec=%s core_op=%s "
                "has_parent=%s has_tree_tokens=%s has_accept=%s "
                "has_draft=%s has_state_ids=%s",
                reason,
                self.use_full_cuda_graph,
                self.use_spec_decode,
                envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP,
                ddtree_parent_ids is not None,
                ddtree_num_tree_tokens_cpu is not None,
                num_accepted_tokens is not None,
                num_decode_draft_tokens_cpu is not None,
                current_state_block_ids is not None,
            )
            return None

        if (
            for_cudagraph_capture
            or not self.use_full_cuda_graph
            or not self.use_spec_decode
            or not envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP
            or ddtree_parent_ids is None
            or ddtree_num_tree_tokens_cpu is None
            or num_accepted_tokens is None
            or num_decode_draft_tokens_cpu is None
        ):
            return _miss("missing_required")

        m = common_attn_metadata
        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        num_reqs = query_start_loc_cpu.numel() - 1
        if num_reqs != 1:
            return None
        if (
            ddtree_parent_ids.ndim != 2
            or ddtree_parent_ids.shape[0] < 1
            or ddtree_num_tree_tokens_cpu.ndim != 1
            or ddtree_num_tree_tokens_cpu.numel() < 1
            or num_accepted_tokens.ndim != 1
            or num_accepted_tokens.numel() < 1
            or num_decode_draft_tokens_cpu.ndim != 1
            or num_decode_draft_tokens_cpu.numel() < 1
            or (
                current_state_block_ids is not None
                and (
                    current_state_block_ids.ndim != 2
                    or current_state_block_ids.shape[0] < 1
                )
            )
        ):
            return _miss("shape")

        query_len = int((query_start_loc_cpu[1] - query_start_loc_cpu[0]).item())
        tree_tokens = int(ddtree_num_tree_tokens_cpu[0].item())
        if (
            int(query_start_loc_cpu[0].item()) != 0
            or query_len <= 1
            or tree_tokens + 1 != query_len
            or int(num_decode_draft_tokens_cpu[0].item()) <= 0
        ):
            return _miss("not_pure_ddtree")

        width = self.num_spec_state_tokens + 1
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        is_mamba_cache_all = self.vllm_config.cache_config.mamba_cache_mode == "all"
        if current_state_block_ids is not None:
            fast_path_state_block_ids = current_state_block_ids[:1, :width]
        elif is_mamba_cache_all:
            seq_lens_cpu_for_state = m._seq_lens_cpu
            if seq_lens_cpu_for_state is not None:
                seq_len = int(seq_lens_cpu_for_state[0].item())
                start_idx = max((seq_len - 1) // self.kv_cache_spec.block_size, 0)
                end_idx = start_idx + width
                if end_idx > block_table_tensor.shape[1]:
                    return _miss("state_block_table_capacity")
                fast_path_state_block_ids = block_table_tensor[:1, start_idx:end_idx]
            else:
                fast_path_state_block_ids = gather_gdn_state_block_ids(
                    block_table_tensor[:1],
                    m.seq_lens[:1],
                    self.kv_cache_spec.block_size,
                    width,
                )
        elif width <= block_table_tensor.shape[1]:
            fast_path_state_block_ids = block_table_tensor[:1, :width]
        else:
            return _miss("state_block_table_capacity")

        batch_size = int(m.num_actual_tokens)
        if (
            batch_size < 1
            or batch_size > self.decode_cudagraph_max_bs
            or query_len > self.decode_cudagraph_max_bs
            or self.spec_state_indices_tensor.shape[1] < width
        ):
            return _miss("capacity")

        spec_token_size = min(width, query_len)
        if spec_token_size > self.spec_token_indx.numel():
            return _miss("token_index_capacity")
        common_buffers = self._ddtree_fast_common_buffers
        if common_buffers is not None:
            spec_token_capacity = common_buffers.spec_token_indx.numel()
            token_index_initialized_size = common_buffers.token_index_initialized_size
        else:
            spec_token_capacity = self.spec_token_indx.numel()
            token_index_initialized_size = self._spec_token_indx_initialized_size
        if spec_token_size > spec_token_capacity:
            return _miss("token_index_capacity")
        if spec_token_size > token_index_initialized_size:
            start_idx = token_index_initialized_size
            token_index_buffer = (
                common_buffers.spec_token_indx
                if common_buffers is not None
                else self.spec_token_indx
            )
            token_index_buffer[start_idx:spec_token_size].copy_(
                torch.arange(
                    start_idx,
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                ),
                non_blocking=True,
            )
            if common_buffers is not None:
                common_buffers.token_index_initialized_size = spec_token_size
            else:
                self._spec_token_indx_initialized_size = spec_token_size

        static_key = (batch_size, query_len, width)
        tail_initialized = getattr(self, "_ddtree_fast_tail_key", None) == static_key
        common_tail_initialized = (
            common_buffers is not None and common_buffers.initialized_key == static_key
        )
        update_common = True
        if common_buffers is not None and fast_build_epoch is not None:
            update_common = (
                common_buffers.updated_epoch != fast_build_epoch
                or not common_tail_initialized
            )

        profile_graph_buffers_t0 = time.perf_counter() if metadata_profile else 0.0
        spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
        spec_sequence_masks_source = (
            common_buffers.spec_sequence_masks
            if common_buffers is not None
            else self.spec_sequence_masks
        )
        spec_token_indx_source = (
            common_buffers.spec_token_indx
            if common_buffers is not None
            else self.spec_token_indx
        )
        non_spec_token_indx_source = (
            common_buffers.non_spec_token_indx
            if common_buffers is not None
            else self.non_spec_token_indx
        )
        spec_query_start_loc_source = (
            common_buffers.spec_query_start_loc
            if common_buffers is not None
            else self.spec_query_start_loc
        )
        num_accepted_tokens_source = (
            common_buffers.num_accepted_tokens
            if common_buffers is not None
            else self.num_accepted_tokens
        )
        spec_state_slot_selectors_source = (
            common_buffers.spec_state_slot_selectors
            if common_buffers is not None
            else self.spec_state_slot_selectors
        )
        spec_sequence_masks = spec_sequence_masks_source[:batch_size]
        spec_token_indx = spec_token_indx_source[:spec_token_size]
        non_spec_token_indx = non_spec_token_indx_source[:0]
        spec_query_start_loc = spec_query_start_loc_source[: batch_size + 1]
        num_accepted_tokens_padded = num_accepted_tokens_source[:batch_size]

        selector_source = (
            num_accepted_tokens
            if spec_state_slot_selectors is None
            else spec_state_slot_selectors
        )
        if selector_source.ndim != 1 or selector_source.numel() < 1:
            return _miss("selector_shape")
        spec_state_slot_selectors_padded = spec_state_slot_selectors_source[:batch_size]
        use_triton_update = (
            _dflash_ddtree_gdn_fast_build_triton_enabled()
            and fast_path_state_block_ids.is_cuda
            and num_accepted_tokens.is_cuda
            and selector_source.is_cuda
            and spec_state_indices_tensor.is_contiguous()
            and fast_path_state_block_ids.stride(-1) == 1
        )
        if use_triton_update:
            state_block = (
                1 << max(batch_size * width, batch_size + 1, width).bit_length()
            )
            _ddtree_gdn_fast_metadata_kernel[(1,)](
                fast_path_state_block_ids,
                spec_state_indices_tensor,
                spec_sequence_masks,
                spec_query_start_loc,
                num_accepted_tokens_padded,
                spec_state_slot_selectors_padded,
                num_accepted_tokens,
                selector_source,
                width,
                batch_size,
                query_len,
                tail_initialized,
                update_common,
                common_tail_initialized
                if common_buffers is not None
                else tail_initialized,
                state_block,
            )
        else:
            spec_state_indices_tensor[:1].copy_(
                fast_path_state_block_ids, non_blocking=True
            )
            if not tail_initialized:
                spec_state_indices_tensor[1:].fill_(PAD_SLOT_ID)

            if update_common:
                spec_sequence_masks[:1].fill_(True)
                if not (
                    common_tail_initialized
                    if common_buffers is not None
                    else tail_initialized
                ):
                    spec_sequence_masks[1:].fill_(False)

                spec_query_start_loc[:1].fill_(0)
                if not (
                    common_tail_initialized
                    if common_buffers is not None
                    else tail_initialized
                ):
                    spec_query_start_loc[1:].fill_(query_len)

                num_accepted_tokens_padded[:1].copy_(
                    num_accepted_tokens[:1], non_blocking=True
                )
                if not (
                    common_tail_initialized
                    if common_buffers is not None
                    else tail_initialized
                ):
                    num_accepted_tokens_padded[1:].fill_(1)

                spec_state_slot_selectors_padded[:1].copy_(
                    selector_source[:1], non_blocking=True
                )
                if not (
                    common_tail_initialized
                    if common_buffers is not None
                    else tail_initialized
                ):
                    spec_state_slot_selectors_padded[1:].fill_(1)
        self._ddtree_fast_tail_key = static_key
        if common_buffers is not None and update_common:
            common_buffers.initialized_key = static_key
            common_buffers.updated_epoch = fast_build_epoch

        profile_graph_buffers_ms = 0.0
        if metadata_profile:
            profile_graph_buffers_ms = (
                time.perf_counter() - profile_graph_buffers_t0
            ) * 1000.0

        profile_register_t0 = time.perf_counter() if metadata_profile else 0.0
        metadata_key = (
            static_key,
            int(ddtree_parent_ids.data_ptr()),
            tuple(ddtree_parent_ids.shape),
            int(ddtree_num_tree_tokens_cpu.data_ptr()),
            tuple(ddtree_num_tree_tokens_cpu.shape),
        )
        cache_metadata = _dflash_ddtree_gdn_fast_build_cache_enabled()
        cached_metadata = None
        if (
            cache_metadata
            and getattr(self, "_ddtree_fast_metadata_key", None) == metadata_key
        ):
            cached_metadata = getattr(self, "_ddtree_fast_metadata", None)
        if cached_metadata is None:
            attn_metadata = GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=1,
                num_spec_decode_tokens=query_len,
                num_actual_tokens=m.num_actual_tokens,
                has_initial_state=None,
                chunk_indices=None,
                chunk_offsets=None,
                spec_query_start_loc=spec_query_start_loc,
                non_spec_query_start_loc=None,
                spec_state_indices_tensor=spec_state_indices_tensor,
                non_spec_state_indices_tensor=None,
                spec_sequence_masks=spec_sequence_masks,
                spec_token_indx=spec_token_indx,
                non_spec_token_indx=non_spec_token_indx,
                num_accepted_tokens=num_accepted_tokens_padded,
                spec_state_slot_selectors=spec_state_slot_selectors_padded,
                ddtree_parent_ids=ddtree_parent_ids,
                ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
                nums_dict=None,
                batch_ptr=None,
                token_chunk_offset_ptr=None,
            )
            register_gdn_spec_metadata_tensors(
                self.layer_names,
                gdn_spec_metadata_tensors(attn_metadata, query_start_loc.device),
            )
            if cache_metadata:
                self._ddtree_fast_metadata_key = metadata_key
                self._ddtree_fast_metadata = attn_metadata
        else:
            attn_metadata = cached_metadata
        profile_register_ms = 0.0
        if metadata_profile:
            profile_register_ms = (time.perf_counter() - profile_register_t0) * 1000.0
            logger.info(
                "DFLASH_DDTREE_METADATA_PROFILE gdn_build_fast total_ms=%.3f "
                "graph_buffers_ms=%.3f register_ms=%.3f tail_cached=%s "
                "common_cached=%s common_updated=%s cache_hit=%s "
                "num_spec_decodes=%d num_spec_decode_tokens=%d "
                "num_actual_tokens=%d full_graph=%s ddtree=%s layers=%d",
                (time.perf_counter() - metadata_profile_t0) * 1000.0,
                profile_graph_buffers_ms,
                profile_register_ms,
                tail_initialized,
                common_tail_initialized,
                update_common,
                cached_metadata is not None,
                1,
                query_len,
                m.num_actual_tokens,
                self.use_full_cuda_graph,
                True,
                len(self.layer_names),
            )

        return attn_metadata

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        spec_state_slot_selectors: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        spec_sequence_masks_cpu: torch.Tensor | None = None,
        current_state_block_ids: torch.Tensor | None = None,
        ddtree_parent_ids: torch.Tensor | None = None,
        ddtree_num_tree_tokens_cpu: torch.Tensor | None = None,
        ddtree_fast_build_epoch: int | None = None,
        for_cudagraph_capture: bool = False,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        metadata_profile = _dflash_ddtree_metadata_profile_enabled()
        metadata_profile_t0 = time.perf_counter() if metadata_profile else 0.0
        profile_state_contract_ms = 0.0
        profile_graph_buffers_ms = 0.0
        profile_register_ms = 0.0
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        if fast_build and _dflash_ddtree_gdn_fast_build_enabled():
            fast_metadata = self._build_fast_pure_ddtree_full_graph(
                common_attn_metadata=common_attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                spec_state_slot_selectors=spec_state_slot_selectors,
                num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                current_state_block_ids=current_state_block_ids,
                ddtree_parent_ids=ddtree_parent_ids,
                ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
                for_cudagraph_capture=for_cudagraph_capture,
                fast_build_epoch=ddtree_fast_build_epoch,
                metadata_profile=metadata_profile,
                metadata_profile_t0=metadata_profile_t0,
            )
            if fast_metadata is not None:
                return fast_metadata
        self._ddtree_fast_tail_key = None
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        is_mamba_cache_all = self.vllm_config.cache_config.mamba_cache_mode == "all"

        num_reqs = query_start_loc_cpu.numel() - 1
        if spec_sequence_masks_cpu is not None:
            assert spec_sequence_masks_cpu.dtype == torch.bool
            assert spec_sequence_masks_cpu.ndim == 1
            assert spec_sequence_masks_cpu.numel() == num_reqs, (
                f"spec_sequence_masks_cpu.shape={tuple(spec_sequence_masks_cpu.shape)} "
                f"must align with num_reqs={num_reqs}"
            )
        if num_decode_draft_tokens_cpu is not None:
            assert num_decode_draft_tokens_cpu.ndim == 1
            assert num_decode_draft_tokens_cpu.numel() == num_reqs, (
                "num_decode_draft_tokens_cpu must align with query_start_loc"
            )
        if num_accepted_tokens is not None:
            assert num_accepted_tokens.ndim == 1
            assert num_accepted_tokens.numel() == num_reqs, (
                "num_accepted_tokens must align with query_start_loc"
            )
        if spec_state_slot_selectors is not None:
            assert spec_state_slot_selectors.ndim == 1
            assert spec_state_slot_selectors.numel() == num_reqs, (
                "spec_state_slot_selectors must align with query_start_loc"
            )
        if ddtree_parent_ids is not None:
            assert ddtree_parent_ids.ndim == 2
            assert ddtree_parent_ids.shape[0] == num_reqs, (
                "ddtree_parent_ids must align with query_start_loc"
            )
            assert ddtree_num_tree_tokens_cpu is not None
            assert ddtree_num_tree_tokens_cpu.ndim == 1
            assert ddtree_num_tree_tokens_cpu.numel() == num_reqs, (
                "ddtree_num_tree_tokens_cpu must align with query_start_loc"
            )

        if not self.use_spec_decode:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            if spec_sequence_masks_cpu is None:
                if num_decode_draft_tokens_cpu is None:
                    spec_sequence_masks = None
                    num_spec_decodes = 0
                    spec_sequence_masks_cpu = None
                else:
                    spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            if (
                spec_sequence_masks_cpu is None
                or spec_sequence_masks_cpu.sum().item() == 0
            ):
                spec_sequence_masks = None
                num_spec_decodes = 0
                spec_sequence_masks_cpu = None
            else:
                if num_decode_draft_tokens_cpu is not None:
                    num_spec_draft_tokens = (
                        num_decode_draft_tokens_cpu[spec_sequence_masks_cpu]
                        .sum()
                        .item()
                    )
                    if num_spec_draft_tokens == 0:
                        spec_sequence_masks = None
                        num_spec_decodes = 0
                        spec_sequence_masks_cpu = None
                    else:
                        num_spec_decodes = spec_sequence_masks_cpu.sum().item()
                        spec_sequence_masks = spec_sequence_masks_cpu.to(
                            query_start_loc.device, non_blocking=True
                        )
                else:
                    num_spec_decodes = spec_sequence_masks_cpu.sum().item()
                    spec_sequence_masks = spec_sequence_masks_cpu.to(
                        query_start_loc.device, non_blocking=True
                    )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            if is_mamba_cache_all:
                non_spec_state_indices_tensor = gather_gdn_state_block_ids(
                    block_table_tensor,
                    m.seq_lens,
                    self.kv_cache_spec.block_size,
                    1,
                ).squeeze(1)
            else:
                non_spec_state_indices_tensor = select_gdn_state_block_ids(
                    block_table_tensor,
                    num_accepted_tokens,
                    self.num_spec_state_tokens,
                )
            if num_prefills == 0:
                query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
                if torch.any(query_lens_cpu == 0):
                    decode_lane_mask = (query_lens_cpu > 0).to(
                        device=non_spec_state_indices_tensor.device,
                        non_blocking=True,
                    )
                    non_spec_state_indices_tensor = torch.where(
                        decode_lane_mask,
                        non_spec_state_indices_tensor,
                        torch.full_like(non_spec_state_indices_tensor, PAD_SLOT_ID),
                    )
            if for_cudagraph_capture and num_prefills == 0:
                # Capture/dummy runs must not mutate real recurrent state.
                # State slot 0 is live, so padding has to use PAD_SLOT_ID.
                non_spec_state_indices_tensor = torch.full_like(
                    non_spec_state_indices_tensor, PAD_SLOT_ID
                )
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
            if (
                self.use_full_cuda_graph
                and self.use_spec_decode
                and envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP
            ):
                placeholder_rows = min(
                    self.num_spec_state_tokens + 1,
                    self.decode_cudagraph_max_bs,
                )
                self.spec_state_indices_tensor[:placeholder_rows].fill_(PAD_SLOT_ID)
                spec_state_indices_tensor = self.spec_state_indices_tensor[
                    :placeholder_rows
                ]
                self.spec_sequence_masks[:placeholder_rows].fill_(False)
                spec_sequence_masks = self.spec_sequence_masks[:placeholder_rows]
                self.spec_query_start_loc[: placeholder_rows + 1].fill_(0)
                spec_query_start_loc = self.spec_query_start_loc[: placeholder_rows + 1]
                self.spec_token_indx[:placeholder_rows].copy_(
                    torch.arange(
                        placeholder_rows,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    ),
                    non_blocking=True,
                )
                spec_token_indx = self.spec_token_indx[:placeholder_rows]
                self.num_accepted_tokens[:placeholder_rows].fill_(1)
                num_accepted_tokens = self.num_accepted_tokens[:placeholder_rows]
        else:
            assert spec_sequence_masks_cpu is not None
            assert num_accepted_tokens is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            pure_ddtree_spec_fast_path_candidate = (
                ddtree_parent_ids is not None
                and num_spec_decodes > 0
                and non_spec_query_lens_cpu.size(0) == num_zero_len
            )
            fast_path_state_block_ids: torch.Tensor | None = None
            if pure_ddtree_spec_fast_path_candidate:
                if current_state_block_ids is not None:
                    fast_path_state_block_ids = current_state_block_ids[
                        :num_spec_decodes, : self.num_spec_state_tokens + 1
                    ]
                elif bool(torch.all(spec_sequence_masks_cpu).item()):
                    fast_path_width = self.num_spec_state_tokens + 1
                    if is_mamba_cache_all:
                        seq_lens_cpu_for_state = m._seq_lens_cpu
                        if seq_lens_cpu_for_state is not None and num_spec_decodes == 1:
                            seq_len = int(seq_lens_cpu_for_state[0].item())
                            start_idx = max(
                                (seq_len - 1) // self.kv_cache_spec.block_size,
                                0,
                            )
                            end_idx = start_idx + fast_path_width
                            if end_idx <= block_table_tensor.shape[1]:
                                fast_path_state_block_ids = block_table_tensor[
                                    :1, start_idx:end_idx
                                ]
                        else:
                            fast_path_state_block_ids = gather_gdn_state_block_ids(
                                block_table_tensor[:num_spec_decodes],
                                m.seq_lens[:num_spec_decodes],
                                self.kv_cache_spec.block_size,
                                fast_path_width,
                            )
                    elif fast_path_width <= block_table_tensor.shape[1]:
                        fast_path_state_block_ids = block_table_tensor[
                            :num_spec_decodes, :fast_path_width
                        ]

            pure_ddtree_spec_fast_path = fast_path_state_block_ids is not None
            if pure_ddtree_spec_fast_path:
                profile_state_contract_t0 = (
                    time.perf_counter() if metadata_profile else 0.0
                )
                num_decodes = 0
                num_prefills = 0
                num_decode_tokens = 0
                num_prefill_tokens = 0
                num_spec_decode_tokens = query_lens_cpu.sum().item()
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec_state_tokens + 1),
                    query_start_loc_cpu[-1].item(),
                )
                if spec_token_size <= self.spec_token_indx.numel():
                    if spec_token_size > self._spec_token_indx_initialized_size:
                        start_idx = self._spec_token_indx_initialized_size
                        self.spec_token_indx[start_idx:spec_token_size].copy_(
                            torch.arange(
                                start_idx,
                                spec_token_size,
                                dtype=torch.int32,
                                device=query_start_loc.device,
                            ),
                            non_blocking=True,
                        )
                        self._spec_token_indx_initialized_size = spec_token_size
                    spec_token_indx = self.spec_token_indx[:spec_token_size]
                else:
                    spec_token_indx = torch.arange(
                        spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                non_spec_token_indx = self.non_spec_token_indx[:0]
                spec_state_indices_tensor = fast_path_state_block_ids
                if for_cudagraph_capture:
                    spec_state_indices_tensor = torch.full_like(
                        spec_state_indices_tensor, PAD_SLOT_ID
                    )
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
                if spec_state_slot_selectors is None:
                    spec_state_slot_selectors = num_accepted_tokens
                else:
                    spec_state_slot_selectors = spec_state_slot_selectors[
                        :num_spec_decodes
                    ]
                if metadata_profile:
                    profile_state_contract_ms = (
                        time.perf_counter() - profile_state_contract_t0
                    ) * 1000.0
            else:
                # query_start_loc may be padded for CUDA graph replay. The CPU
                # metadata is authoritative for the live request count here.
                query_lens = query_lens_cpu.to(
                    query_start_loc.device, non_blocking=True
                )
                profile_state_contract_t0 = (
                    time.perf_counter() if metadata_profile else 0.0
                )
                state_contract = build_gdn_spec_decode_state_contract(
                    block_table_tensor=block_table_tensor,
                    seq_lens=m.seq_lens,
                    block_size=self.kv_cache_spec.block_size,
                    num_spec=self.num_spec_state_tokens,
                    spec_sequence_masks_cpu=spec_sequence_masks_cpu,
                    num_accepted_tokens=num_accepted_tokens,
                    current_state_block_ids=current_state_block_ids,
                    is_mamba_cache_all=is_mamba_cache_all,
                    spec_state_slot_selectors=spec_state_slot_selectors,
                )
                if metadata_profile:
                    profile_state_contract_ms = (
                        time.perf_counter() - profile_state_contract_t0
                    ) * 1000.0

                if envs.VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING:
                    # 0.0.3 kept ordinary query_len==1 rows on the decode path even
                    # when another row was running speculative verification. This
                    # is an A/B guard for MTP-only recurrent-state corruption.
                    num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
                    num_prefills = (
                        non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
                    )
                    num_decode_tokens = num_decodes
                    num_prefill_tokens = (
                        non_spec_query_lens_cpu.sum().item() - num_decode_tokens
                    )
                else:
                    # When active spec decodes are present, route non-spec requests
                    # through the prefill path so mixed batches keep separate GDN
                    # state metadata for spec and non-spec tokens.
                    num_decodes = 0
                    num_prefills = non_spec_query_lens_cpu.size(0) - num_zero_len
                    num_decode_tokens = 0
                    num_prefill_tokens = non_spec_query_lens_cpu.sum().item()
                num_spec_decode_tokens = (
                    query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
                )

                if num_prefills == 0 and num_decodes == 0:
                    spec_token_size = min(
                        num_spec_decodes * (self.num_spec_state_tokens + 1),
                        query_start_loc_cpu[-1].item(),
                    )
                    spec_token_indx = torch.arange(
                        spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    non_spec_token_indx = torch.empty(
                        0, dtype=torch.int32, device=query_start_loc.device
                    )
                    spec_state_indices_tensor = state_contract.spec_state_indices_tensor
                    if for_cudagraph_capture:
                        spec_state_indices_tensor = torch.full_like(
                            spec_state_indices_tensor, PAD_SLOT_ID
                        )
                    non_spec_state_indices_tensor = None
                    # Padded sequences are always at the back, so the first
                    # num_spec_decodes + 1 entries of query_start_loc already
                    # contain the correct cumulative token counts.
                    spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                    non_spec_query_start_loc = None
                    non_spec_query_start_loc_cpu = None
                else:
                    spec_token_masks = torch.repeat_interleave(
                        spec_sequence_masks,
                        query_lens,
                        output_size=query_start_loc_cpu[-1].item(),
                    )
                    index = torch.argsort(spec_token_masks, stable=True)
                    num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                    non_spec_token_indx = index[:num_non_spec_tokens]
                    spec_token_indx = index[num_non_spec_tokens:]

                    spec_state_indices_tensor = state_contract.spec_state_indices_tensor
                    non_spec_state_indices_tensor = (
                        state_contract.non_spec_state_indices_tensor
                    )
                    if for_cudagraph_capture:
                        spec_state_indices_tensor = torch.full_like(
                            spec_state_indices_tensor, PAD_SLOT_ID
                        )
                        non_spec_state_indices_tensor = torch.full_like(
                            non_spec_state_indices_tensor, PAD_SLOT_ID
                        )

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
                        query_lens.size(0) - num_spec_decodes + 1,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    torch.cumsum(
                        query_lens[~spec_sequence_masks],
                        dim=0,
                        out=non_spec_query_start_loc[1:],
                    )
                    non_spec_query_start_loc_cpu = torch.zeros(
                        query_lens_cpu.size(0) - num_spec_decodes + 1,
                        dtype=torch.int32,
                        device="cpu",
                    )
                    torch.cumsum(
                        query_lens_cpu[~spec_sequence_masks_cpu],
                        dim=0,
                        out=non_spec_query_start_loc_cpu[1:],
                    )

                num_accepted_tokens = state_contract.num_accepted_tokens
                spec_state_slot_selectors = state_contract.spec_state_slot_selectors
            assert spec_query_start_loc is not None
            assert spec_query_start_loc[-1].item() == num_spec_decode_tokens
            assert spec_state_indices_tensor is not None
            assert spec_state_indices_tensor.shape[0] == num_spec_decodes

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        flashqla_original_prefill = (
            self.gdn_prefill_backend == "flashqla_sm70"
            and _sm70_flashqla_original_prefill_enabled()
        )
        if num_prefills > 0 and (
            self.gdn_prefill_backend != "flashqla_sm70" or flashqla_original_prefill
        ):
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            if self.gdn_prefill_backend == "cutedsl":
                from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                    prepare_metadata_cutedsl,
                )

                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                total_tokens = int(non_spec_query_start_loc_cpu[-1].item())
                chunk_indices, chunk_offsets = prepare_metadata_cutedsl(
                    non_spec_query_start_loc,
                    total_tokens,
                    FLA_CHUNK_SIZE,
                )
            else:
                gpu_device = query_start_loc.device
                # Only prefill batches use FLA chunk ops.
                # Pre-compute on CPU and async-copy to GPU to avoid
                # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
                from vllm.model_executor.layers.fla.ops.index import (
                    prepare_chunk_indices,
                    prepare_chunk_offsets,
                )

                assert non_spec_query_start_loc_cpu is not None
                chunk_indices = prepare_chunk_indices(
                    non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE
                ).to(device=gpu_device, non_blocking=True)
                chunk_offsets = prepare_chunk_offsets(
                    non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE
                ).to(device=gpu_device, non_blocking=True)

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            profile_graph_buffers_t0 = time.perf_counter() if metadata_profile else 0.0
            common_buffers = self._ddtree_fast_common_buffers
            spec_sequence_masks_buffer = (
                common_buffers.spec_sequence_masks
                if common_buffers is not None
                else self.spec_sequence_masks
            )
            spec_token_indx_buffer = (
                common_buffers.spec_token_indx
                if common_buffers is not None
                else self.spec_token_indx
            )
            non_spec_token_indx_buffer = (
                common_buffers.non_spec_token_indx
                if common_buffers is not None
                else self.non_spec_token_indx
            )
            spec_query_start_loc_buffer = (
                common_buffers.spec_query_start_loc
                if common_buffers is not None
                else self.spec_query_start_loc
            )
            num_accepted_tokens_buffer = (
                common_buffers.num_accepted_tokens
                if common_buffers is not None
                else self.num_accepted_tokens
            )
            spec_state_slot_selectors_buffer = (
                common_buffers.spec_state_slot_selectors
                if common_buffers is not None
                else self.spec_state_slot_selectors
            )
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            spec_sequence_masks_buffer[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = spec_sequence_masks_buffer[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            if non_spec_token_indx.numel() > 0:
                non_spec_token_indx_buffer[: non_spec_token_indx.size(0)].copy_(
                    non_spec_token_indx, non_blocking=True
                )
            non_spec_token_indx = non_spec_token_indx_buffer[
                : non_spec_token_indx.size(0)
            ]

            if (
                spec_token_indx.numel() > 0
                and spec_token_indx.data_ptr() != spec_token_indx_buffer.data_ptr()
            ):
                spec_token_indx_buffer[: spec_token_indx.size(0)].copy_(
                    spec_token_indx, non_blocking=True
                )
            if common_buffers is not None:
                common_buffers.token_index_initialized_size = max(
                    common_buffers.token_index_initialized_size,
                    spec_token_indx.size(0),
                )
            spec_token_indx = spec_token_indx_buffer[: spec_token_indx.size(0)]

            spec_query_start_loc_buffer[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = spec_query_start_loc_buffer[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            num_accepted_tokens_buffer[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = num_accepted_tokens_buffer[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

            spec_state_slot_selectors_buffer[:num_spec_decodes].copy_(
                spec_state_slot_selectors, non_blocking=True
            )
            spec_state_slot_selectors = spec_state_slot_selectors_buffer[:batch_size]
            spec_state_slot_selectors[num_spec_decodes:].fill_(1)
            if common_buffers is not None:
                common_buffers.initialized_key = (
                    batch_size,
                    int(num_spec_decode_tokens),
                    self.num_spec_state_tokens + 1,
                )
            if metadata_profile:
                profile_graph_buffers_ms = (
                    time.perf_counter() - profile_graph_buffers_t0
                ) * 1000.0

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor,
                non_blocking=True,
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            spec_state_slot_selectors=spec_state_slot_selectors,
            ddtree_parent_ids=ddtree_parent_ids,
            ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        if ddtree_parent_ids is not None and _ddtree_trace_path():
            _write_ddtree_trace_event(
                "gdn_metadata",
                {
                    "for_cudagraph_capture": for_cudagraph_capture,
                    "num_prefills": num_prefills,
                    "num_decodes": num_decodes,
                    "num_spec_decodes": num_spec_decodes,
                    "num_spec_decode_tokens": num_spec_decode_tokens,
                    "num_actual_tokens": m.num_actual_tokens,
                    "seq_lens": _trace_tensor(m.seq_lens),
                    "query_start_loc_cpu": _trace_tensor(query_start_loc_cpu),
                    "num_accepted_tokens": _trace_tensor(num_accepted_tokens),
                    "spec_state_slot_selectors": _trace_tensor(
                        spec_state_slot_selectors
                    ),
                    "spec_query_start_loc": _trace_tensor(spec_query_start_loc),
                    "spec_state_indices_tensor": _trace_tensor(
                        spec_state_indices_tensor
                    ),
                    "current_state_block_ids": _trace_tensor(current_state_block_ids),
                    "ddtree_parent_ids": _trace_tensor(ddtree_parent_ids),
                    "ddtree_num_tree_tokens_cpu": _trace_tensor(
                        ddtree_num_tree_tokens_cpu
                    ),
                },
            )
        if self.use_spec_decode and envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP:
            profile_register_t0 = time.perf_counter() if metadata_profile else 0.0
            register_gdn_spec_metadata_tensors(
                self.layer_names,
                gdn_spec_metadata_tensors(attn_metadata, query_start_loc.device),
            )
            if metadata_profile:
                profile_register_ms = (
                    time.perf_counter() - profile_register_t0
                ) * 1000.0
        if metadata_profile:
            logger.info(
                "DFLASH_DDTREE_METADATA_PROFILE gdn_build total_ms=%.3f "
                "state_contract_ms=%.3f graph_buffers_ms=%.3f "
                "register_ms=%.3f num_prefills=%d num_decodes=%d "
                "num_spec_decodes=%d num_spec_decode_tokens=%d "
                "num_actual_tokens=%d full_graph=%s ddtree=%s layers=%d",
                (time.perf_counter() - metadata_profile_t0) * 1000.0,
                profile_state_contract_ms,
                profile_graph_buffers_ms,
                profile_register_ms,
                num_prefills,
                num_decodes,
                num_spec_decodes,
                num_spec_decode_tokens,
                m.num_actual_tokens,
                self.use_full_cuda_graph,
                ddtree_parent_ids is not None,
                len(self.layer_names),
            )
        if os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_DIR"):

            def _cpu(t: torch.Tensor | None) -> torch.Tensor | None:
                return None if t is None else t.detach().cpu()

            dump_path = _dump_sm70_gdn_state_table(
                {
                    "num_spec": self.num_spec,
                    "num_spec_state_tokens": self.num_spec_state_tokens,
                    "layer_names": self.layer_names,
                    "use_full_cuda_graph": self.use_full_cuda_graph,
                    "decode_cudagraph_max_bs": self.decode_cudagraph_max_bs,
                    "num_prefills": num_prefills,
                    "num_prefill_tokens": num_prefill_tokens,
                    "num_decodes": num_decodes,
                    "num_decode_tokens": num_decode_tokens,
                    "num_spec_decodes": num_spec_decodes,
                    "num_spec_decode_tokens": num_spec_decode_tokens,
                    "num_actual_tokens": m.num_actual_tokens,
                    "query_start_loc": _cpu(query_start_loc),
                    "query_start_loc_cpu": query_start_loc_cpu.detach().cpu(),
                    "seq_lens": _cpu(m.seq_lens),
                    "block_table_tensor": _cpu(block_table_tensor),
                    "current_state_block_ids": _cpu(current_state_block_ids),
                    "num_decode_draft_tokens_cpu": _cpu(num_decode_draft_tokens_cpu),
                    "spec_sequence_masks_cpu": _cpu(spec_sequence_masks_cpu),
                    "spec_sequence_masks": _cpu(spec_sequence_masks),
                    "num_accepted_tokens": _cpu(num_accepted_tokens),
                    "spec_query_start_loc": _cpu(spec_query_start_loc),
                    "non_spec_query_start_loc": _cpu(non_spec_query_start_loc),
                    "spec_token_indx": _cpu(spec_token_indx),
                    "non_spec_token_indx": _cpu(non_spec_token_indx),
                    "spec_state_indices_tensor": _cpu(spec_state_indices_tensor),
                    "non_spec_state_indices_tensor": _cpu(
                        non_spec_state_indices_tensor
                    ),
                    "ddtree_parent_ids": _cpu(ddtree_parent_ids),
                    "ddtree_num_tree_tokens_cpu": _cpu(ddtree_num_tree_tokens_cpu),
                },
                m.seq_lens,
                num_prefills,
                num_decodes,
            )
            if dump_path:
                logger.warning(
                    "Saved SM70 GDN state table diagnostics to %s", dump_path
                )
        if envs.VLLM_DFLASH_DEBUG_STATE_TABLE and self.use_spec_decode:

            def _cpu(t: torch.Tensor | None) -> torch.Tensor | None:
                return None if t is None else t.detach().cpu()

            dump_path = _dump_dflash_state_table(
                {
                    "num_spec": self.num_spec,
                    "num_spec_state_tokens": self.num_spec_state_tokens,
                    "layer_names": self.layer_names,
                    "use_full_cuda_graph": self.use_full_cuda_graph,
                    "decode_cudagraph_max_bs": self.decode_cudagraph_max_bs,
                    "num_prefills": num_prefills,
                    "num_prefill_tokens": num_prefill_tokens,
                    "num_decodes": num_decodes,
                    "num_decode_tokens": num_decode_tokens,
                    "num_spec_decodes": num_spec_decodes,
                    "num_spec_decode_tokens": num_spec_decode_tokens,
                    "num_actual_tokens": m.num_actual_tokens,
                    "query_start_loc": _cpu(query_start_loc),
                    "query_start_loc_cpu": query_start_loc_cpu.detach().cpu(),
                    "seq_lens": _cpu(m.seq_lens),
                    "block_table_tensor": _cpu(block_table_tensor),
                    "num_decode_draft_tokens_cpu": _cpu(num_decode_draft_tokens_cpu),
                    "spec_sequence_masks_cpu": _cpu(spec_sequence_masks_cpu),
                    "spec_sequence_masks": _cpu(spec_sequence_masks),
                    "num_accepted_tokens": _cpu(num_accepted_tokens),
                    "spec_query_start_loc": _cpu(spec_query_start_loc),
                    "non_spec_query_start_loc": _cpu(non_spec_query_start_loc),
                    "spec_token_indx": _cpu(spec_token_indx),
                    "non_spec_token_indx": _cpu(non_spec_token_indx),
                    "spec_state_indices_tensor": _cpu(spec_state_indices_tensor),
                    "non_spec_state_indices_tensor": _cpu(
                        non_spec_state_indices_tensor
                    ),
                    "ddtree_parent_ids": _cpu(ddtree_parent_ids),
                    "ddtree_num_tree_tokens_cpu": _cpu(ddtree_num_tree_tokens_cpu),
                }
            )
            logger.warning("Saved DFlash/GDN state table diagnostics to %s", dump_path)
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
        spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0

        return self.build(
            common_prefix_len=0,
            common_attn_metadata=m,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            for_cudagraph_capture=True,
        )
