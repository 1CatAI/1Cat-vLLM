# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

import os
from dataclasses import dataclass
from typing import Literal

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
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
    )


def gdn_spec_metadata_tensors(
    attn_metadata: GDNAttentionMetadata | None,
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
]:
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
) -> GDNSpecDecodeStateContract:
    """Build the state-index/count contract consumed by active-MTP GDN.

    ``current_state_block_ids`` is authoritative for align-mode replay because
    it is materialized from the live ``mamba_state_idx`` after preprocess
    rollover. The accepted count selects the committed speculative slot as
    ``num_accepted_tokens - 1`` in the recurrent kernels.
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
        spec_state_indices_tensor = block_table_tensor[
            block_mask, : num_spec + 1
        ]
        non_spec_state_indices_tensor = select_gdn_state_block_ids(
            block_table_tensor[~block_mask],
            num_accepted_tokens[~accepted_mask],
            num_spec,
        )

    return GDNSpecDecodeStateContract(
        spec_state_indices_tensor=spec_state_indices_tensor,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens[accepted_mask],
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
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
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
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
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

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        spec_sequence_masks_cpu: torch.Tensor | None = None,
        current_state_block_ids: torch.Tensor | None = None,
        for_cudagraph_capture: bool = False,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
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
                    block_table_tensor, num_accepted_tokens, self.num_spec
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
        else:
            assert spec_sequence_masks_cpu is not None
            assert num_accepted_tokens is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            # query_start_loc may be padded for CUDA graph replay. The CPU
            # metadata is authoritative for the live request count here.
            query_lens = query_lens_cpu.to(query_start_loc.device, non_blocking=True)
            state_contract = build_gdn_spec_decode_state_contract(
                block_table_tensor=block_table_tensor,
                seq_lens=m.seq_lens,
                block_size=self.kv_cache_spec.block_size,
                num_spec=self.num_spec,
                spec_sequence_masks_cpu=spec_sequence_masks_cpu,
                num_accepted_tokens=num_accepted_tokens,
                current_state_block_ids=current_state_block_ids,
                is_mamba_cache_all=is_mamba_cache_all,
            )

            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
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
                    num_spec_decodes * (self.num_spec + 1),
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
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

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
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        if os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_DIR"):

            def _cpu(t: torch.Tensor | None) -> torch.Tensor | None:
                return None if t is None else t.detach().cpu()

            dump_path = _dump_sm70_gdn_state_table(
                {
                    "num_spec": self.num_spec,
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
            0,
            m,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            spec_sequence_masks_cpu,
            for_cudagraph_capture=True,
        )
