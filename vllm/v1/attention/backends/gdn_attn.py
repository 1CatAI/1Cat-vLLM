# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass
import os

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

logger = init_logger(__name__)


def _dflash_debug_state_table_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DEBUG_STATE_TABLE", "0") == "1"


def _dump_dflash_state_table(payload) -> str:
    dump_path = f"/tmp/dflash_state_table_pid{os.getpid()}.pt"
    torch.save(payload, dump_path)
    return dump_path


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder


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

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


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

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens = torch.empty(
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

        def _gather_state_block_ids(
            block_table: torch.Tensor,
            seq_lens: torch.Tensor,
            width: int,
        ) -> torch.Tensor:
            block_size = self.kv_cache_spec.block_size
            current_block_idx = torch.clamp((seq_lens - 1) // block_size, min=0)
            offsets = torch.arange(width, device=block_table.device, dtype=torch.int32)
            gather_indices = current_block_idx.unsqueeze(1) + offsets
            gather_indices = torch.clamp(gather_indices, max=block_table.shape[1] - 1)
            return torch.gather(block_table, 1, gather_indices)

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
                    num_spec_draft_tokens = num_decode_draft_tokens_cpu[
                        spec_sequence_masks_cpu
                    ].sum().item()
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
                non_spec_state_indices_tensor = _gather_state_block_ids(
                    block_table_tensor, m.seq_lens, 1
                ).squeeze(1)
            else:
                non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # Keep normal decode tokens on the decode fast path even when
            # speculative decode tokens are present in the same step.
            #
            # Baseline decode-only performance on SM70 depends heavily on the
            # num_decodes > 0 branch in qwen3_next.py, which uses the
            # recurrent decode update path. Historically we collapsed the
            # non-spec decode portion into the prefill bucket whenever
            # num_spec_decodes > 0, forcing those tokens through the generic
            # prefill/mixed path instead. That preserves correctness but can
            # erase most of the expected DFlash speedup on V100, because the
            # verifier loses the decode-only fast path on the majority GDN
            # layers. qwen3_next.py already has separate handling for
            # spec/non-spec tokens and can merge the outputs, so keep the
            # decode and spec-decode counts distinct here.

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
                if is_mamba_cache_all:
                    spec_state_indices_tensor = _gather_state_block_ids(
                        block_table_tensor[spec_sequence_masks_cpu],
                        m.seq_lens[spec_sequence_masks_cpu],
                        self.num_spec + 1,
                    )
                else:
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
                non_spec_state_indices_tensor = None
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

                if is_mamba_cache_all:
                    spec_state_indices_tensor = _gather_state_block_ids(
                        block_table_tensor[spec_sequence_masks_cpu],
                        m.seq_lens[spec_sequence_masks_cpu],
                        self.num_spec + 1,
                    )
                    non_spec_state_indices_tensor = _gather_state_block_ids(
                        block_table_tensor[~spec_sequence_masks_cpu],
                        m.seq_lens[~spec_sequence_masks_cpu],
                        1,
                    ).squeeze(1)
                else:
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
                    non_spec_state_indices_tensor = block_table_tensor[
                        ~spec_sequence_masks_cpu, 0
                    ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
            assert spec_query_start_loc is not None
            assert spec_query_start_loc[-1].item() == num_spec_decode_tokens
            assert spec_state_indices_tensor is not None
            assert spec_state_indices_tensor.shape[0] == num_spec_decodes

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
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

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
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

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
        if _dflash_debug_state_table_enabled() and self.use_spec_decode:
            dump_path = _dump_dflash_state_table(
                {
                    "block_table_tensor": block_table_tensor.detach().cpu(),
                    "current_state_block_ids": None,
                    "query_start_loc": query_start_loc.detach().cpu(),
                    "query_start_loc_cpu": query_start_loc_cpu.clone(),
                    "seq_lens": m.seq_lens.detach().cpu(),
                    "context_lens": context_lens_tensor.detach().cpu(),
                    "num_accepted_tokens": (
                        None
                        if num_accepted_tokens is None
                        else num_accepted_tokens.detach().cpu()
                    ),
                    "num_decode_draft_tokens_cpu": (
                        None
                        if num_decode_draft_tokens_cpu is None
                        else num_decode_draft_tokens_cpu.clone()
                    ),
                    "spec_sequence_masks_cpu": (
                        None
                        if spec_sequence_masks_cpu is None
                        else spec_sequence_masks_cpu.clone()
                    ),
                    "spec_sequence_masks": (
                        None
                        if spec_sequence_masks is None
                        else spec_sequence_masks.detach().cpu()
                    ),
                    "spec_state_indices_tensor": (
                        None
                        if spec_state_indices_tensor is None
                        else spec_state_indices_tensor.detach().cpu()
                    ),
                    "non_spec_state_indices_tensor": (
                        None
                        if non_spec_state_indices_tensor is None
                        else non_spec_state_indices_tensor.detach().cpu()
                    ),
                    "spec_query_start_loc": (
                        None
                        if spec_query_start_loc is None
                        else spec_query_start_loc.detach().cpu()
                    ),
                    "non_spec_query_start_loc": (
                        None
                        if non_spec_query_start_loc is None
                        else non_spec_query_start_loc.detach().cpu()
                    ),
                }
            )
            logger.info("Saved DFlash state-table debug to %s", dump_path)
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
        )
