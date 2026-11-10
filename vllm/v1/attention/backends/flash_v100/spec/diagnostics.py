# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix verification diagnostics, subscribed through common events."""

from __future__ import annotations

import os

import torch

from vllm.diagnostics import diagnostic_engine_tag, write_payload
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100.config import (
    diagnostic_seen as log_once_seen,
)
from vllm.v1.attention.backends.flash_v100.config import (
    mark_diagnostic as set_log_once_state,
)
from vllm.v1.attention.backends.flash_v100.plan import events as _events

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def _layer_debug_info(layer: torch.nn.Module) -> dict[str, object]:
    return {
        "layer_name": getattr(layer, "layer_name", None),
        "is_dflash_draft_attn": getattr(layer, "is_dflash_draft_attn", False),
        "kv_sharing_target_layer_name": getattr(
            layer, "kv_sharing_target_layer_name", None
        ),
        "impl_kv_sharing_target_layer_name": getattr(
            getattr(layer, "impl", None), "kv_sharing_target_layer_name", None
        ),
    }


class PrefixReferenceObserver:
    def __call__(self, event: _events.PrefillDebugEvent) -> None:
        k_cont, v_cont = _kv_layout.extract_contiguous_kv_from_paged_cache(
            kv_cache=event.kv_cache,
            block_table=event.attn_metadata.block_table[event.i : event.i + 1],
            seq_lens=event.attn_metadata.seq_lens[event.i : event.i + 1],
            num_kv_heads=event.num_kv_heads,
            head_dim=event.head_dim,
            block_size=event.block_size,
            total_tokens=event.seq_len,
        )
        k_cont, v_cont = _kv_layout.dequantize_fp8_contiguous_kv(
            k_cont,
            v_cont,
            event.kv_cache_dtype,
            float(event.layer._k_scale_float),
            float(event.layer._v_scale_float),
        )
        if bool(getattr(event.layer, "is_dflash_draft_attn", False)):
            ref_out = event.torch_reference(
                event.query[event.start : event.end],
                k_cont,
                v_cont,
                causal=event.causal,
                softmax_scale=event.scale,
                window_size=event.window_size,
            )
        else:
            ref_out = event.dense(
                event.query[event.start : event.end].unsqueeze(0),
                k_cont.unsqueeze(0),
                v_cont.unsqueeze(0),
                causal=event.causal,
                softmax_scale=event.scale,
                window_size=event.window_size,
            )
        diff = (event.out_seq - ref_out).abs()
        nan_count = int(torch.isnan(event.out_seq).sum().item())
        if event.debug_compare and (
            not log_once_seen("flash_v100._logged_prefill_compare")
        ):
            logger.warning(
                "FLASH_ATTN_V100 debug prefix compare: "
                "query_len=%d seq_len=%d max_diff=%.8f mean_diff=%.8f "
                "nan_count=%d q_absmax=%.6f k_absmax=%.6f "
                "v_absmax=%.6f kv_cache_shape=%s key_shape=%s "
                "key_stride=%s value_stride=%s key_contig=%s "
                "value_contig=%s",
                event.end - event.start,
                event.seq_len,
                float(diff.max().item()),
                float(diff.mean().item()),
                nan_count,
                float(event.query[event.start : event.end].abs().max().item()),
                float(k_cont.abs().max().item()),
                float(v_cont.abs().max().item()),
                tuple(event.kv_cache.shape),
                tuple(event.key_cache.shape),
                tuple(event.key_cache.stride()),
                tuple(event.value_cache.stride()),
                str(event.key_cache.is_contiguous()),
                str(event.value_cache.is_contiguous()),
            )
        event.reference = _events.PrefillReference(
            k_cont, v_cont, ref_out, diff, nan_count
        )


class PrefixReportObserver:
    def __call__(self, event: _events.PrefillDebugEvent) -> None:
        reference = event.reference
        assert reference is not None
        k_cont, v_cont, ref_out, diff, nan_count = reference
        if event.dump_enabled:
            slot_mapping = getattr(event.attn_metadata, "slot_mapping", None)
            slot_slice = None
            cache_k_by_slot = None
            cache_v_by_slot = None
            key_input = None
            value_input = None
            slot_k_diff = None
            slot_v_diff = None
            tail_k_diff = None
            tail_v_diff = None
            if (
                slot_mapping is not None
                and event.key is not None
                and (event.value is not None)
                and (event.key_cache.dtype != torch.uint8)
            ):
                slot_slice = slot_mapping[event.start : event.end].to(torch.long)
                valid_slots = slot_slice >= 0
                if bool(valid_slots.all().item()):
                    slot_blocks = torch.div(
                        slot_slice, event.block_size, rounding_mode="floor"
                    )
                    slot_offsets = torch.remainder(slot_slice, event.block_size)
                    cache_k_by_slot = event.key_cache[slot_blocks, slot_offsets]
                    cache_v_by_slot = event.value_cache[slot_blocks, slot_offsets]
                    cache_k_by_slot, cache_v_by_slot = (
                        _kv_layout.dequantize_fp8_contiguous_kv(
                            cache_k_by_slot,
                            cache_v_by_slot,
                            event.kv_cache_dtype,
                            float(event.layer._k_scale_float),
                            float(event.layer._v_scale_float),
                        )
                    )
                    key_input = event.key[event.start : event.end]
                    value_input = event.value[event.start : event.end]
                    slot_k_diff = (cache_k_by_slot - key_input).abs()
                    slot_v_diff = (cache_v_by_slot - value_input).abs()
                    tail_start = max(0, event.seq_len - (event.end - event.start))
                    tail_k = k_cont[tail_start : event.seq_len]
                    tail_v = v_cont[tail_start : event.seq_len]
                    if tail_k.shape == key_input.shape:
                        tail_k_diff = (tail_k - key_input).abs()
                        tail_v_diff = (tail_v - value_input).abs()
            dump_path = (
                f"/tmp/flash_v100_dflash_prefix_dump_pid{os.getpid()}_seq{event.i}.pt"
            )
            dump_path = write_payload(
                os.path.dirname(dump_path),
                os.path.basename(dump_path),
                {
                    "layer_name": event.layer_info(event.layer).get("layer_name"),
                    "causal": event.causal,
                    "window_size": event.window_size,
                    "query_start_loc": event.query_start_loc.detach().cpu(),
                    "seq_lens": event.seq_lens.detach().cpu(),
                    "attn_seq_lens": event.attn_metadata.seq_lens.detach().cpu(),
                    "block_table": event.attn_metadata.block_table[
                        event.i : event.i + 1
                    ]
                    .detach()
                    .cpu(),
                    "slot_mapping": None
                    if slot_slice is None
                    else slot_slice.detach().cpu(),
                    "query": event.query[event.start : event.end].detach().cpu(),
                    "key_input": None
                    if key_input is None
                    else key_input.detach().cpu(),
                    "value_input": None
                    if value_input is None
                    else value_input.detach().cpu(),
                    "cache_k_by_slot": None
                    if cache_k_by_slot is None
                    else cache_k_by_slot.detach().cpu(),
                    "cache_v_by_slot": None
                    if cache_v_by_slot is None
                    else cache_v_by_slot.detach().cpu(),
                    "k_cont_tail": k_cont[
                        max(
                            0, event.seq_len - (event.end - event.start)
                        ) : event.seq_len
                    ]
                    .detach()
                    .cpu(),
                    "v_cont_tail": v_cont[
                        max(
                            0, event.seq_len - (event.end - event.start)
                        ) : event.seq_len
                    ]
                    .detach()
                    .cpu(),
                    "k_cont": k_cont.detach().cpu(),
                    "v_cont": v_cont.detach().cpu(),
                    "out_seq": event.out_seq.detach().cpu(),
                    "ref_out": ref_out.detach().cpu(),
                    "paged_vs_dense_max": float(diff.max().item()),
                    "paged_vs_dense_mean": float(diff.mean().item()),
                    "slot_k_max": None
                    if slot_k_diff is None
                    else float(slot_k_diff.max().item()),
                    "slot_v_max": None
                    if slot_v_diff is None
                    else float(slot_v_diff.max().item()),
                    "tail_k_max": None
                    if tail_k_diff is None
                    else float(tail_k_diff.max().item()),
                    "tail_v_max": None
                    if tail_v_diff is None
                    else float(tail_v_diff.max().item()),
                    "kv_cache_shape": tuple(event.kv_cache.shape),
                    "key_cache_shape": tuple(event.key_cache.shape),
                    "key_cache_stride": tuple(event.key_cache.stride()),
                    "value_cache_stride": tuple(event.value_cache.stride()),
                },
                diagnostic_engine_tag(),
            )
            logger.warning(
                "FLASH_ATTN_V100 saved DFlash prefix dump to %s "
                "(paged_vs_dense_max=%.8f slot_k_max=%s tail_k_max=%s)",
                dump_path,
                float(diff.max().item()),
                "n/a"
                if slot_k_diff is None
                else f"{float(slot_k_diff.max().item()):.8f}",
                "n/a"
                if tail_k_diff is None
                else f"{float(tail_k_diff.max().item()):.8f}",
            )
            set_log_once_state("flash_v100.prefix_dump", True)
        if (
            event.debug_compare
            and (not log_once_seen("flash_v100._logged_prefill_compare"))
            and (nan_count > 0)
        ):
            dump_path = f"/tmp/flash_v100_prefill_nan_dump_pid{os.getpid()}.pt"
            dump_path = write_payload(
                os.path.dirname(dump_path),
                os.path.basename(dump_path),
                {
                    "query": event.query[event.start : event.end].detach().cpu(),
                    "key_cache": event.key_cache.detach().cpu(),
                    "value_cache": event.value_cache.detach().cpu(),
                    "block_table": event.attn_metadata.block_table[
                        event.i : event.i + 1
                    ]
                    .detach()
                    .cpu(),
                    "seq_lens": event.attn_metadata.seq_lens[event.i : event.i + 1]
                    .detach()
                    .cpu(),
                    "k_cont": k_cont.detach().cpu(),
                    "v_cont": v_cont.detach().cpu(),
                    "out_seq": event.out_seq.detach().cpu(),
                    "ref_out": ref_out.detach().cpu(),
                },
                diagnostic_engine_tag(),
            )
            logger.warning(
                "FLASH_ATTN_V100 saved failing prefix prefill dump to %s", dump_path
            )
        if event.debug_compare and (
            not log_once_seen("flash_v100._logged_prefill_compare")
        ):
            set_log_once_state("flash_v100._logged_prefill_compare", True)


_events.prefill_debug.subscribe(PrefixReferenceObserver())
_events.prefill_debug.subscribe(PrefixReportObserver())
