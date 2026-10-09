# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comparison calculations with owned counters and explicit operator inputs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import state as _state
from vllm.v1.attention.backends.flash_v100.plan import events as _events
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def _maybe_compare_bhmd_out(
    self: ComparisonExecutor,
    layer: torch.nn.Module,
    q_bhmd: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    safe_bmhd: torch.Tensor,
) -> None:
    call_idx = self._reserve_bhmd_compare_call()
    if call_idx is None or self.flash_attn_prefill_paged_bhmd is None:
        return

    raw_bmhd = torch.empty_like(safe_bmhd)
    raw_bhmd = raw_bmhd.permute(0, 2, 1, 3)
    self.flash_attn_prefill_paged_bhmd(
        q_bhmd,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale=self.scale,
        out=raw_bhmd,
        kv_cache_dtype=self.kv_cache_dtype,
        k_scale=float(layer._k_scale_float),
        v_scale=float(layer._v_scale_float),
        causal=True,
    )
    self._write_bhmd_compare_report(
        raw_bmhd,
        safe_bmhd,
        call_idx,
        "scratch_raw_vs_safe",
        {
            "q_bhmd_stride": list(q_bhmd.stride()),
            "raw_bhmd_stride": list(raw_bhmd.stride()),
            "raw_bhmd_contiguous": raw_bhmd.is_contiguous(),
        },
    )


def _reserve_bhmd_compare_call(self: ComparisonExecutor) -> int | None:
    if (
        not self.compare_bhmd_out_dir
        or self.compare_bhmd_out_max_calls <= 0
        or self._compare_bhmd_out_calls >= self.compare_bhmd_out_max_calls
    ):
        return None

    call_idx = self._compare_bhmd_out_calls
    self._compare_bhmd_out_calls += 1
    return call_idx


def _reserve_triton_compare_call(self: ComparisonExecutor) -> int | None:
    if (
        not self.compare_triton_out_dir
        or self.compare_triton_out_max_calls <= 0
        or self._compare_triton_out_calls >= self.compare_triton_out_max_calls
    ):
        return None

    call_idx = self._compare_triton_out_calls
    self._compare_triton_out_calls += 1
    return call_idx


def _write_bhmd_compare_report(
    self: ComparisonExecutor,
    candidate_bmhd: torch.Tensor,
    reference_bmhd: torch.Tensor,
    call_idx: int,
    mode: str,
    extra: dict[str, object],
) -> None:
    assert self.compare_bhmd_out_dir is not None
    diff = candidate_bmhd - reference_bmhd
    report = {
        "call_idx": call_idx,
        "mode": mode,
        "equal": bool(torch.equal(candidate_bmhd, reference_bmhd)),
        "max_diff": float(diff.abs().max().item()),
        "mean_diff": float(diff.abs().float().mean().item()),
        "num_different": int((candidate_bmhd != reference_bmhd).sum().item()),
        "shape_bmhd": list(reference_bmhd.shape),
        "pid": os.getpid(),
    }
    report.update(extra)

    os.makedirs(self.compare_bhmd_out_dir, exist_ok=True)
    file_name = f"bhmd_compare_pid{os.getpid()}_call{call_idx}_{time.time_ns()}.json"
    path = os.path.join(self.compare_bhmd_out_dir, file_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_triton_compare_report(
    self: ComparisonExecutor,
    candidate: torch.Tensor,
    reference: torch.Tensor,
    call_idx: int,
    stage: str,
    extra: dict[str, object],
) -> None:
    assert self.compare_triton_out_dir is not None
    diff = candidate.float() - reference.float()
    abs_diff = diff.abs()
    report = {
        "call_idx": call_idx,
        "stage": stage,
        "equal": bool(torch.equal(candidate, reference)),
        "max_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_diff": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "num_different": int((candidate != reference).sum().item()),
        "shape": list(candidate.shape),
        "dtype": str(candidate.dtype),
        "candidate_nan_count": int(torch.isnan(candidate).sum().item()),
        "reference_nan_count": int(torch.isnan(reference).sum().item()),
        "pid": os.getpid(),
    }
    report.update(extra)

    os.makedirs(self.compare_triton_out_dir, exist_ok=True)
    file_name = (
        f"triton_out_compare_pid{os.getpid()}_call{call_idx}_"
        f"{stage}_{time.time_ns()}.json"
    )
    path = os.path.join(self.compare_triton_out_dir, file_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def _maybe_write_triton_tensor_dump(
    self: ComparisonExecutor,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    candidate: torch.Tensor,
    reference: torch.Tensor,
    call_idx: int,
    stage: str,
    num_actual_tokens: int,
) -> dict[str, object]:
    if not self.compare_triton_tensor_dump_dir:
        return {}
    if num_actual_tokens > self.compare_triton_tensor_dump_max_tokens:
        return {
            "tensor_dump_skipped": "num_actual_tokens_exceeds_limit",
            "tensor_dump_max_tokens": self.compare_triton_tensor_dump_max_tokens,
        }

    payload: dict[str, object] = {
        "call_idx": call_idx,
        "stage": stage,
        "num_actual_tokens": num_actual_tokens,
        "scale": self.scale,
        "kv_cache_dtype": self.kv_cache_dtype,
        "layer": self._layer_debug_info(layer),
        "query": query[:num_actual_tokens].detach().cpu(),
        "raw_key": key[:num_actual_tokens].detach().cpu(),
        "raw_value": value[:num_actual_tokens].detach().cpu(),
        "candidate_output": candidate[:num_actual_tokens].detach().cpu(),
        "triton_reference_output": reference[:num_actual_tokens].detach().cpu(),
        "query_start_loc": attn_metadata.query_start_loc.detach().cpu(),
        "seq_lens": attn_metadata.seq_lens.detach().cpu(),
        "block_table": attn_metadata.block_table.detach().cpu(),
    }

    if stage in ("prefill_no_prefix", "prefill_no_prefix_paged_cache"):
        key_cache, _ = _kv_layout._split_paged_kv_cache(kv_cache)
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        num_seqs = len(query_start_loc) - 1
        k_cont, v_cont = _kv_layout._extract_contiguous_kv_from_paged_cache(
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table[:num_seqs],
            seq_lens=attn_metadata.seq_lens[:num_seqs],
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            total_tokens=num_actual_tokens,
        )
        k_cont, v_cont = _kv_layout._dequantize_fp8_contiguous_kv(
            k_cont,
            v_cont,
            self.kv_cache_dtype,
            float(layer._k_scale_float),
            float(layer._v_scale_float),
        )
        payload["cache_key"] = k_cont.detach().cpu()
        payload["cache_value"] = v_cont.detach().cpu()

    os.makedirs(self.compare_triton_tensor_dump_dir, exist_ok=True)
    file_name = (
        f"triton_tensor_dump_pid{os.getpid()}_call{call_idx}_"
        f"{stage}_{time.time_ns()}.pt"
    )
    path = os.path.join(self.compare_triton_tensor_dump_dir, file_name)
    torch.save(payload, path)
    return {"tensor_dump_path": path}


def _small_tensor_list(
    tensor: torch.Tensor | None,
    limit: int = 32,
) -> list[int] | None:
    if tensor is None:
        return None
    flat = tensor.detach().cpu().reshape(-1)
    return [int(x) for x in flat[:limit].tolist()]


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


def _tensor_compare_stats(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, object]:
    if candidate.shape != reference.shape:
        return {
            "shape_mismatch": True,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
        }

    diff = candidate.float() - reference.float()
    abs_diff = diff.abs()
    return {
        "shape_mismatch": False,
        "equal": bool(torch.equal(candidate, reference)),
        "max_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_diff": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "num_different": int((candidate != reference).sum().item()),
        "candidate_dtype": str(candidate.dtype),
        "reference_dtype": str(reference.dtype),
        "candidate_abs_max": float(candidate.float().abs().max().item())
        if candidate.numel()
        else 0.0,
        "reference_abs_max": float(reference.float().abs().max().item())
        if reference.numel()
        else 0.0,
        "candidate_mean": float(candidate.float().mean().item())
        if candidate.numel()
        else 0.0,
        "reference_mean": float(reference.float().mean().item())
        if reference.numel()
        else 0.0,
        "shape": list(candidate.shape),
    }


def _prefill_raw_kv_cache_compare_stats(
    self: ComparisonExecutor,
    layer: torch.nn.Module,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    num_actual_tokens: int,
) -> dict[str, object]:
    key_cache, _ = _kv_layout._split_paged_kv_cache(kv_cache)

    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    query_start_loc = (
        query_start_loc_cpu
        if query_start_loc_cpu is not None
        else attn_metadata.query_start_loc
    )
    num_seqs = len(query_start_loc) - 1
    k_cont, v_cont = _kv_layout._extract_contiguous_kv_from_paged_cache(
        kv_cache=kv_cache,
        block_table=attn_metadata.block_table[:num_seqs],
        seq_lens=attn_metadata.seq_lens[:num_seqs],
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        total_tokens=num_actual_tokens,
    )
    k_cont, v_cont = _kv_layout._dequantize_fp8_contiguous_kv(
        k_cont,
        v_cont,
        self.kv_cache_dtype,
        float(layer._k_scale_float),
        float(layer._v_scale_float),
    )
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
    return {
        "raw_key_vs_cache": self._tensor_compare_stats(key[:num_actual_tokens], k_cont),
        "raw_value_vs_cache": self._tensor_compare_stats(
            value[:num_actual_tokens], v_cont
        ),
        "kv_cache_dtype": str(kv_cache.dtype),
        "kv_cache_shape": list(kv_cache.shape),
        "query_start_loc": self._small_tensor_list(attn_metadata.query_start_loc),
        "query_start_loc_cpu": self._small_tensor_list(query_start_loc_cpu),
        "seq_lens": self._small_tensor_list(attn_metadata.seq_lens),
        "seq_lens_cpu": self._small_tensor_list(seq_lens_cpu),
        "block_table_shape": list(attn_metadata.block_table.shape),
        "block_table_first_row": self._small_tensor_list(attn_metadata.block_table[:1]),
    }


def _maybe_compare_triton_output(
    self: ComparisonExecutor,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    output: torch.Tensor,
    output_scale: torch.Tensor | None,
    output_block_scale: torch.Tensor | None,
    stage: str,
) -> None:
    call_idx = self._reserve_triton_compare_call()
    if call_idx is None:
        return
    if query.is_cuda and torch.cuda.is_current_stream_capturing():
        return

    reference = torch.empty_like(output)
    self.ops.triton_forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        reference,
        output_scale,
        output_block_scale,
    )
    if query.is_cuda:
        # Diagnostic-only: the Triton reference path may update KV cache
        # asynchronously before this hook extracts raw-vs-cache tensors.
        torch.accelerator.synchronize(query.device)
    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    extra = {
        "num_actual_tokens": num_actual_tokens,
        "max_query_len": int(attn_metadata.max_query_len),
        "max_seq_len": int(attn_metadata.max_seq_len),
        "layer_type": type(layer).__name__,
    }
    extra.update(self._layer_debug_info(layer))
    if stage in ("prefill_no_prefix", "prefill_no_prefix_paged_cache"):
        extra.update(
            self._prefill_raw_kv_cache_compare_stats(
                layer,
                key,
                value,
                kv_cache,
                attn_metadata,
                num_actual_tokens,
            )
        )
    extra.update(
        self._maybe_write_triton_tensor_dump(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            reference,
            call_idx,
            stage,
            num_actual_tokens,
        )
    )
    self._write_triton_compare_report(
        output[:num_actual_tokens],
        reference[:num_actual_tokens],
        call_idx,
        stage,
        extra,
    )


class PrefixReferenceObserver:
    def __call__(self, event: _events.PrefillDebugEvent) -> None:
        k_cont, v_cont = _kv_layout._extract_contiguous_kv_from_paged_cache(
            kv_cache=event.kv_cache,
            block_table=event.attn_metadata.block_table[event.i : event.i + 1],
            seq_lens=event.attn_metadata.seq_lens[event.i : event.i + 1],
            num_kv_heads=event.num_kv_heads,
            head_dim=event.head_dim,
            block_size=event.block_size,
            total_tokens=event.seq_len,
        )
        k_cont, v_cont = _kv_layout._dequantize_fp8_contiguous_kv(
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
        if event.debug_compare and (not _state._logged_prefill_compare):
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
                        _kv_layout._dequantize_fp8_contiguous_kv(
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
            torch.save(
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
                dump_path,
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
            _state._logged_dflash_prefix_dump = True
        if (
            event.debug_compare
            and (not _state._logged_prefill_compare)
            and (nan_count > 0)
        ):
            dump_path = f"/tmp/flash_v100_prefill_nan_dump_pid{os.getpid()}.pt"
            torch.save(
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
                dump_path,
            )
            logger.warning(
                "FLASH_ATTN_V100 saved failing prefix prefill dump to %s", dump_path
            )
        if event.debug_compare and (not _state._logged_prefill_compare):
            _state._logged_prefill_compare = True


_events.prefill_debug.subscribe(PrefixReferenceObserver())
_events.prefill_debug.subscribe(PrefixReportObserver())


COUNTER_FIELDS = frozenset({"_compare_bhmd_out_calls", "_compare_triton_out_calls"})


class ComparisonState:
    # No defaults: partial legacy objects retain missing-field errors if a
    # diagnostic is enabled before the original constructor initializes it.
    _compare_bhmd_out_calls: int
    _compare_triton_out_calls: int


@dataclass(frozen=True)
class ComparisonOps:
    triton_forward: Any
    paged_bhmd: Any


class ComparisonExecutor:
    def __init__(self, policy, scale, kv_cache_dtype, ops, state, overrides=None):
        self.policy = policy
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype
        self.ops = ops
        self.state = state
        if overrides:
            vars(self).update(overrides)

    @property
    def flash_attn_prefill_paged_bhmd(self):
        return self.ops.paged_bhmd

    def __getattr__(self, name):
        if name in COUNTER_FIELDS:
            return getattr(self.state, name)
        return getattr(self.policy, name)

    def __setattr__(self, name, value):
        if name in COUNTER_FIELDS:
            setattr(self.state, name, value)
        else:
            super().__setattr__(name, value)

    _maybe_compare_bhmd_out = _maybe_compare_bhmd_out
    _reserve_bhmd_compare_call = _reserve_bhmd_compare_call
    _reserve_triton_compare_call = _reserve_triton_compare_call
    _write_bhmd_compare_report = _write_bhmd_compare_report
    _write_triton_compare_report = _write_triton_compare_report
    _maybe_write_triton_tensor_dump = _maybe_write_triton_tensor_dump
    _prefill_raw_kv_cache_compare_stats = _prefill_raw_kv_cache_compare_stats
    _maybe_compare_triton_output = _maybe_compare_triton_output
    _small_tensor_list = staticmethod(_small_tensor_list)
    _layer_debug_info = staticmethod(_layer_debug_info)
    _tensor_compare_stats = staticmethod(_tensor_compare_stats)


LEGACY_METHODS = (
    "_maybe_compare_bhmd_out",
    "_reserve_bhmd_compare_call",
    "_reserve_triton_compare_call",
    "_write_bhmd_compare_report",
    "_write_triton_compare_report",
    "_maybe_write_triton_tensor_dump",
    "_prefill_raw_kv_cache_compare_stats",
    "_maybe_compare_triton_output",
)

STATIC_METHODS = ("_small_tensor_list", "_layer_debug_info", "_tensor_compare_stats")
