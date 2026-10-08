# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 debug compare methods, bound by impl."""

from __future__ import annotations

import json
import os
import time

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import impl as _impl
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def _maybe_compare_bhmd_out(
    self: _impl.FlashAttnV100Impl,
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


def _reserve_bhmd_compare_call(self: _impl.FlashAttnV100Impl) -> int | None:
    if (
        not self.compare_bhmd_out_dir
        or self.compare_bhmd_out_max_calls <= 0
        or self._compare_bhmd_out_calls >= self.compare_bhmd_out_max_calls
    ):
        return None

    call_idx = self._compare_bhmd_out_calls
    self._compare_bhmd_out_calls += 1
    return call_idx


def _reserve_triton_compare_call(self: _impl.FlashAttnV100Impl) -> int | None:
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
    self: _impl.FlashAttnV100Impl,
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
    self: _impl.FlashAttnV100Impl,
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
    self: _impl.FlashAttnV100Impl,
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
    self: _impl.FlashAttnV100Impl,
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
    self: _impl.FlashAttnV100Impl,
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
    super(_impl._super_owner, self).forward(
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
