# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 27B dense prefill accelerators.

Provides:
- TP2/TP4 NVLink zero-memcpy PUSH P2P RS + GemmaRMSNorm + AG (with automatic
  PCIe fallback when native peer atomics are unavailable on any rank)
- KN-cached FP32-accum cuBLASLt GEMM + 16B vectorized SiLU+Mul
- Automatic double-buffered side-stream weight dequantization overlap
  (NVFP4 & FP8)
- Dynamo-safe generic D=128 strided-z GDN RMSNormGated
"""

from __future__ import annotations

import ctypes
import importlib
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_LOADED: bool = False
_DYNAMO_OPS_REGISTERED: bool = False
_GATE_UP_STAGING: dict[int, torch.Tensor] = {}


def _init_sm70_27b_prefill_ops() -> bool:
    global _LOADED
    if not envs.VLLM_SM70_27B_FAST_PREFILL:
        _LOADED = False
        return False
    ns = getattr(torch.ops, "vllm_sm70_27b_prefill", None)
    if ns is None or not hasattr(ns, "p2p_push_rs_gemma_norm_ag"):
        try:
            importlib.import_module("vllm._sm70_27b_prefill_C")
        except ImportError:
            _LOADED = False
            return False
    ns = getattr(torch.ops, "vllm_sm70_27b_prefill", None)
    _LOADED = bool(ns is not None and hasattr(ns, "p2p_push_rs_gemma_norm_ag"))
    if _LOADED:
        _ensure_dynamo_ops()
    return _LOADED


@torch.compiler.assume_constant_result
def has_sm70_27b_prefill_ops() -> bool:
    return _LOADED


def _prefetch_dequant_enabled() -> bool:
    return bool(envs.VLLM_SM70_27B_PREFETCH_DEQUANT)


def _get_gate_up_staging(device: torch.device, m: int, n: int) -> torch.Tensor:
    dev_idx = device.index or 0
    need = m * n
    buf = _GATE_UP_STAGING.get(dev_idx)
    if buf is None or buf.numel() < need:
        buf = torch.empty((need,), dtype=torch.float16, device=device)
        _GATE_UP_STAGING[dev_idx] = buf
    return buf[:need].view(m, n)


class _WeightDequantPrefetcher:
    """Automatic double-buffered side-stream weight dequantization overlap."""

    def __init__(self, device: torch.device, ws_numel: int):
        self.device = device
        self.ws_alt = torch.empty((ws_numel,), dtype=torch.float16, device=device)
        self.stream = torch.cuda.Stream(device=device)
        self.slot_done = [torch.cuda.Event(), torch.cuda.Event()]
        self.pref_ready = torch.cuda.Event()
        self.next_map: dict[int, tuple[Any, ...]] = {}
        self.prev_ptr: int | None = None
        self.pref_ptr: int | None = None
        self.pref_slot: int = 1
        self.pref_buf: torch.Tensor | None = None
        self.slot_has_event = [False, False]

    def _run_dequant(self, spec: tuple[Any, ...], dst_buf: torch.Tensor) -> None:
        kind, codes, scales, global_scale, k, n = spec
        dense_w = dst_buf[: k * n].view(k, n)
        if kind == 0:
            torch.ops._C.nvfp4_qpn4_dequantize_sm70_out(
                dense_w,
                codes.view(k, n // 2),
                scales.view(torch.float8_e4m3fn).view(k // 16, n),
                float(global_scale),
                False,
            )
        else:
            torch.ops._C.fp8_qpn8_dequantize_sm70_out(dense_w, codes, scales)

    def drain(self) -> None:
        if self.pref_ptr is not None:
            torch.cuda.current_stream(self.device).wait_event(self.pref_ready)
            self.pref_ptr = None
        self.pref_buf = None
        self.prev_ptr = None

    def acquire_and_prefetch(
        self, spec: tuple[Any, ...], workspace: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        _, codes, _, _, k, n = spec
        cur_ptr = int(codes.data_ptr())
        main_stream = torch.cuda.current_stream(self.device)

        if self.pref_ptr == cur_ptr and self.pref_buf is not None:
            main_stream.wait_event(self.pref_ready)
            use_slot = self.pref_slot
            use_buf = self.pref_buf
            self.pref_ptr = None
            self.pref_buf = None
        else:
            if self.pref_ptr is not None:
                main_stream.wait_event(self.pref_ready)
                self.pref_ptr = None
                self.pref_buf = None
            use_slot = 0
            use_buf = workspace
            self._run_dequant(spec, use_buf)

        if self.prev_ptr is not None and self.prev_ptr != cur_ptr:
            self.next_map[self.prev_ptr] = spec
        self.prev_ptr = cur_ptr

        nxt = self.next_map.get(cur_ptr)
        if nxt is not None:
            nxt_ptr = int(nxt[1].data_ptr())
            nxt_numel = int(nxt[4]) * int(nxt[5])
            nxt_slot = use_slot ^ 1
            # Never prefetch into the same backing storage currently in use.
            nxt_buf = (
                workspace
                if (nxt_slot == 0 and workspace.data_ptr() != use_buf.data_ptr())
                else self.ws_alt
            )
            if (
                nxt_ptr != cur_ptr
                and nxt_buf.data_ptr() != use_buf.data_ptr()
                and nxt_numel <= nxt_buf.numel()
            ):
                if self.slot_has_event[nxt_slot]:
                    self.stream.wait_event(self.slot_done[nxt_slot])
                with torch.cuda.stream(self.stream):
                    self._run_dequant(nxt, nxt_buf)
                    self.pref_ready.record(self.stream)
                self.pref_ptr = nxt_ptr
                self.pref_slot = nxt_slot
                self.pref_buf = nxt_buf

        return use_buf[: k * n].view(k, n), use_slot

    def mark_consumed(self, slot: int) -> None:
        main_stream = torch.cuda.current_stream(self.device)
        self.slot_done[slot].record(main_stream)
        self.slot_has_event[slot] = True


_PREFETCHERS: dict[int, _WeightDequantPrefetcher] = {}


def _get_prefetcher(
    device: torch.device, ws_numel: int
) -> _WeightDequantPrefetcher | None:
    if not _prefetch_dequant_enabled():
        return None
    dev_idx = device.index or 0
    pf = _PREFETCHERS.get(dev_idx)
    if pf is None:
        pf = _WeightDequantPrefetcher(device, ws_numel)
        _PREFETCHERS[dev_idx] = pf
    elif pf.ws_alt.numel() < ws_numel:
        pf.drain()
        pf = _WeightDequantPrefetcher(device, ws_numel)
        _PREFETCHERS[dev_idx] = pf
    return pf


def drain_prefill_prefetcher(device: torch.device) -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    pf = _PREFETCHERS.get(device.index or 0)
    if pf is not None and pf.pref_ptr is not None:
        pf.drain()


def clear_sm70_27b_prefill_workspaces() -> None:
    for pf in _PREFETCHERS.values():
        pf.drain()
    _PREFETCHERS.clear()
    _GATE_UP_STAGING.clear()


def fast_nvfp4_prefill_out(
    out: torch.Tensor,
    workspace: torch.Tensor,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    gated_silu: bool,
) -> None:
    m, k = int(x.shape[0]), int(x.shape[1])
    n = int(out.shape[1]) * (2 if gated_silu else 1)
    pf = _get_prefetcher(x.device, workspace.numel())
    if pf is not None:
        spec = (0, codes, scales, float(global_scale), k, n)
        dense_w, slot = pf.acquire_and_prefetch(spec, workspace)
    else:
        dense_w = workspace[: k * n].view(k, n)
        torch.ops._C.nvfp4_qpn4_dequantize_sm70_out(
            dense_w,
            codes.view(k, n // 2),
            scales.view(torch.float8_e4m3fn).view(k // 16, n),
            float(global_scale),
            False,
        )
        slot = -1

    if not gated_silu:
        torch.ops.vllm_sm70_27b_prefill.cublaslt_fp32_mm_out(out, x, dense_w)
        if pf is not None:
            pf.mark_consumed(slot)
        return
    gate_up = _get_gate_up_staging(x.device, m, n)
    torch.ops.vllm_sm70_27b_prefill.cublaslt_fp32_mm_out(gate_up, x, dense_w)
    if pf is not None:
        pf.mark_consumed(slot)
    torch.ops.vllm_sm70_27b_prefill.silu_and_mul_out(out, gate_up)


def fast_fp8_prefill_out(
    out: torch.Tensor,
    workspace: torch.Tensor,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    gated_silu: bool,
) -> None:
    m, k = int(x.shape[0]), int(x.shape[1])
    n = int(codes.shape[1])
    pf = _get_prefetcher(x.device, workspace.numel())
    if pf is not None:
        spec = (1, codes, scales, 1.0, k, n)
        dense_w, slot = pf.acquire_and_prefetch(spec, workspace)
    else:
        dense_w = workspace[: k * n].view(k, n)
        torch.ops._C.fp8_qpn8_dequantize_sm70_out(dense_w, codes, scales)
        slot = -1

    if not gated_silu:
        torch.ops.vllm_sm70_27b_prefill.cublaslt_fp32_mm_out(out, x, dense_w)
        if pf is not None:
            pf.mark_consumed(slot)
        return
    gate_up = _get_gate_up_staging(x.device, m, n)
    torch.ops.vllm_sm70_27b_prefill.cublaslt_fp32_mm_out(gate_up, x, dense_w)
    if pf is not None:
        pf.mark_consumed(slot)
    torch.ops.vllm_sm70_27b_prefill.silu_and_mul_out(out, gate_up)


class Sm7027BPeerRuntime:
    """Lockless double-buffered PUSH NVLink P2P RS + GemmaRMSNorm + AG.

    Targets TP2/TP4 H=5120 prefill batches and strictly verifies NVLink/P2P
    native peer atomics across every TP rank; returns None cleanly on PCIe or
    non-mesh topologies so the standard communicator fallback handles them.
    """

    @classmethod
    def try_create(
        cls,
        cpu_group: Any,
        device: torch.device,
        rank: int,
        world_size: int,
        ca_comm: Any,
        max_rows: int = 4096,
    ) -> Sm7027BPeerRuntime | None:
        if envs.VLLM_SM70_27B_DISABLE_P2P_NORM:
            logger.info_once(
                "SM70 27B P2P RS+GemmaRMSNorm+AG disabled via "
                "VLLM_SM70_27B_DISABLE_P2P_NORM=1; using standard communicator "
                "fallback."
            )
            return None
        if world_size not in (2, 4) or not has_sm70_27b_prefill_ops():
            return None

        local_dev = device.index or 0
        local_ok = bool(
            ca_comm is not None
            and not getattr(ca_comm, "disabled", True)
            and getattr(ca_comm, "fully_connected", False)
            and hasattr(torch.ops._C, "sm70_ring_native_peer_atomics")
            and torch.accelerator.device_count() >= world_size
        )
        try:
            from vllm.platforms import current_platform

            my_uuid = current_platform.get_device_uuid(local_dev)
            uuids = [""] * world_size
            dist.all_gather_object(uuids, my_uuid, group=cpu_group)
            if local_ok:
                visible = {
                    current_platform.get_device_uuid(i): i
                    for i in range(torch.accelerator.device_count())
                }
                local_ok = all(
                    u in visible
                    and (
                        visible[u] == local_dev
                        or bool(
                            torch.ops._C.sm70_ring_native_peer_atomics(
                                local_dev, visible[u]
                            )
                        )
                    )
                    for u in uuids
                )
            flags = [False] * world_size
            dist.all_gather_object(flags, local_ok, group=cpu_group)
            if not all(flags):
                logger.info_once(
                    "SM70 27B P2P NVLink RS+GemmaRMSNorm+AG unavailable (PCIe "
                    "or non-mesh P2P); falling back cleanly to standard "
                    "communicator."
                )
                return None
            return cls(cpu_group, device, rank, world_size, max_rows=max_rows)
        except Exception as exc:
            logger.warning_once(
                "SM70 27B P2P NVLink RS+GemmaRMSNorm+AG init failed (%s); "
                "falling back cleanly to standard communicator.",
                exc,
            )
            return None

    def __init__(
        self,
        cpu_group: Any,
        device: torch.device,
        rank: int,
        world_size: int,
        max_rows: int = 4096,
    ):
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_rows = (
            (int(max_rows) + self.world_size - 1) // self.world_size
        ) * self.world_size
        self.max_blocks = 80
        self.bank_bytes = self.max_rows * 5120 * 2
        self.flag_bytes = 4 * self.max_blocks * self.world_size * 4
        self.total_bytes = 4 * self.bank_bytes + ((self.flag_bytes + 255) // 256) * 256
        self.pointers: list[int] | None = CustomAllreduce.create_shared_buffer(
            self.total_bytes, cpu_group
        )
        flag_off = 4 * self.bank_bytes
        ctypes.CDLL("libcudart.so").cudaMemset(
            ctypes.c_void_p(int(self.pointers[self.rank]) + flag_off),
            0,
            self.flag_bytes,
        )
        self.rs_ptrs = [
            [int(p) + 0 * self.bank_bytes for p in self.pointers],
            [int(p) + 1 * self.bank_bytes for p in self.pointers],
        ]
        self.out_ptrs = [
            [int(p) + 2 * self.bank_bytes for p in self.pointers],
            [int(p) + 3 * self.bank_bytes for p in self.pointers],
        ]
        self.flag_ptrs = [int(p) + flag_off for p in self.pointers]
        self._epoch = 0
        torch.accelerator.synchronize(device)
        dist.barrier(group=cpu_group)
        logger.info_once(
            "SM70 27B zero-memcpy PUSH P2P RS+GemmaRMSNorm+AG enabled "
            "(TP=%d, max_rows=%d, %.1f MiB/GPU).",
            self.world_size,
            self.max_rows,
            self.total_bytes / (1024 * 1024),
        )

    def close(self) -> None:
        ptrs = self.pointers
        if ptrs is None:
            return
        self.pointers = None
        try:
            from vllm.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            CustomAllreduce.free_shared_buffer(ptrs, rank=self.rank)
        except Exception:
            pass

    def can_use(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        return bool(
            self.pointers is not None
            and not torch.cuda.is_current_stream_capturing()
            and inp.is_cuda
            and inp.ndim == 2
            and 256 <= inp.shape[0] <= self.max_rows
            and inp.shape[1] == 5120
            and inp.dtype == torch.float16
            and residual.shape == inp.shape
            and residual.dtype == torch.float32
            and weight.ndim == 1
            and weight.numel() == 5120
            and weight.dtype in (torch.float16, torch.float32)
            and inp.is_contiguous()
            and residual.is_contiguous()
            and weight.is_contiguous()
            and inp.device == residual.device == weight.device == self.device
        )

    def push_rs_gemma_norm_ag(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        tail_fallback: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, float],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = int(inp.shape[0])
        ws = self.world_size
        aligned = (m // ws) * ws
        if aligned == m:
            residual_out = torch.empty_like(residual, dtype=torch.float32)
            self._epoch += 1
            slot = self._epoch & 1
            norm_out = torch.ops.vllm_sm70_27b_prefill.p2p_push_rs_gemma_norm_ag(
                inp,
                residual,
                weight,
                residual_out,
                self.rs_ptrs[slot],
                self.out_ptrs[slot],
                self.flag_ptrs,
                self.rank,
                float(epsilon),
                slot,
                self._epoch,
                self.max_blocks,
            )
            return norm_out, residual_out

        norm_out = torch.empty_like(inp)
        residual_out = torch.empty_like(residual, dtype=torch.float32)
        self._epoch += 1
        slot = self._epoch & 1
        aligned_norm = torch.ops.vllm_sm70_27b_prefill.p2p_push_rs_gemma_norm_ag(
            inp[:aligned],
            residual[:aligned],
            weight,
            residual_out[:aligned],
            self.rs_ptrs[slot],
            self.out_ptrs[slot],
            self.flag_ptrs,
            self.rank,
            float(epsilon),
            slot,
            self._epoch,
            self.max_blocks,
        )
        norm_out[:aligned].copy_(aligned_norm)
        norm_tail, res_tail = tail_fallback(
            inp[aligned:], residual[aligned:], weight, epsilon
        )
        norm_out[aligned:].copy_(norm_tail)
        residual_out[aligned:].copy_(res_tail)
        return norm_out, residual_out


def _sm70_27b_gdn_rmsnorm_gated_impl(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str,
    rmsnorm_onepass: bool,
) -> torch.Tensor:
    num_tokens = int(core_attn_out.shape[0])
    num_heads = int(core_attn_out.shape[1])
    if (
        num_tokens >= 256
        and core_attn_out.ndim == 3
        and core_attn_out.shape[2] == 128
        and core_attn_out.dtype == torch.float16
        and z.ndim == 3
        and z.shape == core_attn_out.shape
        and z.dtype == torch.float16
        and z.stride(1) == 128
        and z.stride(2) == 1
        and (z.stride(0) & 3) == 0
        and weight.shape == (128,)
        and weight.dtype in (torch.float16, torch.float32)
        and weight.is_contiguous()
        and activation in ("silu", "swish")
        and not torch.cuda.is_current_stream_capturing()
        and has_sm70_27b_prefill_ops()
    ):
        x_c = (
            core_attn_out
            if core_attn_out.is_contiguous()
            else core_attn_out.contiguous()
        )
        out_2d = torch.empty(
            (num_tokens, num_heads * 128),
            dtype=torch.float16,
            device=core_attn_out.device,
        )
        torch.ops.vllm_sm70_27b_prefill.gdn_rmsnorm_gated_d128_out(
            out_2d, x_c, z, weight, float(eps)
        )
        return out_2d

    # Decode / small-M / CUDA-graph exact fallback.
    core_2d = core_attn_out.reshape(-1, 128)
    if not core_2d.is_contiguous():
        core_2d = core_2d.contiguous()
    z_2d = z.contiguous().reshape(-1, 128)
    out = torch.ops.vllm.sm70_qwen_gdn_rmsnorm_gated_configured(
        core_2d,
        z_2d,
        weight,
        float(eps),
        -1,
        True,
        activation,
        rmsnorm_onepass,
    )
    return out.reshape(num_tokens, num_heads * 128)


def _sm70_27b_gdn_rmsnorm_gated_fake(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str,
    rmsnorm_onepass: bool,
) -> torch.Tensor:
    del z, weight, eps, activation, rmsnorm_onepass
    return core_attn_out.new_empty(
        (core_attn_out.shape[0], core_attn_out.shape[1] * core_attn_out.shape[2])
    )


def _ensure_dynamo_ops() -> None:
    global _DYNAMO_OPS_REGISTERED
    if _DYNAMO_OPS_REGISTERED:
        return
    if not hasattr(torch.ops.vllm, "sm70_27b_gdn_rmsnorm_gated"):
        direct_register_custom_op(
            "sm70_27b_gdn_rmsnorm_gated",
            _sm70_27b_gdn_rmsnorm_gated_impl,
            mutates_args=[],
            fake_impl=_sm70_27b_gdn_rmsnorm_gated_fake,
        )
    _DYNAMO_OPS_REGISTERED = True


_init_sm70_27b_prefill_ops()
