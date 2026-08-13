# SPDX-License-Identifier: MIT
# Adapted from ref/V100-SM_70-Flash-Attn-2-v1 (fa2_sm70), Copyright (c) 2026 xormal
# (Alexander Romanoff), MIT license. See csrc/sm70_chunk_delta_h.cu for provenance.
#
# SM70 (Volta) tensor-core replacement for chunk_gated_delta_rule_fwd_h's Triton kernel
# (patches/0100). Triton targeting sm_70 emits no HMMA at all — tl.dot lowers to scalar
# FP32 FMA — so the chunk-state GEMMs run at the FMA ceiling while the fp16 tensor cores
# sit idle. This wrapper JIT-compiles the hand-written HMMA kernel (the flash_qla
# precedent: torch.utils.cpp_extension.load, no prebuilt artifact) and is dispatched from
# chunk_delta_h.py behind a strict shape gate with the Triton kernel as fallback.

from __future__ import annotations

import os
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_EXT = None
_LOAD_FAILED = False

# BV (the V-slice a block owns) and PIPE (register-staged chunk-window pipeline) select
# among compiled instantiations. Defaults are the measured winner at the served TP4
# per-rank shape (Hg=4, H=8, K=V=128, BT=64): BV=16/PIPE=1 won at EVERY swept T
# (2048/8192/65536, patches/0100 PoC, V100 board, 150 W cap) — at per-rank H=8 the
# grid is (V/BV)*N*H blocks, so BV=16 is the only choice that covers 64 of 80 SMs at
# N=1; the reference implementation's BV=64 optimum was measured at H=24 and does NOT
# transfer. Env overrides keep the other instantiations reachable for diagnostics,
# mirroring the SM70 Triton-schedule precedent in chunk_delta_h.py.
_BV_CHOICES = (16, 32, 64)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def sm70_delta_h_tc_enabled() -> bool:
    return os.getenv("VLLM_SM70_GDN_DELTA_H_TC", "1") == "1"


def _load_ext():
    global _EXT, _LOAD_FAILED
    if _EXT is not None:
        return _EXT
    if _LOAD_FAILED:
        return None
    try:
        from torch.utils.cpp_extension import load

        src = Path(__file__).with_name("csrc") / "sm70_chunk_delta_h.cu"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
        _EXT = load(
            name="monico_sm70_gdn_delta_h",
            sources=[str(src)],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=bool(int(os.environ.get("VLLM_SM70_GDN_DELTA_H_TC_VERBOSE", "0"))),
        )
    except Exception:
        # Any build/load failure disables the path permanently for this process; the
        # caller falls back to the Triton kernel, which is always correct.
        logger.warning(
            "SM70 tensor-core chunk_delta_h JIT load failed; "
            "falling back to the Triton kernel.",
            exc_info=True,
        )
        _LOAD_FAILED = True
        return None
    return _EXT


def sm70_delta_h_tc_supported(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    chunk_size: int,
    save_new_value: bool,
    cu_seqlens: torch.Tensor | None,
    use_exp2: bool,
) -> bool:
    """Strict gate: exactly the configuration the kernel implements and patches/0100
    proved. Anything else routes to the Triton kernel (correct for every shape)."""
    if not sm70_delta_h_tc_enabled():
        return False
    if cu_seqlens is None or g is None or gk is not None or use_exp2:
        return False
    if not save_new_value:
        return False
    if chunk_size != 64:
        return False
    if k.dtype != torch.float16 or w.dtype != torch.float16 or u.dtype != torch.float16:
        return False
    if g.dtype != torch.float32:
        return False
    if k.shape[0] != 1:  # varlen packs the batch into T with B == 1
        return False
    if k.shape[-1] != 128 or u.shape[-1] != 128:
        return False
    H, Hg = u.shape[-2], k.shape[-2]
    if Hg <= 0 or H % Hg != 0:
        return False
    if initial_state is not None and (
        initial_state.dtype != torch.float32 or not initial_state.is_contiguous()
    ):
        return False
    if not (
        k.is_contiguous() and w.is_contiguous() and u.is_contiguous() and g.is_contiguous()
    ):
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(k.device)
    if cap != (7, 0):
        return False
    return _load_ext() is not None


def sm70_chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Same contract as the Triton path of chunk_gated_delta_rule_fwd_h (varlen,
    save_new_value=True): returns (h [1,NT,H,V,K] fp16, v_new like u, final_state
    [N,H,V,K] fp32 or None)."""
    ext = _load_ext()
    assert ext is not None, "call sm70_delta_h_tc_supported first"
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    if chunk_offsets.dtype != torch.int32:
        chunk_offsets = chunk_offsets.to(torch.int32)
    bv = _env_int("VLLM_SM70_GDN_DELTA_H_TC_BV", 16)
    if bv not in _BV_CHOICES:
        bv = 16
    pipe = os.getenv("VLLM_SM70_GDN_DELTA_H_TC_PIPE", "1") == "1"
    h, v_new, final_state = ext.fwd_h(
        k,
        w,
        u,
        g,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        num_chunks,
        output_final_state,
        bv,
        pipe,
    )
    return h, v_new, final_state if output_final_state else None
