# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 SwiGLU and routing boundaries around the original Q8_1 integer dots."""

import os
from typing import Any

import torch

import vllm.envs as envs
from vllm._sm70.policy import register_policy_op

from .gguf_native import native_dense


def v41_fp16_experts(
    x,
    ids,
    probabilities,
    gate,
    up,
    down,
    source_type,
    down_type,
    left,
    swiglu_limit=10.0,
    maximum_bytes=32 * 1024**2,
):
    """Bounded FP16 tail: one routed expert's matrix at a time, all on GPU.

    GPU index_select gathers one packed carrier using its device-side slot.
    Only that matrix is dequantized for cuBLAS. No CPU routes, matrices, or
    full decoded expert bank are materialized. This intentionally favors a
    small correctness fallback over permanently reserving another library.
    """
    if (
        not x.is_cuda
        or x.dtype != torch.float16
        or x.ndim != 2
        or not 1 <= x.shape[0] <= 6
        or x.shape[1] != 5120
        or ids.shape != (x.shape[0], 6)
        or ids.device != x.device
        or ids.dtype not in (torch.int32, torch.int64)
        or probabilities.shape != ids.shape
        or probabilities.dtype != torch.float32
        or probabilities.device != x.device
        or swiglu_limit != 10.0
        or source_type not in (16, 17, 29)
        or down_type not in (16, 17, 29)
        or gate.ndim != 3
        or gate.shape[1] != 576
        or up.shape != gate.shape
        or down.ndim != 3
        or down.shape[1] != 5120
        or down.shape[0] != gate.shape[0]
        or left not in (0, 64, 128, 192)
        or maximum_bytes < 1024**2
        or any(
            bank.device != x.device
            or bank.dtype != torch.uint8
            or not bank.is_contiguous()
            for bank in (gate, up, down)
        )
    ):
        raise ValueError("V4.1 FP16 tail requires M1..6, top-6 and TP4 carriers")
    from vllm.transformers_utils.gguf_tensor_reader import quant_size

    block, size = quant_size(down_type)
    physical_k = down.shape[2] // size * block
    if physical_k != 768:
        raise ValueError("V4.1 FP16 down carrier must cover 768 columns")
    flat = ids.reshape(-1).long()
    outputs = torch.empty(x.shape[0] * 6, 5120, device=x.device, dtype=x.dtype)

    def projection(bank, kind, route, values):
        packed = torch.index_select(bank, 0, flat[route : route + 1]).squeeze(0)
        output = native_dense(
            values, packed, kind, prefill_min_m=1, max_dequant_bytes=maximum_bytes
        )
        if output is None:
            raise RuntimeError("V4.1 FP16 expert GEMM is unavailable")
        return output

    for route in range(flat.numel()):
        token = route // 6
        values = x[token : token + 1].contiguous()
        g = projection(gate, source_type, route, values).float().clamp(max=10)
        u = projection(up, source_type, route, values).float().clamp(-10, 10)
        hidden = (
            torch.nn.functional.silu(g) * u * probabilities.reshape(-1)[route]
        ).half()
        hidden = torch.nn.functional.pad(hidden, (left, physical_k - left - 576))
        outputs[route : route + 1].copy_(projection(down, down_type, route, hidden))
    return outputs.view(x.shape[0], 6, 5120).float().sum(1).half()


def v41_dp4a_available():
    return all(
        hasattr(torch.ops._C, name)
        for name in (
            "gguf_v41_dp4a_gate_up_sm70_out",
            "gguf_v41_dp4a_gate_adaptive_sm70_out",
            "gguf_v41_dp4a_down_sm70_out",
            "gguf_v41_dp4a_q8_down_sm70_out",
            "gguf_v41_dp4a_q8_down_tiled_sm70_out",
            "gguf_v41_swiglu_q8_sm70_out",
        )
    )


def default_v41k_library():
    """VLLM_V41K_LIB, else the in-tree build (benchmarks/deepseek_v41/v41k/build.py)."""
    return envs.VLLM_V41K_LIB or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "benchmarks",
        "deepseek_v41",
        "v41k",
        "build",
        "v41k.so",
    )


_EM: Any = {"loaded": None, "books": {}}
_EM5 = envs.VLLM_V41_EM5 == "1"


def _expert_major_ready():
    """Expert-major V4.1 kernels (one decode of each distinct expert per layer)."""
    if _EM["loaded"] is None:
        _EM["loaded"] = False  # type: ignore[assignment]
        if envs.VLLM_V41_EM == "1":
            path = default_v41k_library()
            if os.path.exists(path):
                torch.ops.load_library(path)
                _EM["loaded"] = True  # type: ignore[assignment]
    return _EM["loaded"]


_EM7 = envs.VLLM_V41_EM7 == "1"
os.environ.setdefault("VLLM_V41_EM8", "0")  # single-barrier down: no gain measured


def _plan(device):
    key = ("plan", device)
    if key not in _EM["books"]:  # type: ignore[operator]
        _EM["books"][key] = torch.zeros(512, dtype=torch.int32, device=device)  # type: ignore[index]
    return _EM["books"][key]  # type: ignore[index]


def _book(kind, device):
    key = (kind, device)
    if key not in _EM["books"]:  # type: ignore[operator]
        _EM["books"][key] = torch.ops.v41k.table(kind, device)  # type: ignore[index]
    return _EM["books"][key]  # type: ignore[index]


_ROWS = {}
# Peer gather bank: four rank slots of at most 36 routes x 648 B, then flags.
ROWS_STRIDE_MAX = (36 * 648 + 255) // 256 * 256
ROWS_FLAGS = 4 * ROWS_STRIDE_MAX
ROWS_BYTES = ROWS_FLAGS + 256
_PEER = {}


def register_rows_peer(device, rank, pointers, own):
    """Called once at load with the IPC bank (own: uint8 view of this rank's)."""
    own.zero_()
    _PEER[device] = (
        rank,
        tuple(int(p) for p in pointers),
        own,
        torch.zeros(1, dtype=torch.int32, device=device),
    )


def _rows_context(device):
    """TP rank and the NCCL communicator for the row-split down gather."""
    if device not in _ROWS:
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        _ROWS[device] = (group.rank_in_group, group.device_communicator.pynccl_comm)  # type: ignore[union-attr]
    return _ROWS[device]


def _v41_experts_rows(
    x, ids32, probabilities, gate, up, down, source_type, down_type, rows, warps
):
    """Row-split down: local Q8 h groups are gathered, then 1280 rows per rank."""
    m = x.shape[0]
    routes = m * 6
    stride = (routes * 648 + 255) // 256 * 256
    xq = torch.empty((m, 160, 32), dtype=torch.int8, device=x.device)
    xd = torch.empty((m, 160), dtype=torch.float32, device=x.device)
    torch.ops.v41k.q8_aligned(xq, xd, x.contiguous())
    peer = _PEER.get(x.device)
    if peer is not None and hasattr(torch.ops.v41k, "peer_gather"):
        rank, pointers, own, epoch = peer
        slot = own[rank * stride : rank * stride + routes * 648]
        hq = slot[: routes * 576].view(torch.int8).view(routes, 18, 32)
        hd = slot[routes * 576 :].view(torch.float32).view(routes, 18)
        plan = _plan(x.device)
        torch.ops.v41k.gate_up7(
            xq,
            xd,
            hq,
            hd,
            ids32,
            probabilities.contiguous(),
            gate,
            up,
            source_type,
            _book(source_type, x.device),
            plan,
        )
        torch.ops.v41k.peer_gather(*pointers, rank, stride, ROWS_FLAGS, epoch)
        result = torch.empty_like(x)  # down_rows writes zeros in other rows
        torch.ops.v41k.down_rows(
            result,
            own,
            stride,
            routes,
            down,
            down_type,
            rank * down.shape[1],
            _book(down_type, x.device),
            rows,
            warps,
            plan,
        )
        return result
    rank, nccl = _rows_context(x.device)
    packed = torch.empty(stride, dtype=torch.uint8, device=x.device)
    hq = packed[: routes * 576].view(torch.int8).view(routes, 18, 32)
    hd = packed[routes * 576 : routes * 648].view(torch.float32).view(routes, 18)
    plan = _plan(x.device)
    torch.ops.v41k.gate_up7(
        xq,
        xd,
        hq,
        hd,
        ids32,
        probabilities.contiguous(),
        gate,
        up,
        source_type,
        _book(source_type, x.device),
        plan,
    )
    gathered = torch.empty(4 * stride, dtype=torch.uint8, device=x.device)
    nccl.all_gather(gathered, packed)
    result = torch.empty_like(x)
    torch.ops.v41k.down_rows(
        result,
        gathered,
        stride,
        routes,
        down,
        down_type,
        rank * down.shape[1],
        _book(down_type, x.device),
        rows,
        warps,
        plan,
    )
    return result


def _v41_experts(
    x, ids, probabilities, gate, up, down, source_type, down_type, left, swiglu_limit
):
    if (
        not x.is_cuda
        or x.dtype != torch.float16
        or x.ndim != 2
        or not 1 <= x.shape[0] <= 6
        or x.shape[1] != 5120
        or ids.shape != (x.shape[0], 6)
        or probabilities.shape != ids.shape
        or probabilities.dtype != torch.float32
        or swiglu_limit != 10.0
        or source_type not in (16, 17, 29)
        or down_type not in (16, 17, 29)
        or gate.ndim != 3
        or gate.shape[1] != 576
    ):
        raise ValueError("V4.1 integer experts require FP16 M1..6, top-6 and clamp 10")
    if _expert_major_ready() and down.shape[1] == 1280:
        # Row tile sweep (V100, 36 routes): 4 rows x 8 lanes of K, 8 warps.
        rows, warps = (int(v) for v in envs.VLLM_V41_ROWS_TILE.split(","))  # type: ignore[union-attr]
        return _v41_experts_rows(
            x,
            ids.to(torch.int32).contiguous(),
            probabilities,
            gate,
            up,
            down,
            source_type,
            down_type,
            rows,
            warps,
        )
    if _expert_major_ready() and _EM5:
        m = x.shape[0]
        xq = torch.empty((m, 160, 32), dtype=torch.int8, device=x.device)
        xd = torch.empty((m, 160), dtype=torch.float32, device=x.device)
        hq = torch.empty((m * 6, 18, 32), dtype=torch.int8, device=x.device)
        hd = torch.empty((m * 6, 18), dtype=torch.float32, device=x.device)
        torch.ops.v41k.q8_aligned(xq, xd, x.contiguous())
        result = torch.empty_like(x)
        # Down tiles per type (V100 sweep): IQ1_M 8 warps x 16 rows, IQ2 4 x 8.
        rows, warps = (16, 8) if down_type == 29 else (8, 4)
        args = (
            result,
            xq,
            xd,
            hq,
            hd,
            ids.to(torch.int32).contiguous(),
            probabilities.contiguous(),
            gate,
            up,
            down,
            source_type,
            down_type,
            left,
            _book(source_type, x.device),
            _book(down_type, x.device),
            rows,
            warps,
        )
        if _EM7 and hasattr(torch.ops.v41k, "moe7"):
            # One warp plans distinct experts per layer instead of every CTA.
            torch.ops.v41k.moe7(*args, _plan(x.device))
        else:
            torch.ops.v41k.moe5w(*args)
        return result
    q8 = torch.empty((x.shape[0], 160, 36), dtype=torch.uint8, device=x.device)
    torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x.contiguous())
    if _expert_major_ready():
        ids32 = ids.to(torch.int32).contiguous()
        down_q8 = torch.empty(
            (x.shape[0], 6, 18, 36), dtype=torch.uint8, device=x.device
        )
        torch.ops.v41k.gate_up4(
            down_q8,
            q8,
            ids32,
            probabilities.contiguous(),
            gate,
            up,
            source_type,
            _book(source_type, x.device),
        )
        result = torch.empty_like(x)
        torch.ops.v41k.down3(
            result,
            down_q8,
            ids32,
            down,
            down_type,
            left,
            _book(down_type, x.device),
            16,
        )
        return result
    projections = x.new_empty((x.shape[0], 6, 2, 576))
    if x.shape[0] == 6 and source_type == 29:
        # IQ1_M benefits across the measured route distribution. A device
        # classifier retains the original path for insufficient repetition.
        # IQ2 retains its original kernel after worst-route ABBA checks.
        torch.ops._C.gguf_v41_dp4a_gate_adaptive_sm70_out(
            projections, q8, ids, gate, up, source_type
        )
    else:
        torch.ops._C.gguf_v41_dp4a_gate_up_sm70_out(
            projections,
            q8,
            ids,
            gate,
            up,
            source_type,
            False,
            16,
            False,
        )
    # The checkpoint rounds each linear result to the compute dtype, then
    # clamps and activates in FP32. Routing multiplies the down input before
    # its FP16/Q8_1 boundary, rather than weighting the down output afterward.
    # Quantize each routed row once, rather than repeating the warp
    # reductions in every down-projection output tile.
    down_q8 = torch.empty((x.shape[0], 6, 18, 36), dtype=torch.uint8, device=x.device)
    torch.ops._C.gguf_v41_swiglu_q8_sm70_out(down_q8, projections, probabilities)
    result = torch.empty_like(x)
    torch.ops._C.gguf_v41_dp4a_q8_down_tiled_sm70_out(
        result,
        down_q8,
        ids,
        torch.ones_like(probabilities),
        down,
        down_type,
        left,
    )
    return result


def _v41_experts_fake(
    x, ids, probabilities, gate, up, down, source_type, down_type, left, swiglu_limit
):
    return torch.empty_like(x)


register_policy_op(
    "gguf_v41_original_expert_dp4a",
    "(Tensor x, Tensor ids, Tensor probabilities, Tensor gate, Tensor up, "
    "Tensor down, int source_type, int down_type, int left, "
    "float swiglu_limit) -> Tensor",
    _v41_experts,
    _v41_experts_fake,
)
