# SPDX-License-Identifier: Apache-2.0
"""Strict exactness harness for the GDN/FLA chunk prefill op on SM70.

This script intentionally does not import vLLM at module import time. Run the
same saved input case in two different source trees/environments, then compare
the saved outputs with torch.equal and max_diff == 0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _dtype(name: str) -> torch.dtype:
    by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return by_name[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _resolve_state_dtype(name: str, tensor_dtype: torch.dtype) -> torch.dtype:
    if name == "same":
        return tensor_dtype
    return _dtype(name)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    finite = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": _tensor_sha256(tensor),
        "min": float(finite.min().item()) if finite.numel() else None,
        "max": float(finite.max().item()) if finite.numel() else None,
        "mean": float(finite.mean().item()) if finite.numel() else None,
    }


def _call_supported(fn: Any, **kwargs: Any) -> Any:
    params = inspect.signature(fn).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return fn(**kwargs)
    filtered = {key: val for key, val in kwargs.items() if key in params}
    return fn(**filtered)


def _vllm_version() -> str:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "case_name": args.case_name,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm_version": _vllm_version(),
        "device_name": torch.cuda.get_device_name()
        if torch.cuda.is_available()
        else None,
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
                "VLLM_SM70_GDN_KKT_SCHEDULE",
                "VLLM_SM70_GDN_KKT_BK",
                "VLLM_SM70_GDN_KKT_WARPS",
                "VLLM_SM70_GDN_KKT_STAGES",
                "VLLM_SM70_GDN_DELTA_H_SCHEDULE",
                "VLLM_SM70_GDN_DELTA_H_BV",
                "VLLM_SM70_GDN_DELTA_H_WARPS",
                "VLLM_SM70_GDN_DELTA_H_STAGES",
                "VLLM_SM70_GDN_CHUNK_O_SCHEDULE",
                "VLLM_SM70_GDN_CHUNK_O_BK",
                "VLLM_SM70_GDN_CHUNK_O_BV",
                "VLLM_SM70_GDN_CHUNK_O_WARPS",
                "VLLM_SM70_GDN_CHUNK_O_STAGES",
                "VLLM_SM70_FUSED_SIGMOID_GATING_SCHED",
                "VLLM_SM70_FUSED_SIGMOID_GATING_BV",
                "VLLM_SM70_FUSED_SIGMOID_GATING_WARPS",
                "VLLM_SM70_FUSED_SIGMOID_GATING_STAGES",
                "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV",
                "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV_COMPARE",
                "VLLM_SM70_FLA_RECURRENT_SCHEDULE",
                "VLLM_SM70_FLA_BV",
                "VLLM_SM70_FLA_WARPS",
                "VLLM_SM70_FLA_STAGES",
                "VLLM_SM70_FLA_TARGET_WAVES",
                "VLLM_SM70_FLA_BV_CANDIDATES",
            )
        },
    }


def _make_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    state_dtype = _resolve_state_dtype(args.state_dtype, dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    q = torch.randn(
        (args.batch, args.seqlen, args.k_heads, args.head_k_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        (args.batch, args.seqlen, args.k_heads, args.head_k_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if args.l2norm_inputs:
        q = F.normalize(q.float(), p=2, dim=-1).to(dtype)
        k = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    else:
        q = (q * args.input_scale).to(dtype)
        k = (k * args.input_scale).to(dtype)

    v = (
        torch.randn(
            (args.batch, args.seqlen, args.v_heads, args.head_v_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.input_scale
    ).to(dtype)
    g = -torch.rand(
        (args.batch, args.seqlen, args.v_heads),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) * args.gate_scale
    beta = torch.sigmoid(
        torch.randn(
            (args.batch, args.seqlen, args.v_heads),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * args.beta_scale
    )
    if args.random_initial_state:
        initial_state = (
            torch.randn(
                (
                    args.batch,
                    args.v_heads,
                    args.head_v_dim,
                    args.head_k_dim,
                ),
                device=device,
                dtype=state_dtype,
                generator=generator,
            )
            * args.input_scale
        ).to(state_dtype)
    else:
        initial_state = torch.zeros(
            (args.batch, args.v_heads, args.head_v_dim, args.head_k_dim),
            device=device,
            dtype=state_dtype,
        )

    result = {
        "q": q.cpu(),
        "k": k.cpu(),
        "v": v.cpu(),
        "g": g.cpu(),
        "beta": beta.cpu(),
        "initial_state": initial_state.cpu(),
    }
    if args.cu_seqlens:
        result["cu_seqlens"] = torch.tensor(
            [0, args.seqlen], dtype=torch.int32
        )
    return result


def _load_or_create_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    if args.input_file:
        payload = _torch_load(Path(args.input_file))
        if "inputs" not in payload:
            raise ValueError(f"{args.input_file} does not contain saved inputs")
        return payload["inputs"]
    return _make_inputs(args)


def _make_packed_recurrent_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    state_dtype = _resolve_state_dtype(args.state_dtype, dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    q = torch.randn(
        (args.batch, args.k_heads, args.head_k_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        (args.batch, args.k_heads, args.head_k_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if args.l2norm_inputs:
        q = F.normalize(q.float(), p=2, dim=-1).to(dtype)
        k = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    else:
        q = (q * args.input_scale).to(dtype)
        k = (k * args.input_scale).to(dtype)
    v = (
        torch.randn(
            (args.batch, args.v_heads, args.head_v_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.input_scale
    ).to(dtype)
    mixed_qkv = torch.cat(
        [
            q.reshape(args.batch, -1),
            k.reshape(args.batch, -1),
            v.reshape(args.batch, -1),
        ],
        dim=-1,
    ).contiguous()
    a = (
        torch.randn(
            (args.batch, args.v_heads),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.gate_scale
    ).contiguous()
    b = (
        torch.randn(
            (args.batch, args.v_heads),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.beta_scale
    ).contiguous()
    a_log = (
        torch.randn(
            (args.v_heads,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
        - 1.0
    ).contiguous()
    dt_bias = (
        torch.randn(
            (args.v_heads,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * args.gate_scale
    ).contiguous()
    initial_state = torch.zeros(
        (args.batch + 1, args.v_heads, args.head_v_dim, args.head_k_dim),
        device=device,
        dtype=state_dtype,
    )
    if args.random_initial_state:
        initial_state[1:] = (
            torch.randn(
                (
                    args.batch,
                    args.v_heads,
                    args.head_v_dim,
                    args.head_k_dim,
                ),
                device=device,
                dtype=state_dtype,
                generator=generator,
            )
            * args.input_scale
        ).to(state_dtype)
    ssm_state_indices = torch.arange(
        1, args.batch + 1, device=device, dtype=torch.int32
    )
    return {
        "mixed_qkv": mixed_qkv.cpu(),
        "a": a.cpu(),
        "b": b.cpu(),
        "A_log": a_log.cpu(),
        "dt_bias": dt_bias.cpu(),
        "initial_state": initial_state.cpu(),
        "ssm_state_indices": ssm_state_indices.cpu(),
    }


def _load_or_create_packed_recurrent_inputs(
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if args.input_file:
        payload = _torch_load(Path(args.input_file))
        if "inputs" not in payload:
            raise ValueError(f"{args.input_file} does not contain saved inputs")
        return payload["inputs"]
    return _make_packed_recurrent_inputs(args)


def _make_packed_recurrent_sequence_inputs(
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    state_dtype = _resolve_state_dtype(args.state_dtype, dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    q = torch.randn(
        (args.seqlen, args.batch, args.k_heads, args.head_k_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        (args.seqlen, args.batch, args.k_heads, args.head_k_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if args.l2norm_inputs:
        q = F.normalize(q.float(), p=2, dim=-1).to(dtype)
        k = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    else:
        q = (q * args.input_scale).to(dtype)
        k = (k * args.input_scale).to(dtype)
    v = (
        torch.randn(
            (args.seqlen, args.batch, args.v_heads, args.head_v_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.input_scale
    ).to(dtype)
    mixed_qkv = torch.cat(
        [
            q.reshape(args.seqlen, args.batch, -1),
            k.reshape(args.seqlen, args.batch, -1),
            v.reshape(args.seqlen, args.batch, -1),
        ],
        dim=-1,
    ).contiguous()
    a = (
        torch.randn(
            (args.seqlen, args.batch, args.v_heads),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.gate_scale
    ).contiguous()
    b = (
        torch.randn(
            (args.seqlen, args.batch, args.v_heads),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.beta_scale
    ).contiguous()
    a_log = (
        torch.randn(
            (args.v_heads,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
        - 1.0
    ).contiguous()
    dt_bias = (
        torch.randn(
            (args.v_heads,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * args.gate_scale
    ).contiguous()
    initial_state = torch.zeros(
        (args.batch + 1, args.v_heads, args.head_v_dim, args.head_k_dim),
        device=device,
        dtype=state_dtype,
    )
    if args.random_initial_state:
        initial_state[1:] = (
            torch.randn(
                (
                    args.batch,
                    args.v_heads,
                    args.head_v_dim,
                    args.head_k_dim,
                ),
                device=device,
                dtype=state_dtype,
                generator=generator,
            )
            * args.input_scale
        ).to(state_dtype)
    ssm_state_indices = torch.arange(
        1, args.batch + 1, device=device, dtype=torch.int32
    )
    return {
        "mixed_qkv_seq": mixed_qkv.cpu(),
        "a_seq": a.cpu(),
        "b_seq": b.cpu(),
        "A_log": a_log.cpu(),
        "dt_bias": dt_bias.cpu(),
        "initial_state": initial_state.cpu(),
        "ssm_state_indices": ssm_state_indices.cpu(),
    }


def _make_conv_update_sequence_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    state_dtype = _resolve_state_dtype(args.state_dtype, dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    conv_dim = args.conv_dim
    if conv_dim is None:
        conv_dim = 2 * args.k_heads * args.head_k_dim + args.v_heads * args.head_v_dim

    x_seq = (
        torch.randn(
            (args.seqlen, args.batch, conv_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.input_scale
    ).contiguous()
    conv_state = torch.zeros(
        (args.batch + 1, conv_dim, args.conv_width - 1),
        device=device,
        dtype=state_dtype,
    )
    if args.random_initial_state:
        conv_state[1:] = (
            torch.randn(
                (args.batch, conv_dim, args.conv_width - 1),
                device=device,
                dtype=state_dtype,
                generator=generator,
            )
            * args.input_scale
        ).to(state_dtype)
    weight = (
        torch.randn(
            (conv_dim, args.conv_width),
            device=device,
            dtype=state_dtype,
            generator=generator,
        )
        * args.input_scale
    ).contiguous()
    bias = (
        torch.randn(
            (conv_dim,),
            device=device,
            dtype=state_dtype,
            generator=generator,
        )
        * args.input_scale
    ).contiguous()
    conv_state_indices = torch.arange(
        1, args.batch + 1, device=device, dtype=torch.int32
    )
    return {
        "x_seq": x_seq.cpu(),
        "conv_state": conv_state.cpu(),
        "weight": weight.cpu(),
        "bias": bias.cpu(),
        "conv_state_indices": conv_state_indices.cpu(),
    }


def _load_or_create_packed_recurrent_sequence_inputs(
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if args.input_file:
        payload = _torch_load(Path(args.input_file))
        if "inputs" not in payload:
            raise ValueError(f"{args.input_file} does not contain saved inputs")
        return payload["inputs"]
    return _make_packed_recurrent_sequence_inputs(args)


def _load_or_create_conv_update_sequence_inputs(
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if args.input_file:
        payload = _torch_load(Path(args.input_file))
        if "inputs" not in payload:
            raise ValueError(f"{args.input_file} does not contain saved inputs")
        return payload["inputs"]
    return _make_conv_update_sequence_inputs(args)


def _make_sigmoid_gating_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    inputs = _make_inputs(args)
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    state_dtype = _resolve_state_dtype(args.state_dtype, dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 17)
    a = (
        torch.randn(
            (args.batch, args.seqlen, args.v_heads),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * args.gate_scale
    )
    b = (
        torch.randn(
            (args.batch, args.seqlen, args.v_heads),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * args.beta_scale
    )
    a_log = (
        torch.randn(
            (args.v_heads,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
        - 1.0
    )
    dt_bias = (
        torch.randn(
            (args.v_heads,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * args.gate_scale
    )
    inputs.update({
        "a": a.cpu(),
        "b": b.cpu(),
        "A_log": a_log.cpu(),
        "dt_bias": dt_bias.cpu(),
    })
    if getattr(args, "decode_segments", False):
        if args.batch != 1:
            raise ValueError("decode-segments fixture requires --batch 1")
        if args.random_initial_state:
            initial_state = (
                torch.randn(
                    (
                        args.seqlen + 1,
                        args.v_heads,
                        args.head_v_dim,
                        args.head_k_dim,
                    ),
                    device=device,
                    dtype=state_dtype,
                    generator=generator,
                )
                * args.input_scale
            ).to(state_dtype)
        else:
            initial_state = torch.zeros(
                (args.seqlen + 1, args.v_heads, args.head_v_dim, args.head_k_dim),
                device=device,
                dtype=state_dtype,
            )
        inputs["initial_state"] = initial_state.cpu()
        inputs["cu_seqlens"] = torch.arange(
            0, args.seqlen + 1, dtype=torch.int32
        )
        inputs["ssm_state_indices"] = torch.arange(
            1, args.seqlen + 1, dtype=torch.int32
        )
    return inputs


def _load_or_create_sigmoid_gating_inputs(
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if args.input_file:
        payload = _torch_load(Path(args.input_file))
        if "inputs" not in payload:
            raise ValueError(f"{args.input_file} does not contain saved inputs")
        return payload["inputs"]
    return _make_sigmoid_gating_inputs(args)


def _to_cuda_inputs(
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    cuda_inputs = {
        name: tensor.cuda(non_blocking=False)
        for name, tensor in inputs.items()
        if name != "cu_seqlens"
    }
    cu_seqlens = inputs.get("cu_seqlens")
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.cuda(non_blocking=False)
    return cuda_inputs, cu_seqlens


def _run_chunk_pipeline(
    cuda_inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor | None,
    scale: float,
) -> dict[str, torch.Tensor | None]:
    from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o
    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
    from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
    from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd

    chunk_size = 64
    g_cumsum = _call_supported(
        chunk_local_cumsum,
        g=cuda_inputs["g"],
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
    )
    a_kkt = _call_supported(
        chunk_scaled_dot_kkt_fwd,
        k=cuda_inputs["k"],
        beta=cuda_inputs["beta"],
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    a_solved = _call_supported(
        solve_tril,
        A=a_kkt,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        output_dtype=cuda_inputs["k"].dtype,
    )
    w, u = _call_supported(
        recompute_w_u_fwd,
        k=cuda_inputs["k"],
        v=cuda_inputs["v"],
        beta=cuda_inputs["beta"],
        A=a_solved,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
    )
    h, v_new, final_state = _call_supported(
        chunk_gated_delta_rule_fwd_h,
        k=cuda_inputs["k"],
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=cuda_inputs["initial_state"],
        output_final_state=True,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_offsets=None,
    )
    out = _call_supported(
        chunk_fwd_o,
        q=cuda_inputs["q"],
        k=cuda_inputs["k"],
        v=v_new,
        h=h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    return {
        "stage_g_cumsum": g_cumsum,
        "stage_A_kkt": a_kkt,
        "stage_A_solved": a_solved,
        "stage_w": w,
        "stage_u": u,
        "stage_h": h,
        "stage_v_new": v_new,
        "stage_final_state": final_state,
        "stage_out": out,
    }


def run_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule

    inputs = _load_or_create_inputs(args)
    cuda_inputs, cu_seqlens = _to_cuda_inputs(inputs)
    scale = args.scale
    if scale is None:
        scale = cuda_inputs["k"].shape[-1] ** -0.5

    def run_op() -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_gated_delta_rule(
            q=cuda_inputs["q"],
            k=cuda_inputs["k"],
            v=cuda_inputs["v"],
            g=cuda_inputs["g"],
            beta=cuda_inputs["beta"],
            scale=scale,
            initial_state=cuda_inputs["initial_state"],
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )

    out = None
    final_state = None
    timings_s: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            out, final_state = run_op()
            torch.cuda.synchronize()
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out, final_state = run_op()
            torch.cuda.synchronize()
            timings_s.append(time.perf_counter() - start)

    assert out is not None
    elapsed_s = sum(timings_s)

    outputs: dict[str, torch.Tensor | None] = {
        "out": out.detach().cpu(),
        "final_state": final_state.detach().cpu()
        if final_state is not None
        else None,
    }
    if args.save_intermediates:
        pipeline_outputs = _run_chunk_pipeline(cuda_inputs, cu_seqlens, scale)
        torch.cuda.synchronize()
        outputs.update(
            {
                name: tensor.detach().cpu() if tensor is not None else None
                for name, tensor in pipeline_outputs.items()
            }
        )

    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "batch": args.batch,
            "seqlen": args.seqlen,
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "cu_seqlens": args.cu_seqlens,
            "scale": scale,
            "l2norm_inputs": args.l2norm_inputs,
            "random_initial_state": args.random_initial_state,
            "save_intermediates": args.save_intermediates,
        },
        "elapsed_s": elapsed_s,
        "timings_s": timings_s,
        "timing_summary": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "avg_s": elapsed_s / len(timings_s) if timings_s else None,
            "min_s": min(timings_s) if timings_s else None,
            "max_s": max(timings_s) if timings_s else None,
        },
        "inputs": saved_inputs,
        "input_summaries": {
            name: _tensor_summary(tensor) for name, tensor in saved_inputs.items()
        },
        "outputs": outputs,
        "output_summaries": {
            name: _tensor_summary(tensor) if tensor is not None else None
            for name, tensor in outputs.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(json.dumps({
        "out": str(out_path),
        "elapsed_s": elapsed_s,
        "timing_summary": payload["timing_summary"],
        "output_summaries": payload["output_summaries"],
    }, indent=2))
    return 0


def run_recurrent_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.batch != 1:
        raise ValueError(
            "`run-recurrent` uses the non-continuous single-sequence recurrent "
            "path; use --batch 1."
        )

    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    inputs = _load_or_create_inputs(args)
    cuda_inputs, cu_seqlens = _to_cuda_inputs(inputs)
    scale = args.scale
    if scale is None:
        scale = cuda_inputs["k"].shape[-1] ** -0.5
    initial_state = cuda_inputs["initial_state"].clone()

    def run_op() -> tuple[torch.Tensor, torch.Tensor]:
        state = initial_state.clone()
        return fused_recurrent_gated_delta_rule(
            q=cuda_inputs["q"],
            k=cuda_inputs["k"],
            v=cuda_inputs["v"],
            g=cuda_inputs["g"],
            beta=cuda_inputs["beta"],
            scale=scale,
            initial_state=state,
            inplace_final_state=False,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=args.l2norm_inputs,
        )

    out_result = None
    final_state = None
    timings_s: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            out_result, final_state = run_op()
            torch.cuda.synchronize()
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out_result, final_state = run_op()
            torch.cuda.synchronize()
            timings_s.append(time.perf_counter() - start)

    assert out_result is not None
    assert final_state is not None
    elapsed_s = sum(timings_s)
    outputs: dict[str, torch.Tensor | None] = {
        "out": out_result.detach().cpu(),
        "final_state": final_state.detach().cpu(),
    }
    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "fused_recurrent_gated_delta_rule",
            "batch": args.batch,
            "seqlen": args.seqlen,
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "cu_seqlens": args.cu_seqlens,
            "scale": scale,
            "l2norm_inputs": args.l2norm_inputs,
            "inplace_final_state": False,
            "random_initial_state": args.random_initial_state,
        },
        "elapsed_s": elapsed_s,
        "timings_s": timings_s,
        "timing_summary": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "avg_s": elapsed_s / len(timings_s) if timings_s else None,
            "min_s": min(timings_s) if timings_s else None,
            "max_s": max(timings_s) if timings_s else None,
        },
        "inputs": saved_inputs,
        "input_summaries": {
            name: _tensor_summary(tensor) for name, tensor in saved_inputs.items()
        },
        "outputs": outputs,
        "output_summaries": {
            name: _tensor_summary(tensor) if tensor is not None else None
            for name, tensor in outputs.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(json.dumps({
        "out": str(out_path),
        "elapsed_s": elapsed_s,
        "timing_summary": payload["timing_summary"],
        "output_summaries": payload["output_summaries"],
    }, indent=2))
    return 0


def run_sigmoid_gating_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.batch != 1:
        raise ValueError(
            "`run-sigmoid-gating` uses the non-continuous single-sequence "
            "path; use --batch 1."
        )

    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    inputs = _load_or_create_sigmoid_gating_inputs(args)
    cuda_inputs, cu_seqlens = _to_cuda_inputs(inputs)
    scale = args.scale
    if scale is None:
        scale = cuda_inputs["k"].shape[-1] ** -0.5
    initial_state = cuda_inputs["initial_state"].clone()
    ssm_state_indices = cuda_inputs.get("ssm_state_indices")
    inplace_final_state = ssm_state_indices is not None

    def run_op() -> tuple[torch.Tensor, torch.Tensor]:
        state = initial_state.clone()
        return fused_sigmoid_gating_delta_rule_update(
            A_log=cuda_inputs["A_log"],
            a=cuda_inputs["a"],
            b=cuda_inputs["b"],
            dt_bias=cuda_inputs["dt_bias"],
            q=cuda_inputs["q"],
            k=cuda_inputs["k"],
            v=cuda_inputs["v"],
            beta=1.0,
            threshold=20.0,
            scale=scale,
            initial_state=state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=args.l2norm_inputs,
        )

    out_result = None
    final_state = None
    timings_s: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            out_result, final_state = run_op()
            torch.cuda.synchronize()
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out_result, final_state = run_op()
            torch.cuda.synchronize()
            timings_s.append(time.perf_counter() - start)

    assert out_result is not None
    assert final_state is not None
    elapsed_s = sum(timings_s)
    outputs: dict[str, torch.Tensor | None] = {
        "out": out_result.detach().cpu(),
        "final_state": final_state.detach().cpu(),
    }
    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "fused_sigmoid_gating_delta_rule_update",
            "batch": args.batch,
            "seqlen": args.seqlen,
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "cu_seqlens": args.cu_seqlens,
            "scale": scale,
            "l2norm_inputs": args.l2norm_inputs,
            "inplace_final_state": inplace_final_state,
            "decode_segments": args.decode_segments,
            "random_initial_state": args.random_initial_state,
        },
        "elapsed_s": elapsed_s,
        "timings_s": timings_s,
        "timing_summary": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "avg_s": elapsed_s / len(timings_s) if timings_s else None,
            "min_s": min(timings_s) if timings_s else None,
            "max_s": max(timings_s) if timings_s else None,
        },
        "inputs": saved_inputs,
        "input_summaries": {
            name: _tensor_summary(tensor) for name, tensor in saved_inputs.items()
        },
        "outputs": outputs,
        "output_summaries": {
            name: _tensor_summary(tensor) if tensor is not None else None
            for name, tensor in outputs.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(json.dumps({
        "out": str(out_path),
        "elapsed_s": elapsed_s,
        "timing_summary": payload["timing_summary"],
        "output_summaries": payload["output_summaries"],
    }, indent=2))
    return 0


def run_sigmoid_gating_mixed_qkv_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.batch != 1:
        raise ValueError(
            "`run-sigmoid-gating-mixed-qkv` uses a packed single-sequence "
            "mixed_qkv input; use --batch 1."
        )

    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update_mixed_qkv,
    )

    inputs = _load_or_create_sigmoid_gating_inputs(args)
    cuda_inputs, cu_seqlens = _to_cuda_inputs(inputs)
    scale = args.scale
    if scale is None:
        scale = cuda_inputs["k"].shape[-1] ** -0.5
    initial_state = cuda_inputs["initial_state"].clone()
    ssm_state_indices = cuda_inputs.get("ssm_state_indices")
    inplace_final_state = ssm_state_indices is not None
    seq_len = cuda_inputs["q"].shape[1]
    mixed_qkv = torch.cat(
        [
            cuda_inputs["q"].reshape(seq_len, -1),
            cuda_inputs["k"].reshape(seq_len, -1),
            cuda_inputs["v"].reshape(seq_len, -1),
        ],
        dim=-1,
    ).contiguous()

    def run_op() -> tuple[torch.Tensor, torch.Tensor]:
        state = initial_state.clone()
        return fused_sigmoid_gating_delta_rule_update_mixed_qkv(
            A_log=cuda_inputs["A_log"],
            a=cuda_inputs["a"],
            b=cuda_inputs["b"],
            dt_bias=cuda_inputs["dt_bias"],
            mixed_qkv=mixed_qkv,
            num_q_heads=args.k_heads,
            num_v_heads=args.v_heads,
            head_k_dim=args.head_k_dim,
            head_v_dim=args.head_v_dim,
            beta=1.0,
            threshold=20.0,
            scale=scale,
            initial_state=state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=args.l2norm_inputs,
        )

    out_result = None
    final_state = None
    timings_s: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            out_result, final_state = run_op()
            torch.cuda.synchronize()
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out_result, final_state = run_op()
            torch.cuda.synchronize()
            timings_s.append(time.perf_counter() - start)

    assert out_result is not None
    assert final_state is not None
    elapsed_s = sum(timings_s)
    outputs: dict[str, torch.Tensor | None] = {
        "out": out_result.detach().cpu(),
        "final_state": final_state.detach().cpu(),
    }
    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "fused_sigmoid_gating_delta_rule_update_mixed_qkv",
            "batch": args.batch,
            "seqlen": seq_len,
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "cu_seqlens": args.cu_seqlens,
            "scale": scale,
            "l2norm_inputs": args.l2norm_inputs,
            "inplace_final_state": inplace_final_state,
            "decode_segments": args.decode_segments,
            "random_initial_state": args.random_initial_state,
        },
        "elapsed_s": elapsed_s,
        "timings_s": timings_s,
        "timing_summary": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "avg_s": elapsed_s / len(timings_s) if timings_s else None,
            "min_s": min(timings_s) if timings_s else None,
            "max_s": max(timings_s) if timings_s else None,
        },
        "inputs": saved_inputs,
        "input_summaries": {
            name: _tensor_summary(tensor) for name, tensor in saved_inputs.items()
        },
        "outputs": outputs,
        "output_summaries": {
            name: _tensor_summary(tensor) if tensor is not None else None
            for name, tensor in outputs.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(json.dumps({
        "out": str(out_path),
        "elapsed_s": elapsed_s,
        "timing_summary": payload["timing_summary"],
        "output_summaries": payload["output_summaries"],
    }, indent=2))
    return 0


def run_model_mixed_qkv_route_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.batch != 1:
        raise ValueError("model mixed-qkv route smoke requires --batch 1")

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen_gdn
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    inputs = _load_or_create_sigmoid_gating_inputs(args)
    cuda_inputs, _ = _to_cuda_inputs(inputs)
    seq_len = cuda_inputs["q"].shape[1]
    q_flat = cuda_inputs["q"].reshape(seq_len, -1)
    k_flat = cuda_inputs["k"].reshape(seq_len, -1)
    v_flat = cuda_inputs["v"].reshape(seq_len, -1)
    mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1).contiguous()
    state_shape = (
        seq_len + 1,
        args.v_heads,
        args.head_v_dim,
        args.head_k_dim,
    )
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=seq_len,
        num_decode_tokens=seq_len,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=seq_len,
        non_spec_query_start_loc=torch.arange(
            0, seq_len + 1, device=mixed_qkv.device, dtype=torch.int32
        ),
        non_spec_state_indices_tensor=torch.arange(
            1, seq_len + 1, device=mixed_qkv.device, dtype=torch.int32
        ),
    )

    original_get_forward_context = qwen_gdn.get_forward_context
    original_causal_conv1d_update = qwen_gdn.causal_conv1d_update
    original_mixed = qwen_gdn.fused_sigmoid_gating_delta_rule_update_mixed_qkv
    original_ref = qwen_gdn.fused_sigmoid_gating_delta_rule_update

    def fake_get_forward_context() -> SimpleNamespace:
        return SimpleNamespace(attn_metadata={"sm70_gdn_route": metadata})

    def identity_causal_conv1d_update(x: torch.Tensor, *unused: Any, **kwargs: Any):
        del unused, kwargs
        return x.contiguous()

    def make_layer(
        *,
        enable_packed_recurrent_decode: bool,
        enable_mixed_qkv: bool,
        compare_mixed_qkv: bool,
        counters: dict[str, int],
    ) -> QwenGatedDeltaNetAttention:
        layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
        layer.prefix = "sm70_gdn_route"
        layer.num_k_heads = args.k_heads
        layer.num_v_heads = args.v_heads
        layer.tp_size = 1
        layer.head_k_dim = args.head_k_dim
        layer.head_v_dim = args.head_v_dim
        layer.key_dim = args.k_heads * args.head_k_dim
        layer.value_dim = args.v_heads * args.head_v_dim
        layer.A_log = cuda_inputs["A_log"]
        layer.dt_bias = cuda_inputs["dt_bias"]
        layer.activation = "silu"
        layer.conv1d = SimpleNamespace(
            weight=torch.empty(
                (mixed_qkv.shape[1], 1, 1),
                device=mixed_qkv.device,
                dtype=mixed_qkv.dtype,
            ),
            bias=None,
        )
        layer.kv_cache = (
            torch.empty(
                (seq_len + 1, mixed_qkv.shape[1], 1),
                device=mixed_qkv.device,
                dtype=mixed_qkv.dtype,
            ),
            torch.zeros(state_shape, device=mixed_qkv.device, dtype=mixed_qkv.dtype),
        )
        layer.enable_packed_recurrent_decode = enable_packed_recurrent_decode
        layer.enable_sm70_fused_sigmoid_mixed_qkv = enable_mixed_qkv
        layer.compare_sm70_fused_sigmoid_mixed_qkv = compare_mixed_qkv

        def counted_mixed(*call_args: Any, **call_kwargs: Any):
            counters["mixed"] += 1
            return original_mixed(*call_args, **call_kwargs)

        def counted_ref(*call_args: Any, **call_kwargs: Any):
            counters["ref"] += 1
            return original_ref(*call_args, **call_kwargs)

        qwen_gdn.fused_sigmoid_gating_delta_rule_update_mixed_qkv = counted_mixed
        qwen_gdn.fused_sigmoid_gating_delta_rule_update = counted_ref
        return layer

    def run_layer(
        *,
        enable_packed_recurrent_decode: bool,
        enable_mixed_qkv: bool,
        compare_mixed_qkv: bool,
    ) -> dict[str, Any]:
        counters = {"mixed": 0, "ref": 0}
        layer = make_layer(
            enable_packed_recurrent_decode=enable_packed_recurrent_decode,
            enable_mixed_qkv=enable_mixed_qkv,
            compare_mixed_qkv=compare_mixed_qkv,
            counters=counters,
        )
        core_attn_out = torch.empty(
            (seq_len, args.v_heads, args.head_v_dim),
            device=mixed_qkv.device,
            dtype=mixed_qkv.dtype,
        )
        QwenGatedDeltaNetAttention._forward_core(
            layer,
            mixed_qkv=mixed_qkv,
            b=cuda_inputs["b"].reshape(seq_len, -1),
            a=cuda_inputs["a"].reshape(seq_len, -1),
            core_attn_out=core_attn_out,
        )
        return {
            "out": core_attn_out.detach().cpu(),
            "state": layer.kv_cache[1].detach().cpu(),
            "counters": counters,
        }

    try:
        qwen_gdn.get_forward_context = fake_get_forward_context
        qwen_gdn.causal_conv1d_update = identity_causal_conv1d_update
        with torch.inference_mode():
            generic = run_layer(
                enable_packed_recurrent_decode=False,
                enable_mixed_qkv=False,
                compare_mixed_qkv=False,
            )
            mixed = run_layer(
                enable_packed_recurrent_decode=True,
                enable_mixed_qkv=True,
                compare_mixed_qkv=False,
            )
            mixed_compare = run_layer(
                enable_packed_recurrent_decode=True,
                enable_mixed_qkv=True,
                compare_mixed_qkv=True,
            )
            torch.cuda.synchronize()
    finally:
        qwen_gdn.get_forward_context = original_get_forward_context
        qwen_gdn.causal_conv1d_update = original_causal_conv1d_update
        qwen_gdn.fused_sigmoid_gating_delta_rule_update_mixed_qkv = original_mixed
        qwen_gdn.fused_sigmoid_gating_delta_rule_update = original_ref

    mixed_vs_generic = {
        "out": _compare_tensor(generic["out"], mixed["out"]),
        "state": _compare_tensor(generic["state"], mixed["state"]),
    }
    compare_vs_generic = {
        "out": _compare_tensor(generic["out"], mixed_compare["out"]),
        "state": _compare_tensor(generic["state"], mixed_compare["state"]),
    }
    route_ok = (
        generic["counters"] == {"mixed": 0, "ref": 1}
        and mixed["counters"] == {"mixed": 1, "ref": 0}
        and mixed_compare["counters"] == {"mixed": 1, "ref": 1}
    )
    strict_ok = route_ok and all(
        item["torch_equal"] and item["max_diff"] == 0.0
        for group in (mixed_vs_generic, compare_vs_generic)
        for item in group.values()
    )
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "qwen_gdn_linear_attn_mixed_qkv_route_smoke",
            "batch": args.batch,
            "seqlen": seq_len,
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "l2norm_inputs": args.l2norm_inputs,
            "conv_update": "identity_mock",
        },
        "strict_ok": strict_ok,
        "route_ok": route_ok,
        "counters": {
            "generic": generic["counters"],
            "mixed": mixed["counters"],
            "mixed_compare": mixed_compare["counters"],
        },
        "comparisons": {
            "mixed_vs_generic": mixed_vs_generic,
            "mixed_compare_vs_generic": compare_vs_generic,
        },
        "input_summaries": {
            name: _tensor_summary(tensor) for name, tensor in inputs.items()
        },
        "output_summaries": {
            "generic_out": _tensor_summary(generic["out"]),
            "mixed_out": _tensor_summary(mixed["out"]),
            "mixed_compare_out": _tensor_summary(mixed_compare["out"]),
            "generic_state": _tensor_summary(generic["state"]),
            "mixed_state": _tensor_summary(mixed["state"]),
            "mixed_compare_state": _tensor_summary(mixed_compare["state"]),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if strict_ok else 1


def run_packed_recurrent_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    inputs = _load_or_create_packed_recurrent_inputs(args)
    cuda_inputs, _ = _to_cuda_inputs(inputs)
    initial_state = cuda_inputs["initial_state"].clone()
    out = torch.empty(
        (
            cuda_inputs["mixed_qkv"].shape[0],
            1,
            args.v_heads,
            args.head_v_dim,
        ),
        device=cuda_inputs["mixed_qkv"].device,
        dtype=cuda_inputs["mixed_qkv"].dtype,
    )
    scale = args.scale
    if scale is None:
        scale = args.head_k_dim**-0.5

    def run_op() -> tuple[torch.Tensor, torch.Tensor]:
        state = initial_state.clone()
        out_buf = torch.empty_like(out)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=cuda_inputs["mixed_qkv"],
            a=cuda_inputs["a"],
            b=cuda_inputs["b"],
            A_log=cuda_inputs["A_log"],
            dt_bias=cuda_inputs["dt_bias"],
            scale=scale,
            initial_state=state,
            out=out_buf,
            ssm_state_indices=cuda_inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=args.l2norm_inputs,
        )
        return out_buf, state

    out_result = None
    final_state = None
    timings_s: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            out_result, final_state = run_op()
            torch.cuda.synchronize()
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out_result, final_state = run_op()
            torch.cuda.synchronize()
            timings_s.append(time.perf_counter() - start)

    assert out_result is not None
    assert final_state is not None
    elapsed_s = sum(timings_s)
    outputs: dict[str, torch.Tensor | None] = {
        "out": out_result.detach().cpu(),
        "final_state": final_state.detach().cpu(),
    }
    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "fused_recurrent_gated_delta_rule_packed_decode",
            "batch": args.batch,
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "scale": scale,
            "l2norm_inputs": args.l2norm_inputs,
            "random_initial_state": args.random_initial_state,
        },
        "elapsed_s": elapsed_s,
        "timings_s": timings_s,
        "timing_summary": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "avg_s": elapsed_s / len(timings_s) if timings_s else None,
            "min_s": min(timings_s) if timings_s else None,
            "max_s": max(timings_s) if timings_s else None,
        },
        "inputs": saved_inputs,
        "input_summaries": {
            name: _tensor_summary(tensor) for name, tensor in saved_inputs.items()
        },
        "outputs": outputs,
        "output_summaries": {
            name: _tensor_summary(tensor) if tensor is not None else None
            for name, tensor in outputs.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(json.dumps({
        "out": str(out_path),
        "elapsed_s": elapsed_s,
        "timing_summary": payload["timing_summary"],
        "output_summaries": payload["output_summaries"],
    }, indent=2))
    return 0


def _split_packed_mixed_qkv(
    mixed_qkv: torch.Tensor,
    *,
    k_heads: int,
    v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_dim = k_heads * head_k_dim
    k_dim = k_heads * head_k_dim
    v_dim = v_heads * head_v_dim
    q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    seq_len = mixed_qkv.shape[0]
    return (
        q.contiguous().view(1, seq_len, k_heads, head_k_dim),
        k.contiguous().view(1, seq_len, k_heads, head_k_dim),
        v.contiguous().view(1, seq_len, v_heads, head_v_dim),
    )


def run_flashqla_decode_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.dtype != "float16":
        raise ValueError("SM70 FlashQLA production decode benchmark requires fp16")
    if args.head_k_dim != 128 or args.head_v_dim != 128:
        raise ValueError("SM70 FlashQLA decode currently requires K=V=128")

    from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import (
        gdn_decode_mixed_qkv_global_state_sm70,
    )

    inputs = _load_or_create_packed_recurrent_inputs(args)
    cuda_inputs, _ = _to_cuda_inputs(inputs)
    initial_state = cuda_inputs["initial_state"]
    state_indices = cuda_inputs["ssm_state_indices"].contiguous()
    mixed_qkv = cuda_inputs["mixed_qkv"].contiguous()
    tokens = mixed_qkv.shape[0]
    out = torch.empty(
        (tokens, args.v_heads, args.head_v_dim),
        device=mixed_qkv.device,
        dtype=mixed_qkv.dtype,
    )

    def run_op() -> tuple[torch.Tensor, torch.Tensor]:
        state = initial_state.clone()
        out_buf = torch.empty_like(out)
        gdn_decode_mixed_qkv_global_state_sm70(
            mixed_qkv=mixed_qkv,
            a=cuda_inputs["a"],
            b=cuda_inputs["b"],
            A_log=cuda_inputs["A_log"],
            dt_bias=cuda_inputs["dt_bias"],
            state=state,
            state_indices=state_indices,
            output=out_buf,
            scale=args.head_k_dim**-0.5,
            use_qk_l2norm_in_kernel=args.l2norm_inputs,
        )
        return out_buf.unsqueeze(1), state

    out_result = None
    final_state = None
    timings_s: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            out_result, final_state = run_op()
            torch.cuda.synchronize()
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out_result, final_state = run_op()
            torch.cuda.synchronize()
            timings_s.append(time.perf_counter() - start)

    assert out_result is not None
    assert final_state is not None
    elapsed_s = sum(timings_s)
    outputs: dict[str, torch.Tensor | None] = {
        "out": out_result.detach().cpu(),
        "final_state": final_state.detach().cpu(),
    }
    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "flashqla_sm70_gdn_decode_full_route",
            "route_components": (
                "fused_mixed_qkv+gating+l2norm+global_state_update"
            ),
            "batch": args.batch,
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "l2norm_inputs": args.l2norm_inputs,
            "random_initial_state": args.random_initial_state,
        },
        "elapsed_s": elapsed_s,
        "timings_s": timings_s,
        "timing_summary": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "avg_s": elapsed_s / len(timings_s) if timings_s else None,
            "min_s": min(timings_s) if timings_s else None,
            "max_s": max(timings_s) if timings_s else None,
        },
        "inputs": saved_inputs,
        "input_summaries": {
            name: _tensor_summary(tensor) for name, tensor in saved_inputs.items()
        },
        "outputs": outputs,
        "output_summaries": {
            name: _tensor_summary(tensor) if tensor is not None else None
            for name, tensor in outputs.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(json.dumps({
        "out": str(out_path),
        "elapsed_s": elapsed_s,
        "timing_summary": payload["timing_summary"],
        "output_summaries": payload["output_summaries"],
    }, indent=2))
    return 0


def run_packed_recurrent_cudagraph_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    inputs = _load_or_create_packed_recurrent_sequence_inputs(args)
    cuda_inputs, _ = _to_cuda_inputs(inputs)
    mixed_qkv_seq = cuda_inputs["mixed_qkv_seq"]
    a_seq = cuda_inputs["a_seq"]
    b_seq = cuda_inputs["b_seq"]
    initial_state = cuda_inputs["initial_state"]
    state_indices = cuda_inputs["ssm_state_indices"]
    scale = args.scale
    if scale is None:
        scale = args.head_k_dim**-0.5

    def launch_one(
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state: torch.Tensor,
        out: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=cuda_inputs["A_log"],
            dt_bias=cuda_inputs["dt_bias"],
            scale=scale,
            initial_state=state,
            out=out,
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=args.l2norm_inputs,
        )

    eager_state = initial_state.clone()
    eager_out = torch.empty(
        (
            args.seqlen,
            args.batch,
            1,
            args.v_heads,
            args.head_v_dim,
        ),
        device=mixed_qkv_seq.device,
        dtype=mixed_qkv_seq.dtype,
    )
    graph_state = initial_state.clone()
    graph_out = torch.empty_like(eager_out)
    step_out = torch.empty_like(eager_out[0])
    graph_step_out = torch.empty_like(eager_out[0])
    mixed_buf = mixed_qkv_seq[0].clone()
    a_buf = a_seq[0].clone()
    b_buf = b_seq[0].clone()
    graph_state_indices = (
        torch.zeros_like(state_indices)
        if args.capture_null_indices
        else state_indices.clone()
    )

    with torch.inference_mode():
        warm_state = initial_state.clone()
        warm_out = torch.empty_like(step_out)
        launch_one(
            mixed_qkv_seq[0],
            a_seq[0],
            b_seq[0],
            warm_state,
            warm_out,
            state_indices,
        )
        torch.cuda.synchronize()

        for step in range(args.seqlen):
            launch_one(
                mixed_qkv_seq[step],
                a_seq[step],
                b_seq[step],
                eager_state,
                step_out,
                state_indices,
            )
            eager_out[step].copy_(step_out)
        torch.cuda.synchronize()

        capture_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture_graph):
            launch_one(
                mixed_buf,
                a_buf,
                b_buf,
                graph_state,
                graph_step_out,
                graph_state_indices,
            )
        torch.cuda.synchronize()

        graph_state.copy_(initial_state)
        graph_state_indices.copy_(state_indices)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(args.seqlen):
            mixed_buf.copy_(mixed_qkv_seq[step])
            a_buf.copy_(a_seq[step])
            b_buf.copy_(b_seq[step])
            capture_graph.replay()
            graph_out[step].copy_(graph_step_out)
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

    outputs: dict[str, torch.Tensor | None] = {
        "eager_out": eager_out.detach().cpu(),
        "graph_out": graph_out.detach().cpu(),
        "eager_state": eager_state.detach().cpu(),
        "graph_state": graph_state.detach().cpu(),
    }
    comparisons = {
        "out": _compare_tensor(outputs["eager_out"], outputs["graph_out"]),
        "final_state": _compare_tensor(
            outputs["eager_state"], outputs["graph_state"]
        ),
    }
    strict_ok = all(
        item["torch_equal"] and item["max_diff"] == 0.0
        for item in comparisons.values()
    )
    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "fused_recurrent_gated_delta_rule_packed_decode_cudagraph",
            "batch": args.batch,
            "seqlen": args.seqlen,
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "scale": scale,
            "l2norm_inputs": args.l2norm_inputs,
            "random_initial_state": args.random_initial_state,
            "capture_null_indices": args.capture_null_indices,
        },
        "strict_ok": strict_ok,
        "elapsed_s": elapsed_s,
        "inputs": saved_inputs,
        "input_summaries": {
            name: _tensor_summary(tensor) for name, tensor in saved_inputs.items()
        },
        "outputs": outputs,
        "output_summaries": {
            name: _tensor_summary(tensor) if tensor is not None else None
            for name, tensor in outputs.items()
        },
        "comparisons": comparisons,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(json.dumps({
        "out": str(out_path),
        "strict_ok": strict_ok,
        "elapsed_s": elapsed_s,
        "comparisons": comparisons,
    }, indent=2))
    return 0 if strict_ok else 1


def run_conv_update_cudagraph_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    inputs = _load_or_create_conv_update_sequence_inputs(args)
    cuda_inputs, _ = _to_cuda_inputs(inputs)
    x_seq = cuda_inputs["x_seq"]
    eager_x_seq = x_seq.clone()
    graph_x_seq = x_seq.clone()
    initial_state = cuda_inputs["conv_state"]
    state_indices = cuda_inputs["conv_state_indices"]
    activation = "silu" if args.silu else None
    graph_state_indices = (
        torch.zeros_like(state_indices)
        if args.capture_null_indices
        else state_indices.clone()
    )

    def launch_one(
        x: torch.Tensor,
        state: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return causal_conv1d_update(
            x,
            state,
            cuda_inputs["weight"],
            cuda_inputs["bias"],
            activation,
            conv_state_indices=indices,
            validate_data=False,
        )

    eager_state = initial_state.clone()
    eager_out = torch.empty_like(x_seq)
    graph_state = initial_state.clone()
    graph_out = torch.empty_like(x_seq)
    x_buf = graph_x_seq[0].clone()

    with torch.inference_mode():
        warm_state = initial_state.clone()
        _ = launch_one(x_seq[0].clone(), warm_state, state_indices)
        torch.cuda.synchronize()

        for step in range(args.seqlen):
            eager_out[step].copy_(
                launch_one(eager_x_seq[step], eager_state, state_indices)
            )
        torch.cuda.synchronize()

        capture_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture_graph):
            graph_step_out = launch_one(x_buf, graph_state, graph_state_indices)
        torch.cuda.synchronize()

        graph_state.copy_(initial_state)
        graph_state_indices.copy_(state_indices)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(args.seqlen):
            x_buf.copy_(graph_x_seq[step])
            capture_graph.replay()
            graph_out[step].copy_(graph_step_out)
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

    outputs: dict[str, torch.Tensor | None] = {
        "eager_out": eager_out.detach().cpu(),
        "graph_out": graph_out.detach().cpu(),
        "eager_state": eager_state.detach().cpu(),
        "graph_state": graph_state.detach().cpu(),
    }
    comparisons = {
        "out": _compare_tensor(outputs["eager_out"], outputs["graph_out"]),
        "final_state": _compare_tensor(
            outputs["eager_state"], outputs["graph_state"]
        ),
    }
    strict_ok = all(
        item["torch_equal"] and item["max_diff"] == 0.0
        for item in comparisons.values()
    )
    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "causal_conv1d_update_cudagraph",
            "batch": args.batch,
            "seqlen": args.seqlen,
            "conv_dim": x_seq.shape[-1],
            "conv_width": args.conv_width,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "silu": args.silu,
            "random_initial_state": args.random_initial_state,
            "capture_null_indices": args.capture_null_indices,
        },
        "strict_ok": strict_ok,
        "elapsed_s": elapsed_s,
        "inputs": saved_inputs,
        "input_summaries": {
            name: _tensor_summary(tensor) for name, tensor in saved_inputs.items()
        },
        "outputs": outputs,
        "output_summaries": {
            name: _tensor_summary(tensor) if tensor is not None else None
            for name, tensor in outputs.items()
        },
        "comparisons": comparisons,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(json.dumps({
        "out": str(out_path),
        "strict_ok": strict_ok,
        "elapsed_s": elapsed_s,
        "comparisons": comparisons,
    }, indent=2))
    return 0 if strict_ok else 1


def _compare_tensor(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> dict[str, Any]:
    if left is None or right is None:
        return {
            "both_present": left is not None and right is not None,
            "torch_equal": left is right,
            "max_diff": None,
            "mean_diff": None,
            "num_different": None,
            "first_mismatch_flat_index": None,
            "left": None if left is None else _tensor_summary(left),
            "right": None if right is None else _tensor_summary(right),
        }
    shape_equal = tuple(left.shape) == tuple(right.shape)
    dtype_equal = left.dtype == right.dtype
    if not shape_equal:
        return {
            "shape_equal": False,
            "dtype_equal": dtype_equal,
            "torch_equal": False,
            "max_diff": None,
            "mean_diff": None,
            "num_different": None,
            "first_mismatch_flat_index": None,
            "left": _tensor_summary(left),
            "right": _tensor_summary(right),
        }

    torch_equal = bool(torch.equal(left, right))
    diff = (left.float() - right.float()).abs()
    mismatch = torch.ne(left, right)
    num_different = int(mismatch.sum().item())
    first_mismatch = None
    if num_different:
        first_mismatch = int(torch.nonzero(mismatch.flatten(), as_tuple=False)[0])
    return {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "torch_equal": torch_equal,
        "max_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "num_different": num_different,
        "first_mismatch_flat_index": first_mismatch,
        "left": _tensor_summary(left),
        "right": _tensor_summary(right),
    }


def compare_cases(args: argparse.Namespace) -> int:
    left = _torch_load(Path(args.left))
    right = _torch_load(Path(args.right))

    input_matches = {}
    for name, left_tensor in left.get("inputs", {}).items():
        right_tensor = right.get("inputs", {}).get(name)
        input_matches[name] = _compare_tensor(left_tensor, right_tensor)

    output_matches = {}
    for name, left_tensor in left["outputs"].items():
        right_tensor = right["outputs"].get(name)
        output_matches[name] = _compare_tensor(left_tensor, right_tensor)

    strict_ok = all(
        item.get("torch_equal") and item.get("max_diff") in (0.0, None)
        for item in output_matches.values()
    )
    inputs_ok = all(item.get("torch_equal") for item in input_matches.values())
    report = {
        "strict_ok": bool(strict_ok and inputs_ok),
        "inputs_ok": bool(inputs_ok),
        "left": {
            "path": args.left,
            "metadata": left.get("metadata"),
            "elapsed_s": left.get("elapsed_s"),
            "timing_summary": left.get("timing_summary"),
        },
        "right": {
            "path": args.right,
            "metadata": right.get("metadata"),
            "elapsed_s": right.get("elapsed_s"),
            "timing_summary": right.get("timing_summary"),
        },
        "input_matches": input_matches,
        "output_matches": output_matches,
    }

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["strict_ok"] or args.no_strict else 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--input-file")
    run.add_argument("--case-name", default="gdn_exactness")
    run.add_argument("--batch", type=int, default=1)
    run.add_argument("--seqlen", type=int, default=1024)
    run.add_argument("--k-heads", type=int, default=4)
    run.add_argument("--v-heads", type=int, default=8)
    run.add_argument("--head-k-dim", type=int, default=128)
    run.add_argument("--head-v-dim", type=int, default=128)
    run.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    run.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--input-scale", type=float, default=0.02)
    run.add_argument("--gate-scale", type=float, default=0.01)
    run.add_argument("--beta-scale", type=float, default=0.25)
    run.add_argument("--scale", type=float)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--cu-seqlens", action="store_true")
    run.add_argument("--no-l2norm-inputs", dest="l2norm_inputs", action="store_false")
    run.add_argument("--random-initial-state", action="store_true")
    run.add_argument("--save-intermediates", action="store_true")
    run.set_defaults(func=run_case, l2norm_inputs=True)

    recurrent = subparsers.add_parser("run-recurrent")
    recurrent.add_argument("--out", required=True)
    recurrent.add_argument("--input-file")
    recurrent.add_argument("--case-name", default="gdn_recurrent_exactness")
    recurrent.add_argument("--batch", type=int, default=1)
    recurrent.add_argument("--seqlen", type=int, default=16)
    recurrent.add_argument("--k-heads", type=int, default=4)
    recurrent.add_argument("--v-heads", type=int, default=8)
    recurrent.add_argument("--head-k-dim", type=int, default=128)
    recurrent.add_argument("--head-v-dim", type=int, default=128)
    recurrent.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    recurrent.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    recurrent.add_argument("--seed", type=int, default=1234)
    recurrent.add_argument("--input-scale", type=float, default=0.02)
    recurrent.add_argument("--gate-scale", type=float, default=0.01)
    recurrent.add_argument("--beta-scale", type=float, default=0.25)
    recurrent.add_argument("--scale", type=float)
    recurrent.add_argument("--warmup", type=int, default=0)
    recurrent.add_argument("--repeat", type=int, default=1)
    recurrent.add_argument("--cu-seqlens", action="store_true")
    recurrent.add_argument(
        "--no-l2norm-inputs", dest="l2norm_inputs", action="store_false"
    )
    recurrent.add_argument("--random-initial-state", action="store_true")
    recurrent.set_defaults(func=run_recurrent_case, l2norm_inputs=True)

    sigmoid = subparsers.add_parser("run-sigmoid-gating")
    sigmoid.add_argument("--out", required=True)
    sigmoid.add_argument("--input-file")
    sigmoid.add_argument("--case-name", default="gdn_sigmoid_gating_exactness")
    sigmoid.add_argument("--batch", type=int, default=1)
    sigmoid.add_argument("--seqlen", type=int, default=16)
    sigmoid.add_argument("--k-heads", type=int, default=4)
    sigmoid.add_argument("--v-heads", type=int, default=8)
    sigmoid.add_argument("--head-k-dim", type=int, default=128)
    sigmoid.add_argument("--head-v-dim", type=int, default=128)
    sigmoid.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    sigmoid.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    sigmoid.add_argument("--seed", type=int, default=1234)
    sigmoid.add_argument("--input-scale", type=float, default=0.02)
    sigmoid.add_argument("--gate-scale", type=float, default=0.01)
    sigmoid.add_argument("--beta-scale", type=float, default=0.25)
    sigmoid.add_argument("--scale", type=float)
    sigmoid.add_argument("--warmup", type=int, default=0)
    sigmoid.add_argument("--repeat", type=int, default=1)
    sigmoid.add_argument("--cu-seqlens", action="store_true")
    sigmoid.add_argument(
        "--no-l2norm-inputs", dest="l2norm_inputs", action="store_false"
    )
    sigmoid.add_argument("--random-initial-state", action="store_true")
    sigmoid.add_argument("--decode-segments", action="store_true")
    sigmoid.set_defaults(func=run_sigmoid_gating_case, l2norm_inputs=True)

    mixed_sigmoid = subparsers.add_parser("run-sigmoid-gating-mixed-qkv")
    mixed_sigmoid.add_argument("--out", required=True)
    mixed_sigmoid.add_argument("--input-file")
    mixed_sigmoid.add_argument(
        "--case-name", default="gdn_sigmoid_gating_mixed_qkv_exactness"
    )
    mixed_sigmoid.add_argument("--batch", type=int, default=1)
    mixed_sigmoid.add_argument("--seqlen", type=int, default=16)
    mixed_sigmoid.add_argument("--k-heads", type=int, default=4)
    mixed_sigmoid.add_argument("--v-heads", type=int, default=8)
    mixed_sigmoid.add_argument("--head-k-dim", type=int, default=128)
    mixed_sigmoid.add_argument("--head-v-dim", type=int, default=128)
    mixed_sigmoid.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    mixed_sigmoid.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    mixed_sigmoid.add_argument("--seed", type=int, default=1234)
    mixed_sigmoid.add_argument("--input-scale", type=float, default=0.02)
    mixed_sigmoid.add_argument("--gate-scale", type=float, default=0.01)
    mixed_sigmoid.add_argument("--beta-scale", type=float, default=0.25)
    mixed_sigmoid.add_argument("--scale", type=float)
    mixed_sigmoid.add_argument("--warmup", type=int, default=0)
    mixed_sigmoid.add_argument("--repeat", type=int, default=1)
    mixed_sigmoid.add_argument("--cu-seqlens", action="store_true")
    mixed_sigmoid.add_argument(
        "--no-l2norm-inputs", dest="l2norm_inputs", action="store_false"
    )
    mixed_sigmoid.add_argument("--random-initial-state", action="store_true")
    mixed_sigmoid.add_argument("--decode-segments", action="store_true")
    mixed_sigmoid.set_defaults(
        func=run_sigmoid_gating_mixed_qkv_case, l2norm_inputs=True
    )

    model_mixed = subparsers.add_parser("run-model-mixed-qkv-route")
    model_mixed.add_argument("--out", required=True)
    model_mixed.add_argument("--input-file")
    model_mixed.add_argument(
        "--case-name", default="gdn_model_mixed_qkv_route_smoke"
    )
    model_mixed.add_argument("--batch", type=int, default=1)
    model_mixed.add_argument("--seqlen", type=int, default=16)
    model_mixed.add_argument("--k-heads", type=int, default=4)
    model_mixed.add_argument("--v-heads", type=int, default=8)
    model_mixed.add_argument("--head-k-dim", type=int, default=128)
    model_mixed.add_argument("--head-v-dim", type=int, default=128)
    model_mixed.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    model_mixed.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    model_mixed.add_argument("--seed", type=int, default=1234)
    model_mixed.add_argument("--input-scale", type=float, default=0.02)
    model_mixed.add_argument("--gate-scale", type=float, default=0.01)
    model_mixed.add_argument("--beta-scale", type=float, default=0.25)
    model_mixed.add_argument(
        "--no-l2norm-inputs", dest="l2norm_inputs", action="store_false"
    )
    model_mixed.add_argument("--random-initial-state", action="store_true")
    model_mixed.set_defaults(
        func=run_model_mixed_qkv_route_case,
        l2norm_inputs=True,
        decode_segments=True,
        cu_seqlens=False,
    )

    packed = subparsers.add_parser("run-packed-recurrent")
    packed.add_argument("--out", required=True)
    packed.add_argument("--input-file")
    packed.add_argument("--case-name", default="gdn_packed_recurrent_exactness")
    packed.add_argument("--batch", type=int, default=16)
    packed.add_argument("--k-heads", type=int, default=4)
    packed.add_argument("--v-heads", type=int, default=8)
    packed.add_argument("--head-k-dim", type=int, default=128)
    packed.add_argument("--head-v-dim", type=int, default=128)
    packed.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    packed.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    packed.add_argument("--seed", type=int, default=1234)
    packed.add_argument("--input-scale", type=float, default=0.02)
    packed.add_argument("--gate-scale", type=float, default=0.01)
    packed.add_argument("--beta-scale", type=float, default=0.25)
    packed.add_argument("--scale", type=float)
    packed.add_argument("--warmup", type=int, default=0)
    packed.add_argument("--repeat", type=int, default=1)
    packed.add_argument(
        "--no-l2norm-inputs", dest="l2norm_inputs", action="store_false"
    )
    packed.add_argument("--random-initial-state", action="store_true")
    packed.set_defaults(func=run_packed_recurrent_case, l2norm_inputs=True)

    qla_decode = subparsers.add_parser("run-flashqla-decode")
    qla_decode.add_argument("--out", required=True)
    qla_decode.add_argument("--input-file")
    qla_decode.add_argument("--case-name", default="gdn_flashqla_decode_exactness")
    qla_decode.add_argument("--batch", type=int, default=16)
    qla_decode.add_argument("--k-heads", type=int, default=4)
    qla_decode.add_argument("--v-heads", type=int, default=8)
    qla_decode.add_argument("--head-k-dim", type=int, default=128)
    qla_decode.add_argument("--head-v-dim", type=int, default=128)
    qla_decode.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    qla_decode.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    qla_decode.add_argument("--seed", type=int, default=1234)
    qla_decode.add_argument("--input-scale", type=float, default=0.02)
    qla_decode.add_argument("--gate-scale", type=float, default=0.01)
    qla_decode.add_argument("--beta-scale", type=float, default=0.25)
    qla_decode.add_argument("--scale", type=float)
    qla_decode.add_argument("--warmup", type=int, default=0)
    qla_decode.add_argument("--repeat", type=int, default=1)
    qla_decode.add_argument(
        "--no-l2norm-inputs", dest="l2norm_inputs", action="store_false"
    )
    qla_decode.add_argument("--random-initial-state", action="store_true")
    qla_decode.set_defaults(func=run_flashqla_decode_case, l2norm_inputs=True)

    packed_cg = subparsers.add_parser("run-packed-recurrent-cudagraph")
    packed_cg.add_argument("--out", required=True)
    packed_cg.add_argument("--input-file")
    packed_cg.add_argument(
        "--case-name", default="gdn_packed_recurrent_cudagraph_exactness"
    )
    packed_cg.add_argument("--batch", type=int, default=2)
    packed_cg.add_argument("--seqlen", type=int, default=1024)
    packed_cg.add_argument("--k-heads", type=int, default=4)
    packed_cg.add_argument("--v-heads", type=int, default=8)
    packed_cg.add_argument("--head-k-dim", type=int, default=128)
    packed_cg.add_argument("--head-v-dim", type=int, default=128)
    packed_cg.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    packed_cg.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    packed_cg.add_argument("--seed", type=int, default=1234)
    packed_cg.add_argument("--input-scale", type=float, default=0.02)
    packed_cg.add_argument("--gate-scale", type=float, default=0.01)
    packed_cg.add_argument("--beta-scale", type=float, default=0.25)
    packed_cg.add_argument("--scale", type=float)
    packed_cg.add_argument(
        "--no-l2norm-inputs", dest="l2norm_inputs", action="store_false"
    )
    packed_cg.add_argument("--random-initial-state", action="store_true")
    packed_cg.add_argument("--capture-null-indices", action="store_true")
    packed_cg.set_defaults(
        func=run_packed_recurrent_cudagraph_case, l2norm_inputs=True
    )

    conv_cg = subparsers.add_parser("run-conv-update-cudagraph")
    conv_cg.add_argument("--out", required=True)
    conv_cg.add_argument("--input-file")
    conv_cg.add_argument(
        "--case-name", default="gdn_conv_update_cudagraph_exactness"
    )
    conv_cg.add_argument("--batch", type=int, default=2)
    conv_cg.add_argument("--seqlen", type=int, default=1024)
    conv_cg.add_argument("--k-heads", type=int, default=4)
    conv_cg.add_argument("--v-heads", type=int, default=8)
    conv_cg.add_argument("--head-k-dim", type=int, default=128)
    conv_cg.add_argument("--head-v-dim", type=int, default=128)
    conv_cg.add_argument("--conv-dim", type=int)
    conv_cg.add_argument("--conv-width", type=int, default=4)
    conv_cg.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    conv_cg.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    conv_cg.add_argument("--seed", type=int, default=1234)
    conv_cg.add_argument("--input-scale", type=float, default=0.02)
    conv_cg.add_argument("--random-initial-state", action="store_true")
    conv_cg.add_argument("--capture-null-indices", action="store_true")
    conv_cg.add_argument("--no-silu", dest="silu", action="store_false")
    conv_cg.set_defaults(func=run_conv_update_cudagraph_case, silu=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--json-out")
    compare.add_argument("--no-strict", action="store_true")
    compare.set_defaults(func=compare_cases)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
