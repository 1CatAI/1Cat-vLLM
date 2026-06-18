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


def _make_conv_update_sequence_inputs(
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
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


def _make_spec_decode_cudagraph_inputs(
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    state_dtype = _resolve_state_dtype(args.state_dtype, dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    max_query_len = args.num_spec + 1
    graph_rows = args.graph_rows or max_query_len
    if graph_rows < max_query_len:
        raise ValueError(
            f"--graph-rows must be >= num_spec + 1 ({max_query_len})"
        )
    if args.state_slots < max_query_len:
        raise ValueError(
            f"--state-slots must be >= num_spec + 1 ({max_query_len})"
        )
    if args.fixed_query_len is not None and not (
        1 <= args.fixed_query_len <= max_query_len
    ):
        raise ValueError(
            f"--fixed-query-len must be in [1, {max_query_len}]"
        )
    if args.fixed_accepted is not None and not (
        1 <= args.fixed_accepted <= max_query_len
    ):
        raise ValueError(
            f"--fixed-accepted must be in [1, {max_query_len}]"
        )

    q_dim = args.k_heads * args.head_k_dim
    k_dim = args.k_heads * args.head_k_dim
    v_dim = args.v_heads * args.head_v_dim
    qkv_dim = q_dim + k_dim + v_dim

    mixed_qkv_seq = (
        torch.randn(
            (args.steps, max_query_len, qkv_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.input_scale
    ).contiguous()
    a_seq = (
        torch.randn(
            (args.steps, max_query_len, args.v_heads),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.gate_scale
    ).contiguous()
    b_seq = (
        torch.randn(
            (args.steps, max_query_len, args.v_heads),
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

    conv_state_len = args.conv_width - 1 + args.num_spec
    conv_state = torch.zeros(
        (args.state_slots, qkv_dim, conv_state_len),
        device=device,
        dtype=state_dtype,
    )
    ssm_state = torch.zeros(
        (
            args.state_slots,
            args.v_heads,
            args.head_v_dim,
            args.head_k_dim,
        ),
        device=device,
        dtype=state_dtype,
    )
    if args.random_initial_state:
        conv_state = (
            torch.randn(
                (args.state_slots, qkv_dim, conv_state_len),
                device=device,
                dtype=state_dtype,
                generator=generator,
            )
            * args.input_scale
        ).contiguous()
        ssm_state = (
            torch.randn(
                (
                    args.state_slots,
                    args.v_heads,
                    args.head_v_dim,
                    args.head_k_dim,
                ),
                device=device,
                dtype=state_dtype,
                generator=generator,
            )
            * args.input_scale
        ).contiguous()

    conv_weight = (
        torch.randn(
            (qkv_dim, args.conv_width),
            device=device,
            dtype=state_dtype,
            generator=generator,
        )
        * args.input_scale
    ).contiguous()
    conv_bias = (
        torch.randn(
            (qkv_dim,),
            device=device,
            dtype=state_dtype,
            generator=generator,
        )
        * args.input_scale
    ).contiguous()

    steps = torch.arange(args.steps, device=device, dtype=torch.int64)
    if args.fixed_query_len is None:
        if args.vary_query_lens:
            query_lens = (steps * 2 % max_query_len + 1).to(torch.int32)
        else:
            query_lens = torch.full(
                (args.steps,), max_query_len, device=device, dtype=torch.int32
            )
    else:
        query_lens = torch.full(
            (args.steps,),
            args.fixed_query_len,
            device=device,
            dtype=torch.int32,
        )

    if args.fixed_accepted is None:
        accepted = (steps * 3 % max_query_len + 1).to(torch.int32)
        if args.vary_query_lens:
            accepted = torch.minimum(accepted, query_lens)
    else:
        accepted = torch.full(
            (args.steps,), args.fixed_accepted, device=device, dtype=torch.int32
        )

    query_start_loc_seq = torch.empty(
        (args.steps, graph_rows + 1), device=device, dtype=torch.int32
    )
    query_start_loc_seq[:, 0] = 0
    query_start_loc_seq[:, 1:] = query_lens[:, None]

    num_accepted_tokens_seq = torch.ones(
        (args.steps, graph_rows), device=device, dtype=torch.int32
    )
    num_accepted_tokens_seq[:, 0] = accepted

    state_indices_seq = torch.full(
        (args.steps, graph_rows, max_query_len),
        -1,
        device=device,
        dtype=torch.int32,
    )
    slot_base = (steps * args.slot_stride) % args.state_slots
    slot_offsets = torch.arange(max_query_len, device=device, dtype=torch.int64)
    state_indices_seq[:, 0, :] = (
        (slot_base[:, None] + slot_offsets[None, :]) % args.state_slots
    ).to(torch.int32)

    return {
        "mixed_qkv_seq": mixed_qkv_seq.cpu(),
        "a_seq": a_seq.cpu(),
        "b_seq": b_seq.cpu(),
        "A_log": a_log.cpu(),
        "dt_bias": dt_bias.cpu(),
        "conv_state": conv_state.cpu(),
        "ssm_state": ssm_state.cpu(),
        "conv_weight": conv_weight.cpu(),
        "conv_bias": conv_bias.cpu(),
        "query_start_loc_seq": query_start_loc_seq.cpu(),
        "state_indices_seq": state_indices_seq.cpu(),
        "num_accepted_tokens_seq": num_accepted_tokens_seq.cpu(),
        "query_lens": query_lens.cpu(),
    }


def _load_or_create_spec_decode_cudagraph_inputs(
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if args.input_file:
        payload = _torch_load(Path(args.input_file))
        if "inputs" not in payload:
            raise ValueError(f"{args.input_file} does not contain saved inputs")
        return payload["inputs"]
    return _make_spec_decode_cudagraph_inputs(args)


def _add_spec_projection_inputs(
    args: argparse.Namespace,
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    required = {
        "z_seq",
        "norm_weight",
        "out_proj_weight",
        "logits_weight",
    }
    if required.issubset(inputs):
        return inputs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to create projection inputs")

    mixed_qkv_seq = inputs["mixed_qkv_seq"]
    steps = mixed_qkv_seq.shape[0]
    max_query_len = mixed_qkv_seq.shape[1]
    value_dim = args.v_heads * args.head_v_dim
    hidden_size = args.hidden_size or value_dim
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 913)

    z_seq = (
        torch.randn(
            (steps, max_query_len, args.v_heads, args.head_v_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.input_scale
    ).contiguous()
    norm_weight = (
        1.0
        + torch.randn(
            (args.head_v_dim,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.02
    ).contiguous()
    out_proj_weight = (
        torch.randn(
            (hidden_size, value_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.proj_scale
    ).contiguous()
    logits_weight = (
        torch.randn(
            (args.vocab_size, hidden_size),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * args.logits_scale
    ).contiguous()

    updated = dict(inputs)
    updated.update(
        {
            "z_seq": z_seq.cpu(),
            "norm_weight": norm_weight.cpu(),
            "out_proj_weight": out_proj_weight.cpu(),
            "logits_weight": logits_weight.cpu(),
        }
    )
    return updated


def _load_or_create_spec_projection_inputs(
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if args.input_file:
        payload = _torch_load(Path(args.input_file))
        if "inputs" not in payload:
            raise ValueError(f"{args.input_file} does not contain saved inputs")
        inputs = payload["inputs"]
    else:
        inputs = _make_spec_decode_cudagraph_inputs(args)
    return _add_spec_projection_inputs(args, inputs)


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
        layer.enable_flashqla_decode = False

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
            kv_cache=layer.kv_cache,
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


def run_spec_decode_cudagraph_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fused_gdn_gating,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    inputs = _load_or_create_spec_decode_cudagraph_inputs(args)
    cuda_inputs, _ = _to_cuda_inputs(inputs)
    mixed_qkv_seq = cuda_inputs["mixed_qkv_seq"]
    a_seq = cuda_inputs["a_seq"]
    b_seq = cuda_inputs["b_seq"]
    state_indices_seq = cuda_inputs["state_indices_seq"]
    num_accepted_tokens_seq = cuda_inputs["num_accepted_tokens_seq"]
    query_start_loc_seq = cuda_inputs["query_start_loc_seq"]
    initial_conv_state = cuda_inputs["conv_state"]
    initial_ssm_state = cuda_inputs["ssm_state"]
    steps = mixed_qkv_seq.shape[0]
    max_query_len = mixed_qkv_seq.shape[1]
    graph_rows = query_start_loc_seq.shape[1] - 1
    scale = args.scale
    if scale is None:
        scale = args.head_k_dim**-0.5
    activation = "silu" if args.silu else None

    def launch_one(
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        out: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        mixed_qkv_conv = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            cuda_inputs["conv_weight"],
            cuda_inputs["conv_bias"],
            activation,
            conv_state_indices=state_indices[:, 0],
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            max_query_len=max_query_len,
            validate_data=False,
        )
        query, key, value = _split_packed_mixed_qkv(
            mixed_qkv_conv,
            k_heads=args.k_heads,
            v_heads=args.v_heads,
            head_k_dim=args.head_k_dim,
            head_v_dim=args.head_v_dim,
        )
        g, beta = fused_gdn_gating(
            cuda_inputs["A_log"],
            a,
            b,
            cuda_inputs["dt_bias"],
        )
        core_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=query_start_loc,
            ssm_state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=args.l2norm_inputs,
        )
        out.copy_(core_out.squeeze(0))

    eager_conv_state = initial_conv_state.clone()
    eager_ssm_state = initial_ssm_state.clone()
    graph_conv_state = initial_conv_state.clone()
    graph_ssm_state = initial_ssm_state.clone()
    eager_out = torch.empty(
        (steps, max_query_len, args.v_heads, args.head_v_dim),
        device=mixed_qkv_seq.device,
        dtype=mixed_qkv_seq.dtype,
    )
    graph_out = torch.empty_like(eager_out)
    eager_mixed_after = torch.empty_like(mixed_qkv_seq)
    graph_mixed_after = torch.empty_like(mixed_qkv_seq)

    mixed_buf = mixed_qkv_seq[0].clone()
    a_buf = a_seq[0].clone()
    b_buf = b_seq[0].clone()
    state_indices_buf = state_indices_seq[0].clone()
    num_accepted_tokens_buf = num_accepted_tokens_seq[0].clone()
    query_start_loc_buf = query_start_loc_seq[0].clone()
    graph_step_out = torch.empty_like(eager_out[0])

    with torch.inference_mode():
        warm_conv_state = initial_conv_state.clone()
        warm_ssm_state = initial_ssm_state.clone()
        warm_out = torch.empty_like(eager_out[0])
        launch_one(
            mixed_qkv_seq[0].clone(),
            a_seq[0],
            b_seq[0],
            warm_conv_state,
            warm_ssm_state,
            warm_out,
            state_indices_seq[0],
            num_accepted_tokens_seq[0],
            query_start_loc_seq[0],
        )
        torch.cuda.synchronize()

        for step in range(steps):
            mixed_step = mixed_qkv_seq[step].clone()
            step_out = torch.empty_like(eager_out[step])
            launch_one(
                mixed_step,
                a_seq[step],
                b_seq[step],
                eager_conv_state,
                eager_ssm_state,
                step_out,
                state_indices_seq[step],
                num_accepted_tokens_seq[step],
                query_start_loc_seq[step],
            )
            eager_out[step].copy_(step_out)
            eager_mixed_after[step].copy_(mixed_step)
        torch.cuda.synchronize()

        capture_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture_graph):
            launch_one(
                mixed_buf,
                a_buf,
                b_buf,
                graph_conv_state,
                graph_ssm_state,
                graph_step_out,
                state_indices_buf,
                num_accepted_tokens_buf,
                query_start_loc_buf,
            )
        torch.cuda.synchronize()

        graph_conv_state.copy_(initial_conv_state)
        graph_ssm_state.copy_(initial_ssm_state)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(steps):
            mixed_buf.copy_(mixed_qkv_seq[step])
            a_buf.copy_(a_seq[step])
            b_buf.copy_(b_seq[step])
            state_indices_buf.copy_(state_indices_seq[step])
            num_accepted_tokens_buf.copy_(num_accepted_tokens_seq[step])
            query_start_loc_buf.copy_(query_start_loc_seq[step])
            capture_graph.replay()
            graph_out[step].copy_(graph_step_out)
            graph_mixed_after[step].copy_(mixed_buf)
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

    query_lens = inputs["query_lens"].to(torch.int64)
    live_token_mask = (
        torch.arange(max_query_len, dtype=torch.int64)[None, :]
        < query_lens[:, None]
    )
    outputs: dict[str, torch.Tensor | None] = {
        "eager_live_out": eager_out.detach().cpu()[live_token_mask],
        "graph_live_out": graph_out.detach().cpu()[live_token_mask],
        "eager_out": eager_out.detach().cpu(),
        "graph_out": graph_out.detach().cpu(),
        "eager_mixed_after": eager_mixed_after.detach().cpu(),
        "graph_mixed_after": graph_mixed_after.detach().cpu(),
        "eager_conv_state": eager_conv_state.detach().cpu(),
        "graph_conv_state": graph_conv_state.detach().cpu(),
        "eager_ssm_state": eager_ssm_state.detach().cpu(),
        "graph_ssm_state": graph_ssm_state.detach().cpu(),
    }
    comparisons = {
        "live_out": _compare_tensor(
            outputs["eager_live_out"], outputs["graph_live_out"]
        ),
        "full_out": _compare_tensor(outputs["eager_out"], outputs["graph_out"]),
        "mixed_after": _compare_tensor(
            outputs["eager_mixed_after"], outputs["graph_mixed_after"]
        ),
        "conv_state": _compare_tensor(
            outputs["eager_conv_state"], outputs["graph_conv_state"]
        ),
        "ssm_state": _compare_tensor(
            outputs["eager_ssm_state"], outputs["graph_ssm_state"]
        ),
    }
    strict_ok = all(
        comparisons[name]["torch_equal"] and comparisons[name]["max_diff"] == 0.0
        for name in ("live_out", "mixed_after", "conv_state", "ssm_state")
    )
    accepted_hist = torch.bincount(
        inputs["num_accepted_tokens_seq"][:, 0].to(torch.int64),
        minlength=max_query_len + 1,
    )
    query_len_hist = torch.bincount(
        inputs["query_lens"].to(torch.int64),
        minlength=max_query_len + 1,
    )
    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "qwen_gdn_spec_decode_conv_recurrent_cudagraph",
            "steps": steps,
            "num_spec": max_query_len - 1,
            "graph_rows": graph_rows,
            "state_slots": initial_ssm_state.shape[0],
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "conv_width": args.conv_width,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "scale": scale,
            "silu": args.silu,
            "l2norm_inputs": args.l2norm_inputs,
            "random_initial_state": args.random_initial_state,
            "vary_query_lens": args.vary_query_lens,
            "slot_stride": args.slot_stride,
            "accepted_hist": accepted_hist.tolist(),
            "query_len_hist": query_len_hist.tolist(),
            "pad_slot_id": -1,
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
        "config": payload["config"],
        "comparisons": comparisons,
    }, indent=2))
    return 0 if strict_ok else 1


def run_spec_commit_vs_standard_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen_gdn
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    inputs = _load_or_create_spec_decode_cudagraph_inputs(args)
    cuda_inputs, _ = _to_cuda_inputs(inputs)
    mixed_qkv_seq = cuda_inputs["mixed_qkv_seq"]
    a_seq = cuda_inputs["a_seq"]
    b_seq = cuda_inputs["b_seq"]
    state_indices_seq = cuda_inputs["state_indices_seq"]
    num_accepted_tokens_seq = cuda_inputs["num_accepted_tokens_seq"]
    query_start_loc_seq = cuda_inputs["query_start_loc_seq"]
    steps = mixed_qkv_seq.shape[0]
    max_query_len = mixed_qkv_seq.shape[1]
    graph_rows = query_start_loc_seq.shape[1] - 1
    qkv_dim = mixed_qkv_seq.shape[-1]
    layer_name = "sm70_gdn_spec_commit_exactness"

    conv_state_kernel = cuda_inputs["conv_state"]
    if qwen_gdn.is_conv_state_dim_first():
        conv_state_cache = conv_state_kernel
    else:
        conv_state_cache = conv_state_kernel.transpose(-1, -2).contiguous()
    ssm_state_cache = cuda_inputs["ssm_state"]
    empty_i32 = torch.empty(0, device=mixed_qkv_seq.device, dtype=torch.int32)
    spec_token_indx = torch.arange(
        max_query_len, device=mixed_qkv_seq.device, dtype=torch.int32
    )
    non_spec_token_indx = empty_i32
    spec_sequence_masks = torch.zeros(
        graph_rows, device=mixed_qkv_seq.device, dtype=torch.bool
    )
    spec_sequence_masks[0] = True

    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = layer_name
    layer.num_k_heads = args.k_heads
    layer.num_v_heads = args.v_heads
    layer.tp_size = 1
    layer.head_k_dim = args.head_k_dim
    layer.head_v_dim = args.head_v_dim
    layer.key_dim = args.k_heads * args.head_k_dim
    layer.value_dim = args.v_heads * args.head_v_dim
    layer.A_log = cuda_inputs["A_log"]
    layer.dt_bias = cuda_inputs["dt_bias"]
    layer.activation = "silu" if args.silu else None
    layer.conv1d = SimpleNamespace(
        weight=cuda_inputs["conv_weight"].view(qkv_dim, 1, args.conv_width),
        bias=cuda_inputs["conv_bias"],
    )
    layer.kv_cache = (conv_state_cache, ssm_state_cache)
    layer.enable_packed_recurrent_decode = False
    layer.enable_sm70_fused_sigmoid_mixed_qkv = False
    layer.compare_sm70_fused_sigmoid_mixed_qkv = False
    layer.enable_flashqla_decode = False
    layer.enable_sm70_legacy_prefill_prep = False
    layer.gdn_prefill_backend = "triton"

    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=max_query_len,
        num_actual_tokens=max_query_len,
        spec_query_start_loc=query_start_loc_seq[0],
        non_spec_query_start_loc=empty_i32,
        spec_state_indices_tensor=state_indices_seq[0],
        non_spec_state_indices_tensor=empty_i32,
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=num_accepted_tokens_seq[0],
    )
    forward_context = SimpleNamespace(
        attn_metadata={layer_name: metadata},
        no_compile_layers={layer_name: layer},
    )
    original_get_forward_context = qwen_gdn.get_forward_context

    def fake_get_forward_context() -> SimpleNamespace:
        return forward_context

    def call_standard_spec(
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        conv_cache: torch.Tensor,
        ssm_cache: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        torch.ops.vllm.qwen_gdn_attention_core_standard_spec(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_cache,
            ssm_cache,
            empty_i32,
            empty_i32,
            query_start_loc,
            state_indices,
            spec_token_indx,
            non_spec_token_indx,
            spec_sequence_masks,
            num_accepted_tokens,
            layer_name,
        )

    def call_spec_commit(
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        conv_cache: torch.Tensor,
        ssm_cache: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        torch.ops.vllm.qwen_gdn_attention_core_spec_commit(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_cache,
            ssm_cache,
            empty_i32,
            empty_i32,
            query_start_loc,
            state_indices,
            spec_token_indx,
            non_spec_token_indx,
            spec_sequence_masks,
            num_accepted_tokens,
            layer_name,
        )

    standard_conv = conv_state_cache.clone()
    standard_ssm = ssm_state_cache.clone()
    spec_eager_conv = conv_state_cache.clone()
    spec_eager_ssm = ssm_state_cache.clone()
    spec_graph_conv = conv_state_cache.clone()
    spec_graph_ssm = ssm_state_cache.clone()
    standard_out = torch.empty(
        (steps, max_query_len, args.v_heads, args.head_v_dim),
        device=mixed_qkv_seq.device,
        dtype=mixed_qkv_seq.dtype,
    )
    spec_eager_out = torch.empty_like(standard_out)
    spec_graph_out = torch.empty_like(standard_out)
    standard_mixed_after = torch.empty_like(mixed_qkv_seq)
    spec_eager_mixed_after = torch.empty_like(mixed_qkv_seq)
    spec_graph_mixed_after = torch.empty_like(mixed_qkv_seq)

    mixed_buf = mixed_qkv_seq[0].clone()
    a_buf = a_seq[0].clone()
    b_buf = b_seq[0].clone()
    state_indices_buf = state_indices_seq[0].clone()
    num_accepted_tokens_buf = num_accepted_tokens_seq[0].clone()
    query_start_loc_buf = query_start_loc_seq[0].clone()
    graph_step_out = torch.empty_like(spec_graph_out[0])

    try:
        qwen_gdn.get_forward_context = fake_get_forward_context
        with torch.inference_mode():
            warm_out = torch.empty_like(standard_out[0])
            call_spec_commit(
                mixed_qkv_seq[0].clone(),
                b_seq[0],
                a_seq[0],
                warm_out,
                conv_state_cache.clone(),
                ssm_state_cache.clone(),
                query_start_loc_seq[0],
                state_indices_seq[0],
                num_accepted_tokens_seq[0],
            )
            torch.cuda.synchronize()

            for step in range(steps):
                mixed_step = mixed_qkv_seq[step].clone()
                step_out = torch.empty_like(standard_out[step])
                call_standard_spec(
                    mixed_step,
                    b_seq[step],
                    a_seq[step],
                    step_out,
                    standard_conv,
                    standard_ssm,
                    query_start_loc_seq[step],
                    state_indices_seq[step],
                    num_accepted_tokens_seq[step],
                )
                standard_out[step].copy_(step_out)
                standard_mixed_after[step].copy_(mixed_step)

                mixed_step = mixed_qkv_seq[step].clone()
                step_out = torch.empty_like(spec_eager_out[step])
                call_spec_commit(
                    mixed_step,
                    b_seq[step],
                    a_seq[step],
                    step_out,
                    spec_eager_conv,
                    spec_eager_ssm,
                    query_start_loc_seq[step],
                    state_indices_seq[step],
                    num_accepted_tokens_seq[step],
                )
                spec_eager_out[step].copy_(step_out)
                spec_eager_mixed_after[step].copy_(mixed_step)
            torch.cuda.synchronize()

            capture_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture_graph):
                call_spec_commit(
                    mixed_buf,
                    b_buf,
                    a_buf,
                    graph_step_out,
                    spec_graph_conv,
                    spec_graph_ssm,
                    query_start_loc_buf,
                    state_indices_buf,
                    num_accepted_tokens_buf,
                )
            torch.cuda.synchronize()

            spec_graph_conv.copy_(conv_state_cache)
            spec_graph_ssm.copy_(ssm_state_cache)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for step in range(steps):
                mixed_buf.copy_(mixed_qkv_seq[step])
                a_buf.copy_(a_seq[step])
                b_buf.copy_(b_seq[step])
                state_indices_buf.copy_(state_indices_seq[step])
                num_accepted_tokens_buf.copy_(num_accepted_tokens_seq[step])
                query_start_loc_buf.copy_(query_start_loc_seq[step])
                capture_graph.replay()
                spec_graph_out[step].copy_(graph_step_out)
                spec_graph_mixed_after[step].copy_(mixed_buf)
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - start
    finally:
        qwen_gdn.get_forward_context = original_get_forward_context

    query_lens = inputs["query_lens"].to(torch.int64)
    live_token_mask = (
        torch.arange(max_query_len, dtype=torch.int64)[None, :]
        < query_lens[:, None]
    )
    outputs: dict[str, torch.Tensor | None] = {
        "standard_live_out": standard_out.detach().cpu()[live_token_mask],
        "spec_eager_live_out": spec_eager_out.detach().cpu()[live_token_mask],
        "spec_graph_live_out": spec_graph_out.detach().cpu()[live_token_mask],
        "standard_out": standard_out.detach().cpu(),
        "spec_eager_out": spec_eager_out.detach().cpu(),
        "spec_graph_out": spec_graph_out.detach().cpu(),
        "standard_mixed_after": standard_mixed_after.detach().cpu(),
        "spec_eager_mixed_after": spec_eager_mixed_after.detach().cpu(),
        "spec_graph_mixed_after": spec_graph_mixed_after.detach().cpu(),
        "standard_conv_state": standard_conv.detach().cpu(),
        "spec_eager_conv_state": spec_eager_conv.detach().cpu(),
        "spec_graph_conv_state": spec_graph_conv.detach().cpu(),
        "standard_ssm_state": standard_ssm.detach().cpu(),
        "spec_eager_ssm_state": spec_eager_ssm.detach().cpu(),
        "spec_graph_ssm_state": spec_graph_ssm.detach().cpu(),
    }
    comparisons = {
        "eager_live_out": _compare_tensor(
            outputs["standard_live_out"], outputs["spec_eager_live_out"]
        ),
        "graph_live_out": _compare_tensor(
            outputs["standard_live_out"], outputs["spec_graph_live_out"]
        ),
        "eager_full_out": _compare_tensor(
            outputs["standard_out"], outputs["spec_eager_out"]
        ),
        "graph_full_out": _compare_tensor(
            outputs["standard_out"], outputs["spec_graph_out"]
        ),
        "eager_mixed_after": _compare_tensor(
            outputs["standard_mixed_after"], outputs["spec_eager_mixed_after"]
        ),
        "graph_mixed_after": _compare_tensor(
            outputs["standard_mixed_after"], outputs["spec_graph_mixed_after"]
        ),
        "eager_conv_state": _compare_tensor(
            outputs["standard_conv_state"], outputs["spec_eager_conv_state"]
        ),
        "graph_conv_state": _compare_tensor(
            outputs["standard_conv_state"], outputs["spec_graph_conv_state"]
        ),
        "eager_ssm_state": _compare_tensor(
            outputs["standard_ssm_state"], outputs["spec_eager_ssm_state"]
        ),
        "graph_ssm_state": _compare_tensor(
            outputs["standard_ssm_state"], outputs["spec_graph_ssm_state"]
        ),
    }
    strict_keys = (
        "eager_live_out",
        "graph_live_out",
        "eager_mixed_after",
        "graph_mixed_after",
        "eager_conv_state",
        "graph_conv_state",
        "eager_ssm_state",
        "graph_ssm_state",
    )
    strict_ok = all(
        comparisons[name]["torch_equal"] and comparisons[name]["max_diff"] == 0.0
        for name in strict_keys
    )
    accepted_hist = torch.bincount(
        inputs["num_accepted_tokens_seq"][:, 0].to(torch.int64),
        minlength=max_query_len + 1,
    )
    query_len_hist = torch.bincount(
        inputs["query_lens"].to(torch.int64),
        minlength=max_query_len + 1,
    )
    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "qwen_gdn_spec_commit_vs_standard_spec_cudagraph",
            "steps": steps,
            "num_spec": max_query_len - 1,
            "graph_rows": graph_rows,
            "state_slots": ssm_state_cache.shape[0],
            "conv_state_layout_dim_first": qwen_gdn.is_conv_state_dim_first(),
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "conv_width": args.conv_width,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "silu": args.silu,
            "random_initial_state": args.random_initial_state,
            "vary_query_lens": args.vary_query_lens,
            "slot_stride": args.slot_stride,
            "accepted_hist": accepted_hist.tolist(),
            "query_len_hist": query_len_hist.tolist(),
            "pad_slot_id": -1,
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
        "config": payload["config"],
        "comparisons": comparisons,
    }, indent=2))
    return 0 if strict_ok else 1


def run_spec_commit_projection_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    from vllm.model_executor.layers.layernorm import RMSNormGated
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen_gdn
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    class _FakeRowParallelLinear:
        def __init__(self, weight: torch.Tensor):
            self.weight = weight

        def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return F.linear(x, self.weight), None

    class _FakeRMSNormGated:
        def __init__(
            self,
            weight: torch.Tensor,
            eps: float,
            activation: str,
        ):
            self.weight = weight
            self.eps = eps
            self.activation = activation

        def __call__(
            self,
            x: torch.Tensor,
            z: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return RMSNormGated.forward_static(
                x,
                z,
                self.weight,
                self.eps,
                x.dtype,
                group_size=None,
                norm_before_gate=True,
                activation=self.activation,
            )

    inputs = _load_or_create_spec_projection_inputs(args)
    cuda_inputs, _ = _to_cuda_inputs(inputs)
    mixed_qkv_seq = cuda_inputs["mixed_qkv_seq"]
    a_seq = cuda_inputs["a_seq"]
    b_seq = cuda_inputs["b_seq"]
    z_seq = cuda_inputs["z_seq"]
    state_indices_seq = cuda_inputs["state_indices_seq"]
    num_accepted_tokens_seq = cuda_inputs["num_accepted_tokens_seq"]
    query_start_loc_seq = cuda_inputs["query_start_loc_seq"]
    logits_weight = cuda_inputs["logits_weight"]
    steps = mixed_qkv_seq.shape[0]
    max_query_len = mixed_qkv_seq.shape[1]
    graph_rows = query_start_loc_seq.shape[1] - 1
    qkv_dim = mixed_qkv_seq.shape[-1]
    hidden_size = logits_weight.shape[-1]
    vocab_size = logits_weight.shape[0]
    layer_name = "sm70_gdn_spec_commit_projection_exactness"

    conv_state_kernel = cuda_inputs["conv_state"]
    if qwen_gdn.is_conv_state_dim_first():
        conv_state_cache = conv_state_kernel
    else:
        conv_state_cache = conv_state_kernel.transpose(-1, -2).contiguous()
    ssm_state_cache = cuda_inputs["ssm_state"]
    empty_i32 = torch.empty(0, device=mixed_qkv_seq.device, dtype=torch.int32)
    spec_token_indx = torch.arange(
        max_query_len, device=mixed_qkv_seq.device, dtype=torch.int32
    )
    non_spec_token_indx = empty_i32
    spec_sequence_masks = torch.zeros(
        graph_rows, device=mixed_qkv_seq.device, dtype=torch.bool
    )
    spec_sequence_masks[0] = True

    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = layer_name
    layer.num_k_heads = args.k_heads
    layer.num_v_heads = args.v_heads
    layer.tp_size = 1
    layer.head_k_dim = args.head_k_dim
    layer.head_v_dim = args.head_v_dim
    layer.key_dim = args.k_heads * args.head_k_dim
    layer.value_dim = args.v_heads * args.head_v_dim
    layer.A_log = cuda_inputs["A_log"]
    layer.dt_bias = cuda_inputs["dt_bias"]
    layer.activation = "silu" if args.silu else None
    layer.conv1d = SimpleNamespace(
        weight=cuda_inputs["conv_weight"].view(qkv_dim, 1, args.conv_width),
        bias=cuda_inputs["conv_bias"],
    )
    layer.kv_cache = (conv_state_cache, ssm_state_cache)
    layer.enable_packed_recurrent_decode = False
    layer.enable_sm70_fused_sigmoid_mixed_qkv = False
    layer.compare_sm70_fused_sigmoid_mixed_qkv = False
    layer.enable_flashqla_decode = False
    layer.enable_sm70_legacy_prefill_prep = False
    layer.gdn_prefill_backend = "triton"
    layer.norm = _FakeRMSNormGated(
        cuda_inputs["norm_weight"],
        args.norm_eps,
        args.output_gate,
    )
    layer.out_proj = _FakeRowParallelLinear(cuda_inputs["out_proj_weight"])

    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=max_query_len,
        num_actual_tokens=max_query_len,
        spec_query_start_loc=query_start_loc_seq[0],
        non_spec_query_start_loc=empty_i32,
        spec_state_indices_tensor=state_indices_seq[0],
        non_spec_state_indices_tensor=empty_i32,
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=num_accepted_tokens_seq[0],
    )
    forward_context = SimpleNamespace(
        attn_metadata={layer_name: metadata},
        no_compile_layers={layer_name: layer},
    )
    original_get_forward_context = qwen_gdn.get_forward_context

    def fake_get_forward_context() -> SimpleNamespace:
        return forward_context

    def call_standard_spec(
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        conv_cache: torch.Tensor,
        ssm_cache: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        torch.ops.vllm.qwen_gdn_attention_core_standard_spec(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_cache,
            ssm_cache,
            empty_i32,
            empty_i32,
            query_start_loc,
            state_indices,
            spec_token_indx,
            non_spec_token_indx,
            spec_sequence_masks,
            num_accepted_tokens,
            layer_name,
        )

    def call_spec_commit(
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        conv_cache: torch.Tensor,
        ssm_cache: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        torch.ops.vllm.qwen_gdn_attention_core_spec_commit(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_cache,
            ssm_cache,
            empty_i32,
            empty_i32,
            query_start_loc,
            state_indices,
            spec_token_indx,
            non_spec_token_indx,
            spec_sequence_masks,
            num_accepted_tokens,
            layer_name,
        )

    def direct_projection(
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        layer._output_projection(core_attn_out, z, output, max_query_len)

    def opaque_projection(
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        torch.ops.vllm.qwen_gdn_output_projection(
            core_attn_out,
            z,
            output,
            max_query_len,
            layer_name,
        )

    def compute_logits(output: torch.Tensor, logits: torch.Tensor) -> None:
        logits.copy_(F.linear(output, logits_weight))

    standard_conv = conv_state_cache.clone()
    standard_ssm = ssm_state_cache.clone()
    spec_direct_eager_conv = conv_state_cache.clone()
    spec_direct_eager_ssm = ssm_state_cache.clone()
    spec_direct_graph_conv = conv_state_cache.clone()
    spec_direct_graph_ssm = ssm_state_cache.clone()
    spec_opaque_graph_conv = conv_state_cache.clone()
    spec_opaque_graph_ssm = ssm_state_cache.clone()
    spec_direct_compile_conv = conv_state_cache.clone()
    spec_direct_compile_ssm = ssm_state_cache.clone()

    output_shape = (steps, max_query_len, hidden_size)
    logits_shape = (steps, max_query_len, vocab_size)
    core_shape = (steps, max_query_len, args.v_heads, args.head_v_dim)
    standard_output = torch.empty(
        output_shape, device=mixed_qkv_seq.device, dtype=mixed_qkv_seq.dtype
    )
    spec_direct_eager_output = torch.empty_like(standard_output)
    spec_direct_graph_output = torch.empty_like(standard_output)
    spec_opaque_graph_output = torch.empty_like(standard_output)
    spec_direct_compile_output = (
        torch.empty_like(standard_output) if args.compile_direct else None
    )
    standard_logits = torch.empty(
        logits_shape, device=mixed_qkv_seq.device, dtype=mixed_qkv_seq.dtype
    )
    spec_direct_eager_logits = torch.empty_like(standard_logits)
    spec_direct_graph_logits = torch.empty_like(standard_logits)
    spec_opaque_graph_logits = torch.empty_like(standard_logits)
    spec_direct_compile_logits = (
        torch.empty_like(standard_logits) if args.compile_direct else None
    )
    standard_core = torch.empty(
        core_shape, device=mixed_qkv_seq.device, dtype=mixed_qkv_seq.dtype
    )
    spec_direct_graph_core = torch.empty_like(standard_core)
    spec_opaque_graph_core = torch.empty_like(standard_core)
    spec_direct_compile_core = (
        torch.empty_like(standard_core) if args.compile_direct else None
    )

    mixed_buf = mixed_qkv_seq[0].clone()
    a_buf = a_seq[0].clone()
    b_buf = b_seq[0].clone()
    z_buf = z_seq[0].clone()
    state_indices_buf = state_indices_seq[0].clone()
    num_accepted_tokens_buf = num_accepted_tokens_seq[0].clone()
    query_start_loc_buf = query_start_loc_seq[0].clone()
    graph_direct_core = torch.empty_like(standard_core[0])
    graph_direct_output = torch.empty_like(standard_output[0])
    graph_direct_logits = torch.empty_like(standard_logits[0])
    graph_opaque_core = torch.empty_like(standard_core[0])
    graph_opaque_output = torch.empty_like(standard_output[0])
    graph_opaque_logits = torch.empty_like(standard_logits[0])
    compile_step_error: str | None = None

    try:
        qwen_gdn.get_forward_context = fake_get_forward_context
        with torch.inference_mode():
            warm_core = torch.empty_like(standard_core[0])
            warm_output = torch.empty_like(standard_output[0])
            call_spec_commit(
                mixed_qkv_seq[0].clone(),
                b_seq[0],
                a_seq[0],
                warm_core,
                conv_state_cache.clone(),
                ssm_state_cache.clone(),
                query_start_loc_seq[0],
                state_indices_seq[0],
                num_accepted_tokens_seq[0],
            )
            direct_projection(warm_core, z_seq[0], warm_output)
            torch.cuda.synchronize()

            for step in range(steps):
                mixed_step = mixed_qkv_seq[step].clone()
                step_core = torch.empty_like(standard_core[step])
                step_output = torch.empty_like(standard_output[step])
                step_logits = torch.empty_like(standard_logits[step])
                call_standard_spec(
                    mixed_step,
                    b_seq[step],
                    a_seq[step],
                    step_core,
                    standard_conv,
                    standard_ssm,
                    query_start_loc_seq[step],
                    state_indices_seq[step],
                    num_accepted_tokens_seq[step],
                )
                opaque_projection(step_core, z_seq[step], step_output)
                compute_logits(step_output, step_logits)
                standard_core[step].copy_(step_core)
                standard_output[step].copy_(step_output)
                standard_logits[step].copy_(step_logits)

                mixed_step = mixed_qkv_seq[step].clone()
                step_core = torch.empty_like(standard_core[step])
                step_output = torch.empty_like(standard_output[step])
                step_logits = torch.empty_like(standard_logits[step])
                call_spec_commit(
                    mixed_step,
                    b_seq[step],
                    a_seq[step],
                    step_core,
                    spec_direct_eager_conv,
                    spec_direct_eager_ssm,
                    query_start_loc_seq[step],
                    state_indices_seq[step],
                    num_accepted_tokens_seq[step],
                )
                direct_projection(step_core, z_seq[step], step_output)
                compute_logits(step_output, step_logits)
                spec_direct_eager_output[step].copy_(step_output)
                spec_direct_eager_logits[step].copy_(step_logits)
            torch.cuda.synchronize()

            direct_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(direct_graph):
                call_spec_commit(
                    mixed_buf,
                    b_buf,
                    a_buf,
                    graph_direct_core,
                    spec_direct_graph_conv,
                    spec_direct_graph_ssm,
                    query_start_loc_buf,
                    state_indices_buf,
                    num_accepted_tokens_buf,
                )
                direct_projection(graph_direct_core, z_buf, graph_direct_output)
                compute_logits(graph_direct_output, graph_direct_logits)
            torch.cuda.synchronize()

            opaque_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(opaque_graph):
                call_spec_commit(
                    mixed_buf,
                    b_buf,
                    a_buf,
                    graph_opaque_core,
                    spec_opaque_graph_conv,
                    spec_opaque_graph_ssm,
                    query_start_loc_buf,
                    state_indices_buf,
                    num_accepted_tokens_buf,
                )
                opaque_projection(graph_opaque_core, z_buf, graph_opaque_output)
                compute_logits(graph_opaque_output, graph_opaque_logits)
            torch.cuda.synchronize()

            spec_direct_graph_conv.copy_(conv_state_cache)
            spec_direct_graph_ssm.copy_(ssm_state_cache)
            spec_opaque_graph_conv.copy_(conv_state_cache)
            spec_opaque_graph_ssm.copy_(ssm_state_cache)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for step in range(steps):
                mixed_buf.copy_(mixed_qkv_seq[step])
                a_buf.copy_(a_seq[step])
                b_buf.copy_(b_seq[step])
                z_buf.copy_(z_seq[step])
                state_indices_buf.copy_(state_indices_seq[step])
                num_accepted_tokens_buf.copy_(num_accepted_tokens_seq[step])
                query_start_loc_buf.copy_(query_start_loc_seq[step])
                direct_graph.replay()
                spec_direct_graph_core[step].copy_(graph_direct_core)
                spec_direct_graph_output[step].copy_(graph_direct_output)
                spec_direct_graph_logits[step].copy_(graph_direct_logits)
            for step in range(steps):
                mixed_buf.copy_(mixed_qkv_seq[step])
                a_buf.copy_(a_seq[step])
                b_buf.copy_(b_seq[step])
                z_buf.copy_(z_seq[step])
                state_indices_buf.copy_(state_indices_seq[step])
                num_accepted_tokens_buf.copy_(num_accepted_tokens_seq[step])
                query_start_loc_buf.copy_(query_start_loc_seq[step])
                opaque_graph.replay()
                spec_opaque_graph_core[step].copy_(graph_opaque_core)
                spec_opaque_graph_output[step].copy_(graph_opaque_output)
                spec_opaque_graph_logits[step].copy_(graph_opaque_logits)
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - start

            if args.compile_direct:

                def compile_direct_step(
                    mixed_qkv: torch.Tensor,
                    b: torch.Tensor,
                    a: torch.Tensor,
                    z: torch.Tensor,
                    core_attn_out: torch.Tensor,
                    core_snapshot: torch.Tensor,
                    output: torch.Tensor,
                    logits: torch.Tensor,
                    conv_cache: torch.Tensor,
                    ssm_cache: torch.Tensor,
                    query_start_loc: torch.Tensor,
                    state_indices: torch.Tensor,
                    num_accepted_tokens: torch.Tensor,
                ) -> None:
                    call_spec_commit(
                        mixed_qkv,
                        b,
                        a,
                        core_attn_out,
                        conv_cache,
                        ssm_cache,
                        query_start_loc,
                        state_indices,
                        num_accepted_tokens,
                    )
                    core_snapshot.copy_(core_attn_out)
                    direct_projection(core_attn_out, z, output)
                    compute_logits(output, logits)

                try:
                    compiled_step = torch.compile(
                        compile_direct_step,
                        fullgraph=args.compile_fullgraph,
                        dynamic=False,
                    )
                    warm_core = torch.empty_like(standard_core[0])
                    warm_core_snapshot = torch.empty_like(standard_core[0])
                    warm_output = torch.empty_like(standard_output[0])
                    warm_logits = torch.empty_like(standard_logits[0])
                    compiled_step(
                        mixed_qkv_seq[0].clone(),
                        b_seq[0],
                        a_seq[0],
                        z_seq[0],
                        warm_core,
                        warm_core_snapshot,
                        warm_output,
                        warm_logits,
                        spec_direct_compile_conv,
                        spec_direct_compile_ssm,
                        query_start_loc_seq[0],
                        state_indices_seq[0],
                        num_accepted_tokens_seq[0],
                    )
                    torch.cuda.synchronize()
                    spec_direct_compile_conv.copy_(conv_state_cache)
                    spec_direct_compile_ssm.copy_(ssm_state_cache)
                    torch.cuda.synchronize()
                    assert spec_direct_compile_core is not None
                    assert spec_direct_compile_output is not None
                    assert spec_direct_compile_logits is not None
                    start = time.perf_counter()
                    for step in range(steps):
                        mixed_step = mixed_qkv_seq[step].clone()
                        step_core = torch.empty_like(standard_core[step])
                        step_core_snapshot = torch.empty_like(standard_core[step])
                        step_output = torch.empty_like(standard_output[step])
                        step_logits = torch.empty_like(standard_logits[step])
                        compiled_step(
                            mixed_step,
                            b_seq[step],
                            a_seq[step],
                            z_seq[step],
                            step_core,
                            step_core_snapshot,
                            step_output,
                            step_logits,
                            spec_direct_compile_conv,
                            spec_direct_compile_ssm,
                            query_start_loc_seq[step],
                            state_indices_seq[step],
                            num_accepted_tokens_seq[step],
                        )
                        spec_direct_compile_core[step].copy_(step_core_snapshot)
                        spec_direct_compile_output[step].copy_(step_output)
                        spec_direct_compile_logits[step].copy_(step_logits)
                    torch.cuda.synchronize()
                    elapsed_s += time.perf_counter() - start
                except Exception as exc:
                    compile_step_error = repr(exc)
    finally:
        qwen_gdn.get_forward_context = original_get_forward_context

    query_lens = inputs["query_lens"].to(torch.int64)
    live_token_mask = (
        torch.arange(max_query_len, dtype=torch.int64)[None, :]
        < query_lens[:, None]
    )
    outputs: dict[str, torch.Tensor | None] = {
        "standard_live_output": standard_output.detach().cpu()[live_token_mask],
        "spec_direct_eager_live_output": spec_direct_eager_output.detach().cpu()[
            live_token_mask
        ],
        "spec_direct_graph_live_output": spec_direct_graph_output.detach().cpu()[
            live_token_mask
        ],
        "spec_opaque_graph_live_output": spec_opaque_graph_output.detach().cpu()[
            live_token_mask
        ],
        "spec_direct_compile_live_output": (
            spec_direct_compile_output.detach().cpu()[live_token_mask]
            if spec_direct_compile_output is not None
            else None
        ),
        "standard_live_logits": standard_logits.detach().cpu()[live_token_mask],
        "spec_direct_eager_live_logits": spec_direct_eager_logits.detach().cpu()[
            live_token_mask
        ],
        "spec_direct_graph_live_logits": spec_direct_graph_logits.detach().cpu()[
            live_token_mask
        ],
        "spec_opaque_graph_live_logits": spec_opaque_graph_logits.detach().cpu()[
            live_token_mask
        ],
        "spec_direct_compile_live_logits": (
            spec_direct_compile_logits.detach().cpu()[live_token_mask]
            if spec_direct_compile_logits is not None
            else None
        ),
        "standard_output": standard_output.detach().cpu(),
        "spec_direct_eager_output": spec_direct_eager_output.detach().cpu(),
        "spec_direct_graph_output": spec_direct_graph_output.detach().cpu(),
        "spec_opaque_graph_output": spec_opaque_graph_output.detach().cpu(),
        "spec_direct_compile_output": spec_direct_compile_output.detach().cpu()
        if spec_direct_compile_output is not None
        else None,
        "standard_logits": standard_logits.detach().cpu(),
        "spec_direct_eager_logits": spec_direct_eager_logits.detach().cpu(),
        "spec_direct_graph_logits": spec_direct_graph_logits.detach().cpu(),
        "spec_opaque_graph_logits": spec_opaque_graph_logits.detach().cpu(),
        "spec_direct_compile_logits": spec_direct_compile_logits.detach().cpu()
        if spec_direct_compile_logits is not None
        else None,
        "standard_core": standard_core.detach().cpu(),
        "spec_direct_graph_core": spec_direct_graph_core.detach().cpu(),
        "spec_opaque_graph_core": spec_opaque_graph_core.detach().cpu(),
        "spec_direct_compile_core": spec_direct_compile_core.detach().cpu()
        if spec_direct_compile_core is not None
        else None,
        "standard_conv_state": standard_conv.detach().cpu(),
        "spec_direct_eager_conv_state": spec_direct_eager_conv.detach().cpu(),
        "spec_direct_graph_conv_state": spec_direct_graph_conv.detach().cpu(),
        "spec_opaque_graph_conv_state": spec_opaque_graph_conv.detach().cpu(),
        "spec_direct_compile_conv_state": (
            spec_direct_compile_conv.detach().cpu() if args.compile_direct else None
        ),
        "standard_ssm_state": standard_ssm.detach().cpu(),
        "spec_direct_eager_ssm_state": spec_direct_eager_ssm.detach().cpu(),
        "spec_direct_graph_ssm_state": spec_direct_graph_ssm.detach().cpu(),
        "spec_opaque_graph_ssm_state": spec_opaque_graph_ssm.detach().cpu(),
        "spec_direct_compile_ssm_state": (
            spec_direct_compile_ssm.detach().cpu() if args.compile_direct else None
        ),
    }
    comparisons = {
        "direct_eager_live_output": _compare_tensor(
            outputs["standard_live_output"],
            outputs["spec_direct_eager_live_output"],
        ),
        "direct_graph_live_output": _compare_tensor(
            outputs["standard_live_output"],
            outputs["spec_direct_graph_live_output"],
        ),
        "opaque_graph_live_output": _compare_tensor(
            outputs["standard_live_output"],
            outputs["spec_opaque_graph_live_output"],
        ),
        "direct_graph_vs_opaque_graph_live_output": _compare_tensor(
            outputs["spec_direct_graph_live_output"],
            outputs["spec_opaque_graph_live_output"],
        ),
        "direct_compile_live_output": _compare_tensor(
            outputs["standard_live_output"],
            outputs["spec_direct_compile_live_output"],
        )
        if args.compile_direct
        else None,
        "direct_eager_live_logits": _compare_tensor(
            outputs["standard_live_logits"],
            outputs["spec_direct_eager_live_logits"],
        ),
        "direct_graph_live_logits": _compare_tensor(
            outputs["standard_live_logits"],
            outputs["spec_direct_graph_live_logits"],
        ),
        "opaque_graph_live_logits": _compare_tensor(
            outputs["standard_live_logits"],
            outputs["spec_opaque_graph_live_logits"],
        ),
        "direct_graph_vs_opaque_graph_live_logits": _compare_tensor(
            outputs["spec_direct_graph_live_logits"],
            outputs["spec_opaque_graph_live_logits"],
        ),
        "direct_compile_live_logits": _compare_tensor(
            outputs["standard_live_logits"],
            outputs["spec_direct_compile_live_logits"],
        )
        if args.compile_direct
        else None,
        "direct_graph_core": _compare_tensor(
            outputs["standard_core"], outputs["spec_direct_graph_core"]
        ),
        "opaque_graph_core": _compare_tensor(
            outputs["standard_core"], outputs["spec_opaque_graph_core"]
        ),
        "direct_compile_core": _compare_tensor(
            outputs["standard_core"], outputs["spec_direct_compile_core"]
        )
        if args.compile_direct
        else None,
        "direct_eager_conv_state": _compare_tensor(
            outputs["standard_conv_state"],
            outputs["spec_direct_eager_conv_state"],
        ),
        "direct_graph_conv_state": _compare_tensor(
            outputs["standard_conv_state"],
            outputs["spec_direct_graph_conv_state"],
        ),
        "opaque_graph_conv_state": _compare_tensor(
            outputs["standard_conv_state"],
            outputs["spec_opaque_graph_conv_state"],
        ),
        "direct_compile_conv_state": _compare_tensor(
            outputs["standard_conv_state"],
            outputs["spec_direct_compile_conv_state"],
        )
        if args.compile_direct
        else None,
        "direct_eager_ssm_state": _compare_tensor(
            outputs["standard_ssm_state"],
            outputs["spec_direct_eager_ssm_state"],
        ),
        "direct_graph_ssm_state": _compare_tensor(
            outputs["standard_ssm_state"],
            outputs["spec_direct_graph_ssm_state"],
        ),
        "opaque_graph_ssm_state": _compare_tensor(
            outputs["standard_ssm_state"],
            outputs["spec_opaque_graph_ssm_state"],
        ),
        "direct_compile_ssm_state": _compare_tensor(
            outputs["standard_ssm_state"],
            outputs["spec_direct_compile_ssm_state"],
        )
        if args.compile_direct
        else None,
    }
    strict_keys = [
        "direct_eager_live_output",
        "direct_graph_live_output",
        "opaque_graph_live_output",
        "direct_graph_vs_opaque_graph_live_output",
        "direct_eager_live_logits",
        "direct_graph_live_logits",
        "opaque_graph_live_logits",
        "direct_graph_vs_opaque_graph_live_logits",
        "direct_graph_core",
        "opaque_graph_core",
        "direct_eager_conv_state",
        "direct_graph_conv_state",
        "opaque_graph_conv_state",
        "direct_eager_ssm_state",
        "direct_graph_ssm_state",
        "opaque_graph_ssm_state",
    ]
    if args.compile_direct:
        strict_keys.extend(
            [
                "direct_compile_live_output",
                "direct_compile_live_logits",
                "direct_compile_core",
                "direct_compile_conv_state",
                "direct_compile_ssm_state",
            ]
        )
    strict_ok = all(
        comparisons[name] is not None
        and comparisons[name]["torch_equal"]
        and comparisons[name]["max_diff"] == 0.0
        for name in strict_keys
    ) and compile_step_error is None
    accepted_hist = torch.bincount(
        inputs["num_accepted_tokens_seq"][:, 0].to(torch.int64),
        minlength=max_query_len + 1,
    )
    query_len_hist = torch.bincount(
        inputs["query_lens"].to(torch.int64),
        minlength=max_query_len + 1,
    )
    saved_inputs = {name: tensor.cpu() for name, tensor in inputs.items()}
    payload = {
        "metadata": _metadata(args),
        "config": {
            "op": "qwen_gdn_spec_commit_projection_vs_opaque_cudagraph",
            "steps": steps,
            "num_spec": max_query_len - 1,
            "graph_rows": graph_rows,
            "state_slots": ssm_state_cache.shape[0],
            "conv_state_layout_dim_first": qwen_gdn.is_conv_state_dim_first(),
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": args.head_v_dim,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "conv_width": args.conv_width,
            "dtype": args.dtype,
            "state_dtype": args.state_dtype,
            "seed": args.seed,
            "silu": args.silu,
            "output_gate": args.output_gate,
            "norm_eps": args.norm_eps,
            "compile_direct": args.compile_direct,
            "compile_fullgraph": args.compile_fullgraph,
            "compile_step_error": compile_step_error,
            "random_initial_state": args.random_initial_state,
            "vary_query_lens": args.vary_query_lens,
            "slot_stride": args.slot_stride,
            "accepted_hist": accepted_hist.tolist(),
            "query_len_hist": query_len_hist.tolist(),
            "pad_slot_id": -1,
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
        "config": payload["config"],
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

    spec_cg = subparsers.add_parser("run-spec-decode-cudagraph")
    spec_cg.add_argument("--out", required=True)
    spec_cg.add_argument("--input-file")
    spec_cg.add_argument(
        "--case-name", default="gdn_spec_decode_cudagraph_exactness"
    )
    spec_cg.add_argument("--steps", type=int, default=512)
    spec_cg.add_argument("--num-spec", type=int, default=4)
    spec_cg.add_argument("--graph-rows", type=int)
    spec_cg.add_argument("--state-slots", type=int, default=64)
    spec_cg.add_argument("--slot-stride", type=int, default=3)
    spec_cg.add_argument("--k-heads", type=int, default=4)
    spec_cg.add_argument("--v-heads", type=int, default=8)
    spec_cg.add_argument("--head-k-dim", type=int, default=128)
    spec_cg.add_argument("--head-v-dim", type=int, default=128)
    spec_cg.add_argument("--conv-width", type=int, default=4)
    spec_cg.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    spec_cg.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    spec_cg.add_argument("--seed", type=int, default=1234)
    spec_cg.add_argument("--input-scale", type=float, default=0.02)
    spec_cg.add_argument("--gate-scale", type=float, default=0.01)
    spec_cg.add_argument("--beta-scale", type=float, default=0.25)
    spec_cg.add_argument("--scale", type=float)
    spec_cg.add_argument("--fixed-query-len", type=int)
    spec_cg.add_argument("--fixed-accepted", type=int)
    spec_cg.add_argument("--vary-query-lens", action="store_true")
    spec_cg.add_argument("--random-initial-state", action="store_true")
    spec_cg.add_argument(
        "--no-l2norm-inputs", dest="l2norm_inputs", action="store_false"
    )
    spec_cg.add_argument("--no-silu", dest="silu", action="store_false")
    spec_cg.set_defaults(
        func=run_spec_decode_cudagraph_case,
        l2norm_inputs=True,
        silu=True,
    )

    spec_op = subparsers.add_parser("run-spec-commit-vs-standard")
    spec_op.add_argument("--out", required=True)
    spec_op.add_argument("--input-file")
    spec_op.add_argument(
        "--case-name", default="gdn_spec_commit_vs_standard_exactness"
    )
    spec_op.add_argument("--steps", type=int, default=512)
    spec_op.add_argument("--num-spec", type=int, default=4)
    spec_op.add_argument("--graph-rows", type=int)
    spec_op.add_argument("--state-slots", type=int, default=64)
    spec_op.add_argument("--slot-stride", type=int, default=3)
    spec_op.add_argument("--k-heads", type=int, default=4)
    spec_op.add_argument("--v-heads", type=int, default=8)
    spec_op.add_argument("--head-k-dim", type=int, default=128)
    spec_op.add_argument("--head-v-dim", type=int, default=128)
    spec_op.add_argument("--conv-width", type=int, default=4)
    spec_op.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    spec_op.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    spec_op.add_argument("--seed", type=int, default=1234)
    spec_op.add_argument("--input-scale", type=float, default=0.02)
    spec_op.add_argument("--gate-scale", type=float, default=0.01)
    spec_op.add_argument("--beta-scale", type=float, default=0.25)
    spec_op.add_argument("--scale", type=float)
    spec_op.add_argument("--fixed-query-len", type=int)
    spec_op.add_argument("--fixed-accepted", type=int)
    spec_op.add_argument("--vary-query-lens", action="store_true")
    spec_op.add_argument("--random-initial-state", action="store_true")
    spec_op.add_argument(
        "--no-l2norm-inputs", dest="l2norm_inputs", action="store_false"
    )
    spec_op.add_argument("--no-silu", dest="silu", action="store_false")
    spec_op.set_defaults(
        func=run_spec_commit_vs_standard_case,
        l2norm_inputs=True,
        silu=True,
    )

    spec_proj = subparsers.add_parser("run-spec-commit-projection")
    spec_proj.add_argument("--out", required=True)
    spec_proj.add_argument("--input-file")
    spec_proj.add_argument(
        "--case-name", default="gdn_spec_commit_projection_exactness"
    )
    spec_proj.add_argument("--steps", type=int, default=512)
    spec_proj.add_argument("--num-spec", type=int, default=4)
    spec_proj.add_argument("--graph-rows", type=int)
    spec_proj.add_argument("--state-slots", type=int, default=64)
    spec_proj.add_argument("--slot-stride", type=int, default=3)
    spec_proj.add_argument("--k-heads", type=int, default=4)
    spec_proj.add_argument("--v-heads", type=int, default=8)
    spec_proj.add_argument("--head-k-dim", type=int, default=128)
    spec_proj.add_argument("--head-v-dim", type=int, default=128)
    spec_proj.add_argument("--hidden-size", type=int)
    spec_proj.add_argument("--vocab-size", type=int, default=1024)
    spec_proj.add_argument("--conv-width", type=int, default=4)
    spec_proj.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    spec_proj.add_argument(
        "--state-dtype",
        choices=("same", "float16", "bfloat16", "float32"),
        default="same",
    )
    spec_proj.add_argument("--seed", type=int, default=1234)
    spec_proj.add_argument("--input-scale", type=float, default=0.02)
    spec_proj.add_argument("--gate-scale", type=float, default=0.01)
    spec_proj.add_argument("--beta-scale", type=float, default=0.25)
    spec_proj.add_argument("--proj-scale", type=float, default=0.02)
    spec_proj.add_argument("--logits-scale", type=float, default=0.02)
    spec_proj.add_argument("--norm-eps", type=float, default=1e-5)
    spec_proj.add_argument(
        "--output-gate", choices=("silu", "swish", "sigmoid"), default="silu"
    )
    spec_proj.add_argument("--scale", type=float)
    spec_proj.add_argument("--fixed-query-len", type=int)
    spec_proj.add_argument("--fixed-accepted", type=int)
    spec_proj.add_argument("--vary-query-lens", action="store_true")
    spec_proj.add_argument("--random-initial-state", action="store_true")
    spec_proj.add_argument("--compile-direct", action="store_true")
    spec_proj.add_argument(
        "--no-compile-fullgraph",
        dest="compile_fullgraph",
        action="store_false",
    )
    spec_proj.add_argument(
        "--no-l2norm-inputs", dest="l2norm_inputs", action="store_false"
    )
    spec_proj.add_argument("--no-silu", dest="silu", action="store_false")
    spec_proj.set_defaults(
        func=run_spec_commit_projection_case,
        l2norm_inputs=True,
        silu=True,
        compile_fullgraph=True,
    )

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
