# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-free DCP1/DCP2 QSA arithmetic comparison, not a text-quality gate.

Simulate two cache owners on one GPU with identical quantized K/V and queries.
Compare the actual G6/page4 route, unsharded generic G12, and sharded G12 with
an FP64 oracle. No model weights, private extensions, or NCCL are required.
"""

import argparse
import json
import math

import torch
from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    _qsa_output_gate,
    qsa_sparse_paged_attention,
)


def error(actual, reference):
    delta = actual.double() - reference.double()
    return {
        "max_abs": delta.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / reference.double().norm()).item(),
        "equal_fraction": (actual == reference).double().mean().item(),
    }


@torch.inference_mode()
def run(rows, seed):
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    dim, heads, page, width = 256, 12, 1568, 2051
    length = 97 if rows == 1 else rows
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.float16)
    gate = torch.randn_like(q)
    # First target QSA layer's published calibration scales (non-power-of-two).
    k_scale, v_scale = 0.02015904150903225, 0.01712472178041935

    def quantized_values(scale):
        return (
            (torch.randn(page, 1, dim, device="cuda") / scale)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )

    k, v = quantized_values(k_scale), quantized_values(v_scale)
    # Page 0 is a null page, as in the real allocator. Page 1 stores this request.
    cache = torch.zeros(2, 2, page, 1, dim, device="cuda", dtype=torch.uint8)
    cache[1, 0], cache[1, 1] = k, v
    table = torch.tensor([[1]], device="cuda", dtype=torch.int32)
    req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    positions = (
        torch.full((1,), length - 1, device="cuda", dtype=torch.int64)
        if rows == 1
        else torch.arange(rows, device="cuda", dtype=torch.int64)
    )
    lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
    columns = torch.arange(width, device="cuda", dtype=torch.int32)
    ids = torch.where(columns[None] <= positions[:, None], columns[None], -1)
    baseline = torch.empty_like(q)
    generic_g6 = torch.empty_like(q)
    for first in (0, 6):
        qs = q[:, first : first + 6].contiguous()
        gs = gate[:, first : first + 6].contiguous()
        kwargs = dict(kv_cache_dtype="fp8_e4m3", k_scale=k_scale, v_scale=v_scale)
        baseline[:, first : first + 6] = qsa_sparse_paged_attention(
            qs,
            cache[:, 0],
            cache[:, 1],
            ids,
            table,
            req,
            output_gate=gs,
            query_positions=positions,
            sequence_lengths=lengths,
            **kwargs,
        )
        generic_g6[:, first : first + 6] = qsa_sparse_paged_attention(
            qs, cache[:, 0], cache[:, 1], ids, table, req, output_gate=gs, **kwargs
        )

    def partial(keys, values, selected):
        out = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty((rows, heads), device="cuda", dtype=torch.float32)
        qsa_sparse_paged_attention(
            q,
            keys,
            values,
            selected,
            table,
            req,
            out=out,
            lse=lse,
            kv_cache_dtype="fp8_e4m3",
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return out, lse

    unsharded, _ = partial(cache[:, 0], cache[:, 1], ids)
    outputs, lses = [], []
    for rank in range(2):
        local = torch.zeros(2, 2, page // 2, 1, dim, device="cuda", dtype=k.dtype)
        local[1, 0], local[1, 1] = k[rank::2], v[rank::2]
        selected = torch.empty_like(ids)
        qsa_localize_dcp_indices(
            ids,
            selected,
            dcp_world_size=2,
            dcp_rank=rank,
            interleave_size=1,
            local_block_size=page // 2,
        )
        out, lse = partial(local[:, 0], local[:, 1], selected[:, :1026])
        outputs.append(out)
        lses.append(lse)
    lses = torch.stack(lses)
    weights = torch.softmax(lses * math.log(2), dim=0).nan_to_num()
    sharded = (torch.stack(outputs) * weights[..., None]).sum(0)
    dk = k[:length, 0].view(torch.float8_e4m3fn).double() * k_scale
    dv = v[:length, 0].view(torch.float8_e4m3fn).double() * v_scale
    scores = torch.einsum("rhd,kd->rhk", q.double(), dk) / math.sqrt(dim)
    scores.masked_fill_(
        torch.arange(length, device="cuda")[None, None] > positions[:, None, None],
        -torch.inf,
    )
    oracle = torch.einsum("rhk,kd->rhd", scores.softmax(-1), dv)

    def gated(value):
        result = value.half()
        _qsa_output_gate(result, gate)
        return result

    reference = gated(oracle)
    candidates = {
        "dcp1_runtime": baseline,
        "dcp1_generic_g6": generic_g6,
        "unsharded_g12": gated(unsharded),
        "dcp2_sharded_g12": gated(sharded),
    }
    assert all(torch.isfinite(value).all() for value in candidates.values())
    return {
        "rows": rows,
        "seed": seed,
        "vs_fp64_oracle": {
            name: error(value, reference) for name, value in candidates.items()
        },
        "dcp2_vs_dcp1": error(candidates["dcp2_sharded_g12"], baseline),
        "generic_vs_runtime": error(generic_g6, baseline),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    for rows in (1, 69):
        for seed in args.seeds:
            print(json.dumps(run(rows, seed)), flush=True)
