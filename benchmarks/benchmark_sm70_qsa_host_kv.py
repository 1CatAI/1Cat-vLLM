# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident versus host-backed QSA write/attention graphs at M5/M20.

Selections are synthetic, shared by each request's five verification queries.
Run under the shared GPU lock. This measures one layer, not a model round.
"""

import argparse
import json
from pathlib import Path

import torch

import vllm
from vllm import _custom_ops as ops
from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import host_qsa_attention
from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention


def captured(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    return graph


def timing(graph, iterations):
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / iterations


def point(context, rows, hot_tokens, iterations):
    requests = rows // 5
    page = 1568
    per_request = (context + page - 1) // page
    blocks = per_request * requests
    device = torch.device("cuda:0")
    state = HostQSAKV(blocks, page, 256, device, hot_tokens=hot_tokens, rows=rows)
    cache = torch.randn((blocks, 2, page, 1, 256), device=device, dtype=torch.float16)
    key, value = cache.unbind(1)
    key_rows, value_rows = (
        key.contiguous().view(-1, 1, 256),
        value.contiguous().view(-1, 1, 256),
    )
    state.write(key_rows, value_rows, torch.arange(blocks * page, device=device))
    table = torch.arange(blocks, dtype=torch.int32, device=device).view(requests, -1)
    owners = torch.arange(rows, dtype=torch.int32, device=device) // 5
    lengths = torch.full((requests,), context, dtype=torch.int32, device=device)
    positions = torch.full((rows,), context - 1, dtype=torch.int64, device=device)
    # Exactly 512 native four-token pages, with boundary positions appended.
    count = min(512, context // 4)
    pages = torch.linspace(0, context // 4 - 1, count, device=device).long()
    selected = (pages[:, None] * 4 + torch.arange(4, device=device)).flatten()
    selected = torch.cat(
        (selected, torch.full((2051 - selected.numel(),), -1, device=device))
    )
    indices = selected.to(torch.int32).repeat(rows, 1)
    query = torch.randn((rows, 6, 256), dtype=torch.float16, device=device)
    output_resident, output_host = torch.empty_like(query), torch.empty_like(query)
    output_direct = torch.empty_like(query)
    slots = (
        owners.long() * per_request * page
        + context
        - 5
        + torch.arange(rows, device=device) % 5
    )
    new_key = key_rows.index_select(0, slots)
    new_value = value_rows.index_select(0, slots)
    scale = torch.ones((), device=device)

    def resident():
        ops.reshape_and_cache_flash(
            new_key, new_value, key, value, slots, "auto", scale, scale
        )
        qsa_sparse_paged_attention(
            query,
            key,
            value,
            indices,
            table,
            owners,
            output_resident,
            query_positions=positions,
            sequence_lengths=lengths,
        )

    def host():
        state.write(new_key, new_value, slots)
        k, v, remapped = state.gather(indices, table, owners, positions, lengths)
        qsa_sparse_paged_attention(
            query,
            k,
            v,
            remapped,
            state.table,
            state.requests,
            output_host,
            query_positions=state.positions,
            sequence_lengths=state.lengths,
        )

    def direct():
        state.write(new_key, new_value, slots)
        host_qsa_attention(
            query, state, indices, table, owners, positions, lengths, output_direct
        )

    a, b, c = captured(resident), captured(host), captured(direct)
    a.replay()
    b.replay()
    c.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output_direct, output_host, rtol=0, atol=0)
    if (
        not torch.isfinite(output_resident).all()
        or not torch.isfinite(output_host).all()
    ):
        raise RuntimeError(
            "Nonfinite attention: "
            f"resident={torch.isfinite(output_resident).all().item()}, "
            f"host={torch.isfinite(output_host).all().item()}"
        )
    physical_tokens = owners[
        :, None
    ].long() * per_request * page + indices.long().clamp_min(0)
    rk = key_rows[physical_tokens, 0].float()
    rv = value_rows[physical_tokens, 0].float()
    hk = state.staging[:, 0, :2051, 0].float()
    hv = state.staging[:, 1, :2051, 0].float()
    mask = indices[:, None, :] >= 0

    def oracle(k, v):
        scores = torch.einsum("mhd,mkd->mhk", query.float(), k) / 16
        probabilities = scores.masked_fill(~mask, -float("inf")).softmax(-1)
        return torch.einsum("mhk,mkd->mhd", probabilities, v)

    ro, ho = oracle(rk, rv), oracle(hk, hv)
    diagnostics = dict(
        resident_oracle_l2=((output_resident.float() - ro).norm() / ro.norm()).item(),
        host_oracle_l2=((output_host.float() - ho).norm() / ho.norm()).item(),
        quant_oracle_l2=((ho - ro).norm() / ro.norm()).item(),
        key_l2=(
            (hk - rk).masked_fill(~mask.transpose(1, 2), 0).norm() / rk.norm()
        ).item(),
    )
    physical_cpu = physical_tokens.cpu()
    ck = state.host[:, 0, :, 0].reshape(-1, 256)[physical_cpu]
    cs = state.host_scales[physical_cpu, 0]
    expected = (
        (ck.view(torch.float8_e4m3fn).float() * cs.unsqueeze(-1)).half().to(device)
    )
    valid_key = mask.transpose(1, 2)
    stage_diff = (hk - expected.float()).masked_fill(~valid_key, 0)
    diagnostics["stage_reference_max_abs"] = stage_diff.abs().max().item()
    diagnostics["stage_reference_nonzero"] = torch.count_nonzero(stage_diff).item()
    diagnostics["fp8_only_key_l2"] = (
        (expected.float() - rk).masked_fill(~valid_key, 0).norm() / rk.norm()
    ).item()
    error = (output_host.float() - output_resident.float()).norm().item()
    error /= output_resident.float().norm().item()
    measurements = []
    previous = state.stats.clone()
    for _ in range(3):
        measurements.append([timing(g, iterations) for g in (a, b, c, c, b, a)])
    delta = (state.stats - previous).cpu().tolist()
    resident_us = sum(v[0] + v[5] for v in measurements) / 6
    host_us = sum(v[1] + v[4] for v in measurements) / 6
    direct_us = sum(v[2] + v[3] for v in measurements) / 6
    return dict(
        context=context,
        rows=rows,
        hot_tokens=hot_tokens,
        selection=f"{count} fixed pages per request; five shared queries",
        resident_us=resident_us,
        host_us=host_us,
        direct_us=direct_us,
        direct_stage_exactly_equal=True,
        relative_output_l2=error,
        counters=delta,
        hit_rate=delta[0] / max(delta[0] + delta[1], 1),
        host_bytes=state.history.nbytes + state.host_scales.nbytes,
        device_hot_bytes=sum(
            t.nbytes
            for t in (
                state.hot_values,
                state.tags,
                state.stamps,
                state.locks,
                state.hands,
                state.epoch,
                state._stats,
            )
        ),
        shared_staging_bytes=state.staging.untyped_storage().nbytes(),
        epochs_abc_cba_us=measurements,
        diagnostics=diagnostics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--rows", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--hot-tokens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    torch.manual_seed(20261007)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    if "site-packages" not in vllm.__file__:
        raise RuntimeError("Use the packaged runtime")
    report = dict(
        version=vllm.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        scope="synthetic single-layer graph",
        points=[],
    )
    for context in args.contexts:
        for rows in args.rows:
            if rows not in (5, 20):
                raise ValueError("Use M5 or M20 verification shapes")
            for hot in args.hot_tokens:
                result = point(context, rows, hot, args.iterations)
                report["points"].append(result)
                print(json.dumps(result), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
