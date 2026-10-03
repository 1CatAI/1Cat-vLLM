# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer cost of the resident decode/verify rows inside a mixed batch.

A mixed batch holds one prefill chunk and a few resident requests (plain
decodes or DFlash2 verification spans). The chunk row is identical in every
arm; this measures the small-query rows only, on the route a DFlash2 E4M3
target takes in the batch:

* ``grouped``  - request-major grouped FP32 operator (current route);
* ``scalar``   - the per-token scalar paged decoder the rows used before;
* ``uniform``  - the same rows as a batch without a prefill chunk, which is the
  verification route a pure decode step already takes.

Run on one V100 holding the GPU lock, for example::

    python benchmarks/benchmark_sm70_mixed_decode_rows.py --context 131072
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

HEADS = 6
KV_HEADS = 1
DIM = 256


def _build(context: int, requests: int, verify: int, block_size: int, chunk: int):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = FlashAttnV100Impl(
        num_heads=HEADS,
        head_size=DIM,
        scale=DIM**-0.5,
        num_kv_heads=KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_e4m3",
    )
    q_lens = [chunk] + [verify] * requests
    seq_lens = [chunk + 1000] + [context + i * 17 for i in range(requests)]
    blocks = [(s + block_size - 1) // block_size for s in seq_lens]
    kv = torch.randn(
        2,
        sum(blocks) + 1,
        block_size,
        KV_HEADS,
        DIM,
        dtype=torch.float16,
        device="cuda",
    )
    kv = kv.to(torch.float8_e4m3fn).view(torch.uint8)
    table = torch.zeros(len(seq_lens), max(blocks), dtype=torch.int32)
    nxt = 1
    for row, n in enumerate(blocks):
        table[row, :n] = torch.arange(nxt, nxt + n)
        nxt += n
    qsl = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), 0)), dtype=torch.int32
    )
    seq_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    query = torch.randn(int(qsl[-1]), HEADS, DIM, dtype=torch.float16, device="cuda")

    def metadata(rows_only: bool):
        sel = slice(1, None) if rows_only else slice(None)
        qs = qsl if not rows_only else (qsl[1:] - qsl[1])
        return SimpleNamespace(
            query_start_loc=qs.to("cuda"),
            query_start_loc_cpu=qs,
            seq_lens=seq_cpu[sel].to("cuda"),
            seq_lens_cpu=seq_cpu[sel],
            block_table=table[sel].to("cuda"),
            num_actual_tokens=int(qs[-1]),
            max_query_len=max(q_lens[1:] if rows_only else q_lens),
            causal=True,
            max_model_len=262144,
            is_dflash_selector_target=True,
        )

    return impl, query, kv, qsl, metadata


def _time(fn, iterations: int) -> float:
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iterations


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--verify", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=1648)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    impl, query, kv, qsl, make_metadata = _build(
        args.context, args.requests, args.verify, args.block_size, args.chunk
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    k = kv[0]
    window = (-1, -1)

    def rows(metadata, query_rows, query_start_loc, forced_scalar=False):
        out = torch.empty_like(query_rows)
        impl_forced = impl
        original = impl.__class__._run_mixed_rows_grouped_e4m3
        if forced_scalar:
            impl.__class__._run_mixed_rows_grouped_e4m3 = lambda *a, **kw: False
        try:
            # A fresh metadata object per call: the per-step plan is built by
            # the first layer of a step, so include that cost once per call.
            def call():
                metadata_step = SimpleNamespace(**vars(metadata))
                impl_forced._run_prefill_prefix_decode_rows(
                    layer,
                    query_rows,
                    k,
                    kv[1],
                    metadata_step,
                    out,
                    query_start_loc,
                    metadata.seq_lens_cpu,
                    window,
                )

            return _time(call, args.iterations)
        finally:
            impl.__class__._run_mixed_rows_grouped_e4m3 = original

    mixed = make_metadata(False)
    grouped_ms = rows(mixed, query, qsl)
    scalar_ms = rows(mixed, query, qsl, forced_scalar=True)

    uniform = make_metadata(True)
    rows_query = query[args.chunk :]
    uniform_ms = _time(
        lambda: impl._flash_v100_small_query_prefill_as_decode(
            layer,
            rows_query,
            k,
            kv[1],
            uniform,
            torch.empty_like(rows_query),
            uniform.query_start_loc_cpu,
            uniform.seq_lens_cpu,
        ),
        args.iterations,
    )
    print(
        f"context={args.context} requests={args.requests} verify={args.verify} "
        f"page={args.block_size}\n"
        f"  mixed rows, grouped FP32 : {grouped_ms:8.3f} ms/layer\n"
        f"  mixed rows, scalar       : {scalar_ms:8.3f} ms/layer "
        f"({scalar_ms / grouped_ms:.1f}x)\n"
        f"  uniform verify batch     : {uniform_ms:8.3f} ms/layer"
    )


if __name__ == "__main__":
    main()
