# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU check of v41pidx against today's prefill index path (one GPU, no timing).

1. Per head-sum order: ``v41pidx.index_head_sum`` against the per-rank partial
   sums of ``select_compressed_positions`` (``torch.mm`` -> ``relu_`` ->
   ``mul_`` -> ``sum(1)``), bit for bit over the live columns. Prints the
   orders that match; VLLM_V41_INDEX_FUSE may only name one of those.
2. For every matching order: ``index_fuse.select`` (the real scorer and
   the real ``index_topk``) against ``select_compressed_positions``, with an
   identity all-reduce (one rank), over several shapes and block budgets: the
   selected index rows must be equal.

    python check.py /path/to/v41pidx.so
Exit status 0 only when some order matches and every select comparison is
equal.
"""

import sys

import torch

from vllm.models.deepseek_v41 import attention_ops, index_fuse


def today_partial(q, w, keys):
    rows, heads = q.shape[:2]
    s = torch.mm(q.reshape(-1, 128), keys.T).view(rows, heads, -1)
    s.relu_().mul_(w[:, :, None])
    return s.sum(1)


def inputs(tokens, width, live, ratio, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(tokens, 8, 128, device="cuda", generator=g) * 0.3).half()
    w = (torch.rand(tokens, 8, device="cuda", generator=g) * 0.5).half()
    keys = (torch.randn(width, 128, device="cuda", generator=g) * 0.3).half()
    positions = torch.randint(
        ratio, live * ratio, (tokens,), device="cuda", generator=g
    )
    valid = torch.ones(tokens, dtype=torch.bool, device="cuda")
    valid[::17] = False
    return q, w, keys, positions, valid


def main(lib):
    index_fuse.load_library(lib)
    matching = []
    for name, order in index_fuse.ORDERS.items():
        bad = 0
        for tokens, width, live, seed in ((64, 4096, 4000, 1), (200, 9000, 8800, 2)):
            q, w, keys, _, _ = inputs(tokens, width, live, 4, seed)
            out = torch.empty(tokens, width, dtype=torch.float16, device="cuda")
            torch.ops.v41pidx.index_head_sum(out, q, w, keys, order, live)
            want = today_partial(q, w, keys)
            bad += int(
                (
                    out[:, :live].view(torch.int16) != want[:, :live].view(torch.int16)
                ).sum()
            )
        print(f"order {name}: {bad} head-sum mismatches")
        if bad == 0:
            matching.append(name)
    if not matching:
        print("FAIL: no order matches today's head sums")
        return 1
    from vllm.models.deepseek_v41.index_topk import index_topk

    failed = 0
    for name in matching:
        scorer = index_fuse.cuda_scorer(index_fuse.ORDERS[name])
        for tokens, width, live, ratio, tiles in (
            (300, 4096, 4000, 4, 1),
            (1500, 2048, 1900, 4, 2),
            (513, 9000, 8800, 4, 1),
            (2000, 9000, 8800, 4, 2),
            (2000, 9000, 8800, 4, 5),
        ):
            q, w, keys, positions, valid = inputs(tokens, width, live, ratio, tokens)
            tile = index_fuse.today_tile(attention_ops._INDEX_TILE_BYTES, 8, width)
            want = attention_ops.select_compressed_positions(
                q, w, keys, positions, valid, ratio, lambda x: x, top_k=512
            )
            got = index_fuse.select(
                q,
                w,
                keys,
                live,
                positions,
                valid,
                ratio,
                lambda x: x,
                top_k=512,
                tile=tile,
                budget=tile * width * 2 * tiles,
                scorer=scorer,
                index_topk=index_topk,
            )
            torch.cuda.synchronize()
            diff = int((got != want).any(dim=1).sum())
            print(
                f"select {name} tokens={tokens} keys={width} live={live} "
                f"tile={tile} block_tiles={tiles}: {diff} rows differ"
            )
            failed += diff != 0
    print("PASS" if failed == 0 else f"FAIL: {failed} select cases differ")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
