# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Segmented page4 decode preserves packed output and CUDA graph replay."""

import pytest
import torch

device = torch.device("cuda:0")


def e4m3_bytes(gen, shape, allow_nan=False):
    b = torch.randint(0, 256, shape, generator=gen, dtype=torch.int64)
    if not allow_nan:
        b = torch.where((b & 0x7F) == 0x7F, b ^ 0x01, b)  # no NaN codes (0x7F / 0xFF)
    return b.to(torch.uint8)


def random_case(
    seed, groups, seg_lens, width_pages, order="category", fp16=False, special=()
):
    gen = torch.Generator().manual_seed(seed)
    rows = 8 * groups
    q = (torch.randn(rows, 6, 256, generator=gen) * 0.5).half()
    t2r, req = [], 0
    for g in range(groups):
        lens = seg_lens[g % len(seg_lens)]
        for n in lens:
            t2r += [req if n > 0 else -1] * abs(n)
            req += 1
    t2r = torch.tensor(t2r, dtype=torch.int32)
    assert t2r.numel() == rows
    nblk = groups * width_pages + 8
    if fp16:
        pk = (torch.randn(nblk, 4, 1, 256, generator=gen) * 0.5).half()
        pv = (torch.randn(nblk, 4, 1, 256, generator=gen) * 0.5).half()
    else:
        pk = e4m3_bytes(gen, (nblk, 4, 1, 256)).view(nblk, 4, 1, 256)
        pv = e4m3_bytes(gen, (nblk, 4, 1, 256)).view(nblk, 4, 1, 256)
        pk[0] = 0x7F  # NaN in masked null-block padding.
        pv[0] = 0xFF
    pages = torch.zeros(groups, 4160, dtype=torch.int32)
    masks = torch.zeros(groups, 4160, dtype=torch.int64)
    sl = torch.zeros(groups, dtype=torch.int32)
    for g in range(groups):
        if "empty_group" in special and g == groups - 1:
            continue
        toks = t2r[g * 8 : (g + 1) * 8].tolist()
        segs: list[tuple[int, list[int]]] = []
        for t, r in enumerate(toks):
            if r < 0:
                continue
            if not segs or segs[-1][0] != r:
                segs.append((r, []))
            segs[-1][1].append(t)
        w = width_pages
        ents = []
        for e in range(w):
            owners = (
                [segs[int(torch.randint(0, len(segs), (1,), generator=gen))]]
                if segs
                else []
            )
            if "shared" in special and segs and e % 5 == 0:
                owners = segs
            bm = 0
            for _, qs in owners:
                for t in qs:
                    if "silent_token" in special and t == qs[0]:
                        continue
                    bm |= int(torch.randint(0, 16, (1,), generator=gen)) << (t * 4)
            ents.append((bm, 1 + int(torch.randint(0, nblk - 1, (1,), generator=gen))))
        if "cancel" in special and len(ents) >= 2:
            ents[0] = (
                ents[0][0],
                1,
            )  # Adjacent equal keys with opposite values.
            ents[1] = (ents[0][0], 2)
        if order == "category":

            def cat(bm):
                c = 0
                for t in range(8):
                    if bm & (0xF << (t * 4)):
                        c |= 1 << ((t * 6) // 16)
                        c |= 1 << ((t * 6 + 5) // 16)
                return c

            ents.sort(key=lambda x: cat(x[0]))
        padded = (len(ents) + 7) // 8 * 8
        ents += [(0, 0)] * (padded - len(ents))
        for e, (bm, pid) in enumerate(ents):
            masks[g, e] = bm
            pages[g, e] = pid
        sl[g] = padded * 4
    if "cancel" in special and not fp16:
        # Equal keys with opposite values give an exact-zero contribution.
        pk[2] = pk[1]
        pv[2] = pv[1] ^ 0x80
    if "negzero" in special and not fp16:
        pv[3] = 0x80
    masks32 = (
        torch.where(masks >= 2**31, masks - 2**32, masks)
        .to(torch.int32)
        .view(torch.uint32)
    )
    kvd = "auto" if fp16 else "fp8_e4m3"
    ks = 1.0 if fp16 else float(0.02 + torch.rand(1, generator=gen) * 2)
    vs = 1.0 if fp16 else float(0.02 + torch.rand(1, generator=gen) * 2)
    return (
        q.to(device),
        pk.to(device),
        pv.to(device),
        pages.to(device),
        masks32.to(device),
        sl.to(device),
        kvd,
        ks,
        vs,
        t2r.to(device),
    )


@pytest.mark.parametrize(
    "groups,width", [(1, 8), (3, 40), (5, 520), (8, 520), (3, 4160)]
)
@pytest.mark.parametrize("segments", [[[8]], [[5, 3], [2, 5, 1]], [[1] * 8], [[5, -3]]])
@pytest.mark.parametrize("fp16", [False, True])
def test_segmented_page4_matches_packed(groups, width, segments, fp16):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")
    module = pytest.importorskip("flash_attn_v100_cuda")
    if not hasattr(module, "grouped_sparse_page4_split_fwd"):
        pytest.skip("Requires segmented page4 extension")
    assert module.grouped_sparse_page4_split_abi_version() >= 1
    case = random_case(
        1729,
        groups,
        segments,
        width,
        "category",
        fp16,
        ("shared", "silent_token", "empty_group", "negzero"),
    )
    q, pk, pv, pages, masks, lengths, dtype, ks, vs, mapping = case
    packed, split = torch.empty_like(q), torch.empty_like(q)
    packed_lse = torch.empty(q.shape[:2], dtype=torch.float32, device=device)
    split_lse = torch.empty_like(packed_lse)

    def packed_call():
        module.grouped_sparse_page4_fwd(
            q,
            pk,
            pv,
            packed,
            pages,
            masks,
            lengths,
            packed_lse,
            q.shape[2] ** -0.5,
            dtype,
            ks,
            vs,
        )

    def split_call():
        module.grouped_sparse_page4_split_fwd(
            q,
            pk,
            pv,
            split,
            pages,
            masks,
            lengths,
            split_lse,
            q.shape[2] ** -0.5,
            dtype,
            ks,
            vs,
            mapping,
        )

    packed_call()
    split_call()
    assert torch.equal(packed.view(torch.int16), split.view(torch.int16))
    assert torch.equal(packed_lse.view(torch.int32), split_lse.view(torch.int32))
    expected, expected_lse = split.clone(), split_lse.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        split_call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        split_call()
    split.fill_(float("nan"))
    split_lse.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(expected.view(torch.int16), split.view(torch.int16))
    assert torch.equal(expected_lse.view(torch.int32), split_lse.view(torch.int32))
