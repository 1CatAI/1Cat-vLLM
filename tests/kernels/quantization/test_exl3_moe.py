# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 routed-expert MoE: the pure-torch reference decoder against exllamav3,
and the SM70 kernels against that decoder.

Run: pytest tests/kernels/quantization/test_exl3_moe.py -v
"""

import hashlib

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.exl3 import (
    MCG_MARKER,
    Exl3Config,
    exl3_decode_tiles,
    exl3_group_slots,
    exl3_moe_forward,
    exl3_reconstruct,
    tile_perm,
)

requires_kernels = pytest.mark.skipif(
    not torch.cuda.is_available() or not ops.exl3_moe_available(),
    reason="EXL3 MoE kernels are built for compute capability 7.0",
)


def _strip_words(K: int) -> torch.Tensor:
    """A 16 x 128 strip of trellis words from an integer hash (no RNG, so the
    golden values below do not depend on the torch version)."""
    n = 8 * 16 * K
    v = [((i * 2654435761 + K * 40503) >> 9) & 0xFFFF for i in range(n)]
    v = [x - 65536 if x >= 32768 else x for x in v]
    return torch.tensor(v, dtype=torch.int16).view(1, 8, 16 * K)


# sha256 of exllamav3's ext.reconstruct(w, trellis, K, mcg=True, mul1=False)
# for _strip_words(K): the fp16 [16, 128] tile-basis weight, sm80 tile order.
EXLLAMAV3_RECONSTRUCT_SHA256 = {
    2: "e45f494665e37c04ff3b8f9672471f0fb00126cac4f020bfd7f84aa06fb5de72",
    3: "cda4e8323139c8702a8465a813d161f54037922d759d8d5f5cd0e10afe62dd85",
    4: "14fbf140e51b3105515eaf58d240df57bfd628d97a13fbcd3214737871c30437",
}


@pytest.mark.parametrize("K", [2, 3, 4])
def test_decoder_matches_exllamav3(K):
    w = exl3_decode_tiles(_strip_words(K), "mcg", "sm80")
    digest = hashlib.sha256(w.numpy().tobytes()).hexdigest()
    assert digest == EXLLAMAV3_RECONSTRUCT_SHA256[K]


def test_tile_orders():
    for order in ("sm80", "colmajor"):
        perm = tile_perm(order)
        assert torch.equal(perm.sort().values, torch.arange(256))
    p = torch.arange(256)
    assert torch.equal(tile_perm("colmajor"), (p % 16) * 16 + p // 16)
    assert torch.equal(tile_perm("sm70_colmajor"), tile_perm("colmajor"))
    # The same stream decodes to the same values in both orders, at the
    # elements each order assigns to the positions.
    words = _strip_words(2)
    a = exl3_decode_tiles(words, "mcg", "sm80").view(16, 8, 16).permute(1, 0, 2)
    b = exl3_decode_tiles(words, "mcg", "colmajor").view(16, 8, 16).permute(1, 0, 2)
    a, b = a.reshape(8, 256), b.reshape(8, 256)
    assert torch.equal(a[:, tile_perm("sm80")], b[:, tile_perm("colmajor")])


@pytest.mark.parametrize("T,topk,E", [(1, 8, 16), (70, 8, 16), (300, 8, 288)])
def test_group_slots(T, topk, E):
    torch.manual_seed(0)
    ids = torch.stack([torch.randperm(E)[:topk] for _ in range(T)]).int()
    gexp, rows = exl3_group_slots(ids, E)
    flat = ids.view(-1)
    used = rows[rows >= 0]
    assert torch.equal(used.sort().values, torch.arange(T * topk, dtype=torch.int32))
    for g in range(rows.shape[0]):
        slots = rows[g][rows[g] >= 0].long()
        if slots.numel():
            assert rows[g, 0] >= 0  # filled from the front
            assert bool((flat[slots] == gexp[g]).all())


def _make_layer(E, H, inter, K, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def trellis(*shape):
        return torch.randint(-32768, 32768, shape, dtype=torch.int16, generator=g)

    def signs(*shape):
        s = torch.randint(0, 2, shape, generator=g).half() * 2 - 1
        return s * (0.5 + torch.rand(shape, generator=g).half())

    w = dict(
        w13_trellis=trellis(2, E, H // 16, inter // 16, 16 * K),
        w13_suh=signs(2, E, H) * 0.05,
        w13_svh=signs(2, E, inter) * 0.05,
        w2_trellis=trellis(E, inter // 16, H // 16, 16 * K),
        w2_suh=signs(E, inter) * 0.05,
        w2_svh=signs(E, H) * 0.05,
    )
    return {k: v.to(device) for k, v in w.items()}


def _oracle(x, ids, wts, layer, order):
    """sum_j w * (silu(x Wg) * (x Wu)) Wd with weights from the decoder."""
    E = layer["w2_trellis"].shape[0]
    W = {}
    for e in ids.unique().tolist():
        W[e] = [
            exl3_reconstruct(
                layer["w13_trellis"][p, e],
                layer["w13_suh"][p, e],
                layer["w13_svh"][p, e],
                "mcg",
                order,
            ).double()
            for p in (0, 1)
        ] + [
            exl3_reconstruct(
                layer["w2_trellis"][e],
                layer["w2_suh"][e],
                layer["w2_svh"][e],
                "mcg",
                order,
            ).double()
        ]
    assert len(W) <= E
    out = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    for t in range(x.shape[0]):
        xt = x[t].double()
        for j in range(ids.shape[1]):
            wg, wu, wd = W[ids[t, j].item()]
            out[t] += wts[t, j].double() * ((F.silu(xt @ wg) * (xt @ wu)) @ wd)
    return out


@requires_kernels
@pytest.mark.parametrize("order", ["sm80", "colmajor"])
@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("T", [1, 5, 70, 300])
def test_exl3_moe_kernel(order, K, T):
    device = torch.device("cuda")
    E, H, inter, topk = 16, 512, 512, 8
    layer = _make_layer(E, H, inter, K, device, seed=K)
    torch.manual_seed(T)
    x = (torch.randn(T, H, device=device) * 0.5).half()
    ids = torch.stack([torch.randperm(E, device=device)[:topk] for _ in range(T)]).int()
    wts = torch.rand(T, topk, device=device)
    y = exl3_moe_forward(x, ids, wts, **layer, colmajor=order == "colmajor")
    # Prefill shapes are checked on a sample of tokens (the oracle is slow).
    rows = list(range(T)) if T <= 5 else [0, 1, T // 2, T - 2, T - 1]
    ref = _oracle(x[rows], ids[rows], wts[rows], layer, order)
    err = ((y[rows].double() - ref).norm() / ref.norm()).item()
    assert torch.isfinite(y).all()
    assert err < 2e-3, err


@requires_kernels
def test_exl3_moe_cuda_graph():
    device = torch.device("cuda")
    E, H, inter, topk, T = 16, 512, 512, 8, 4
    layer = _make_layer(E, H, inter, 2, device)
    x = torch.zeros(T, H, device=device, dtype=torch.half)
    ids = torch.zeros(T, topk, device=device, dtype=torch.int32)
    wts = torch.zeros(T, topk, device=device)
    exl3_moe_forward(x, ids, wts, **layer, colmajor=True)  # warm up
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = exl3_moe_forward(x, ids, wts, **layer, colmajor=True)
    torch.manual_seed(1)
    x.copy_((torch.randn(T, H, device=device) * 0.5).half())
    ids.copy_(torch.stack([torch.randperm(E, device=device)[:topk] for _ in range(T)]))
    wts.copy_(torch.rand(T, topk, device=device))
    graph.replay()
    eager = exl3_moe_forward(x, ids, wts, **layer, colmajor=True)
    torch.testing.assert_close(y, eager, rtol=0, atol=0)


def _write_checkpoint(path, tensors):
    from safetensors.torch import save_file

    save_file(tensors, str(path / "model.safetensors"))


def _expert_tensors(layer, K, E=2, mcg=True):
    out = {}
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            p = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
            out[p + ".trellis"] = torch.zeros(32, 8, 16 * K, dtype=torch.int16)
            out[p + ".suh"] = torch.zeros(512, dtype=torch.half)
            out[p + ".svh"] = torch.zeros(128, dtype=torch.half)
            if mcg:
                out[p + ".mcg"] = torch.tensor(MCG_MARKER, dtype=torch.int32)
            else:
                out[p + ".mul1"] = torch.tensor(1, dtype=torch.int32)
    return out


def test_config_reads_expert_bits(tmp_path):
    _write_checkpoint(tmp_path, {**_expert_tensors(3, 2), **_expert_tensors(45, 4)})
    cfg = Exl3Config.from_config(
        {
            "quant_method": "exl3",
            "bits": 2,
            "codebook": "mcg",
            "tile_order": "sm70_colmajor",
        }
    )
    cfg.maybe_update_config(str(tmp_path))
    assert cfg.expert_bits == {3: 2, 45: 4}
    assert cfg.tile_order == "colmajor"


def test_config_rejects_unsupported_experts(tmp_path):
    _write_checkpoint(tmp_path, _expert_tensors(3, 2, mcg=False))
    cfg = Exl3Config.from_config({"quant_method": "exl3", "codebook": "mul1"})
    with pytest.raises(ValueError, match="mcg codebook"):
        cfg.maybe_update_config(str(tmp_path))
    mixed = {**_expert_tensors(3, 2)}
    mixed["model.language_model.layers.3.mlp.experts.1.down_proj.trellis"] = (
        torch.zeros(32, 8, 48, dtype=torch.int16)
    )
    _write_checkpoint(tmp_path, mixed)
    with pytest.raises(ValueError, match="mixed bit widths"):
        Exl3Config.from_config({"codebook": "mcg"}).maybe_update_config(str(tmp_path))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("order", ["sm80", "colmajor"])
def test_dequantize_dense(order):
    torch.manual_seed(0)
    k, n, K = 256, 384, 4
    trellis = torch.randint(
        -32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int16
    )
    suh, svh = torch.randn(k).half(), torch.randn(n).half()
    cfg = Exl3Config(codebook="mcg", tile_order=order)
    p = "model.layers.45.self_attn.o_proj"
    expert = "model.layers.45.mlp.experts.0.gate_proj.trellis"
    weights = [
        (p + ".trellis", trellis),
        (p + ".suh", suh),
        (p + ".svh", svh),
        (p + ".mcg", torch.tensor(MCG_MARKER, dtype=torch.int32)),
        ("model.norm.weight", torch.ones(4)),
        (expert, trellis),
    ]
    out = dict(cfg.dequantize_dense(iter(weights)))
    assert set(out) == {"model.norm.weight", expert, p + ".weight"}
    ref = exl3_reconstruct(trellis.cuda(), suh.cuda(), svh.cuda(), "mcg", order)
    torch.testing.assert_close(out[p + ".weight"], ref.T.half().cpu())
