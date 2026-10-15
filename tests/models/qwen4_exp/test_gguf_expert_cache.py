# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The routed-expert cache always serves the original expert bytes."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import gguf_expert_cache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="expert cache needs CUDA"
)

EXPERTS = 128
SHAPES = {"w1": (8, 32), "w3": (8, 32), "w2": (16, 16)}
LAYERS = (3, 4, 5)


def _shape(layer, shard):
    # Layers may quantize differently, so bank shapes differ per layer.
    rows, width = SHAPES[shard]
    return (rows, width + 16) if layer == 5 and shard != "w2" else (rows, width)


def _cache(fraction=0.75, spare=4, slack=0.1, experts=EXPERTS, decode_rows=20):
    device = torch.device("cuda", torch.accelerator.current_device_index())
    generator = torch.Generator().manual_seed(0)
    original = {
        (layer, shard): torch.randint(
            0,
            256,
            (experts, *_shape(layer, shard)),
            dtype=torch.uint8,
            generator=generator,
        )
        for layer in LAYERS
        for shard in SHAPES
    }
    banks = [
        (
            layer,
            shard,
            (experts, *_shape(layer, shard)),
            16 if shard == "w2" else 0,
            (0, 0, 0),
        )
        for layer in LAYERS
        for shard in SHAPES
    ]
    config = SimpleNamespace(
        expert_cache_fraction=fraction,
        expert_cache_spare_slots=spare,
        expert_cache_host_slack=slack,
        expert_cache_profile=None,
        expert_cache_prefill="mmq",
        expert_cache_prefill_tokens=4096,
        grouped_gate_up_max_m=decode_rows,
    )
    cache = gguf_expert_cache.create_cache(banks, experts, config, device)
    for (layer, shard), bank in original.items():
        for expert in range(experts):
            cache.store(layer, shard, expert, bank[expert].to(device))
    return cache, {k: v.to(device) for k, v in original.items()}


def _check_decode(cache, original, layer, ids, slots):
    for shard in SHAPES:
        got = cache.bank(layer, shard)[slots.long()]
        torch.testing.assert_close(
            got, original[layer, shard][ids.long()], rtol=0, atol=0
        )


def _random_ids(generator, tokens, hot=None):
    pool = hot if hot is not None else torch.arange(EXPERTS)
    picks = torch.stack(
        [
            pool[torch.randperm(len(pool), generator=generator)[:10]]
            for _ in range(tokens)
        ]
    )
    return picks.to(torch.int32).cuda().contiguous()


@pytest.mark.parametrize("slack", [0.0, 0.25])
def test_decode_resolve_serves_original_experts(slack):
    cache, original = _cache(slack=slack)
    generator = torch.Generator().manual_seed(1)
    for step in range(200):
        layer = LAYERS[step % len(LAYERS)]
        ids = _random_ids(generator, 5)
        slots = cache.resolve(layer, ids)
        _check_decode(cache, original, layer, ids, slots)
    torch.accelerator.synchronize()
    stats = cache.stats()
    assert stats["errors"] == 0 and stats["loaded"] > 0 and stats["evicted"] > 0
    if slack == 0.0:
        assert stats["writebacks"] > 0


def test_resolve_inside_cuda_graph():
    cache, original = _cache()
    generator = torch.Generator().manual_seed(2)
    static = _random_ids(generator, 5)
    cache.resolve(LAYERS[0], static)  # eager first: starts the lane
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        outs = [cache.resolve(layer, static) for layer in LAYERS]
    for _ in range(100):
        static.copy_(_random_ids(generator, 5))
        graph.replay()
        torch.accelerator.synchronize()
        for layer, slots in zip(LAYERS, outs):
            _check_decode(cache, original, layer, static, slots)


@pytest.mark.parametrize("spare", [4, 12])
@pytest.mark.parametrize("rows", [1, 5, 9, 10, 20, 21, 30, 39, 40])
def test_forty_row_cache_lease_changed_graph_preserves_bytes(spare, rows):
    # C8 MTP4 has 400 routes. Force evictions and host write-backs while a
    # single lease protects every route, including ring/plan capacity edges.
    cache, original = _cache(
        fraction=0.82, spare=spare, slack=0.0, experts=512, decode_rows=40
    )
    assert cache.ring.shape[1] == 512
    assert cache.max_ops >= 504 + spare
    ids = torch.arange(rows * 10, dtype=torch.int32, device="cuda").view(rows, 10)
    try:
        cache.resolve(LAYERS[0], ids)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            outs = [cache.resolve(layer, ids) for layer in LAYERS]
        for shift in (0, 113, 257, 511, 32):
            ids.copy_(
                (torch.arange(rows * 10, device="cuda").view_as(ids) + shift) % 512
            )
            # Residency is deliberately shuffled at initialization. Ensure
            # even the one-row graph exercises a miss and spare restoration.
            cold = torch.nonzero(cache.table[0] < 0).flatten()
            ids.view(-1)[0].copy_(cold[0])
            graph.replay()
            torch.accelerator.synchronize()
            for layer, slots in zip(LAYERS, outs):
                _check_decode(cache, original, layer, ids, slots)
        assert cache.stats()["errors"] == 0
        assert cache.stats()["evicted"] > 0
        assert cache.stats()["writebacks"] > 0
    finally:
        torch.accelerator.synchronize()
        cache.close()


def test_prefill_bank_matches_and_interleaves_with_decode():
    cache, original = _cache()
    generator = torch.Generator().manual_seed(3)
    for step in range(20):
        for layer in LAYERS:
            ids = _random_ids(generator, 300)
            w1, w3, w2, mapped = cache.prefill(layer, ids)
            for shard, bank in zip(("w1", "w3", "w2"), (w1, w3, w2)):
                torch.testing.assert_close(
                    bank[mapped.long()],
                    original[layer, shard][ids.long()],
                    rtol=0,
                    atol=0,
                )
            cache.prefill_done(layer)
        for layer in LAYERS:
            ids = _random_ids(generator, 5)
            _check_decode(cache, original, layer, ids, cache.resolve(layer, ids))
    torch.accelerator.synchronize()
    assert cache.stats()["errors"] == 0


@pytest.mark.parametrize("slack", [0.0, 0.25])
def test_cold_stage_uses_snapshot_and_preserves_slot_offsets(slack):
    cache, original = _cache(slack=slack)
    generator = torch.Generator().manual_seed(9683)
    try:
        # Fill holes and exercise write-backs before taking the prefill snapshot.
        for _ in range(10):
            for layer in LAYERS:
                cache.resolve(layer, _random_ids(generator, 5))
        torch.accelerator.synchronize()
        stream = torch.cuda.current_stream().cuda_stream
        for i, layer in enumerate(LAYERS):
            snapshot = torch.empty(2, EXPERTS, dtype=torch.int32, device="cuda")
            cache.lane.residency(i, snapshot.data_ptr(), stream)
            torch.accelerator.synchronize()
            gpu_slots, host_slots = snapshot.cpu()
            cold = torch.nonzero(gpu_slots < 0).flatten()
            selected = host_slots[cold].long()
            # Source selection must use the saved snapshot, not a later CPU map.
            saved = cache.slot_of[i].clone()
            cache.slot_of[i].fill_(0)
            try:
                for s, shard in enumerate(gguf_expert_cache.SHARDS):
                    stage = torch.full_like(
                        cache.host[layer, shard], 0xA5, device="cuda"
                    )
                    cache.lane.stage_cold(i, s, stage.data_ptr(), stream)
                    torch.testing.assert_close(
                        stage[selected], original[layer, shard][cold], rtol=0, atol=0
                    )
                    holes = torch.ones(cache.host_slots, dtype=torch.bool)
                    holes[selected] = False
                    assert (stage[holes] == 0xA5).all()
            finally:
                cache.slot_of[i].copy_(saved)
    finally:
        torch.accelerator.synchronize()
        cache.close()


HIDDEN, INTER, TOP_K = 256, 32, 10


def _iq4_nl_bank(generator, rows, k):
    """Valid IQ4_NL rows [EXPERTS, rows, k / 32 * 18] with small scales."""
    blocks = torch.randint(
        0, 256, (EXPERTS, rows, k // 32, 18), dtype=torch.uint8, generator=generator
    )
    scale = (
        torch.rand(EXPERTS, rows, k // 32, generator=generator) * 0.02 + 0.005
    ).half()
    blocks[..., :2] = scale.view(torch.uint8).reshape(*scale.shape, 2)
    return blocks.reshape(EXPERTS, rows, -1)


def _turbomind_cache(fraction=0.75, spare=4, prefill_tokens=4096):
    """IQ4_NL gate/up/down banks of real (decodable) GGUF blocks."""
    device = torch.device("cuda", torch.accelerator.current_device_index())
    generator = torch.Generator().manual_seed(7)
    shapes = {"w1": (INTER, HIDDEN), "w3": (INTER, HIDDEN), "w2": (HIDDEN, INTER)}
    original = {
        (layer, shard): _iq4_nl_bank(generator, *shapes[shard])
        for layer in LAYERS
        for shard in shapes
    }
    banks = [
        (
            layer,
            shard,
            tuple(original[layer, shard].shape),
            0,
            (20, shapes[shard][1], 0),
        )
        for layer in LAYERS
        for shard in shapes
    ]
    config = SimpleNamespace(
        expert_cache_fraction=fraction,
        expert_cache_spare_slots=spare,
        expert_cache_host_slack=0.1,
        expert_cache_profile=None,
        expert_cache_prefill="turbomind",
        expert_cache_prefill_tokens=prefill_tokens,
    )
    cache = gguf_expert_cache.create_cache(banks, EXPERTS, config, device)
    for (layer, shard), bank in original.items():
        for expert in range(EXPERTS):
            cache.store(layer, shard, expert, bank[expert].to(device))
    return cache, original


def _reference(original, layer, x, ids, weights):
    from vllm.transformers_utils.gguf_tensor_reader import dequantize

    def weight(shard):
        bank = original[layer, shard].numpy()
        return torch.stack(
            [torch.from_numpy(dequantize(bank[e], 20)) for e in range(EXPERTS)]
        ).cuda()

    gate, up, down = weight("w1"), weight("w3"), weight("w2")
    x = x.float()
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for j in range(ids.shape[1]):
        e = ids[:, j].long()
        g = torch.einsum("tk,tnk->tn", x, gate[e])
        u = torch.einsum("tk,tnk->tn", x, up[e])
        h = (torch.nn.functional.silu(g) * u).half().float()
        out += weights[:, j, None] * torch.einsum("tk,tnk->tn", h, down[e])
    return out


@pytest.mark.parametrize("prefill_tokens", [4096, 256])
def test_turbomind_prefill_matches_reference_and_follows_decode(prefill_tokens):
    cache, original = _turbomind_cache(prefill_tokens=prefill_tokens)
    generator = torch.Generator().manual_seed(11)
    for step in range(3):
        for layer in LAYERS:
            tokens = 700
            x = (torch.randn(tokens, HIDDEN, generator=generator) * 0.5).half().cuda()
            ids = (
                torch.stack(
                    [
                        torch.randperm(EXPERTS, generator=generator)[:TOP_K]
                        for _ in range(tokens)
                    ]
                )
                .to(torch.int32)
                .cuda()
            )
            weights = torch.rand(tokens, TOP_K, generator=generator).cuda()
            out = cache.prefill_moe(layer, x, ids, weights)
            ref = _reference(original, layer, x, ids, weights)
            # FP16 activations, FP32 accumulation: per-token relative error.
            error = (out.float() - ref).norm(dim=1) / ref.norm(dim=1)
            assert error.max().item() < 3e-3, error.max().item()
        # Decode misses move experts between GPU slots and the host arena.
        for layer in LAYERS:
            decode = _random_ids(generator, 5)
            cache.resolve(layer, decode)
    torch.accelerator.synchronize()
    assert int(cache.tm.error_host[0]) == 0
    assert cache.stats()["errors"] == 0 and cache.stats()["loaded"] > 0


def test_decode_does_not_deadlock_on_one_hardware_queue():
    # With a single hardware queue every stream is serialized. A miss that is
    # released by work submitted after the wait would deadlock here.
    import os
    import subprocess
    import sys

    env = dict(os.environ, CUDA_DEVICE_MAX_CONNECTIONS="1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-x",
            "-q",
            __file__,
            "-k",
            "decode_resolve or inside_cuda_graph or interleaves",
        ],
        env=env,
        timeout=600,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
