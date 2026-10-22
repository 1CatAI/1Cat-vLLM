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


def _cache(fraction=0.75, spare=4, slack=0.1):
    device = torch.device("cuda", torch.accelerator.current_device_index())
    generator = torch.Generator().manual_seed(0)
    original = {
        (layer, shard): torch.randint(
            0,
            256,
            (EXPERTS, *_shape(layer, shard)),
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
            (EXPERTS, *_shape(layer, shard)),
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
    )
    cache = gguf_expert_cache.create_cache(banks, EXPERTS, config, device)
    for (layer, shard), bank in original.items():
        for expert in range(EXPERTS):
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


@pytest.mark.parametrize("slack", [0.0, 0.25])
@pytest.mark.parametrize("selection", ["empty", "sparse", "all"])
@pytest.mark.parametrize("capture", [False, True])
def test_selected_cold_stage_uses_snapshot_and_preserves_holes(
    slack, selection, capture
):
    cache, original = _cache(slack=slack)
    try:
        generator = torch.Generator().manual_seed(9683)
        for layer in LAYERS:
            cache.resolve(layer, _random_ids(generator, 5))
        torch.accelerator.synchronize()
        stream = torch.cuda.current_stream().cuda_stream
        for i, layer in enumerate(LAYERS):
            snapshot = torch.empty(2, EXPERTS, dtype=torch.int32, device="cuda")
            cache.lane.residency(i, snapshot.data_ptr(), stream)
            gpu_slots, host_slots = snapshot.cpu()
            selected = torch.zeros(EXPERTS, dtype=torch.bool)
            if selection == "sparse":
                selected[::3] = True
            elif selection == "all":
                selected.fill_(True)
            cold = torch.nonzero((gpu_slots < 0) & selected).flatten()
            host = host_slots[cold].long()
            saved = cache.slot_of[i].clone()
            cache.slot_of[i].fill_(0)
            try:
                for s, shard in enumerate(gguf_expert_cache.SHARDS):
                    stage = torch.full_like(
                        cache.host[layer, shard], 0xA5, device="cuda"
                    )
                    copy_stream = torch.cuda.Stream()
                    copy_stream.wait_stream(torch.cuda.current_stream())
                    if capture and len(cold):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=copy_stream):
                            cache.lane.stage_cold_selected(
                                i,
                                s,
                                stage.data_ptr(),
                                copy_stream.cuda_stream,
                                selected,
                            )
                        graph.replay()
                    else:
                        cache.lane.stage_cold_selected(
                            i, s, stage.data_ptr(), copy_stream.cuda_stream, selected
                        )
                        torch.cuda.current_stream().wait_stream(copy_stream)
                    torch.testing.assert_close(
                        stage[host],
                        original[layer, shard][cold],
                        rtol=0,
                        atol=0,
                    )
                    holes = torch.ones(cache.host_slots, dtype=torch.bool)
                    holes[host] = False
                    assert (stage[holes] == 0xA5).all()
            finally:
                cache.slot_of[i].copy_(saved)
    finally:
        torch.accelerator.synchronize()
        cache.close()


def test_selected_cold_stage_rejects_invalid_masks():
    cache, _ = _cache()
    try:
        cache.start()
        stream = torch.cuda.current_stream().cuda_stream
        stage = torch.empty_like(cache.host[LAYERS[0], "w1"], device="cuda")
        masks = [
            torch.zeros(EXPERTS, dtype=torch.int32),
            torch.zeros(EXPERTS - 1, dtype=torch.bool),
            torch.zeros(2, EXPERTS, dtype=torch.bool),
            torch.zeros(EXPERTS * 2, dtype=torch.bool)[::2],
            torch.zeros(EXPERTS, dtype=torch.bool, device="cuda"),
        ]
        for mask in masks:
            with pytest.raises(RuntimeError, match="contiguous CPU bool"):
                cache.lane.stage_cold_selected(0, 0, stage.data_ptr(), stream, mask)
    finally:
        torch.accelerator.synchronize()
        cache.close()


@pytest.mark.parametrize("slack", [0.0, 0.25])
@pytest.mark.parametrize("batched", [False, True])
def test_dma_prefill_admission_preserves_original_bytes_and_decode(slack, batched):
    cache, original = _cache(slack=slack)
    cache.start()
    host_plans = torch.empty(
        (len(LAYERS), cache.plan.numel()), dtype=torch.int32, pin_memory=True
    )
    plan = torch.empty_like(cache.plan)
    temporary_bytes = (
        8 * max(sum(cache.sizes[layer, s] for s in SHAPES) for layer in LAYERS)
        if batched
        else max(cache.sizes.values())
    )
    temporary = torch.empty(temporary_bytes, dtype=torch.uint8, device="cuda")
    previous = torch.cuda.current_stream()
    work = torch.cuda.Stream()
    work.wait_stream(previous)
    torch.cuda.set_stream(work)
    generator = torch.Generator().manual_seed(9683)
    try:
        for step in range(16):
            for i, layer in enumerate(LAYERS):
                count = (0, 1, 7, 8, 9, 10, 48, cache.resident0)[step % 8]
                ids_cpu = torch.randperm(EXPERTS, generator=generator)[:count].int()
                ids = ids_cpu.cuda()
                cache.lane.admit_prefill(
                    i,
                    ids_cpu,
                    host_plans[i],
                    plan,
                    temporary,
                    torch.cuda.current_stream().cuda_stream,
                )
                slots = torch.empty_like(ids)
                gguf_expert_cache._remap[(1,)](
                    ids,
                    cache.table,
                    plan,
                    slots,
                    i,
                    count,
                    EXPERTS=EXPERTS,
                    BLOCK=max(32, 1 << (max(1, count) - 1).bit_length()),
                    OPS=gguf_expert_cache._MAX_OPS,
                )
                _check_decode(cache, original, layer, ids, slots)
            torch.accelerator.synchronize()
            torch.testing.assert_close(cache.table.cpu(), cache.slot_of, rtol=0, atol=0)
            for i, layer in enumerate(LAYERS):
                for shard in SHAPES:
                    for mapping, bank in (
                        (cache.slot_of[i], cache.bank(layer, shard)),
                        (cache.host_slot_of[i], cache.host[layer, shard]),
                    ):
                        valid = torch.where(mapping >= 0)[0]
                        torch.testing.assert_close(
                            bank[mapping[valid].long()].cpu(),
                            original[layer, shard][valid].cpu(),
                            rtol=0,
                            atol=0,
                        )
            # Ordinary GPU-polled decode reuses the same LRU state and plan.
            for layer in LAYERS:
                ids = _random_ids(generator, 5)
                _check_decode(cache, original, layer, ids, cache.resolve(layer, ids))
            torch.accelerator.synchronize()
        assert cache.stats()["errors"] == 0
    finally:
        try:
            torch.accelerator.synchronize()
            cache.close()
        finally:
            torch.cuda.set_stream(previous)


def test_dma_prefill_admission_rejects_invalid_storage_before_mutation():
    cache, _ = _cache()
    cache.start()
    ids = torch.tensor([EXPERTS - 1], dtype=torch.int32)
    host_plan = torch.empty_like(cache.reply, pin_memory=True)
    plan = torch.empty_like(cache.plan)
    temporary = torch.empty(max(cache.sizes.values()), dtype=torch.uint8, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    before = cache.slot_of.clone()
    try:
        for invalid in (
            ids.long(),
            ids.cuda(),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([EXPERTS], dtype=torch.int32),
            torch.arange(cache.resident0 + 1, dtype=torch.int32),
        ):
            with pytest.raises(RuntimeError, match="prefill.*(ids|id)"):
                cache.lane.admit_prefill(0, invalid, host_plan, plan, temporary, stream)
        for bad_host, bad_device, bad_temp in (
            (torch.empty(host_plan.numel(), dtype=torch.int32), plan, temporary),
            (host_plan, plan[:-1], temporary),
            (host_plan, plan, temporary[:1]),
        ):
            with pytest.raises(RuntimeError, match="(plan|temporary)"):
                cache.lane.admit_prefill(0, ids, bad_host, bad_device, bad_temp, stream)
        torch.testing.assert_close(cache.slot_of, before, rtol=0, atol=0)
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


def _turbomind_cache(
    fraction=0.75,
    spare=4,
    prefill_tokens=4096,
    prepare="all",
    seed=7,
):
    """IQ4_NL gate/up/down banks of real (decodable) GGUF blocks."""
    device = torch.device("cuda", torch.accelerator.current_device_index())
    generator = torch.Generator().manual_seed(seed)
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
        expert_cache_prefill_prepare=prepare,
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


@pytest.mark.parametrize("prepare", ["predict", "predict_resident"])
def test_predictive_prefill_thresholds_misses_restart_and_decode(prepare):
    cases = []
    generator = torch.Generator(device="cuda").manual_seed(9683)
    for index, rows in enumerate((4095, 4096, 4097, 8191, 8192, 8193)):
        first = 0 if index % 2 == 0 else EXPERTS - 20
        ids = (torch.arange(rows)[:, None] + torch.arange(TOP_K)) % 20 + first
        ids = ids.to(device="cuda", dtype=torch.int32)
        x = (
            torch.randn(rows, HIDDEN, device="cuda", generator=generator) * 0.125
        ).half()
        weights = torch.randn(rows, TOP_K, device="cuda", generator=generator).softmax(
            1
        )
        cases.append((x, ids, weights))
    control, _ = _turbomind_cache()
    references = []
    try:
        for x, ids, weights in cases:
            references.append(
                [control.prefill_moe(layer, x, ids, weights).cpu() for layer in LAYERS]
            )
    finally:
        torch.accelerator.synchronize()
        control.close()
    cache, _ = _turbomind_cache(prepare=prepare)
    assert cache.tm.preparation is not None
    actuals = []
    try:
        for index, (x, ids, weights) in enumerate(cases):
            if index == 2:
                # A completely wrong prediction must be completed without
                # depending on copied canonical bytes from the old pass.
                torch.accelerator.synchronize()
                helper = cache.tm.preparation
                helper.host_hints.zero_()
                helper.pending = None
                cache.prefill_moe(LAYERS[0], x, ids, weights)  # interrupted pass
            actuals.append(
                [cache.prefill_moe(layer, x, ids, weights) for layer in LAYERS]
            )
            assert cache.tm.prediction_step == (4096 <= len(ids) <= 8192)
        torch.accelerator.synchronize()
        for actual, expected in zip(actuals, references):
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)
        for layer in LAYERS:
            cache.resolve(layer, torch.arange(EXPERTS - 32, EXPERTS, device="cuda"))
        torch.accelerator.synchronize()
        x, ids, weights = cases[4]
        outputs = [cache.prefill_moe(layer, x, ids, weights) for layer in LAYERS]
        # Teardown itself must drain mapped-host reads and pinned hint copies.
        cache.close()
        for actual, expected in zip(outputs, references[4]):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert cache.host_arena is None and cache.tm.preparation is None
    finally:
        torch.accelerator.synchronize()
        cache.close()


def test_resident_prefill_fences_preexisting_decode_graph_after_interruption(
    monkeypatch,
):
    cache, original = _turbomind_cache(prepare="predict_resident")
    original = {key: value.cuda() for key, value in original.items()}
    cache.start()
    static = torch.arange(EXPERTS - 20, EXPERTS - 10, device="cuda", dtype=torch.int32)
    for layer in LAYERS:
        cache.resolve(layer, static)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.graph(graph, stream=stream):
            slots = [cache.resolve(layer, static) for layer in LAYERS]
        helper = cache.tm.preparation
        raw_read_done = torch.cuda.Event()
        original_prefetch = cache.tm._prefetch

        def delay_future_reader(key, i):
            original_prefetch(key, i)
            if key == "dn" and i == 1:
                with torch.cuda.stream(cache.tm.cold_stream):
                    torch.cuda._sleep(20000000)
                    raw_read_done.record()

        monkeypatch.setattr(cache.tm, "_prefetch", delay_future_reader)
        # Make the completed hint disjoint from initial residents. The mask is
        # only a hint: every actual prefill route still gets exact completion.
        helper.host_hints.zero_()
        helper.host_hints[:, :, EXPERTS - 20 :].fill_(True)
        rows = 4096
        ids = static.expand(rows, -1).contiguous()
        x = (torch.randn(rows, HIDDEN, device="cuda") * 0.125).half()
        weights = torch.full((rows, TOP_K), 0.1, device="cuda")
        for _ in range(3):
            # Delay the eager producer so graph replay must honor its external
            # event; no host/device synchronization is inserted before replay.
            with torch.cuda.stream(helper.admission_stream):
                torch.cuda._sleep(20000000)
            cache.prefill_moe(LAYERS[0], x, ids, weights)
            graph.replay()
            torch.cuda.current_stream().synchronize()
            assert raw_read_done.query(), "decode replay overtook a raw reader"
            for layer, mapped in zip(LAYERS, slots):
                _check_decode(cache, original, layer, static, mapped)
            static.copy_((static + 17) % EXPERTS)
        assert cache.stats()["errors"] == 0
    finally:
        torch.accelerator.synchronize()
        cache.close()
