# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model


@pytest.mark.parametrize("num_reqs", [1, 2, 3, 4, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_context_pipeline_defers_writes_and_refreshes_accepted_slots(num_reqs):
    """Computing rejected rows early must never write their scratch K/V."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(717)
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    torch.nn.Module.__init__(model)
    layers, hidden, heads, dim, block = 3, 256, 2, 128, 16
    model._num_attn_layers = layers
    model._kv_size = heads * dim
    model._head_dim = dim
    model._num_kv_heads = heads
    model._rms_norm_eps = 1e-6
    model._hidden_norm_weight = torch.ones(hidden, device=device, dtype=torch.float16)
    model._fused_kv_weight = (
        torch.randn(layers * 2 * heads * dim, hidden, generator=gen, device=device)
        * 0.03
    ).half()
    model._fused_kv_bias = None
    model._k_norm_weights = (
        1 + torch.randn(layers, dim, generator=gen, device=device) * 0.1
    ).half()
    model._rope_head_size = dim
    model._rope_is_neox = True
    freq = 10000.0 ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    angles = torch.arange(4096, device=device).float()[:, None] * freq
    model._rope_cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=1).half()
    model._sm70_context_k_debugged = False
    scale = torch.tensor(1.0, device=device)

    def update(attn, key, value, cache, slots):
        ops.reshape_and_cache_flash(
            key, value, cache[:, 0], cache[:, 1], slots, "auto", scale, scale
        )

    caches = [
        torch.full(
            (num_reqs + 3, 2, block, heads, dim),
            0.125,
            device=device,
            dtype=torch.float16,
        )
        for _ in range(layers)
    ]
    model._attn_layers = [
        SimpleNamespace(kv_cache=c, impl=SimpleNamespace(do_kv_cache_update=update))
        for c in caches
    ]
    states = torch.randn(num_reqs * 8, hidden, generator=gen, device=device).half()
    positions = torch.arange(8, device=device).repeat(num_reqs)
    slots = [
        torch.full((num_reqs * 8,), -1, device=device, dtype=torch.int64)
        for _ in caches
    ]
    model.precompute_and_store_context_kv(states, positions, slots)
    compute, write = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(compute):
        key, value = model.compute_context_kv(states, positions)
    with torch.cuda.graph(write):
        model.store_context_kv(key, value, slots)

    for step in range(16):
        accepted = (torch.arange(num_reqs, device=device) + step) % 8 + 1
        valid = torch.arange(8, device=device)[None] < accepted[:, None]
        states.copy_(torch.randn(states.shape, generator=gen, device=device).half())
        positions.copy_(
            torch.arange(4080 - step, 4088 - step, device=device).repeat(num_reqs)
        )
        before = [c.clone() for c in caches]
        compute.replay()
        # Acceptance and cache placement are intentionally unknown until after
        # projection. Change both between every replay, including invalid slots.
        for layer, mapping in enumerate(slots):
            mapping.copy_(
                (
                    block * (1 + torch.arange(num_reqs, device=device)[:, None])
                    + torch.arange(8, device=device).roll(step + layer)[None]
                ).flatten()
            )
            mapping[~valid.flatten()] = -1
        for actual, expected in zip(caches, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        write.replay()
        actual = [c.clone() for c in caches]
        for c, snapshot in zip(caches, before):
            c.copy_(snapshot)
        reference_positions = positions.clone()
        reference_positions[~valid.flatten()] = 0
        model.precompute_and_store_context_kv(states, reference_positions, slots)
        for got, expected in zip(actual, caches):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
