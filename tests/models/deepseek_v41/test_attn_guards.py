# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN fail-loud guards (MC-ATTN F1, F2, F8).

F1: every V4.1 attention site (attention, compressor, indexer, mirror) takes the profile path ONLY on a dummy
forward (ForwardContext.is_dummy_run) without metadata; list metadata (ubatching / DBO) and None metadata on a real
step raise instead of silently zeroing attention and skipping cache writes.
F2: the mirror refuses payloads whose row count differs from the forward's token count, and with
VLLM_DS41_ATTN_MIRROR_CHECK=1 requires and verifies the exporter's kv20_crc.
F8: mirror cache names follow the stage's decoder-layer prefix instead of a hard-coded "model.layers".
"""

from __future__ import annotations

import pytest
import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.deepseek_v41.common.contracts import CAND_TOPK_BLOCKS, StagePlan

from .test_attn_harness import layer_inputs, ref_config, synthetic_attn_weights, topology
from .test_attn_layers import DEV, Stage, dist_env  # noqa: F401  (fixture)

pytestmark = pytest.mark.sm70

STAGE3 = StagePlan(2, 3, 28, 39, (20,), (), ())


def _stage(layer_ids=(2, 20), **kw) -> tuple[Stage, dict]:
    cfg = ref_config()
    weights = {i: synthetic_attn_weights(cfg, topology(cfg, i), DEV, seed=i) for i in layer_ids}
    return Stage(cfg, layer_ids, weights, num_blocks=64, **kw), weights


def _real_metadata(st: Stage, n: int) -> tuple[dict, torch.Tensor]:
    st._batch = [("a", 0, n)]
    return st.sim.metadata(st._batch)


def _payload(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=DEV).manual_seed(seed)
    ckv = torch.randn(n, 512, device=DEV, generator=g).half()
    ik = torch.randn(n, 128, device=DEV, generator=g).half()
    cand = torch.full((n, CAND_TOPK_BLOCKS), -1, dtype=torch.int32, device=DEV)
    return ckv, ik, cand


def _sites(st: Stage, mirror_st: Stage | None):
    """name -> callable that reaches exactly one metadata site."""
    n = 16
    x = torch.randn(n, 5120, device=DEV).half()
    pos = torch.arange(n, device=DEV)
    qr = torch.randn(n, 1280, device=DEV)
    sites = {
        "attention": lambda: st.attn[2].attention_impl(x, pos, torch.zeros(n, 5120, device=DEV)),
        "compressor": lambda: st.attn[2].compressor(x, pos),
        "compressor_state": lambda: st.attn[2].compressor._state_metadata(),
        "indexer": lambda: st.attn[20].indexer.forward(x, qr, pos),
    }
    if mirror_st is not None:
        sites["mirror"] = lambda: mirror_st.mirror.ingest_impl(pos, *_payload(n), None)
    return sites


SITES = ("attention", "compressor", "compressor_state", "indexer", "mirror")


@pytest.mark.parametrize("site", SITES)
def test_list_metadata_raises(dist_env, site) -> None:    # noqa: F811
    st, _ = _stage()
    m3, _ = _stage((28,), stage=STAGE3, mirror=True)
    md, _ = _real_metadata(st if site != "mirror" else m3, 16)
    fc = ForwardContext(no_compile_layers=st.ctx, attn_metadata=[md], slot_mapping={})
    with override_forward_context(fc), pytest.raises(NotImplementedError, match="micro-batching"):
        _sites(st, m3)[site]()


@pytest.mark.parametrize("site", SITES)
def test_none_metadata_on_real_step_raises(dist_env, site) -> None:    # noqa: F811
    st, _ = _stage()
    m3, _ = _stage((28,), stage=STAGE3, mirror=True)
    fc = ForwardContext(no_compile_layers=st.ctx, attn_metadata=None, slot_mapping={}, is_dummy_run=False)
    with override_forward_context(fc), pytest.raises(RuntimeError, match="not a dummy/profile run"):
        _sites(st, m3)[site]()


def test_none_metadata_on_dummy_step_is_profile_path(dist_env) -> None:    # noqa: F811
    """The dummy path: attention returns zeros, the mirror writes nothing, no cache is touched."""
    m3, _ = _stage((28,), stage=STAGE3, mirror=True)
    before = {k: c.tensor.clone() for k, c in m3.sim.caches.items()}
    n = 16
    fc = ForwardContext(no_compile_layers=m3.ctx, attn_metadata=None, slot_mapping={}, is_dummy_run=True)
    with override_forward_context(fc):
        m3.mirror.ingest(torch.arange(n, device=DEV), *_payload(n))
        y = m3.attn[28](torch.arange(n, device=DEV), torch.randn(n, 5120, device=DEV).half())
    assert torch.equal(y, torch.zeros_like(y))
    for k, c in m3.sim.caches.items():
        assert torch.equal(torch.isnan(c.tensor), torch.isnan(before[k])), f"dummy forward wrote cache {k}"


# ------------------------------------------------------------------------------------------------ F2


@pytest.mark.parametrize("extra", [1, -1])
def test_mirror_payload_row_count_must_match_forward(dist_env, extra) -> None:    # noqa: F811
    m3, _ = _stage((28,), stage=STAGE3, mirror=True)
    md, pos = _real_metadata(m3, 16)
    with override_forward_context(ForwardContext(no_compile_layers=m3.ctx, attn_metadata=md, slot_mapping={})):
        with pytest.raises(ValueError, match="exactly one row per forward token"):
            m3.mirror.ingest(pos, *_payload(16 + extra))


def test_mirror_check_requires_crc(dist_env, monkeypatch) -> None:    # noqa: F811
    monkeypatch.setenv("VLLM_DS41_ATTN_MIRROR_CHECK", "1")
    m3, _ = _stage((28,), stage=STAGE3, mirror=True)
    md, pos = _real_metadata(m3, 16)
    with override_forward_context(ForwardContext(no_compile_layers=m3.ctx, attn_metadata=md, slot_mapping={})):
        with pytest.raises(RuntimeError, match="carries no kv20_crc"):
            m3.mirror.ingest(pos, *_payload(16))


def test_mirror_check_crc_detects_corrupt_row_and_accepts_good(dist_env, monkeypatch) -> None:    # noqa: F811
    from vllm.models.deepseek_v41.kv_mirror import kv20_crc

    monkeypatch.setenv("VLLM_DS41_ATTN_MIRROR_CHECK", "1")
    m3, _ = _stage((28,), stage=STAGE3, mirror=True)
    md, pos = _real_metadata(m3, 16)
    ckv, ik, cand = _payload(16)
    crc = kv20_crc(ckv, ik)
    with override_forward_context(ForwardContext(no_compile_layers=m3.ctx, attn_metadata=md, slot_mapping={})):
        m3.mirror.ingest(pos, ckv, ik, cand, crc)                    # good payload passes both checks
        bad = ckv.clone()
        bad.view(torch.int16)[5, 7] ^= 1                              # one flipped mantissa bit in row 5
        with pytest.raises(RuntimeError, match="1/16 payload rows do not match"):
            m3.mirror.ingest(pos, bad, ik, cand, crc)


# ------------------------------------------------------------------------------------------------ F8


def test_mirror_names_follow_stage_prefix(dist_env) -> None:    # noqa: F811
    from vllm.models.deepseek_v41.kv_mirror import DeepseekV41KVSourceMirror

    base = "language_model.model.layers"
    m3, _ = _stage((28,), stage=STAGE3, mirror=True, layers_prefix=base)
    assert m3.mirror.layers_prefix == base
    assert m3.mirror.ckv_cache.layer_name == f"{base}.20.attn"
    assert m3.mirror.ik_cache.layer_name == f"{base}.20.attn.indexer.k_cache"
    assert m3.attn[28].kv_source_name == m3.mirror.ckv_cache.layer_name     # consumer resolves the mirror
    # one forward through the renamed stage reads the mirrored source
    md, pos = _real_metadata(m3, 16)
    cfg = ref_config()
    x = layer_inputs(synthetic_attn_weights(cfg, topology(cfg, 28), DEV, seed=28), 16, seed=1, device=DEV)
    with override_forward_context(ForwardContext(no_compile_layers=m3.ctx, attn_metadata=md, slot_mapping={})):
        m3.mirror.ingest(pos, *_payload(16))
        assert torch.isfinite(m3.attn[28](pos, x)).all()
    # no registered attention layers and no explicit prefix: refuse instead of guessing "model.layers"
    empty = _stage((28,), stage=STAGE3)[0]
    empty.vcfg.compilation_config.static_forward_context.clear()
    with pytest.raises(RuntimeError, match="cannot derive the decoder-layer prefix"):
        DeepseekV41KVSourceMirror(empty.vcfg, 20, empty.shared)
    explicit = DeepseekV41KVSourceMirror(empty.vcfg, 20, empty.shared, layers_prefix="m.layers")
    assert explicit.ckv_cache.layer_name == "m.layers.20.attn"
