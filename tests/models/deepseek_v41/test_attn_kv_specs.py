# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN: V4.1 KV-cache specs, page sizes and grouping (PORT_DESIGN §2.1 rule 4, §3.3, §3.5, §6),
plus a regression gate that DeepSeek-V4 grouping/layout is byte-identical to ds41/c0."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.common.contracts import CAND_SOURCE, KV_SOURCES
from vllm.models.deepseek_v41.sm70 import sparse as s70
from vllm.v1.core import kv_cache_utils as ku
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec, UniformTypeKVCacheSpecs

RATIOS = [0, 0] + [2] * 18 + [1] * 20          # backbone layers 0..39 (ref config)


def _cfg(max_len: int = 65536, tokens: int = 4096) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_len),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens, disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1,
                                        pipeline_parallel_size=1),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        speculative_config=None,
        max_in_flight_tokens=tokens,
    )


# ----------------------------------------------------------------------------- V4 regression
def _v4_specs(ratios: list[int]) -> dict:
    specs = {}
    for i, r in enumerate(ratios):
        p = f"model.layers.{i}.attn"
        specs[f"{p}.swa_cache"] = SlidingWindowMLASpec(
            block_size=64, num_kv_heads=1, head_size=512, dtype=torch.uint8, sliding_window=128,
            cache_dtype_str="fp8_ds_mla", alignment=576, model_version="deepseek_v4")
        if r in (4, 128):
            specs[p] = MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=512, dtype=torch.uint8,
                                        compress_ratio=r, cache_dtype_str="fp8_ds_mla", alignment=576,
                                        model_version="deepseek_v4")
        if r == 4:
            specs[f"{p}.compressor.state_cache"] = SlidingWindowMLASpec(
                block_size=4, num_kv_heads=1, head_size=2048, dtype=torch.float32, sliding_window=8, alignment=576)
            specs[f"{p}.indexer.k_cache"] = MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=132,
                                                             dtype=torch.uint8, compress_ratio=4, alignment=576)
            specs[f"{p}.indexer.compressor.state_cache"] = SlidingWindowMLASpec(
                block_size=4, num_kv_heads=1, head_size=512, dtype=torch.float32, sliding_window=8, alignment=576)
        elif r == 128:
            specs[f"{p}.compressor.state_cache"] = SlidingWindowMLASpec(
                block_size=8, num_kv_heads=1, head_size=1024, dtype=torch.float32, sliding_window=128, alignment=576)
    return specs


def _layout(specs: dict, splits: list[int] | None, mem: int = 8 << 30) -> dict:
    cfg = _cfg()
    groups = ku.get_kv_cache_groups(cfg, copy.deepcopy(specs))
    out = {"groups": [[len(g.layer_names), sorted(set(g.kv_cache_spec.get_page_sizes()))] for g in groups]}
    bounds = [0] + (splits or []) + [10**9]
    workers = [{k: v for k, v in specs.items() if lo <= int(k.split(".")[2]) < hi}
               for lo, hi in zip(bounds, bounds[1:])]
    out["workers"] = []
    for w in workers:
        proj = ku._project_kv_cache_groups_to_worker(groups, w)
        c = ku.get_kv_cache_config_from_groups(cfg, proj, mem)
        out["workers"].append({
            "num_blocks": c.num_blocks,
            "tensors": [[t.size, sorted(t.shared_by)] for t in c.kv_cache_tensors],
            "pool_bytes_per_block": ku._pool_bytes_per_block(proj),
            "max_mem": ku._max_memory_usage_bytes_from_groups(cfg, proj)})
    return out


# sha256 of the canonical JSON produced by this exact helper on the UNMODIFIED ds41/c0 tree
# (/mnt/nvme2/scratch/ds41/attn/v4_kv_snapshot.py, output v4_kv_snapshot_c0.json)
V4_C0_SHA256 = "064d746344ff04efee821f26c859c93825b89cd9ed6f7d75bc9cecc45bef5286"


def test_v4_layout_unchanged() -> None:
    ratios = [0, 0] + [4, 128] * 5
    snap = {"single": _layout(_v4_specs(ratios), None), "pp2": _layout(_v4_specs(ratios), [6])}
    assert [w["num_blocks"] for w in snap["single"]["workers"]] == [35935]
    assert [w["num_blocks"] for w in snap["pp2"]["workers"]] == [89837, 59891]
    assert [w["pool_bytes_per_block"] for w in snap["pp2"]["workers"]] == [95616, 143424]
    digest = hashlib.sha256(json.dumps(snap, sort_keys=True).encode()).hexdigest()
    assert digest == V4_C0_SHA256


def test_v4_specs_never_take_the_v41_path() -> None:
    groups = ku.get_kv_cache_groups(_cfg(), copy.deepcopy(_v4_specs([0, 0, 4, 128])))
    assert not ku._is_deepseek_v41_groups(groups)


# ----------------------------------------------------------------------------- V4.1 specs
def _v41_specs(layers: range, block: int = 256, mirror: bool = False) -> dict:
    specs = {}
    for i in layers:
        p = f"model.layers.{i}.attn"
        specs[f"{p}.swa_cache"] = s70.swa_cache_spec(block)
        r = RATIOS[i]
        if i in KV_SOURCES:
            specs[p] = s70.compressed_cache_spec(block, r)
            specs[f"{p}.indexer.k_cache"] = s70.index_k_cache_spec(block, r)
            if r == 2:
                specs[f"{p}.compressor.state_cache"] = s70.state_cache_spec(block)
    if mirror:                                     # stage-3 replica of source 20, same names + specs
        p = f"model.layers.{CAND_SOURCE}.attn"
        specs[p] = s70.compressed_cache_spec(block, 1)
        specs[f"{p}.indexer.k_cache"] = s70.index_k_cache_spec(block, 1)
    return specs


def test_v41_page_sizes() -> None:
    assert s70.swa_cache_spec(256).block_size == 128 and s70.state_cache_spec(256).block_size == 32
    assert s70.swa_cache_spec(256).page_size_bytes == 128 * 1024
    assert s70.state_cache_spec(256).page_size_bytes == 32 * 4096
    assert s70.compressed_cache_spec(256, 2).page_size_bytes == 128 * 1024
    assert s70.compressed_cache_spec(256, 1).page_size_bytes == 256 * 1024
    assert s70.index_k_cache_spec(256, 2).page_size_bytes == 128 * 256
    assert s70.index_k_cache_spec(256, 1).page_size_bytes == 256 * 256
    for spec in (s70.swa_cache_spec(256), s70.state_cache_spec(256), s70.compressed_cache_spec(256, 2)):
        assert spec.page_size_padded is None
    with pytest.raises(ValueError):
        s70.compressed_cache_spec(255, 2)
    with pytest.raises(ValueError):
        s70.swa_cache_spec(60)
    with pytest.raises(ValueError):
        s70.index_k_cache_spec(256, 4)
    bad = SlidingWindowMLASpec(block_size=8, num_kv_heads=1, head_size=512, dtype=torch.uint8, sliding_window=2,
                               model_version="deepseek_v41")
    with pytest.raises(ValueError):
        _ = bad.page_size_bytes
    # bytes per token of context per stage (PORT_DESIGN §6.2)
    def per_token(specs):
        return sum(sp.page_size_bytes / sp.block_size for sp in specs.values() if isinstance(sp, MLAAttentionSpec))
    assert per_token(_v41_specs(range(0, 20))) == 1920
    assert per_token(_v41_specs(range(20, 40))) == 1280
    assert per_token(_v41_specs(range(14, 28))) == 1920
    assert per_token(_v41_specs(range(28, 40), mirror=True)) == 1280


def _check_worker(specs: dict, cfg_out) -> None:
    """Every worker layer in exactly one tensor whose size = its page x num_blocks; one layer per group."""
    seen = Counter()
    for t in cfg_out.kv_cache_tensors:
        groups_of = [next(gi for gi, g in enumerate(cfg_out.kv_cache_groups) if n in g.layer_names)
                     for n in t.shared_by]
        assert len(groups_of) == len(set(groups_of)), f"two layers of one group share a tensor: {t.shared_by}"
        for n in t.shared_by:
            seen[n] += 1
            assert t.size == specs[n].page_size_bytes * cfg_out.num_blocks, n
    assert set(seen) == set(specs) and set(seen.values()) == {1}


@pytest.mark.parametrize("splits", [None, [20], [14, 28]], ids=["pp1", "pp2", "pp3"])
def test_v41_layouts(splits) -> None:
    bounds = [0] + (splits or []) + [40]
    workers = [_v41_specs(range(lo, hi), mirror=(splits == [14, 28] and lo == 28))
               for lo, hi in zip(bounds, bounds[1:])]
    merged: dict = {}
    for w in workers:                      # exactly get_kv_cache_configs' merge rule
        for name, spec in w.items():
            if name in merged:
                assert merged[name] == spec, name
            else:
                merged[name] = spec
    cfg = _cfg()
    groups = ku.get_kv_cache_groups(cfg, merged)
    assert ku._is_deepseek_v41_groups(groups)
    assert all(isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) for g in groups)
    for w in workers:
        proj = ku._project_kv_cache_groups_to_worker(groups, w)
        out = ku.get_kv_cache_config_from_groups(cfg, proj, 6 << 30)
        assert out.num_blocks > 0
        _check_worker(w, out)
        used = sum(t.size for t in out.kv_cache_tensors)
        assert used <= 6 << 30
        assert ku._pool_bytes_per_block(proj) * out.num_blocks == used
        assert ku._max_memory_usage_bytes_from_groups(cfg, proj) > 0


def test_mirror_names_share_block_ids_with_the_source() -> None:
    """PP3: the mirror's two cache layers carry layer 20's names and equal specs, so the merged model has
    one entry per name and both stages' layers land in the same KV cache group (same block table)."""
    s2 = _v41_specs(range(14, 28))
    s3 = _v41_specs(range(28, 40), mirror=True)
    for name in ("model.layers.20.attn", "model.layers.20.attn.indexer.k_cache"):
        assert s2[name] == s3[name]
    merged = {**s2, **s3}
    groups = ku.get_kv_cache_groups(_cfg(), copy.deepcopy(merged))
    for name in ("model.layers.20.attn", "model.layers.20.attn.indexer.k_cache"):
        assert sum(name in g.layer_names for g in groups) == 1
    proj3 = ku._project_kv_cache_groups_to_worker(groups, s3)
    gid = next(i for i, g in enumerate(proj3) if "model.layers.20.attn" in g.layer_names)
    gid_full = next(i for i, g in enumerate(groups) if "model.layers.20.attn" in g.layer_names)
    assert gid == gid_full


def test_v41_end_to_end_get_kv_cache_configs() -> None:
    """The real entry point with three PP workers (mirror on the last) and unequal memory."""
    workers = [_v41_specs(range(0, 14)), _v41_specs(range(14, 28)), _v41_specs(range(28, 40), mirror=True)]
    cfg = _cfg()
    cfg.model_config.original_max_model_len = 65536
    configs = ku.get_kv_cache_configs(cfg, workers, [5 << 30, 4 << 30, 6 << 30])
    assert len({c.num_blocks for c in configs}) == 1
    for w, c in zip(workers, configs):
        _check_worker(w, c)


def test_v41_pool_memory_at_256k() -> None:
    """Planning-time pool bytes per stage (every block id reserves a page in each tensor): the chosen
    windowed block sizes keep PP3 at 256K context / 4096-token chunks under 800 MiB per stage."""
    stages = [(0, 14, False), (14, 28, False), (28, 40, True)]
    workers = [_v41_specs(range(a, b), mirror=m) for a, b, m in stages]
    merged: dict = {}
    for w in workers:
        merged.update(w)
    cfg = _cfg(max_len=262144)
    groups = ku.get_kv_cache_groups(cfg, merged)
    mib = [ku._max_memory_usage_bytes_from_groups(cfg, ku._project_kv_cache_groups_to_worker(groups, w)) / 2**20
           for w in workers]
    assert mib == pytest.approx([509.0, 764.0, 626.0], abs=1.0)


def test_every_v41_backend_prefers_the_v41_block_size() -> None:
    """L-INTEG: Platform.update_block_size_for_backend() asks the FIRST cache layer it finds; a compressed or state
    layer returned the generic default (16) while the specs were built with PREFERRED_BLOCK_SIZE."""
    from vllm.models.deepseek_v41.sm70.sparse import (PREFERRED_BLOCK_SIZE, DS41CompressedBackend, DS41StateBackend,
                                                       DS41SWABackend)

    for backend in (DS41SWABackend, DS41CompressedBackend, DS41StateBackend):
        assert backend.get_preferred_block_size(16) == PREFERRED_BLOCK_SIZE, backend.get_name()
