# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN test helpers (no tests here): a simulated paged KV manager driving the real V4.1 metadata builders,
an independent FP32 transcription of the reference attention (ref:m.py:369-789, ref:k.py), and weight
loaders (synthetic, or real tensors from VERIFIED official shards)."""

from __future__ import annotations

import json
import math
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.common.contracts import (
    CAND_BLOCK,
    CAND_SOURCE,
    CAND_TOPK_BLOCKS,
    INDEX_SOURCES,
    KV_SOURCES,
    LayerTopology,
)
from vllm.v1.attention.backend import CommonAttentionMetadata

REF_DIR = Path("/mnt/nvme2/scratch/ds41/model-ref")
CKPT_DIR = Path("/mnt/nvme2/models/DeepSeek-V4.1-Flash")
VERIFY = Path("/mnt/nvme2/models/_dl-logs/verify_shard.sh")


# ============================================================================ config / topology
def ref_config() -> SimpleNamespace:
    raw = json.loads((REF_DIR / "config.json").read_text())
    merged = {k: v for k, v in raw.items() if k != "text_config"}
    merged.update(raw["text_config"])
    merged["rope_parameters"] = dict(merged["rope_scaling"])
    return SimpleNamespace(**merged)


def topology(cfg: SimpleNamespace, layer_id: int) -> LayerTopology:
    r = int(cfg.compress_ratios[layer_id])
    if r == 0:
        mode, kvs, ids = "swa", None, None
    else:
        kvs = max(s for s in KV_SOURCES if s <= layer_id)
        ids = max(s for s in INDEX_SOURCES if s <= layer_id)
        mode = "full" if layer_id in KV_SOURCES else ("reindex" if layer_id in INDEX_SOURCES else "reuse")
    owns_idx = layer_id in INDEX_SOURCES
    return LayerTopology(
        layer_id=layer_id, compress_ratio=r, mode=mode, kv_source=kvs, index_source=ids,
        owns_compressor=layer_id in KV_SOURCES, owns_indexer=owns_idx,
        is_candidate_source=layer_id == CAND_SOURCE, uses_candidates=owns_idx and CAND_SOURCE < layer_id,
        has_engram=layer_id in (1, 14), rope_theta=cfg.compress_rope_theta if r else cfg.rope_theta, yarn=r > 0)


def vllm_config(cfg: SimpleNamespace, block_size: int = 256, max_model_len: int = 65536,
                max_tokens: int = 4096, max_seqs: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=cfg, max_model_len=max_model_len),
        cache_config=SimpleNamespace(block_size=block_size, cache_dtype="auto"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens, max_num_seqs=max_seqs,
                                         async_scheduling=False),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1,
                                        pipeline_parallel_size=1, data_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=None, quant_config=None, kv_transfer_config=None)


# ============================================================================ simulated paged KV manager
@dataclass
class _Cache:
    name: str
    spec: object
    backend: type
    tensor: torch.Tensor
    blocks: dict = field(default_factory=dict)        # req_id -> [physical block ids]
    free: list = field(default_factory=list)


class SimKV:
    """Per cache name: a tensor [num_blocks, storage, head], shuffled physical block ids per request, block tables
    and runner-style slot mappings (slot = block * block_size + pos % block_size, token units). One builder per
    name (vLLM shares one per group; per-name metadata is equivalent for the consumers)."""

    def __init__(self, vcfg: SimpleNamespace, device: torch.device, num_blocks: int = 512, seed: int = 0) -> None:
        self.vcfg = vcfg
        self.device = device
        self.num_blocks = num_blocks
        self.caches: dict[str, _Cache] = {}
        self.builders: dict = {}
        self.rng = random.Random(seed)

    def register_all(self) -> None:
        """Allocate + bind a tensor for every cache layer in the static forward context (DS41CacheLayer)."""
        from vllm.models.deepseek_v41.sm70.sparse import DS41CacheLayer
        for name, layer in self.vcfg.compilation_config.static_forward_context.items():
            if isinstance(layer, DS41CacheLayer) and name not in self.caches:
                spec = layer.get_kv_cache_spec(self.vcfg)
                storage = spec.storage_block_size
                t = torch.full((self.num_blocks, storage, spec.head_size), float("nan"), dtype=spec.dtype,
                               device=self.device)
                layer.bind_kv_cache(t)
                free = list(range(1, self.num_blocks))      # block 0 = null block
                self.rng.shuffle(free)
                self.caches[name] = _Cache(name, spec, layer.get_attn_backend(), t, {}, free)
                self.builders[name] = layer.get_attn_backend().get_builder_cls()(spec, [name], self.vcfg, self.device)

    def ensure(self, req: str, upto: int) -> None:
        for c in self.caches.values():
            need = math.ceil(upto / c.spec.block_size)
            lst = c.blocks.setdefault(req, [])
            while len(lst) < need:
                lst.append(c.free.pop())

    def share_prefix(self, new_req: str, old_req: str, num_tokens: int) -> None:
        """Prefix-cache hit: the new request reuses the old one's blocks for positions < num_tokens."""
        for c in self.caches.values():
            assert num_tokens % c.spec.block_size == 0
            c.blocks[new_req] = list(c.blocks[old_req][: num_tokens // c.spec.block_size])

    def drop(self, req: str) -> None:
        for c in self.caches.values():
            c.blocks.pop(req, None)

    def release(self, req: str) -> None:
        """Preemption: return the request's blocks to the free list and poison them with NaN."""
        for c in self.caches.values():
            for b in c.blocks.pop(req, []):
                c.tensor[b].fill_(float("nan"))
                c.free.append(b)

    def link(self, name: str, other: "SimKV") -> None:
        """Make ``name`` use the same block ids as ``other`` (one scheduler-side KV manager across PP stages)."""
        mine, theirs = self.caches[name], other.caches[name]
        assert mine.spec == theirs.spec, f"{name}: specs differ across stages"
        mine.blocks = theirs.blocks
        mine.free = theirs.free

    def metadata(self, batch: list[tuple[str, int, int]]) -> tuple[dict, torch.Tensor]:
        """batch = [(req_id, start_pos, num_tokens)] in runner order -> ({name: metadata}, positions [T])."""
        for req, start, n in batch:
            self.ensure(req, start + n)
        qlens = [n for _, _, n in batch]
        qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
        seq = np.asarray([s + n for _, s, n in batch], dtype=np.int32)
        pos = np.concatenate([np.arange(s, s + n) for _, s, n in batch]).astype(np.int64)
        T = int(qsl[-1])
        dev = self.device
        out = {}
        for name, c in self.caches.items():
            bs = c.spec.block_size
            width = max(len(c.blocks[r]) for r, _, _ in batch)
            bt = np.zeros((len(batch), width), dtype=np.int32)
            for i, (r, _, _) in enumerate(batch):
                bt[i, : len(c.blocks[r])] = c.blocks[r]
            req_of = np.repeat(np.arange(len(batch)), qlens)
            slots = bt[req_of, pos // bs].astype(np.int64) * bs + pos % bs
            cm = CommonAttentionMetadata(
                query_start_loc=torch.from_numpy(qsl).to(dev), query_start_loc_cpu=torch.from_numpy(qsl),
                seq_lens=torch.from_numpy(seq).to(dev), num_reqs=len(batch), num_actual_tokens=T,
                max_query_len=max(qlens), max_seq_len=int(seq.max()),
                block_table_tensor=torch.from_numpy(bt).to(dev), slot_mapping=torch.from_numpy(slots).to(dev),
                positions=torch.from_numpy(pos).to(dev), seq_lens_cpu_upper_bound=torch.from_numpy(seq))
            out[name] = self.builders[name].build(0, cm)
        return out, torch.from_numpy(pos).to(dev)

    def rows_at(self, name: str, req: str, positions: np.ndarray, ratio: int = 1) -> torch.Tensor:
        """Cache rows holding logical entries ``positions`` (token positions for SWA/state; compressed index for
        compressed caches) of ``req``."""
        c = self.caches[name]
        storage = c.spec.storage_block_size
        blocks = np.asarray(c.blocks[req])
        rows = blocks[positions // storage].astype(np.int64) * storage + positions % storage
        return c.tensor.view(-1, c.spec.head_size)[torch.from_numpy(rows).to(self.device)]


# ============================================================================ weights
def verified_shard(basename: str) -> Path:
    proc = subprocess.run(["bash", str(VERIFY), basename], capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not proc.stdout.strip().startswith("OK"):
        raise RuntimeError(f"shard {basename} not verified: {proc.stdout} {proc.stderr}")
    return CKPT_DIR / basename


def _dequant_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    s = s.float().repeat_interleave(32, 0).repeat_interleave(32, 1)[: w.shape[0], : w.shape[1]]
    return w.float() * s


def load_attn_weights(layer_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    """``layers.{i}.attn.*`` + ``attn_norm.weight`` as FP32 tensors (FP8 dequantised exactly), from a verified shard."""
    from safetensors import safe_open
    idx = json.loads((REF_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{layer_id}.attn"
    names = [k for k in idx if k.startswith(prefix + ".") or k == f"layers.{layer_id}.attn_norm.weight"]
    shards = sorted({idx[k] for k in names})
    raw: dict[str, torch.Tensor] = {}
    for sh in shards:
        path = verified_shard(sh)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in names:
                if idx[k] == sh:
                    raw[k] = f.get_tensor(k)
    out = {}
    for k, t in raw.items():
        if k.endswith(".scale"):
            continue
        short = k[len(f"layers.{layer_id}."):]
        sk = k[: -len(".weight")] + ".scale"
        if t.dtype == torch.float8_e4m3fn:
            out[short] = _dequant_fp8(t, raw[sk]).to(device)
        else:
            out[short] = t.float().to(device)
    return out


def synthetic_attn_weights(cfg: SimpleNamespace, topo: LayerTopology, device: torch.device,
                           seed: int) -> dict[str, torch.Tensor]:
    """Random FP16-representable weights with the checkpoint's names and shapes (FP32 tensors)."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def lin(o: int, i: int, gain: float = 1.0) -> torch.Tensor:
        return (torch.randn(o, i, generator=g) * gain / math.sqrt(i)).half().float()

    def norm(d: int) -> torch.Tensor:
        return (1.0 + 0.1 * torch.randn(d, generator=g)).half().float()

    w = {
        "attn.wq_a.weight": lin(1280, 5120), "attn.wkv.weight": lin(512, 5120),
        "attn.q_norm.weight": norm(1280), "attn.kv_norm.weight": norm(512),
        "attn.wq_b.weight": lin(64 * 512, 1280), "attn.wo_a.weight": lin(8 * 1024, 4096),
        "attn.wo_b.weight": lin(5120, 8192), "attn.attn_sink": torch.randn(64, generator=g),
        "attn_norm.weight": norm(5120),
    }
    if topo.owns_compressor:
        w["attn.compressor.wkv.weight"] = lin(512, 5120)
        if topo.compress_ratio == 2:
            w["attn.compressor.wgate.weight"] = lin(512, 5120, gain=4.0)
        w["attn.compressor.norm.weight"] = norm(512)
        w["attn.indexer.wk.weight"] = lin(128, 512)
        w["attn.indexer.k_norm.weight"] = norm(128)
    if topo.owns_indexer:
        w["attn.indexer.wq_b.weight"] = lin(32 * 128, 1280)
        w["attn.indexer.weights_proj.weight"] = lin(32, 5120)
    return {k: v.to(device) for k, v in w.items()}


def load_into_module(attn, w: dict[str, torch.Tensor], tp_rank: int = 0, tp_size: int = 1) -> None:
    """Copy reference-named FP32 weights into a DeepseekV41Attention the way L-CORE's loader maps them
    (PORT_DESIGN §3.7): wq_b rows = local heads, wo_a rows = local o-groups, wo_b cols = local o-groups,
    attn_sink through its TP weight loader; everything else replicated."""
    hq = 64 // tp_size * 512
    ga = 8 // tp_size * 1024
    with torch.no_grad():
        attn.fused_wqa_wkv.weight.copy_(torch.cat([w["attn.wq_a.weight"], w["attn.wkv.weight"]]).half())
        attn.q_norm.weight.copy_(w["attn.q_norm.weight"])
        attn.kv_norm.weight.copy_(w["attn.kv_norm.weight"])
        attn.wq_b.weight.copy_(w["attn.wq_b.weight"][tp_rank * hq:(tp_rank + 1) * hq].half())
        attn.wo_a.weight.copy_(w["attn.wo_a.weight"][tp_rank * ga:(tp_rank + 1) * ga].half())
        attn.wo_b.weight.copy_(w["attn.wo_b.weight"][:, tp_rank * ga:(tp_rank + 1) * ga].half())
        attn._load_attn_sink(attn.attn_sink, w["attn.attn_sink"])
        if attn.compressor is not None:
            c = attn.compressor
            if c.compress_ratio == 2:
                c.fused_wkv_wgate.weight.copy_(torch.cat([w["attn.compressor.wkv.weight"],
                                                          w["attn.compressor.wgate.weight"]]).half())
            else:
                c.wkv.weight.copy_(w["attn.compressor.wkv.weight"].half())
            c.norm.weight.copy_(w["attn.compressor.norm.weight"])
            attn.indexer.wk.weight.copy_(w["attn.indexer.wk.weight"].half())
            attn.indexer.k_norm.weight.copy_(w["attn.indexer.k_norm.weight"])
        if attn.indexer is not None:
            attn.indexer.wq_b.weight.copy_(w["attn.indexer.wq_b.weight"].half())
            attn.indexer.weights_proj.weight.copy_(w["attn.indexer.weights_proj.weight"].half())


def layer_inputs(w: dict[str, torch.Tensor], S: int, seed: int, device: torch.device) -> torch.Tensor:
    """x = attn_norm(random stream row) as FP16 [S, 5120] (unit RMS times the layer's attn_norm weight)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.randn(S, 5120, generator=g).to(device)
    h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-20)
    return (h * w["attn_norm.weight"]).half()


# ============================================================================ FP32 reference transcription
def _freqs_cis(cfg: SimpleNamespace, ratio: int, seqlen: int, device) -> torch.Tensor:
    dim = cfg.qk_rope_head_dim
    rs = cfg.rope_scaling
    base = cfg.compress_rope_theta if ratio else cfg.rope_theta
    orig = rs["original_max_position_embeddings"] if ratio else 0
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if orig > 0:
        def cd(rot):
            return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(cd(rs["beta_fast"])), 0)
        high = min(math.ceil(cd(rs["beta_slow"])), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / rs["factor"] * (1 - smooth) + freqs * smooth
    return torch.polar(torch.ones(seqlen, dim // 2, device=device),
                       torch.outer(torch.arange(seqlen, device=device), freqs))


def _rot(x: torch.Tensor, fc: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """apply_rotary_emb on the last 64 dims; x [S, ..., D] FP32, fc [S, 32] complex -> new tensor."""
    y = x.clone()
    xc = torch.view_as_complex(x[..., -64:].float().contiguous().unflatten(-1, (-1, 2)))
    f = fc.conj() if inverse else fc
    f = f.view(f.shape[0], *([1] * (xc.dim() - 2)), f.shape[-1])
    y[..., -64:] = torch.view_as_real(xc * f).flatten(-2)
    return y


def _rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20) * w


def ref_select_candidate_blocks(logits: torch.Tensor, compress_lens: torch.Tensor) -> torch.Tensor:
    """ref:m.py:583-610 verbatim (prefill form)."""
    width = logits.size(-1)
    scores = torch.nn.functional.pad(logits, (0, -width % CAND_BLOCK), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, CAND_BLOCK)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // CAND_BLOCK
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(CAND_TOPK_BLOCKS, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(CAND_BLOCK, dim=-1)[..., :width]


@dataclass
class RefState:
    """SharedAttentionRuntime of the reference (one sequence): per kv source compressed KV + index K, the
    last published top-k and candidate mask."""
    ckv: dict = field(default_factory=dict)
    ik: dict = field(default_factory=dict)
    topk: torch.Tensor | None = None
    cand: torch.Tensor | None = None


def ref_attention(cfg: SimpleNamespace, topo: LayerTopology, w: dict[str, torch.Tensor], x16: torch.Tensor,
                  st: RefState, record: dict | None = None, q_rows: torch.Tensor | None = None,
                  score_rows: int = 256, ckv_records: torch.Tensor | None = None) -> torch.Tensor:
    """One layer of the reference over a whole sequence (start_pos 0), FP32 math, QAT kept. Returns out [S, 5120]
    FP32 (wo_b output). ``record`` receives intermediates (golden names of §3.8)."""
    rec = record if record is not None else {}
    x = x16.float()
    S = x.shape[0]
    dev = x.device
    r = topo.compress_ratio
    fc = _freqs_cis(cfg, r, S, dev)
    qr = _rms(x @ w["attn.wq_a.weight"].t(), w["attn.q_norm.weight"])
    q = _rot((qr @ w["attn.wq_b.weight"].t()).view(S, 64, 512), fc)
    kv = _rms(x @ w["attn.wkv.weight"].t(), w["attn.kv_norm.weight"])
    kv_win = qat.fp8_block32_qdq(_rot(kv, fc), out_dtype=torch.float32, impl="torch")
    rec.update({"attn.qr": qr, "attn.q": q, "attn.kv_win": kv_win})
    topk = None
    if r:
        n_all = S // r
        lens = (torch.arange(1, S + 1, device=dev) // r)
        if topo.owns_compressor:
            if r == 1:
                latent = _rms(x @ w["attn.compressor.wkv.weight"].t(), w["attn.compressor.norm.weight"])
            else:
                xx = x[: n_all * r]
                kvc = (xx @ w["attn.compressor.wkv.weight"].t()).view(n_all, r, 512)
                sc = (xx @ w["attn.compressor.wgate.weight"].t()).view(n_all, r, 512)
                latent = _rms((kvc * sc.softmax(dim=1)).sum(dim=1), w["attn.compressor.norm.weight"])
            fpos = fc[: n_all * r: r]
            k = _rms(latent @ w["attn.indexer.wk.weight"].t(), w["attn.indexer.k_norm.weight"])
            st.ik[topo.layer_id] = qat.fp4_e8m0_qdq(_rot(k, fpos), out_dtype=torch.float32, impl="torch")
            st.ckv[topo.layer_id] = qat.fp4_e4m3_qdq(_rot(latent, fpos), out_dtype=torch.float32, impl="torch")
            if ckv_records is not None:          # substitution experiments: another path's compressed records
                if ckv_records.shape != st.ckv[topo.layer_id].shape:
                    raise ValueError(f"ckv_records {tuple(ckv_records.shape)} != {tuple(st.ckv[topo.layer_id].shape)}")
                st.ckv[topo.layer_id] = ckv_records.float()
            rec.update({"attn.latent": latent, "attn.ckv": st.ckv[topo.layer_id], "idx.k": st.ik[topo.layer_id]})
        if topo.owns_indexer:
            iq = qat.fp4_e8m0_qdq(_rot((qr @ w["attn.indexer.wq_b.weight"].t()).view(S, 32, 128), fc),
                                  out_dtype=torch.float32, impl="torch")
            wts = (x @ w["attn.indexer.weights_proj.weight"].t()) * (128 ** -0.5 * 32 ** -0.5)
            keys = st.ik[topo.kv_source][:n_all]
            kk = min(512, n_all)
            topk = torch.full((S, 512), -1, dtype=torch.int64, device=dev)
            cand_rows = []
            rows = q_rows if q_rows is not None else torch.arange(S, device=dev)
            scores_rec = {}
            for a in range(0, rows.numel(), score_rows):
                rr = rows[a: a + score_rows]
                sc = torch.einsum("thd,nd->thn", iq[rr], keys).relu_().mul_(wts[rr][..., None]).sum(dim=1)
                sc.masked_fill_(torch.arange(n_all, device=dev)[None, :] >= lens[rr][:, None], -torch.inf)
                if topo.is_candidate_source:
                    cm = ref_select_candidate_blocks(sc, lens[rr][:, None])
                    cand_rows.append((rr, cm))
                elif topo.uses_candidates:
                    sc = sc.masked_fill(~st.cand[rr], -torch.inf)
                idx = sc.topk(kk, dim=-1, sorted=False).indices.sort(dim=-1).values
                topk[rr, :kk] = torch.where(idx < lens[rr][:, None], idx, -1)
                scores_rec[a] = (rr, sc)
            if topo.is_candidate_source:
                st.cand = torch.zeros((S, n_all), dtype=torch.bool, device=dev)
                for rr, cm in cand_rows:
                    st.cand[rr] = cm
            st.topk = topk
            rec.update({"idx.q": iq, "idx.w": wts, "idx.scores": scores_rec})
        topk = st.topk
        rec["idx.topk"] = topk
    rows = q_rows if q_rows is not None else torch.arange(S, device=dev)
    o = ref_attend(q, kv_win, st.ckv[topo.kv_source] if r else None, topk, w["attn.attn_sink"], rows)
    rec["attn.o"] = o
    out = ref_oproj(o, w, fc)
    rec["attn.out"] = out
    return out


def ref_attend(q: torch.Tensor, kv_win: torch.Tensor, ckv: torch.Tensor | None, topk: torch.Tensor | None,
               sink: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """ref:k.py sparse_attn in FP32 for query ``rows``: window [p-127, p] of ``kv_win`` + ``ckv[topk[p]]``."""
    S = q.shape[0]
    dev = q.device
    o = torch.zeros((S, q.shape[1], 512), device=dev)
    sink = sink.float()
    for a in range(0, rows.numel(), 128):
        rr = rows[a: a + 128]
        win = rr[:, None] - 127 + torch.arange(128, device=dev)[None, :]
        wv = win >= 0
        keys, valid = [kv_win[win.clamp(min=0)].float()], [wv]
        if ckv is not None:
            ti = topk[rr]
            keys.append(ckv[ti.clamp(min=0)].float())
            valid.append(ti >= 0)
        kk_ = torch.cat(keys, 1)
        vv = torch.cat(valid, 1)
        kk_ = kk_.masked_fill(~vv[..., None], 0)
        s = torch.einsum("thd,tkd->thk", q[rr].float(), kk_) * 512 ** -0.5
        s.masked_fill_(~vv[:, None, :], -torch.inf)
        m = s.amax(-1).clamp(min=-1e30)
        p = torch.exp(s - m[..., None])
        den = p.sum(-1) + torch.exp(sink[None, :] - m)
        o[rr] = torch.einsum("thk,tkd->thd", p, kk_) / den[..., None]
    return o


def ref_oproj(o: torch.Tensor, w: dict[str, torch.Tensor], fc: torch.Tensor) -> torch.Tensor:
    """inverse RoPE, grouped wo_a einsum, wo_b (ref:m.py:781-788), FP32."""
    S = o.shape[0]
    o = _rot(o, fc, inverse=True)
    z = torch.einsum("sgd,grd->sgr", o.view(S, 8, 4096), w["attn.wo_a.weight"].view(8, 1024, 4096))
    return z.flatten(1) @ w["attn.wo_b.weight"].t()


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp(min=1e-30))
