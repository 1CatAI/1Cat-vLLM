# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-MOE sub-item 3: A/B of the SM70 MoE block on REAL DeepSeek-V4.1 weights (verified shards only) against an
FP32-dequantised reference of the official MoE (ref:m.py:792-903; no activation quant = the ``v100-semantic``
comparator of PORT_DESIGN §3.8/§4.5).

Inputs: real token embeddings (``embed.weight``, shard 2) of a deterministic text, RMS-normalised and scaled by the
layer's ``ffn_norm`` weight, rounded to FP16 -- the same FP16 input goes to port and reference. Downstream top-1:
the next backbone layer's router decision (top-1 and top-6 of ``sqrt(softplus(logits)) + bias``) on
``ffn_norm(x + moe(x))`` -- the LM head shard is used instead once it is verified (``--head``).

pytest runs TP1 on one GPU; the CLI also runs TP4 across four GPUs (one process per rank, NCCL):

    CUDA_VISIBLE_DEVICES=<4 uuids> PYTHONPATH=<worktree> python -m tests.models.deepseek_v41.test_moe_realweights \\
        --layers 3 10 --tp 4 --tokens 1 2 8 64 512 4096 --out /mnt/nvme2/scratch/ds41/l-moe/ab
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

CKPT = Path(os.environ.get("DS41_CHECKPOINT_DIR", "/mnt/nvme2/models/DeepSeek-V4.1-Flash"))
VERIFY = Path(os.environ.get("DS41_VERIFY_SHARD", "/mnt/nvme2/models/_dl-logs/verify_shard.sh"))
TOKENIZER = Path("/mnt/nvme2/scratch/ds41/model-ref/tokenizer.json")
TEXT_FILES = (Path("/mnt/nvme2/scratch/ds41/model-ref/README.md"),
              Path("/mnt/nvme2/scratch/ds41/model-ref/inference/model.py"))
HIDDEN, INTER = 5120, 2304


def shard_of(prefix: str) -> str:
    if prefix.startswith("mtp."):
        return f"model-{44 + int(prefix.split('.')[1]):05d}-of-00048.safetensors"
    return f"model-{int(prefix.split('.')[1]) + 3:05d}-of-00048.safetensors"


def verified(basename: str) -> Path:
    out = subprocess.run(["bash", str(VERIFY), basename], capture_output=True, text=True, timeout=900)
    if out.returncode != 0 or not out.stdout.strip().startswith("OK"):
        raise RuntimeError(f"shard {basename} not verified: {out.stdout.strip()} {out.stderr.strip()}")
    return CKPT / basename


class FFNTensors:
    """Lazy view of ``{prefix}.ffn.*`` / ``{prefix}.ffn_norm.weight`` in a verified shard: each access reads one
    tensor from the mmapped file (no whole-layer copy in host RAM)."""

    def __init__(self, prefix: str) -> None:
        from safetensors import safe_open

        self.prefix = prefix
        self._f = safe_open(str(verified(shard_of(prefix))), framework="pt")

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._f.get_tensor(f"{self.prefix}.{name}")


def load_ffn(prefix: str) -> FFNTensors:
    return FFNTensors(prefix)


def ffn_inputs(num_tokens: int, norm_weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    """FP16 [num_tokens, 5120]: ffn_norm(embed[token ids of a fixed text])."""
    from safetensors import safe_open
    from tokenizers import Tokenizer

    text = "\n".join(p.read_text(errors="replace") for p in TEXT_FILES)
    ids = Tokenizer.from_file(str(TOKENIZER)).encode(text).ids
    while len(ids) < num_tokens:
        ids = ids + ids
    ids = torch.tensor([0] + ids[: num_tokens - 1], dtype=torch.long)  # BOS first
    with safe_open(str(verified("model-00002-of-00048.safetensors")), framework="pt") as f:
        emb = f.get_tensor("embed.weight")[ids].to(device=device, dtype=torch.float32)
    return rms_norm(emb, norm_weight.to(device)).half()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()


def _deq_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    s = scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)[: weight.shape[0], : weight.shape[1]]
    return weight.float() * s


def reference_moe(t: FFNTensors, x16: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
    """Official MoE in FP32 (dequantised weights, unrounded activations), on x16's device."""
    from vllm.models.deepseek_v41.sm70.moe_method import dequant_mxfp4

    dev = x16.device
    x = x16.float()
    logits = x @ t["ffn.gate.weight"].to(dev).float().t()
    scores = F.softplus(logits).sqrt()
    ids = (scores + t["ffn.gate.bias"].to(dev).float()).topk(top_k, dim=-1)[1]
    w = scores.gather(1, ids)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * 1.5
    y = torch.zeros_like(x)
    for e in torch.unique(ids).tolist():
        tok, pos = torch.where(ids == e)

        def deq(n: str) -> torch.Tensor:
            return dequant_mxfp4(t[f"ffn.experts.{e}.{n}.weight"].to(dev).view(torch.uint8),
                                 t[f"ffn.experts.{e}.{n}.scale"].to(dev).view(torch.uint8))

        xe = x[tok]
        gate = (xe @ deq("w1").t()).clamp(max=10.0)
        up = (xe @ deq("w3").t()).clamp(-10.0, 10.0)
        h = F.silu(gate) * up * w[tok, pos, None]
        y.index_add_(0, tok, h @ deq("w2").t())
    sh = {n: _deq_fp8(t[f"ffn.shared_experts.{n}.weight"].to(dev), t[f"ffn.shared_experts.{n}.scale"].to(dev))
          for n in ("w1", "w2", "w3")}
    gate = (x @ sh["w1"].t()).clamp(max=10.0)
    up = (x @ sh["w3"].t()).clamp(-10.0, 10.0)
    y = y + (F.silu(gate) * up) @ sh["w2"].t()
    return {"out": y, "logits": logits, "ids": ids, "weights": w}


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(hidden_size=HIDDEN, moe_intermediate_size=INTER, swiglu_limit=10.0,
                           routed_scaling_factor=1.5, scoring_func="sqrtsoftplus", topk_method="noaux_tc",
                           norm_topk_prob=True, n_shared_experts=1)


def build_port(t: FFNTensors, n_experts: int, top_k: int, layer_id: int, spill=None):
    """DeepseekV41MoE for this TP rank, loaded through the production loaders (expert TP slicing, gate
    exponent bias, shared expert via MergedColumn/RowParallel with the FP8 dequantised exactly to FP16)."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.models.deepseek_v41.sm70 import moe as v41_moe
    from vllm.utils.torch_utils import set_default_torch_dtype

    from .test_moe_layer import _V41TestQuantConfig

    vllm_config = VllmConfig()
    object.__setattr__(vllm_config, "model_config", SimpleNamespace(hf_config=_hf_config(), dtype=torch.float16))
    object.__setattr__(vllm_config, "quant_config", _V41TestQuantConfig())
    vllm_config.compilation_config.static_forward_context = {}
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.float16), torch.device("cuda"):
        block = v41_moe.DeepseekV41MoE(vllm_config, f"model.layers.{layer_id}.ffn", layer_id,
                                       n_routed_experts=n_experts, top_k=top_k, spill=spill)
    block.gate.weight.weight_loader(block.gate.weight, t["ffn.gate.weight"])
    block.gate.e_score_correction_bias.data.copy_(t["ffn.gate.bias"])
    for e in range(n_experts):
        for shard in ("w1", "w2", "w3"):
            base = "w13" if shard in ("w1", "w3") else "w2"
            for suffix, key in (("weight", "weight"), ("weight_scale", "scale")):
                param = getattr(block.experts, f"{base}_{suffix}")
                param.weight_loader(param, t[f"ffn.experts.{e}.{shard}.{key}"], f"experts.{base}_{suffix}",
                                    shard_id=shard, expert_id=e, return_success=True)
    tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    del tp, rank  # slicing is done by the parallel linears' loaders
    gu, dp = block.shared_experts.gate_up_proj, block.shared_experts.down_proj
    for idx, n in enumerate(("w1", "w3")):
        w = _deq_fp8(t[f"ffn.shared_experts.{n}.weight"].cuda(), t[f"ffn.shared_experts.{n}.scale"].cuda())
        assert torch.equal(w.half().float(), w), "dense FP8 x UE8M0 must be exact in FP16"
        gu.weight_loader(gu.weight, w.half(), idx)
    w = _deq_fp8(t["ffn.shared_experts.w2.weight"].cuda(), t["ffn.shared_experts.w2.scale"].cuda())
    assert torch.equal(w.half().float(), w)
    dp.weight_loader(dp.weight, w.half())
    block.experts.quant_method.process_weights_after_loading(block.experts)
    return block


def metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    out, ref = out.float(), ref.float()
    d = out - ref
    cos = F.cosine_similarity(out, ref, dim=-1)
    return {"rel_rms": (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(),
            "max_abs": d.abs().max().item(), "max_rel_to_scale": (d.abs().max() / ref.abs().max()).item(),
            "ref_max_abs": ref.abs().max().item(), "cos_min": cos.min().item(), "cos_median": cos.median().item()}


def downstream(next_t: FFNTensors | None, x16: torch.Tensor, out: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
    """Next layer's router on ffn_norm_next(x + out): (top-7 ids, top-7 biased scores); index 0 = top-1."""
    if next_t is None:
        return None
    h = rms_norm(x16.float() + out.float(), next_t["ffn_norm.weight"].to(x16.device))
    scores = F.softplus(h @ next_t["ffn.gate.weight"].to(x16.device).float().t()).sqrt()
    vals, ids = (scores + next_t["ffn.gate.bias"].to(x16.device).float()).topk(7, dim=-1)
    return ids, vals


def compare(port: dict[str, torch.Tensor], ref: dict[str, torch.Tensor], down_port, down_ref) -> dict:
    res = metrics(port["out"], ref["out"])
    res["logits_rel_rms"] = metrics(port["logits"], ref["logits"])["rel_rms"]
    same = port["ids"].sort(-1)[0].long().eq(ref["ids"].sort(-1)[0].long()).all(-1)
    res["router_topk_set_agree"] = same.float().mean().item()
    if not bool(same.all()):
        # a flip is admissible only for near-ties (§4.5): relative gap between the k-th and (k+1)-th choice score
        s = F.softplus(ref["logits"]).sqrt() + ref["bias"]
        top = s.topk(port["ids"].shape[1] + 1, dim=-1)[0]
        gap = ((top[:, -2] - top[:, -1]) / top[:, -2].abs())[~same]
        res["router_flip_max_rel_gap"] = gap.max().item()
    if down_port is not None:
        (pid, _), (rid, rval) = down_port, down_ref
        top1 = pid[:, 0].eq(rid[:, 0])
        res["downstream_top1_agree"] = top1.float().mean().item()
        res["downstream_top1_flips"] = int((~top1).sum().item())
        top6 = pid[:, :6].sort(-1)[0].eq(rid[:, :6].sort(-1)[0]).all(-1)
        res["downstream_top6_set_agree"] = top6.float().mean().item()
        # reference relative score gaps at the flipped decision boundaries (near-tie evidence)
        if not bool(top1.all()):
            res["downstream_top1_flip_max_rel_gap"] = ((rval[:, 0] - rval[:, 1]) / rval[:, 0].abs())[~top1].max().item()
        if not bool(top6.all()):
            res["downstream_top6_flip_max_rel_gap"] = ((rval[:, 5] - rval[:, 6]) / rval[:, 5].abs())[~top6].max().item()
            res["downstream_top6_flip_rows"] = torch.nonzero(~top6).flatten()[:8].tolist()
    return res


def run_rank(rank: int, world: int, port: int, args: argparse.Namespace) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    torch.cuda.set_device(rank)
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(world, 1)
    report: dict = {"tp": world, "backend": os.environ.get("VLLM_DS41_MOE_BACKEND", "skinny"),
                    "impl": os.environ.get("VLLM_DS41_MOE_IMPL", "sm70"), "layers": {}}
    for prefix in args.layers:
        n_experts, top_k = (128, 3) if prefix.startswith("mtp.") else (384, 6)
        layer_id = 40 + int(prefix.split(".")[1]) if prefix.startswith("mtp.") else int(prefix.split(".")[1])
        t = load_ffn(prefix)
        next_t = None
        if not prefix.startswith("mtp.") and layer_id + 1 < 40 and args.downstream:
            next_t = load_ffn(f"layers.{layer_id + 1}")
        t0 = time.time()
        block = build_port(t, n_experts, top_k, layer_id)
        load_s = time.time() - t0
        oracle = spilled = None
        if args.oracle:  # the in-tree torch impl (FP32-dequantised experts, same rounding points as the kernels)
            saved = os.environ.get("VLLM_DS41_MOE_IMPL")
            os.environ["VLLM_DS41_MOE_IMPL"] = "torch"
            try:
                oracle = build_port(t, n_experts, top_k, layer_id + 100)
            finally:
                if saved is None:
                    os.environ.pop("VLLM_DS41_MOE_IMPL")
                else:
                    os.environ["VLLM_DS41_MOE_IMPL"] = saved
        if args.spill_check and not prefix.startswith("mtp."):
            from vllm.models.deepseek_v41.sm70.moe_method import make_spill_plan

            spilled = build_port(t, n_experts, top_k, layer_id + 200, spill=make_spill_plan(args.spill_check))
        x_all = ffn_inputs(max(args.tokens), t["ffn_norm.weight"], torch.device("cuda"))
        per_t = {}
        for num_tokens in args.tokens:
            x16 = x_all[:num_tokens].contiguous()
            logits = block.gate_logits(x16)
            _, ids = block.route(logits)
            out = block(x16)
            extra: dict[str, object] = {}
            if oracle is not None:
                ref_o = oracle(x16)
                diff = (out - ref_o).abs()
                extra["vs_oracle_rel_rms"] = (diff.pow(2).mean().sqrt() / ref_o.pow(2).mean().sqrt()).item()
                extra["vs_oracle_frac_diff"] = (diff > 0).float().mean().item()
            if spilled is not None:
                extra["spill_bitwise_equal"] = bool(torch.equal(spilled(x16), out))
            if rank == 0:
                ref = reference_moe(t, x16, top_k)
                ref["bias"] = t["ffn.gate.bias"].cuda().float()
                per_t[str(num_tokens)] = compare({"out": out, "logits": logits, "ids": ids}, ref,
                                                 downstream(next_t, x16, out), downstream(next_t, x16, ref["out"]))
                per_t[str(num_tokens)].update(extra)
                print(f"[{prefix} TP{world} T={num_tokens}] {json.dumps(per_t[str(num_tokens)])}", flush=True)
        if rank == 0:
            report["layers"][prefix] = {"load_s": load_s, "gate_weight_exp": block.gate.weight_exp, "T": per_t}
        # free everything of this layer before the next one (several TP1 layers with --oracle otherwise OOM)
        del block, oracle, spilled, t, next_t, x_all, x16, logits, ids, out
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0 and args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        tag = ("_oracle" if args.oracle else "") + (f"_spill{args.spill_check}" if args.spill_check else "")
        name = f"ab_tp{world}_{report['impl']}_{report['backend']}{tag}_{'_'.join(args.layers)}.json"
        (Path(args.out) / name).write_text(json.dumps(report, indent=1))
    torch.distributed.barrier()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", nargs="+", default=["layers.3", "layers.10"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 8, 64, 512, 4096])
    ap.add_argument("--out", default="/mnt/nvme2/scratch/ds41/l-moe/ab")
    ap.add_argument("--no-downstream", dest="downstream", action="store_false")
    ap.add_argument("--oracle", action="store_true", help="also compare against the in-tree torch impl")
    ap.add_argument("--spill-check", type=int, default=0,
                    help="also build a block with this many spilled experts and require bitwise equality")
    args = ap.parse_args(argv)
    args.layers = [p if p.startswith(("layers.", "mtp.")) else f"layers.{p}" for p in args.layers]
    if torch.cuda.device_count() < args.tp:
        raise SystemExit(f"--tp {args.tp} needs {args.tp} visible GPUs, have {torch.cuda.device_count()}")
    torch.multiprocessing.spawn(run_rank, args=(args.tp, _free_port(), args), nprocs=args.tp, join=True)


@pytest.mark.sm70
@pytest.mark.weights
@pytest.mark.parametrize("layer", ["layers.3"])
def test_real_weights_tp1(layer: str, tmp_path: Path) -> None:
    if not (CKPT / shard_of(layer)).exists():
        pytest.skip(f"{shard_of(layer)} not downloaded")
    out = tmp_path
    main(["--layers", layer, "--tp", "1", "--tokens", "1", "8", "512", "--out", str(out)])
    report = json.loads((out / f"ab_tp1_sm70_{os.environ.get('VLLM_DS41_MOE_BACKEND', 'skinny')}_{layer}.json")
                        .read_text())
    for num_tokens, res in report["layers"][layer]["T"].items():
        assert res["rel_rms"] <= 3e-3, (num_tokens, res)          # §4.5 MoE layer output
        assert res["logits_rel_rms"] <= 1e-5, (num_tokens, res)   # §4.5 router logits
        assert res["router_topk_set_agree"] == 1.0 or res["router_flip_max_rel_gap"] < 1e-6, (num_tokens, res)


if __name__ == "__main__":
    sys.exit(main())
