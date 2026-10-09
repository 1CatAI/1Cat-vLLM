# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-CORE P2 item 7: layer-subset smoke at TP4 with REAL weights and the test-only stand-ins (core_stubs.py).

Builds DeepseekV41ForCausalLM for backbone layers {0,1,2,3} (+ embed, + final norm/head when their shard is
verified) on 4 GPUs, loads every tensor of those layers from VERIFIED shards only (verify_shard.sh), runs the
real FP8->FP16 exact dequant, a forward over a 7-token prompt, and records per-GPU memory against PORT_DESIGN §6.1.
Usage (pin the GPUs by UUID):
  CUDA_VISIBLE_DEVICES=<4 UUIDs> PYTHONPATH=<worktree> python tests/models/deepseek_v41/smoke_core_tp4.py OUT.json
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

CKPT = Path("/mnt/nvme2/models/DeepSeek-V4.1-Flash")
ENGRAM_DIR = Path("/home/mermiges/ds41-engram")
VERIFY = "/mnt/nvme2/models/_dl-logs/verify_shard.sh"
INDEX = Path("/mnt/nvme2/scratch/ds41/model-ref/model.safetensors.index.json")
TOKENIZER = Path("/mnt/nvme2/scratch/ds41/model-ref/tokenizer.json")
LAYERS = (0, 1, 2, 3)
GIB = 2**30


def _needed_shards() -> dict[str, list[str]]:
    weight_map = json.loads(INDEX.read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        m = re.match(r"layers\.(\d+)\.", name)
        if (m and int(m.group(1)) in LAYERS and ".engram.embed." not in name) or name in (
                "embed.weight", "head.weight", "norm.weight"):
            shards.setdefault(shard, []).append(name)
    return shards


def _verified(shard: str) -> bool:
    if shard in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"):
        # D5: the Engram shards count as verified only through ranged_dl.py's .sha256-ok sidecar
        if not (ENGRAM_DIR / f"{shard}.sha256-ok").is_file():
            return False
    if not ((CKPT / shard).is_file() or (ENGRAM_DIR / shard).is_file()):
        return False
    proc = subprocess.run(["bash", VERIFY, shard], capture_output=True, text=True, timeout=900)
    return proc.returncode == 0 and proc.stdout.strip().startswith("OK")


def _shard_path(shard: str) -> Path:
    return CKPT / shard if (CKPT / shard).is_file() else ENGRAM_DIR / shard


def _worker(rank: int, port: int, shards: list[str], out_dir: str) -> None:
    from safetensors import safe_open

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tests.models.deepseek_v41 import core_stubs
    from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.models.deepseek_v41.quant_config import DeepseekV41FP8Config
    from vllm.utils.torch_utils import set_default_torch_dtype

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    os.environ["VLLM_DS41_CORE_LAYER_SUBSET"] = ",".join(map(str, LAYERS))
    os.environ["VLLM_DS41_CORE_ALLOW_LAYER_SUBSET"] = "1"
    core_stubs.install()
    report: dict = {"rank": rank, "device": torch.cuda.get_device_name(rank)}
    from vllm.config import ModelConfig

    mc = ModelConfig(model=str(CKPT), dtype="half", skip_tokenizer_init=True, max_model_len=32768)
    hf = mc.hf_config
    vc = VllmConfig()
    object.__setattr__(vc, "model_config", mc)
    object.__setattr__(vc, "quant_config", DeepseekV41FP8Config.from_config(dict(hf.quantization_config)))
    vc.scheduler_config.max_num_batched_tokens = 256
    vc.compilation_config.cudagraph_mode = CUDAGraphMode.NONE       # eager (also what the golden dump needs)
    with set_current_vllm_config(vc):
        init_distributed_environment(world_size=4, rank=rank, local_rank=rank, backend="nccl",
                                     distributed_init_method=f"tcp://127.0.0.1:{port}")
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
        base = torch.cuda.memory_allocated(dev)
        t0 = time.time()
        from vllm.models.deepseek_v41.sm70.model import DeepseekV41ForCausalLM

        with torch.device(dev), set_default_torch_dtype(torch.float16):
            model = DeepseekV41ForCausalLM(vllm_config=vc)
        report["build_s"] = round(time.time() - t0, 2)
        report["alloc_after_build_GiB"] = (torch.cuda.memory_allocated(dev) - base) / GIB

        read_bytes = 0

        def weights():
            nonlocal read_bytes
            for shard in shards:
                with safe_open(str(_shard_path(shard)), framework="pt", device="cpu") as f:
                    for name in f.keys():
                        if model.skip_checkpoint_weight(name):
                            continue
                        tensor = f.get_tensor(name)
                        read_bytes += tensor.numel() * tensor.element_size()
                        yield name, tensor

        t0 = time.time()
        loaded = model.load_weights(weights())
        report["load_s"] = round(time.time() - t0, 2)
        report["read_GiB"] = read_bytes / GIB
        params = dict(model.named_parameters())
        report["params_total"] = len(params)
        report["params_loaded"] = len(loaded)
        report["params_not_loaded"] = sorted(set(params) - loaded)
        del params   # holds the pre-dequant FP8 tensors; drop it before measuring
        t0 = time.time()
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if method is not None and hasattr(method, "process_weights_after_loading"):
                method.process_weights_after_loading(module)
        torch.cuda.synchronize(dev)
        report["dequant_s"] = round(time.time() - t0, 2)
        torch.cuda.empty_cache()
        report["alloc_after_load_GiB"] = (torch.cuda.memory_allocated(dev) - base) / GIB
        params = dict(model.named_parameters())

        # per-module bytes (what §6.1 counts)
        def nbytes(prefix: str) -> int:
            return sum(p.numel() * p.element_size() for n, p in params.items() if n.startswith(prefix))

        report["bytes"] = {
            "embed": nbytes("model.embed_tokens."), "head": nbytes("lm_head."),
            **{f"layer{layer}_dense": nbytes(f"model.layers.{layer}.") - nbytes(f"model.layers.{layer}.ffn.experts.")
               for layer in LAYERS},
            **{f"layer{layer}_experts": nbytes(f"model.layers.{layer}.ffn.experts.") for layer in LAYERS},
        }
        report["dtypes"] = {n: str(p.dtype) for n, p in params.items()
                            if n.startswith("model.layers.2.") and ".experts." not in n}

        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(TOKENIZER))
        ids = [0] + tok.encode("The capital of France is", add_special_tokens=False).ids
        input_ids = torch.tensor(ids, device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
        with torch.inference_mode():
            t0 = time.time()
            hidden = model(input_ids, torch.arange(len(ids), device=dev), None)
            torch.cuda.synchronize(dev)
            report["forward_s"] = round(time.time() - t0, 3)
            report["hidden"] = {"shape": list(hidden.shape), "dtype": str(hidden.dtype),
                                "finite": bool(torch.isfinite(hidden).all()),
                                "absmax": float(hidden.float().abs().max()),
                                "checksum": float(hidden.double().sum())}
            if "lm_head.weight" in loaded:
                logits = model.compute_logits(hidden)
                report["logits"] = {"shape": list(logits.shape), "dtype": str(logits.dtype),
                                    "finite": bool(torch.isfinite(logits).all()),
                                    "top1": logits.argmax(-1).tolist()}
        report["token_ids"] = ids
        report["peak_forward_GiB"] = (torch.cuda.max_memory_allocated(dev) - base) / GIB
        report["torch_reserved_GiB"] = torch.cuda.memory_reserved(dev) / GIB
        free, total = torch.cuda.mem_get_info(dev)
        report["device_used_GiB"] = (total - free) / GIB
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme2/scratch/ds41/l-core/smoke_tp4.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != 4:
        raise SystemExit(f"need exactly 4 visible GPUs (pinned by UUID), got {torch.cuda.device_count()}")
    needed = _needed_shards()
    usable, unverified = [], []
    for shard in sorted(needed):
        (usable if _verified(shard) else unverified).append(shard)
    print("verified shards:", usable, "\nnot verified (skipped):", unverified, flush=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.start_processes(_worker, args=(port, usable, str(out.parent)), nprocs=4, start_method="spawn")
    ranks = [json.loads(Path(out.parent, f"rank{r}.json").read_text()) for r in range(4)]
    summary = {"layers": LAYERS, "verified_shards": usable, "unverified_shards_skipped": unverified,
               "skipped_tensors": {s: needed[s] for s in unverified}, "ranks": ranks,
               "hidden_identical_across_ranks": len({r["hidden"]["checksum"] for r in ranks}) == 1}
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "ranks"}, indent=1))
    for r in ranks:
        print({k: r[k] for k in ("rank", "alloc_after_load_GiB", "peak_forward_GiB", "device_used_GiB",
                                 "load_s", "dequant_s", "forward_s", "hidden")})


if __name__ == "__main__":
    main()
