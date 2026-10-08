# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture real dense-attention request Q/K/V from an installed FP16-KV model.

Diagnostic eager execution only: hooks synchronize/copy and cannot be timed.
Single-request, first-prefill capture avoids guessing prefix/chunk positions.
QSA/MLA and speculative draft capture need their own explicit mask/role adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import torch


def verify_runtime_wheel(wheel, vllm_root):
    """Reject source overlays and a provenance wheel different from the runtime."""
    import flash_attn_v100.flash_attn_v100_cuda as extension

    paths = {
        "vllm/v1/attention/backends/flash_attn_v100.py": (
            vllm_root / "v1/attention/backends/flash_attn_v100.py"
        ),
        "vllm/_C.abi3.so": vllm_root / "_C.abi3.so",
        f"flash_attn_v100/{Path(extension.__file__).name}": Path(extension.__file__),
    }
    hashes = {}
    with ZipFile(wheel) as archive:
        for name, path in paths.items():
            installed = path.read_bytes()
            if "site-packages" not in path.parts or archive.read(name) != installed:
                raise ValueError(f"Installed runtime does not match the wheel: {name}")
            hashes[name] = hashlib.sha256(installed).hexdigest()
    return hashes


def install_capture(model, directory, token_ids, provenance):
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends import flash_attn_v100 as backend

    if not backend._route_summary_enabled():
        raise ValueError("Capture route assertions require VLLM_SM70_DEBUG=routing")

    layers = [module for module in model.modules() if isinstance(module, Attention)]
    if not layers:
        raise ValueError(
            "No dense Attention layers; QSA/MLA capture is not implemented"
        )
    selected = [layers[i] for i in sorted({0, len(layers) // 2, len(layers) - 1})]
    rank = get_tensor_model_parallel_rank()
    tp = get_tensor_model_parallel_world_size()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "handles": [],
        "samples": [],
        "pending": {},
        "layers": selected,
        "route_snapshot": dict(backend._route_counts),
        "rank": rank,
    }
    model._sm70_kv_capture = state
    tokens = torch.tensor(token_ids, dtype=torch.int64)
    token_sha = hashlib.sha256(
        tokens.numpy().astype("<i8", copy=False).tobytes()
    ).hexdigest()

    def before(module, inputs):
        if any(row["layer"] == module.layer_name for row in state["samples"]):
            return
        if (
            module.attn_type != AttentionType.DECODER
            or module.kv_sharing_target_layer_name
        ):
            raise ValueError("Capture requires independent causal decoder KV")
        if module.attn_backend.get_name() != "FLASH_ATTN_V100":
            raise ValueError("The requested Flash-V100 model route was not selected")
        query, key, value = inputs[:3]
        if query.shape[0] != len(token_ids) or key is None or value is None:
            raise ValueError(
                "Expected the complete first prefill; no padding/prefix/chunks"
            )
        if any(t.dtype != torch.float16 for t in (query, key, value)):
            raise ValueError("Capture requires unquantized FP16 Q/K/V")
        metadata = get_forward_context().attn_metadata[module.layer_name]
        boundaries = metadata.query_start_loc.detach().cpu().tolist()
        lengths = metadata.seq_lens.detach().cpu().tolist()
        if boundaries != [0, len(token_ids)] or lengths != [len(token_ids)]:
            raise ValueError(
                "Request metadata does not describe a single first prefill"
            )
        positions = torch.arange(len(token_ids), dtype=torch.int64)
        query_positions = positions[-min(16, len(token_ids)) :]
        allowed = positions[None, :] <= query_positions[:, None]
        if module.sliding_window is not None:
            allowed &= positions[None, :] > (
                query_positions[:, None] - module.sliding_window
            )
        state["pending"][module.layer_name] = {
            "q": query.detach()
            .view(-1, module.num_heads, module.head_size)[
                query_positions.to(query.device)
            ]
            .cpu()
            .contiguous(),
            "k": key.detach()
            .view(-1, module.num_kv_heads, module.head_size)
            .cpu()
            .contiguous(),
            "v": value.detach()
            .view(-1, module.num_kv_heads, module.head_size_v)
            .cpu()
            .contiguous(),
            "allowed": allowed,
            "request_token_ids": tokens,
            "query_positions": query_positions,
            "key_positions": positions,
            "attention_scale": float(module.impl.scale),
        }

    def after(module, inputs, output):
        sample = state["pending"].pop(module.layer_name, None)
        if sample is None:
            return
        # Record the actual loaded scalar after any runtime scale calculation.
        # FP16 cache scales may differ from an independently calibrated E4M3 run.
        sample.update(k_scale=float(module._k_scale), v_scale=float(module._v_scale))
        filename = f"rank{rank}-layer{len(state['samples'])}.pt"
        torch.save(sample, directory / filename)
        state["samples"].append(
            {
                **provenance,
                "layer": module.layer_name,
                "rank": rank,
                "tp": tp,
                "role": "target",
                "capture_stage": "post_rope_pre_quantization",
                "request_origin": "real",
                "request_tokens_sha256": token_sha,
                "tensor_path": filename,
                "backend": module.attn_backend.get_name(),
                "kv_cache_dtype": module.kv_cache_dtype,
                "scale_origin": (
                    "loaded FP16-cache layer scalar; E4M3 calibration unchecked"
                ),
            }
        )

    for module in selected:
        state["handles"].append(module.register_forward_pre_hook(before))
        state["handles"].append(module.register_forward_hook(after))
    return [module.layer_name for module in selected]


def finish_capture(model):
    from vllm.v1.attention.backends import flash_attn_v100 as backend

    state = model._sm70_kv_capture
    for handle in state["handles"]:
        handle.remove()
    del model._sm70_kv_capture
    if len(state["samples"]) != len(state["layers"]) or state["pending"]:
        raise ValueError("Some selected attention layers did not produce a sample")
    delta = {
        route: count - state["route_snapshot"].get(route, 0)
        for route, count in backend._route_counts.items()
        if count > state["route_snapshot"].get(route, 0)
    }
    if delta.get("prefill_no_prefix_dense_flash", 0) <= 0:
        raise ValueError(f"No executed first-prefill Flash-V100 route: {delta}")
    return {"rank": state["rank"], "samples": state["samples"], "route_delta": delta}


def start_capture_on_worker(worker, directory, token_ids, provenance):
    return install_capture(worker.get_model(), directory, token_ids, provenance)


def finish_capture_on_worker(worker):
    return finish_capture(worker.get_model())


def main() -> None:
    import vllm
    from vllm import LLM, SamplingParams, envs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument(
        "--request", type=Path, required=True, help="JSON prompt_token_ids"
    )
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if "site-packages" not in Path(vllm.__file__).parts:
        raise ValueError("Run from a normal installed wheel outside its source tree")
    if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
        raise ValueError(
            "Diagnostic RPC hooks require local callable serialization; "
            "set VLLM_ALLOW_INSECURE_SERIALIZATION=1 for this isolated process"
        )
    if args.out.exists():
        raise ValueError("Use a new sample directory; do not mix request/model runs")
    config = json.loads(args.engine_config.read_text())
    request = json.loads(args.request.read_text())
    ids = request["prompt_token_ids"]
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Need the real request's nonempty integer token IDs")
    if config.get("speculative_config"):
        raise ValueError("Draft/MTP capture needs explicit role and mask adapters")
    required = {
        "dtype": "half",
        "kv_cache_dtype": "float16",
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_num_seqs": 1,
    }
    for name, expected in required.items():
        if name in config and config[name] != expected:
            raise ValueError(f"Capture requires {name}={expected!r}")
        config[name] = expected
    if config.get("max_num_batched_tokens", len(ids)) < len(ids):
        raise ValueError("The whole real request must fit in one prefill")
    config.setdefault("max_num_batched_tokens", len(ids))
    provenance = {
        "model": config["model"],
        "source_sha": args.source_sha,
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "request_json_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
        "runtime_files_sha256": verify_runtime_wheel(
            args.wheel, Path(vllm.__file__).parent
        ),
    }
    llm = LLM(**config)
    args.out.mkdir(parents=True)
    llm.collective_rpc(start_capture_on_worker, args=(str(args.out), ids, provenance))
    outputs = llm.generate(
        [{"prompt_token_ids": ids}],
        SamplingParams(temperature=0, max_tokens=1),
        use_tqdm=False,
    )
    captured = llm.collective_rpc(finish_capture_on_worker)
    samples = [row for rank in captured for row in rank["samples"]]
    manifest = {
        "version": 1,
        "samples": samples,
        "engine_config": config,
        "torch": torch.__version__,
        "diagnostic_eager_capture": True,
        "performance_evidence": False,
        "output_token_ids": outputs[0].outputs[0].token_ids,
        "sampling": {"temperature": 0, "max_tokens": 1, "ignore_eos": False},
        "executed_route_deltas": [
            {"rank": rank["rank"], "routes": rank["route_delta"]} for rank in captured
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"samples": len(samples), "directory": str(args.out)}))


if __name__ == "__main__":
    main()
