# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture real dense/QSA request Q/K/V from an installed FP16-KV model.

Diagnostic eager execution only: hooks synchronize/copy and cannot be timed.
Single-request, first-prefill capture avoids guessing prefix/chunk positions.
QSA captures the executed selection and compressor state. MLA, context-sharded
QSA and speculative draft capture need their own explicit mask/role adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from zipfile import ZipFile

import torch


def qsa_allowed_from_indices(indices, positions, key_count):
    """Represent the actual unique QSA selection; never substitute dense masks."""
    if (
        indices.dtype != torch.int32
        or indices.ndim != 2
        or positions.dtype != torch.int64
        or positions.shape != (indices.shape[0],)
    ):
        raise ValueError("Invalid QSA selection/position metadata")
    allowed = torch.zeros((indices.shape[0], key_count), dtype=torch.bool)
    for row, selected in enumerate(indices.cpu()):
        if (selected < -1).any():
            raise ValueError("QSA padding must use -1")
        live = selected[selected >= 0].long()
        if (
            not live.numel()
            or (live >= key_count).any()
            or (live > positions[row]).any()
            or live.unique().numel() != live.numel()
        ):
            raise ValueError(
                "QSA selection is empty, duplicated, future or out of range"
            )
        allowed[row, live] = True
    return allowed


class ObservedQSAKernel:
    """Diagnostic launch observer delegates to the original installed kernel."""

    def __init__(self, kernel, counts):
        self.kernel, self.counts = kernel, counts

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def observed(*args, **kwargs):
            result = launch(*args, **kwargs)
            route = "qsa_sparse_triton_splitk"
            self.counts[route] = self.counts.get(route, 0) + 1
            return result

        return observed

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def capture_qsa_state(cache, slots):
    """Copy addressed state rows only, including wrapped compressor slots."""
    live = slots[slots >= 0].unique().long()
    if (live >= cache.shape[0] * cache.shape[1]).any():
        raise ValueError("QSA state slot exceeds its bound cache")
    return {
        "slots": live.detach().cpu(),
        "rows": cache[live // cache.shape[1], live % cache.shape[1]].detach().cpu(),
    }


def select_qsa_capture_layers(layers):
    """Keep depth representatives and cover every executed compression ratio."""
    indices = {0, len(layers) // 2, len(layers) - 1}
    ratios = set()
    for index, layer in enumerate(layers):
        ratio = layer.indexer.compress_ratio
        if ratio not in ratios:
            indices.add(index)
            ratios.add(ratio)
    return [layers[index] for index in sorted(indices)]


def install_qsa_capture(model, directory, token_ids, provenance):
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.forward_context import get_forward_context
    from vllm.models.qwen4_exp.nvidia import qsa as owner
    from vllm.models.qwen4_exp.nvidia.ops import qsa as operations

    layers = [m for m in model.modules() if isinstance(m, owner.Qwen4ExpQSAAttention)]
    if not layers:
        raise ValueError("No NVIDIA Qwen4Exp QSA layers")
    selected = select_qsa_capture_layers(layers)
    if any(m.qsa_dcp_sharded or m.impl.dcp_world_size != 1 for m in selected):
        raise ValueError("Context-sharded QSA needs a separate capture adapter")
    rank = get_tensor_model_parallel_rank()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tokens = torch.tensor(token_ids, dtype=torch.int64)
    token_sha = hashlib.sha256(
        tokens.numpy().astype("<i8", copy=False).tobytes()
    ).hexdigest()
    state = {
        "adapter": "qsa",
        "samples": [],
        "layers": selected,
        "rank": rank,
        "routes": {},
        "restore": [],
    }
    model._sm70_kv_capture = state

    def native_observer(original, route):
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            state["routes"][route] = state["routes"].get(route, 0) + 1
            return result

        return observed

    for name, route in (
        ("_qsa_sparse_paged_attention_sm70_grouped_page4", "qsa_sparse_grouped_page4"),
        ("_qsa_sparse_paged_attention_sm70_xqa_page4_batch", "qsa_sparse_xqa_page4"),
    ):
        original = getattr(operations, name)
        state["restore"].append((operations, name, original))
        setattr(operations, name, native_observer(original, route))
    name = "_qsa_sparse_paged_gqa_splitk_kernel"
    original = getattr(operations, name)
    state["restore"].append((operations, name, original))
    setattr(operations, name, ObservedQSAKernel(original, state["routes"]))

    def before(module, query, key, value, metadata, kwargs):
        if any(t.dtype != torch.float16 for t in (query, key, value)):
            raise ValueError("QSA capture requires unencoded FP16 Q/K/V")
        if query.shape[0] != len(token_ids) or key.shape[0] != len(token_ids):
            raise ValueError("QSA capture requires the complete first prefill")
        if (
            metadata.query_start_loc.detach().cpu().tolist() != [0, len(token_ids)]
            or metadata.seq_lens.detach().cpu().tolist() != [len(token_ids)]
            or not (kwargs["token_to_req"] == 0).all()
        ):
            raise ValueError("QSA metadata does not describe one first prefill")
        positions = kwargs["query_positions"].detach().cpu()
        if not torch.equal(positions, torch.arange(len(token_ids))):
            raise ValueError("QSA query positions differ from request positions")
        indices = (
            module.topk_indices_buffer[: len(token_ids)][-min(16, len(token_ids)) :]
            .detach()
            .cpu()
        )
        positions = positions[-indices.shape[0] :]
        forward_metadata = get_forward_context().attn_metadata
        if isinstance(forward_metadata, list):
            forward_metadata = forward_metadata[0]
        raw = forward_metadata[module.indexer.raw_key_cache.prefix]
        compressed = forward_metadata[module.indexer.compressed_key_cache.prefix]
        sample = {
            "q": query[-indices.shape[0] :].detach().cpu().contiguous(),
            "k": key.detach().cpu().contiguous(),
            "v": value.detach().cpu().contiguous(),
            "allowed": qsa_allowed_from_indices(indices, positions, len(token_ids)),
            "qsa_selected_indices": indices,
            "query_positions": positions,
            "key_positions": torch.arange(len(token_ids), dtype=torch.int64),
            "request_token_ids": tokens,
            "attention_scale": float(module.scaling),
            "k_scale": float(module._k_scale),
            "v_scale": float(module._v_scale),
            "qsa_compress_ratio": module.indexer.compress_ratio,
            "qsa_raw_state": capture_qsa_state(
                module.indexer.raw_key_cache.kv_cache, raw.slot_mapping
            ),
            "qsa_compressed_state": capture_qsa_state(
                module.indexer.compressed_key_cache.kv_cache, compressed.slot_mapping
            ),
            "qsa_compressed_seq_lens": compressed.seq_lens.detach().cpu(),
        }
        # Main cache was published before forward_qsa. Check actual page mapping.
        page = module.kv_cache.shape[2]
        logical = torch.arange(len(token_ids), device=key.device)
        physical = metadata.block_table[0, logical // page].long()
        cached_k, cached_v = module.kv_cache.unbind(1)
        if not torch.equal(cached_k[physical, logical % page], key) or not torch.equal(
            cached_v[physical, logical % page], value
        ):
            raise ValueError("QSA first-prefill cache does not match raw FP16 K/V")
        return sample

    def observe_layer(module, original):
        def observed(layer, query, key, value, kv_cache, metadata, output, **kwargs):
            if any(row["layer"] == module.layer_name for row in state["samples"]):
                return original(
                    layer, query, key, value, kv_cache, metadata, output, **kwargs
                )
            sample = before(module, query, key, value, metadata, kwargs)
            route_snapshot = dict(state["routes"])
            result = original(
                layer, query, key, value, kv_cache, metadata, output, **kwargs
            )
            delta = {
                route: count - route_snapshot.get(route, 0)
                for route, count in state["routes"].items()
                if count > route_snapshot.get(route, 0)
            }
            if not delta:
                raise ValueError(
                    "QSA capture observed no executed sparse attention route"
                )
            filename = f"rank{rank}-layer{len(state['samples'])}.pt"
            torch.save(sample, directory / filename)
            state["samples"].append(
                {
                    **provenance,
                    "layer": module.layer_name,
                    "rank": rank,
                    "tp": get_tensor_model_parallel_world_size(),
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
                    "executed_routes": delta,
                    "mask_origin": "actual QSA indexer selections; duplicates rejected",
                }
            )
            return result

        return observed

    for module in selected:
        original = module.impl.forward_qsa
        state["restore"].append((module.impl, "forward_qsa", original))
        module.impl.forward_qsa = observe_layer(module, original)
    return [module.layer_name for module in selected]


def verify_runtime_wheel(wheel, vllm_root):
    """Reject source overlays and a provenance wheel different from the runtime."""
    from email.parser import Parser
    from importlib.metadata import version

    import flash_attn_v100.flash_attn_v100_cuda as extension

    paths = {
        "vllm/v1/attention/backends/flash_attn_v100.py": (
            vllm_root / "v1/attention/backends/flash_attn_v100.py"
        ),
        "vllm/_C.abi3.so": vllm_root / "_C.abi3.so",
        f"flash_attn_v100/{Path(extension.__file__).name}": Path(extension.__file__),
    }
    # The native cache writer is registered in stable-libtorch, separately
    # from _C. A matching distribution version alone cannot detect a stale DSO.
    for name in (
        "_C_stable_libtorch.abi3.so",
        "vllm_flash_attn/_vllm_fa2_C.abi3.so",
        "v1/kv_cache_codec.py",
        "v1/kv_cache_interface.py",
        "v1/attention/backends/flash_v100/cache_view.py",
        "v1/attention/backends/flash_v100/codec.py",
        "v1/attention/backends/flash_v100/decode_policy.py",
        "v1/attention/backends/flash_v100/masking.py",
        "v1/attention/backends/flash_v100/metadata.py",
        "v1/attention/backends/flash_v100/reference.py",
        "models/qwen4_exp/nvidia/qsa.py",
        "models/qwen4_exp/nvidia/ops/qsa.py",
        "models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py",
        "models/qwen4_exp/nvidia/indexer_qsa.py",
        "v1/attention/ops/kv_codec.py",
        "v1/attention/ops/triton_reshape_and_cache_flash.py",
        "model_executor/layers/attention/sm70_qwen38_qk_rope.py",
    ):
        paths[f"vllm/{name}"] = vllm_root / name
    hashes = {}
    with ZipFile(wheel) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        if version(metadata["Name"]) != metadata["Version"]:
            raise ValueError("Installed distribution version does not match the wheel")
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
    if state.get("adapter") == "qsa":
        for target, name, original in reversed(state["restore"]):
            setattr(target, name, original)
        del model._sm70_kv_capture
        if len(state["samples"]) != len(state["layers"]) or not state["routes"]:
            raise ValueError("QSA layers did not produce complete executed samples")
        return {
            "rank": state["rank"],
            "samples": state["samples"],
            "route_delta": state["routes"],
        }
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


def start_capture_on_worker(worker, directory, token_ids, provenance, adapter):
    if adapter == "qsa":
        return install_qsa_capture(worker.get_model(), directory, token_ids, provenance)
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
    parser.add_argument("--adapter", choices=("dense", "qsa"), default="dense")
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
    if args.adapter == "qsa" and config.get("decode_context_parallel_size", 1) != 1:
        raise ValueError("Context-sharded QSA capture is not implemented")
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
        "capture_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "request_json_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
        "runtime_files_sha256": verify_runtime_wheel(
            args.wheel, Path(vllm.__file__).parent
        ),
    }
    durations = {}

    def begin_stage(stage):
        print(json.dumps({"capture_stage": stage, "status": "started"}), flush=True)
        return time.perf_counter()

    def end_stage(stage, started):
        durations[stage] = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "capture_stage": stage,
                    "status": "finished",
                    "seconds": durations[stage],
                }
            ),
            flush=True,
        )

    started = begin_stage("engine_initialization")
    llm = LLM(**config)
    end_stage("engine_initialization", started)
    args.out.mkdir(parents=True)
    started = begin_stage("install_hooks")
    llm.collective_rpc(
        start_capture_on_worker, args=(str(args.out), ids, provenance, args.adapter)
    )
    end_stage("install_hooks", started)
    started = begin_stage("real_request")
    outputs = llm.generate(
        [{"prompt_token_ids": ids}],
        SamplingParams(temperature=0, max_tokens=1),
        use_tqdm=False,
    )
    end_stage("real_request", started)
    started = begin_stage("finish_capture")
    captured = llm.collective_rpc(finish_capture_on_worker)
    end_stage("finish_capture", started)
    samples = [row for rank in captured for row in rank["samples"]]
    manifest = {
        "version": 1,
        "adapter": args.adapter,
        "samples": samples,
        "engine_config": config,
        "torch": torch.__version__,
        "diagnostic_eager_capture": True,
        "performance_evidence": False,
        "diagnostic_stage_seconds": durations,
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
