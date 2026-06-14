# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump and compare SM70 TurboMind op outputs across vLLM trees.

Run this script from the old and latest vLLM environments with the same
arguments to produce output dumps, then compare the dumps. It is meant to
verify migration parity against the existing 0.0.3 TurboMind path, not against
a different mathematical reference.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open

Mode = Literal["awq", "fp8"]


def _import_sm70_ops() -> Any:
    try:
        from vllm import _sm70_ops

        return _sm70_ops
    except ImportError:
        from vllm import _custom_ops

        return _custom_ops


def _vllm_info() -> dict[str, str | None]:
    import vllm

    return {
        "version": getattr(vllm, "__version__", None),
        "file": getattr(vllm, "__file__", None),
    }


def _cache_hint(device: torch.device) -> torch.Tensor:
    return torch.empty((), device=device)


def _maybe_import_cache(args: argparse.Namespace, device: torch.device) -> int | None:
    if args.import_cache is None:
        return None
    ops = _import_sm70_ops()
    return int(ops.sm70_gemm_import_cache(_cache_hint(device), str(args.import_cache)))


def _maybe_export_cache(args: argparse.Namespace, device: torch.device) -> int | None:
    if args.export_cache is None:
        return None
    ops = _import_sm70_ops()
    return int(ops.sm70_gemm_export_cache(_cache_hint(device), str(args.export_cache)))


def _require_sm70(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This check requires a CUDA device.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")


def _make_input(m: int, k: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(m * k, device=device, dtype=torch.int32)
    values = ((values % 1024).to(torch.float32) / 512.0) - 1.0
    return values.reshape(m, k).to(torch.float16)


def _tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _weight_map(model_path: Path) -> dict[str, str]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def _load_awq(
    model_path: Path,
    layer: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_map = _weight_map(model_path)
    with safe_open(model_path / weight_map[f"{layer}.qweight"],
                   framework="pt", device="cpu") as f:
        qweight = f.get_tensor(f"{layer}.qweight").to(device).contiguous()
        scales = f.get_tensor(f"{layer}.scales").to(device).contiguous()
        qzeros = f.get_tensor(f"{layer}.qzeros").to(device).contiguous()
    return qweight, scales, qzeros


def _load_fp8(
    model_path: Path,
    layer: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_map = _weight_map(model_path)
    with safe_open(model_path / weight_map[f"{layer}.weight"],
                   framework="pt", device="cpu") as f:
        weight = f.get_tensor(f"{layer}.weight").to(device).contiguous()
        scales = (
            f.get_tensor(f"{layer}.weight_scale_inv")
            .to(device)
            .to(torch.float32)
            .contiguous()
        )
    return weight, scales


def _dump_awq(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ops = _import_sm70_ops()
    qweight, scales, qzeros = _load_awq(args.model, args.layer, device)
    x = _make_input(args.m, qweight.shape[0], device)
    tm_weight, tm_scales, meta = ops.awq_sm70_prepare(
        qweight, scales, qzeros, args.group_size
    )
    out = ops.awq_gemm_sm70(
        x,
        tm_weight,
        tm_scales,
        args.group_size,
        int(meta[0].item()),
        int(meta[1].item()),
    )
    details = {
        "input": _tensor_digest(x),
        "qweight": _tensor_digest(qweight),
        "scales": _tensor_digest(scales),
        "qzeros": _tensor_digest(qzeros),
        "tm_weight": _tensor_digest(tm_weight),
        "tm_scales": _tensor_digest(tm_scales),
        "meta_values": [int(meta[0].item()), int(meta[1].item())],
        "meta": _tensor_digest(meta),
    }
    return out, details


def _dump_fp8(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ops = _import_sm70_ops()
    weight, scales = _load_fp8(args.model, args.layer, device)
    x = _make_input(args.m, weight.shape[1], device)
    tm_weight, tm_scales, meta = ops.fp8_sm70_prepare(
        weight, scales, args.group_size
    )
    out = torch.empty((args.m, weight.shape[0]), device=device, dtype=torch.float16)
    ops.fp8_gemm_sm70_out_meta(out, x, tm_weight, tm_scales, meta)
    details = {
        "input": _tensor_digest(x),
        "weight": _tensor_digest(weight),
        "scales": _tensor_digest(scales),
        "tm_weight": _tensor_digest(tm_weight),
        "tm_scales": _tensor_digest(tm_scales),
        "meta_values": [int(meta[0].item()), int(meta[1].item())],
        "meta": _tensor_digest(meta),
    }
    return out, details


def _dump(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    _require_sm70(device)
    if args.out is None:
        raise ValueError("--out is required for dump mode.")
    imported_cache_entries = _maybe_import_cache(args, device)
    if args.mode == "awq":
        out, details = _dump_awq(args, device)
    else:
        out, details = _dump_fp8(args, device)
    torch.cuda.synchronize(device)
    exported_cache_entries = _maybe_export_cache(args, device)
    payload = {
        "mode": args.mode,
        "model": str(args.model),
        "layer": args.layer,
        "m": args.m,
        "group_size": args.group_size,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "vllm": _vllm_info(),
        "import_cache": str(args.import_cache) if args.import_cache else None,
        "imported_cache_entries": imported_cache_entries,
        "export_cache": str(args.export_cache) if args.export_cache else None,
        "exported_cache_entries": exported_cache_entries,
        "details": details,
        "output": out.detach().cpu(),
    }
    torch.save(payload, args.out)
    print(json.dumps({k: v for k, v in payload.items() if k != "output"},
                     indent=2, sort_keys=True))
    return 0


def _compare(args: argparse.Namespace) -> int:
    left = torch.load(args.compare[0], map_location="cpu", weights_only=False)
    right = torch.load(args.compare[1], map_location="cpu", weights_only=False)
    left_out = left["output"]
    right_out = right["output"]
    diff = (left_out - right_out).abs()
    result = {
        "left": str(args.compare[0]),
        "right": str(args.compare[1]),
        "equal": bool(torch.equal(left_out, right_out)),
        "max_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_diff": float(diff.float().mean().item()) if diff.numel() else 0.0,
        "shape": list(left_out.shape),
        "left_meta": {k: v for k, v in left.items() if k != "output"},
        "right_meta": {k: v for k, v in right.items() if k != "output"},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")
    return 0 if result["equal"] and result["max_diff"] == 0.0 else 1


def _time_cuda_call(
    fn: Any,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def _bench_awq(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    ops = _import_sm70_ops()
    qweight, scales, qzeros = _load_awq(args.model, args.layer, device)
    x = _make_input(args.m, qweight.shape[0], device)
    tm_weight, tm_scales, meta = ops.awq_sm70_prepare(
        qweight, scales, qzeros, args.group_size
    )
    out = torch.empty((args.m, qweight.shape[1] * 8),
                      device=device,
                      dtype=torch.float16)
    k_ld = int(meta[0].item())
    q_ld = int(meta[1].item())

    def run() -> None:
        ops.awq_gemm_sm70_out(
            out,
            x,
            tm_weight,
            tm_scales,
            args.group_size,
            k_ld,
            q_ld,
            False,
        )

    timing = _time_cuda_call(run, device, args.warmup, args.iters)
    torch.cuda.synchronize(device)
    return timing, out


def _bench_fp8(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    ops = _import_sm70_ops()
    weight, scales = _load_fp8(args.model, args.layer, device)
    x = _make_input(args.m, weight.shape[1], device)
    tm_weight, tm_scales, meta = ops.fp8_sm70_prepare(
        weight, scales, args.group_size
    )
    out = torch.empty((args.m, weight.shape[0]), device=device, dtype=torch.float16)

    def run() -> None:
        ops.fp8_gemm_sm70_out_meta(out, x, tm_weight, tm_scales, meta)

    timing = _time_cuda_call(run, device, args.warmup, args.iters)
    torch.cuda.synchronize(device)
    return timing, out


def _bench(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    _require_sm70(device)
    if args.model is None or not args.layer:
        raise ValueError("--model and --layer are required for bench mode.")

    imported_cache_entries = _maybe_import_cache(args, device)
    if args.mode == "awq":
        timing, out = _bench_awq(args, device)
    else:
        timing, out = _bench_fp8(args, device)
    exported_cache_entries = _maybe_export_cache(args, device)

    result = {
        "mode": args.mode,
        "model": str(args.model),
        "layer": args.layer,
        "m": args.m,
        "group_size": args.group_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "vllm": _vllm_info(),
        "import_cache": str(args.import_cache) if args.import_cache else None,
        "imported_cache_entries": imported_cache_entries,
        "export_cache": str(args.export_cache) if args.export_cache else None,
        "exported_cache_entries": exported_cache_entries,
        "timing": timing,
        "output": _tensor_digest(out),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--mode", choices=["awq", "fp8"], default="awq")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", default="")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--import-cache", type=Path)
    parser.add_argument("--export-cache", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.compare is not None:
        return _compare(args)
    if args.bench:
        return _bench(args)
    if args.model is None or not args.layer:
        raise ValueError("--model and --layer are required for dump mode.")
    return _dump(args)


if __name__ == "__main__":
    raise SystemExit(main())
