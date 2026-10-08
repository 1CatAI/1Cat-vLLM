# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline element/attention errors of candidate KV storage schemes.

Consumes real, unquantized post-RoPE samples described by a provenance manifest.
It never selects a runtime default or admits a kernel. See sm70_kv_codec.md.
"""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class Reconstruction:
    values: torch.Tensor
    storage_bytes: int
    clipped_elements: int = 0


def symmetric_int8(
    x: torch.Tensor, *, group: int, axis: int = -1, scale_dtype=torch.float32
) -> Reconstruction:
    """Group along features or tokens, rounding against the stored scale."""
    original = x.float().movedim(axis, -1)
    length = original.shape[-1]
    padded = F.pad(original, (0, -length % group))
    groups = padded.reshape(*padded.shape[:-1], -1, group)
    scales = (groups.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-6)
    scales = scales.to(scale_dtype)
    encoded = (groups / scales.float()).round().clamp(-128, 127).to(torch.int8)
    decoded = (encoded.float() * scales.float()).reshape(padded.shape)[..., :length]
    return Reconstruction(
        decoded.movedim(-1, axis),
        encoded.numel() + scales.numel() * scales.element_size(),
    )


def legacy_token_head_int8(x: torch.Tensor) -> Reconstruction:
    """FP32 token/head scales and integer truncation of the existing writer.

    This is an offline arithmetic model, not native-kernel bitwise evidence.
    Keep it separate from nearest-even candidates when comparing real samples.
    """
    values = x.float()
    scales = (values.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-6)
    encoded = (values * scales.reciprocal()).clamp(-128, 127).to(torch.int8)
    return Reconstruction(
        encoded.float() * scales, encoded.numel() + 4 * scales.numel()
    )


def asymmetric_int8(x: torch.Tensor) -> Reconstruction:
    """Token/head affine u8, FP32 scale and minimum (8 metadata bytes/head)."""
    x = x.float()
    minimum = x.amin(-1, keepdim=True)
    scales = ((x.amax(-1, keepdim=True) - minimum) / 255).clamp_min(1e-6)
    encoded = ((x - minimum) / scales).round().clamp(0, 255).to(torch.uint8)
    return Reconstruction(
        encoded.float() * scales + minimum, encoded.numel() + 8 * scales.numel()
    )


def e4m3(x: torch.Tensor, scale: float) -> Reconstruction:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("E4M3 requires the captured finite positive layer scale")
    scaled = x.float() / scale
    encoded = scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
    return Reconstruction(
        encoded.float() * scale,
        encoded.numel(),
        int((scaled.abs() > 448).sum()),
    )


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    error = (actual.float() - reference.float()).flatten()
    absolute = error.abs()
    # torch.quantile rejects tensors larger than 2**24 elements. Real long
    # context KV samples exceed that; order statistics avoid that limit and
    # a full sort. Use the conventional linear interpolation at (N - 1) * p.
    quantiles = []
    for probability in (0.5, 0.99, 0.999):
        position = (absolute.numel() - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        lo = absolute.kthvalue(lower + 1).values
        hi = absolute.kthvalue(upper + 1).values if upper != lower else lo
        quantiles.append(lo.lerp(hi, position - lower))
    return {
        "rmse": float(error.square().mean().sqrt()),
        "max_abs": float(absolute.max()),
        "p50_abs": float(quantiles[0]),
        "p99_abs": float(quantiles[1]),
        "p999_abs": float(quantiles[2]),
    }


def attention(q, k, v, allowed, scale):
    """FP32 GQA oracle; the explicit mask can express causal/window/QSA reads."""
    nq, nh, dim = q.shape
    nkv = k.shape[1]
    grouped_q = q.float().reshape(nq, nkv, nh // nkv, dim)
    scores = torch.einsum("qhgd,khd->hgqk", grouped_q, k.float()) * scale
    mask = (
        allowed[None, None]
        if allowed.ndim == 2
        else allowed.reshape(nq, nkv, nh // nkv, k.shape[0]).permute(1, 2, 0, 3)
    )
    scores.masked_fill_(~mask, -torch.inf)
    probs = scores.softmax(-1)
    return torch.einsum("hgqk,khd->qhgd", probs, v.float()).reshape(nq, nh, v.shape[-1])


def evaluate_sample(
    sample: dict, *, ablate_kv: bool = False, simulate_fp16_staging: bool = False
) -> list[dict]:
    q, k, v, allowed = (sample[name] for name in ("q", "k", "v", "allowed"))
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dtype != torch.float16 or tensor.ndim != 3 or tensor.numel() == 0:
            raise ValueError(
                f"{name} must contain nonempty unquantized FP16 [tokens, heads, dim]"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains nonfinite reference values")
    if k.shape[:2] != v.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Incompatible Q/K/V geometry")
    if q.shape[1] % k.shape[1] or allowed.shape not in (
        (q.shape[0], k.shape[0]),
        (q.shape[0], q.shape[1], k.shape[0]),
    ):
        raise ValueError("Invalid GQA geometry or attention mask")
    if allowed.dtype != torch.bool or not allowed.any(-1).all():
        raise ValueError(
            "Every query needs an explicit nonempty Boolean attention mask"
        )
    scale = float(sample["attention_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid captured attention scale")
    reference = attention(q, k, v, allowed, scale)

    def candidates():
        # Keep one candidate pair live at a time at 128K/256K.
        yield (
            "fp16",
            Reconstruction(k.float(), 2 * k.numel()),
            Reconstruction(v.float(), 2 * v.numel()),
        )
        yield (
            "e4m3_layer_scale",
            e4m3(k, float(sample["k_scale"])),
            e4m3(v, float(sample["v_scale"])),
        )
        for scale_dtype, suffix in ((torch.float32, "fp32"), (torch.float16, "fp16")):
            yield (
                f"int8_token_head_{suffix}",
                symmetric_int8(k, group=k.shape[-1], scale_dtype=scale_dtype),
                symmetric_int8(v, group=v.shape[-1], scale_dtype=scale_dtype),
            )
        yield (
            "int8_token_head_legacy_trunc_fp32",
            legacy_token_head_int8(k),
            legacy_token_head_int8(v),
        )
        yield "int8_token_head_affine_fp32", asymmetric_int8(k), asymmetric_int8(v)
        for group in (32, 64):
            yield (
                f"int8_feature_group{group}_fp16",
                symmetric_int8(k, group=group, scale_dtype=torch.float16),
                symmetric_int8(v, group=group, scale_dtype=torch.float16),
            )
            yield (
                f"int8_k_channel_token_group{group}_v_token_fp32",
                symmetric_int8(k, group=group, axis=0),
                symmetric_int8(v, group=v.shape[-1]),
            )
        # The initial real-request ablation shows larger V-only than K-only
        # error for token/head INT8. Evaluate narrower V groups without charging
        # both sides for that metadata; no runtime codec is selected here.
        for group in (32, 64):
            yield (
                f"int8_k_token_fp32_v_feature_group{group}_fp16",
                symmetric_int8(k, group=k.shape[-1]),
                symmetric_int8(v, group=group, scale_dtype=torch.float16),
            )

    results = []
    for name, key, value in candidates():
        output = attention(q, key.values, value.values, allowed, scale)
        measurement = {
            "scheme": name,
            "k": error_metrics(key.values, k),
            "v": error_metrics(value.values, v),
            "attention": error_metrics(output, reference),
            "payload_and_inline_metadata_bytes": key.storage_bytes
            + value.storage_bytes,
            "fixed_layer_scale_bytes": 8 if name == "e4m3_layer_scale" else 0,
            "k_clipped_elements": key.clipped_elements,
            "v_clipped_elements": value.clipped_elements,
        }
        del output
        if ablate_kv:
            # Change one cache side at a time with the same actual request mask.
            # These errors are not additive: quantized K changes the softmax.
            for side, decoded_k, decoded_v in (
                ("k", key.values, v),
                ("v", k, value.values),
            ):
                output = attention(q, decoded_k, decoded_v, allowed, scale)
                measurement[f"attention_{side}_only"] = error_metrics(output, reference)
                del output
            del decoded_k, decoded_v
        if simulate_fp16_staging:
            staged_k = key.values.to(torch.float16).float()
            staged_v = value.values.to(torch.float16).float()
            staging: dict[str, object] = {
                "nonfinite_k": int((~torch.isfinite(staged_k)).sum()),
                "nonfinite_v": int((~torch.isfinite(staged_v)).sum()),
            }
            if not staging["nonfinite_k"] and not staging["nonfinite_v"]:
                output = attention(q, staged_k, staged_v, allowed, scale)
                staging["attention"] = error_metrics(output, reference)
                del output
            # Report invalid staging without feeding infinity to the oracle or
            # emitting NaN JSON. This is not a simulation of a native kernel's
            # QK scale epilogue, FP16 probabilities or split-K summation order.
            measurement["fp16_tile_staging"] = staging
            del staged_k, staged_v
        results.append(measurement)
        del key, value
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--ablate-kv",
        action="store_true",
        help="Also measure attention error with only K or only V quantized",
    )
    parser.add_argument(
        "--simulate-fp16-staging",
        action="store_true",
        help="Cast decoded K/V tiles to FP16; report overflows and FP32 oracle error",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("version") != 1 or not manifest.get("samples"):
        raise ValueError("Expected a version 1 manifest with nonempty samples")
    results = []
    required = {
        "model",
        "layer",
        "rank",
        "tp",
        "role",
        "source_sha",
        "wheel_sha256",
        "request_tokens_sha256",
    }
    for entry in manifest["samples"]:
        if not required <= entry.keys() or any(
            entry[name] is None for name in required
        ):
            raise ValueError("Incomplete request/model/runtime provenance")
        if (
            entry.get("capture_stage") != "post_rope_pre_quantization"
            or entry.get("request_origin") != "real"
        ):
            raise ValueError(
                "Format selection requires real pre-quantization post-RoPE data"
            )
        path = (args.manifest.parent / entry["tensor_path"]).resolve()
        if not path.is_relative_to(args.manifest.parent.resolve()):
            raise ValueError("Tensor sample must be inside the manifest directory")
        sample = torch.load(path, map_location="cpu", weights_only=True)
        tokens = sample["request_token_ids"]
        if tokens.dtype != torch.int64 or tokens.ndim != 1 or tokens.numel() == 0:
            raise ValueError("Sample must retain the real request's token IDs")
        token_sha = hashlib.sha256(
            tokens.numpy().astype("<i8", copy=False).tobytes()
        ).hexdigest()
        if token_sha != entry["request_tokens_sha256"]:
            raise ValueError("Request token IDs do not match the provenance hash")
        for name, count in (
            ("query_positions", sample["q"].shape[0]),
            ("key_positions", sample["k"].shape[0]),
        ):
            positions = sample[name]
            if (
                positions.dtype != torch.int64
                or positions.shape != (count,)
                or not ((positions >= 0) & (positions < tokens.numel())).all()
            ):
                raise ValueError(f"Invalid captured {name}")
        sample = {
            name: tensor.to(args.device) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in sample.items()
        }
        results.append(
            {
                "provenance": entry,
                "sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "measurements": evaluate_sample(
                    sample,
                    ablate_kv=args.ablate_kv,
                    simulate_fp16_staging=args.simulate_fp16_staging,
                ),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "version": 1,
                "torch": torch.__version__,
                "device": args.device,
                "reference": "FP32 masked attention over captured FP16 Q/K/V",
                "runtime_default_selected": False,
                "kv_ablation": args.ablate_kv,
                "fp16_tile_staging_simulated": args.simulate_fp16_staging,
                "samples": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
