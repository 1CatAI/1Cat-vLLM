# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-only sparse FP32 CUDA probe; never registers a runtime backend.

Run on an externally leased SM70 GPU. An optional captured attention input is
compared with independent gathered FP32 QK/global softmax/PV. Operator timings
are diagnostic and cannot satisfy the complete-denoise performance gate.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch


def load_binary(path: Path):
    path = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fp32_reference(q, k, v, block_map, sizes, scale):
    """Independent mathematical oracle, with physical padding masked explicitly.

    Match the frozen oracle's grouping (at most eight queries and 256 MiB per
    gathered operand). No production sparse kernel or geometry helper is used.
    This is an engineering FP32 reference, not the official GPU implementation.
    """
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("FP32 reference requires CUDA autocast to be disabled")
    batch, rows, heads, dim = q.shape
    blocks = rows // 64
    flat = [x.permute(0, 2, 1, 3).reshape(batch * heads, rows, dim) for x in (q, k, v)]
    maps = block_map.reshape(batch * heads, blocks, blocks)
    order = torch.arange(blocks, device=q.device).view(1, 1, blocks).expand_as(maps)
    selected = order.masked_fill(~maps, blocks).sort(dim=-1).values
    counts = maps.sum(-1).amax(0).tolist()
    lanes = torch.arange(64, device=q.device)
    bh = torch.arange(batch * heads, device=q.device)[:, None, None]
    output = torch.zeros_like(flat[0])
    pos = 0
    while pos < blocks:
        keep = int(counts[pos])
        if keep == 0:
            raise ValueError("reference requires nonempty selected rows")
        chunk = max(1, min(8, 256 * 1024**2 // (batch * heads * keep * 64 * dim * 4)))
        end = pos + 1
        while end < min(blocks, pos + chunk) and counts[end] == keep:
            end += 1
        index = selected[:, pos:end, :keep]
        safe = index.clamp_max(blocks - 1)
        tokens = (safe[..., None] * 64 + lanes).flatten(-2)
        valid = (
            (index < blocks)[..., None] & (lanes < sizes[safe][..., None])
        ).flatten(-2)
        keys, values = [flat[i][bh, tokens].float() for i in (1, 2)]
        keys.masked_fill_(~valid[..., None], 0)
        values.masked_fill_(~valid[..., None], 0)
        queries = flat[0][:, pos * 64 : end * 64].reshape(-1, 64, dim).float()
        shape = (batch * heads * (end - pos), keep * 64, dim)
        scores = torch.bmm(queries, keys.reshape(shape).transpose(1, 2)) * scale
        scores.masked_fill_(~valid.reshape(-1, 1, keep * 64), -float("inf"))
        answer = torch.bmm(scores.softmax(-1), values.reshape(shape))
        output[:, pos * 64 : end * 64] = answer.reshape(batch * heads, -1, dim).to(
            q.dtype
        )
        pos = end
    return output.reshape(batch, heads, rows, dim).permute(0, 2, 1, 3).contiguous()


def prepare_capture(path):
    from vllm.model_executor.models.minimax_h3 import vsa

    data = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    q, k, v = [data[name].cuda() for name in ("q", "k", "v")]
    part, sizes, nonpad, _, prefix, video = vsa._get_h3_tile_metadata(
        tuple(data["prefix_segments"]), tuple(data["video_shape"]), q.device
    )
    tiled = []
    for tensor in (q, k, v):
        out = torch.zeros(
            q.size(0), len(sizes) * 64, q.size(2), 128, device=q.device, dtype=q.dtype
        )
        out[:, nonpad] = tensor[:, part]
        tiled.append(out)
    q, k, v = tiled
    scores = (
        torch.matmul(
            vsa._pool_h3_tiles(q, sizes), vsa._pool_h3_tiles(k, sizes).transpose(-2, -1)
        )
        * data["scale"]
    )
    mask = vsa._build_h3_block_map(scores, prefix, video, data["topk"])
    return (q, k, v, mask, sizes, data["scale"], prefix, data["topk"]), nonpad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--build-directory", type=Path)
    parser.add_argument("--cutlass-root", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.binary:
        if args.build_directory or args.cutlass_root:
            parser.error("choose an existing binary or a build directory/CUTLASS root")
        binary = args.binary.resolve(strict=True)
        ops = load_binary(binary)
    else:
        if not args.build_directory or not args.cutlass_root:
            parser.error("building requires --build-directory and --cutlass-root")
        from torch.utils.cpp_extension import load

        args.build_directory.mkdir(parents=True, exist_ok=True)
        ops = load(
            name="h3_vsa_cutlass_fp32",
            sources=[str(Path(__file__).with_name("h3_vsa_fp32.cu"))],
            extra_include_paths=[str(args.cutlass_root / "include")],
            extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
            build_directory=str(args.build_directory),
            verbose=True,
        )
        binary = Path(ops.__file__)
    record = {
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            Path(__file__).with_name("h3_vsa_fp32.cu").read_bytes()
        ).hexdigest(),
        "built_from_reported_source": not bool(args.binary),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "performance_eligible": False,
        "scope": "operator diagnostic only",
    }
    if args.capture:
        with torch.inference_mode():
            values, valid = prepare_capture(args.capture)
            q, k, v, mask, sizes, scale, _, _ = values
            actual = ops.forward(*values)
            expected = fp32_reference(q, k, v, mask, sizes, scale)
            record["bitwise"] = torch.equal(
                actual[:, valid].view(torch.int16), expected[:, valid].view(torch.int16)
            )
            if record["bitwise"]:
                for _ in range(2):
                    ops._forward_prevalidated(*values)
                times = []
                for _ in range(4):
                    start, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    start.record()
                    ops._forward_prevalidated(*values)
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end))
                record.update(times_ms=times, median_ms=statistics.median(times))
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
    if record.get("bitwise") is False:
        raise SystemExit("FP32 comparison failed; candidate is not admitted")


if __name__ == "__main__":
    main()
