# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight full-QPN8/shortlist heads, per-shard four-call CUDA graphs.

This local operator screen excludes TP communication and cannot admit model
speed or acceptance. Both model arms use the existing compact value/ID IPC.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from benchmarks.kernels.benchmark_sm70_mtp_fp32_dense import capture, elapsed
from vllm.models.qwen4_exp.nvidia.sm70_mtp_head import MTPQPN8Head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = next(n for n in index if n.endswith("lm_head.weight"))
    subset = json.loads(args.vocab.read_text())["token_ids"]
    report = {"model_admission": False, "shards": []}
    with safe_open(args.model / index[name], framework="pt") as tensors:
        weight = tensors.get_tensor(name)
    for rank in args.ranks:
        if not 0 <= rank < 4:
            raise ValueError("TP4 rank must be in 0..3")
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            weight[rank * 62080 : (rank + 1) * 62080].cuda().half().contiguous(),
            requires_grad=False,
        )
        layer.shard_indices = SimpleNamespace(
            org_vocab_start_index=rank * 62080,
            org_vocab_end_index=(rank + 1) * 62080,
            num_org_vocab_padding=0,
        )
        layer.quant_method = SimpleNamespace(
            apply=lambda head, x, bias=None: torch.nn.functional.linear(x, head.weight)
        )
        full_view = MTPQPN8Head(layer)
        view = MTPQPN8Head(layer)
        view.prepare_shortlist(subset)
        for rows in (1, 5):
            xs = torch.randn(4, rows, 2560, device="cuda", dtype=torch.float16)

            def full(view=full_view, xs=xs):
                # Compare against the current default segmented selector, not
                # the retired full-row Torch max implementation.
                return [view.maybe_get_sm70_lm_head_top1_pair(x) for x in xs]

            def short(view=view, xs=xs):
                return [view.maybe_get_sm70_lm_head_top1(x) for x in xs]

            graphs = {"full": capture(full), "shortlist": capture(short)}
            checks = []
            for scale in (0.0, 0.03, 1.0, 3.0):
                xs.normal_(0, scale)
                errors, differences = [], 0
                for x in xs:
                    logits = view.apply(view, x)
                    expected, positions = logits.index_select(
                        -1, view.shortlist_ids - rank * 62080
                    ).max(dim=-1)
                    values, ids = view.maybe_get_sm70_lm_head_top1(x)
                    errors.append((values - expected).abs().max().item())
                    differences += (ids != view.shortlist_ids[positions]).sum().item()
                    assert torch.isfinite(values).all()
                    assert torch.isin(ids, view.shortlist_ids).all()
                checks.append(
                    dict(scale=scale, max_error=max(errors), id_differences=differences)
                )
            trials = {name: [] for name in graphs}
            for trial in range(7):
                for arm in (
                    ("full", "shortlist") if trial % 2 else ("shortlist", "full")
                ):
                    trials[arm].append(elapsed(graphs[arm]))
            report["shards"].append(
                dict(
                    rank=rank,
                    rows=rows,
                    calls=4,
                    valid_shortlist_rows=view._shortlist_size,
                    padded_shortlist_rows=view._shortlist_padded,
                    full_weight_bytes=view.codes.numel() + view.scales.numel() * 2,
                    shortlist_weight_bytes=view.shortlist_codes.numel()
                    + view.shortlist_scales.numel() * 2,
                    checks=checks,
                    samples_ms=trials,
                    median_ms={k: statistics.median(v) for k, v in trials.items()},
                )
            )
            del graphs
        del layer, view, full_view
    report["native_sha256"] = hashlib.sha256(
        Path("vllm/_C.abi3.so").read_bytes()
    ).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
