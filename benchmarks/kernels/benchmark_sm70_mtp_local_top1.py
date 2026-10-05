# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four real QPN8 draft heads including value/ID packets, not TP transport.

Screen only: no model dispatch. Parallel local selection preserves original
vocabulary IDs, first-index ties, NaN precedence and padding exclusion.
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


def select(logits, valid, start, pairs, partial):
    torch.ops._C.qwen38_mtp_local_top1_sm70_out(pairs, partial, logits, valid, start)
    return pairs


def ordinary(logits, valid, start):
    values, ids = logits[:, :valid].max(-1)
    return torch.stack((values.float(), (ids + start).float()), -1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    oracle = []
    for m in (1, 4, 5, 16, 128):
        valid, n, start = 62077, 62080, 186240
        logits = torch.randn(m, n, device="cuda", dtype=torch.float16)
        pairs = torch.empty(m, 2, device="cuda", dtype=torch.float32)
        partial = torch.empty(m, (valid + 511) // 512, 2, device="cuda")
        for case in ("finite", "zeros", "minus_inf", "ties", "nans", "padding_nan"):
            logits.normal_()
            if case == "zeros":
                logits.zero_()
            elif case == "minus_inf":
                logits.fill_(-float("inf"))
            elif case == "ties":
                logits[:, [0, 513, valid - 1]] = float("inf")
            elif case == "nans":
                logits[:, [7, 515, valid - 1]] = float("nan")
            elif case == "padding_nan":
                logits[:, valid:] = float("nan")
            expected = ordinary(logits, valid, start)
            actual = select(logits, valid, start, pairs, partial)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            oracle.append(dict(rows=m, case=case, exact_pair=True))
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = next(n for n in index if n.endswith("lm_head.weight"))
    with safe_open(args.model / index[name], framework="pt") as tensors:
        weight = tensors.get_tensor(name)
    samples = []
    for rank in args.ranks:
        assert 0 <= rank < 4
        start = rank * 62080
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            weight[start : start + 62080].cuda().half().contiguous(),
            requires_grad=False,
        )
        layer.shard_indices = SimpleNamespace(
            org_vocab_start_index=start,
            org_vocab_end_index=start + 62080,
            num_org_vocab_padding=0,
        )
        view = MTPQPN8Head(layer)
        for m in (1, 4, 5):
            xs = torch.randn(4, m, 2560, device="cuda", dtype=torch.float16)
            pairs = [torch.empty(m, 2, device="cuda") for _ in xs]
            partials = [torch.empty(m, 122, 2, device="cuda") for _ in xs]

            def control(view=view, xs=xs, start=start):
                return [ordinary(view.apply(view, x), 62080, start) for x in xs]

            def candidate(
                view=view, xs=xs, start=start, pairs=pairs, partials=partials
            ):
                return [
                    select(view.apply(view, x), 62080, start, pair, partial)
                    for x, pair, partial in zip(xs, pairs, partials)
                ]

            for scale in (0, 0.03, 1, 3):
                xs.normal_(0, scale)
                for a, b in zip(control(), candidate()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
            graphs = {"control": capture(control), "candidate": capture(candidate)}
            times = {arm: [] for arm in graphs}
            for trial in range(7):
                for arm in (
                    ("control", "candidate") if trial % 2 else ("candidate", "control")
                ):
                    times[arm].append(elapsed(graphs[arm]))
            samples.append(
                dict(
                    rank=rank,
                    rows=m,
                    calls=4,
                    exact_pair=True,
                    samples_ms=times,
                    median_ms={k: statistics.median(v) for k, v in times.items()},
                )
            )
            del graphs
        del view, layer
    report = dict(
        model_admission=False,
        includes_head_and_packet=True,
        excludes_identical_ipc=True,
        oracle=oracle,
        shards=samples,
        native_sha256=hashlib.sha256(Path("vllm/_C.abi3.so").read_bytes()).hexdigest(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
