# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight M1 output projection policy A/B, no model initialization."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from verify_sm70_qsa_router_exact import capture, paired_timing

from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
    _qwen38_fp16_row_gemv_kernel as kernel,
)


def load_weights(model, rank):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    names = sorted(
        (
            key
            for key in index["weight_map"]
            if key.endswith(
                (".linear_attn.out_proj.weight", ".self_attn.o_proj.weight")
            )
            and key.startswith("model.language_model.layers.")
        ),
        key=lambda key: int(key.split(".layers.")[1].split(".")[0]),
    )
    assert len(names) == 48
    weights = []
    for name in names:
        with safe_open(model / index["weight_map"][name], framework="pt") as f:
            view = f.get_slice(name)
            assert view.get_shape() == [2560, 6144], (name, view.get_shape())
            weights.append(
                view[:, rank * 1536 : (rank + 1) * 1536]
                .to(device="cuda", dtype=torch.float16)
                .contiguous()
            )
    return names, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(4), default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.accelerator.set_device_index(0)
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(20260905)
    names, weights = load_weights(args.model, args.rank)
    x = torch.randn(48, 1536, device="cuda", dtype=torch.float16)
    a = torch.empty(48, 2560, device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)
    role_policy = [int("linear_attn" in name) for name in names]

    def run(dst, role_specific):
        for i, weight in enumerate(weights):
            kernel[(2560,)](
                x[i],
                weight,
                dst[i],
                K=1536,
                BLOCK_K=512,
                LOAD_POLICY=role_policy[i] if role_specific else 0,
                num_warps=4,
            )

    ga, gb = capture(lambda: run(a, False)), capture(lambda: run(b, True))
    for replay in range(64):
        x.normal_()
        if replay % 4 == 0:
            x.mul_(1e-3)
        elif replay % 4 == 1:
            x.mul_(10)
        elif replay % 4 == 2:
            x[:, ::3] = -0.0
        b.fill_(float("nan"))
        ga.replay()
        gb.replay()
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), replay
    result = {
        "scope": "operator-only, real48-layer FP16 runtime output weights",
        "tp_rank": args.rank,
        "weight_shapes": [2560, 1536],
        "gdn_calls": sum(role_policy),
        "qsa_calls": len(names) - sum(role_policy),
        "bitwise_changed_input_graph_replays": 64,
        "times": paired_timing(ga, gb),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    live_outputs = []

    def production():
        live_outputs.clear()
        for i, weight in enumerate(weights):
            live_outputs.append(
                torch.ops.vllm.qwen38_sm70_fp16_gemv(
                    x[i : i + 1], weight, names[i].removesuffix(".weight")
                )
            )

    gp = capture(production)
    for replay in range(16):
        x.normal_()
        for value in live_outputs:
            value.fill_(float("nan"))
        ga.replay()
        gp.replay()
        assert torch.equal(
            a.view(torch.int16), torch.cat(live_outputs).view(torch.int16)
        ), replay
    result["production_custom_op_graph_replays"] = 16
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
