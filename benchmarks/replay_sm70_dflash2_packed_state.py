# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay real q8 GDN inputs through split and packed verification kernels.

Uses StateAuditExtension captures. Exercises every accepted-slot selector,
non-monotonic slot IDs (including zero), an empty padded request and a strided
state pool. This is an operator parity gate, never a complete-round speed or
acceptance-length benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def replay_capture(path: Path, *, row_strided: bool = False) -> list[dict]:
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update_mixed_qkv_out,
    )

    record = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    if record["phase"] != "verify" or record["num_draft_tokens"] != 7:
        raise ValueError(f"{path}: requires a q8 verification capture")
    results = []
    for layer in record["expected_layers"]:
        prefix = f"verify/layer{layer}/recurrent/"
        states = {
            k.removeprefix(prefix).split(":")[0]: v
            for k, v in record["states"].items()
            if k.startswith(prefix)
        }
        q, k, v, g, beta = (
            states[name].cuda() for name in ("q", "k", "v", "g", "beta")
        )
        if q.shape[:2] != (1, 8):
            raise ValueError(f"{path}: expected B1/q8 tensors, got {q.shape}")
        if not bool(states["input_state/valid"][0]):
            raise ValueError(f"{path}: missing live incoming state")
        incoming = states["input_state/values"][0].cuda()
        mixed = torch.cat([tensor.reshape(8, -1) for tensor in (q, k, v)], dim=1)
        projection = mixed
        if row_strided:
            # QPN2 pads the QKVZBA projection width to a multiple of eight.
            row_width = (
                (mixed.shape[1] + v.shape[2] * v.shape[3] + 2 * v.shape[2] + 7) // 8
            ) * 8
            projection = torch.full(
                (8, row_width), -42.0, dtype=mixed.dtype, device=mixed.device
            )
            projection[:, : mixed.shape[1]].copy_(mixed)
            mixed = projection[:, : mixed.shape[1]]
        projection_before = projection.clone()
        heads, width, depth = incoming.shape
        # The extra columns mimic Mamba pool padding and must stay untouched.
        backing = torch.full(
            (16, incoming.numel() + 64), -42.0, device="cuda", dtype=incoming.dtype
        )
        padded_table = torch.tensor(
            [[7, 2, 11, 0, 9, 5, 13, 3], [-1] * 8],
            device="cuda",
            dtype=torch.int32,
        )
        dummy = torch.zeros(heads, device="cuda", dtype=torch.float32)

        def view(storage, size=heads * width * depth, shape=(16, heads, width, depth)):
            return storage[:, :size].view(shape)

        for padded, selector in [
            *((False, s) for s in range(1, 9)),
            (True, 1),
            (True, 8),
        ]:
            table = padded_table if padded else padded_table[:1]
            cu = torch.tensor(
                [0, 8, 8] if padded else [0, 8], device="cuda", dtype=torch.int32
            )
            seed = backing.clone()
            view(seed)[int(table[0, selector - 1])].copy_(incoming)
            left, right = seed.clone(), seed.clone()
            selectors = torch.tensor(
                [selector, 1] if padded else [selector],
                device="cuda",
                dtype=torch.int32,
            )
            reference, _ = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=view(left),
                inplace_final_state=True,
                cu_seqlens=cu,
                ssm_state_indices=table,
                num_accepted_tokens=selectors,
                use_qk_l2norm_in_kernel=True,
            )
            actual = torch.empty((8, 1, heads, width), device="cuda", dtype=v.dtype)
            fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
                A_log=dummy,
                a=g,
                b=beta,
                dt_bias=dummy,
                mixed_qkv=mixed,
                num_q_heads=q.shape[2],
                num_v_heads=heads,
                head_k_dim=depth,
                head_v_dim=width,
                out=actual,
                initial_state=view(right),
                cu_seqlens=cu,
                ssm_state_indices=table,
                num_accepted_tokens=selectors,
                use_qk_l2norm_in_kernel=True,
                precomputed_g=g,
                precomputed_beta=beta,
                quantize_state_each_step=False,
                match_recurrent_schedule=True,
                match_recurrent_numerics=True,
            )
            # Compare backing storage too: an equal live result cannot excuse
            # a write into padding or a retired slot in either implementation.
            live = torch.zeros_like(seed, dtype=torch.bool)
            live[table[0].long(), : incoming.numel()] = True
            output_exact = torch.equal(
                actual.transpose(0, 1).view(torch.uint8), reference.view(torch.uint8)
            )
            states_exact = torch.equal(left.view(torch.uint8), right.view(torch.uint8))
            untouched = all(torch.equal(t[~live], seed[~live]) for t in (left, right))
            result = {
                "capture": path.name,
                "layer": layer,
                "selector": selector,
                "empty_padded_request": padded,
                "qkv_row_stride": mixed.stride(0),
                "input_projection_untouched": torch.equal(
                    projection.view(torch.uint8), projection_before.view(torch.uint8)
                ),
                "reference_matches_capture": None
                if padded
                else torch.equal(
                    reference.view(torch.uint8),
                    states["output"].cuda().view(torch.uint8),
                ),
                "output_bitwise_equal": output_exact,
                "state_bitwise_equal": states_exact,
                "padding_and_retired_slots_untouched": untouched,
            }
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--row-strided", action="store_true")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise ValueError("This audit targets SM70")
    torch.set_num_threads(4)
    results = []
    for path in args.captures:
        results.extend(replay_capture(path, row_strided=args.row_strided))
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Replayed {path.name}", flush=True)
    passed = all(
        row[key]
        for row in results
        for key in (
            "output_bitwise_equal",
            "state_bitwise_equal",
            "padding_and_retired_slots_untouched",
            "input_projection_untouched",
        )
    ) and all(row["reference_matches_capture"] is not False for row in results)
    print(json.dumps({"cases": len(results), "all_exact": passed}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
