# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen QSA score-tile budgets through the existing complete cuBLAS scorer.

Uses synthetic H4/D128 inputs at serving shapes. Scores must meet a 1e-4
relative-L2 gate with exact masks. At most one selected block may change per
row, and at most 1e-5 of selections overall; model quality remains required.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from moe_tiles import artifact_identity, measure, relative_l2

from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.config.sm70_sparse import Sm70SparseConfig
from vllm.models.qwen4_exp.nvidia.ops.qsa import _qsa_mqa_cublas


def score_and_select(q, keys, valid, visible, budget, keep_logits=False):
    """Preserve serving's outer logits cap and its native top-512 selector."""
    rows, columns = q.shape[0], keys.shape[0]
    # qsa_select_paged_tokens bounds each logits chunk independently of the
    # inner per-head score tile. Omitting this loop mismeasures large prompts.
    from vllm.models.qwen4_exp.nvidia.ops.qsa import _LOGITS_WORKSPACE_BYTES

    step = max(1, min(_LOGITS_WORKSPACE_BYTES, 2 * budget * 1024**2) // (4 * columns))
    selected = torch.empty(rows, 512, device=q.device, dtype=torch.int32)
    saved = torch.empty(rows, columns, device=q.device) if keep_logits else None
    for first in range(0, rows, step):
        last = min(rows, first + step)
        scores, _ = _qsa_mqa_cublas(q[first:last], keys, valid, visible[first:last])
        torch.ops._C.qsa_lexicographic_topk(
            scores, visible[first:last], selected[first:last], 512
        )
        if saved is not None:
            saved[first:last].copy_(scores)
    return selected, saved


def selection_error(actual, expected, visible):
    actual = actual.sort(dim=1).values
    expected = expected.sort(dim=1).values
    where = torch.searchsorted(expected, actual).clamp_max(expected.shape[1] - 1)
    changed = (expected.gather(1, where) != actual).sum(dim=1)
    total, maximum = int(changed.sum().item()), int(changed.max().item())
    valid = bool(
        (actual >= 0).all()
        and (actual < visible[:, None]).all()
        and (actual[:, 1:] > actual[:, :-1]).all()
    )
    return {
        "exact": torch.equal(actual, expected),
        "changed_rows": int((changed > 0).sum().item()),
        "replaced_blocks": total,
        "max_replacements_per_row": maximum,
        "replacement_fraction": total / actual.numel(),
        "valid_distinct_indices": valid,
        "within_micro_bound": valid
        and maximum <= 1
        and total * 100000 <= actual.numel(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"artifact": artifact_identity(), "cases": []}
    for m, n in [(7344, 8672), (8976, 8672)]:
        torch.manual_seed(9683 + m)
        q = torch.randn(m, 4, 128, device="cuda", dtype=torch.half) * 0.125
        keys = torch.randn(n, 128, device="cuda", dtype=torch.half) * 0.125
        valid = torch.ones(n, device="cuda", dtype=torch.bool)
        valid[::127] = False
        visible = (
            n * 4 - m + 1 + torch.arange(m, device="cuda", dtype=torch.int32)
        ) // 4
        expected = (q[:8].float() @ keys.float().T).relu().sum(1) / (128**0.5)
        mask = valid[None, :] & (
            torch.arange(n, device="cuda")[None, :] < visible[:8, None]
        )
        expected.masked_fill_(~mask, -float("inf"))
        control = topk = None
        for index, budget in enumerate((32, 64, 128, 32)):
            config = SimpleNamespace(
                kernel_config=KernelConfig(
                    sm70_sparse=Sm70SparseConfig(qsa_score_tile_mb=budget)
                )
            )
            with set_current_vllm_config(config):

                def run(
                    keep_logits=False,
                    inputs=(q, keys, valid, visible),
                    budget=budget,
                ):
                    return score_and_select(*inputs, budget, keep_logits)

                selected, actual = run(keep_logits=True)
                if control is None:
                    control = actual.clone()
                    topk = selected
                same_mask = torch.equal(torch.isneginf(actual), torch.isneginf(control))
                selection = selection_error(selected, topk, visible)
                safe = torch.where(torch.isneginf(actual), 0, actual)
                reference = torch.where(torch.isneginf(expected), 0, expected)
                errors = {
                    "fp32_max_rel_l2": relative_l2(safe[:8], reference),
                    "paired_max_rel_l2": relative_l2(
                        safe, torch.where(torch.isneginf(control), 0, control)
                    ),
                }
                row = {
                    "rows": m,
                    "columns": n,
                    "score_tile_mb": budget,
                    "repeat_control": index == 3,
                    "top512_equal": torch.equal(selected, topk),
                    "selection": selection,
                    "mask_equal": same_mask,
                    **errors,
                    **measure(run, 3),
                    "passed": bool(torch.isfinite(safe).all())
                    and same_mask
                    and selection["within_micro_bound"]
                    and max(errors.values()) <= 0.0001,
                }
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    raise SystemExit(0 if all(row["passed"] for row in report["cases"]) else 1)


if __name__ == "__main__":
    main()
