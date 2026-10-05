# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare mixed-QKV recurrence allocation/copy with caller output buffers."""

import argparse
import json
import statistics
from pathlib import Path

import torch

import vllm
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_mixed_qkv as mixed,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
results = []
for requests in (1, 4):
    m = requests * 5
    qkv = torch.randn(m, 4096, device="cuda", dtype=torch.float16)[:, :2560]
    a = torch.randn(m, 12, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    al = torch.randn(12, device="cuda", dtype=torch.float32)
    bias = torch.randn(12, device="cuda", dtype=torch.float16)
    indices = torch.arange(m, device="cuda", dtype=torch.int32).view(requests, 5)
    cu = torch.arange(requests + 1, device="cuda", dtype=torch.int32) * 5
    accepted = torch.ones(requests, device="cuda", dtype=torch.int32)
    outputs = [
        torch.empty(1, m, 12, 128, device="cuda", dtype=torch.float16)
        for _ in range(36)
    ]
    states = [
        torch.randn(m + 3, 12, 128, 128, device="cuda", dtype=torch.float32) * 0.02
        for _ in range(36)
    ]

    def run(
        candidate,
        states=states,
        outputs=outputs,
        al=al,
        a=a,
        b=b,
        bias=bias,
        cu=cu,
        indices=indices,
        accepted=accepted,
        qkv=qkv,
    ):
        for state, output in zip(states, outputs):
            common = dict(
                A_log=al,
                a=a,
                b=b,
                dt_bias=bias,
                initial_state=state,
                cu_seqlens=cu,
                ssm_state_indices=indices,
                num_accepted_tokens=accepted,
                use_qk_l2norm_in_kernel=True,
            )
            value = mixed(
                mixed_qkv=qkv,
                num_q_heads=4,
                num_v_heads=12,
                head_k_dim=128,
                head_v_dim=128,
                out=output if candidate else None,
                **common,
            )[0]
            if not candidate:
                output.copy_(value)

    medians = []
    for candidate in (False, True):
        for _ in range(5):
            run(candidate)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(candidate)
        for _ in range(10):
            graph.replay()
        samples = []
        for _ in range(5):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 10)
        medians.append(statistics.median(samples))
    results.append(
        dict(
            m=m, old_us=medians[0], new_us=medians[1], saved_us=medians[0] - medians[1]
        )
    )
r = {
    "version": vllm.__version__,
    "cases": results,
    "scope": "36-layer recurrence plus output-copy chain, no model performance claim",
}
args.out.write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r, indent=2))
