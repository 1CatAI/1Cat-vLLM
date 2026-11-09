# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time separate and fused single-request GDN verifier preprocessing."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import vllm._C as core

import vllm
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fused_gdn_gating
from vllm.model_executor.layers.mamba.gdn.sm70_preprocess import conv_gate_zero
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update


def timing(fn):
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(10):
        graph.replay()
    samples = []
    for _ in range(5):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 10)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=(5, 8), default=5)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert "site-packages" in vllm.__file__
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(20261005)
    m = args.tokens
    banks = []
    for _ in range(args.layers):
        x = (torch.randn(m, 4096, device="cuda") * 0.1).half()[:, :2560]
        state = (
            (torch.randn(1, m + 2, 2560, device="cuda") * 0.1).half().transpose(1, 2)
        )
        weight = (torch.randn(2560, 4, device="cuda") * 0.1).half()
        a = torch.randn(m, 12, device="cuda", dtype=torch.float16)
        b = torch.randn_like(a)
        a_log, bias = torch.randn(12, device="cuda"), torch.randn(12, device="cuda")
        slots = torch.zeros(1, device="cuda", dtype=torch.int32)
        accepted = torch.ones(1, device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, m], device="cuda", dtype=torch.int32)
        out = torch.empty(m, 12, 128, device="cuda", dtype=torch.float16)
        banks.append((x, state, weight, slots, accepted, cu, a_log, a, b, bias, out))

    def old():
        result = None
        for x, state, weight, slots, accepted, cu, a_log, a, b, bias, out in banks:
            out.zero_()
            causal_conv1d_update(
                x,
                state,
                weight,
                None,
                "silu",
                conv_state_indices=slots,
                num_accepted_tokens=accepted,
                query_start_loc=cu,
                max_query_len=m,
                validate_data=False,
            )
            result = fused_gdn_gating(a_log, a, b, bias, beta_dtype=torch.float32)
        return result

    def new():
        result = None
        for bank in banks:
            result = conv_gate_zero(*bank)
        return result

    old_us, new_us = timing(old), timing(new)
    report = dict(
        version=vllm.__version__,
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        tokens=m,
        layers=args.layers,
        state_length=m + 2,
        qkv_row_stride=4096,
        old_chain_us=old_us,
        new_chain_us=new_us,
        saved_chain_us=old_us - new_us,
        scope="synthetic exact-shape operator chain; not model-round timing",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
