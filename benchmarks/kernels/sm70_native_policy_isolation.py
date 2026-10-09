# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit native routing beats conflicting legacy env and survives replay."""

import argparse
import json
import os
from pathlib import Path

# Set the conflicting legacy input before loading native code.
os.environ["VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH"] = "1"
import torch
from torch.profiler import ProfilerActivity, profile

from tools.sm70.native_trace import NativeDispatchTrace
from vllm._sm70.policy import NativeBindings, native_policy_abi_available
from vllm.config.sm70_native import Sm70NativeConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert native_policy_abi_available()
policies = []
for enabled in (False, True):
    policy = Sm70NativeConfig(moe_single_token_fastpath=enabled)
    policy.resolve("fp8")
    policies.append(policy)
assert policies[0].hash_options() != policies[1].hash_options()
x = (torch.arange(256, device="cuda", dtype=torch.float16) / 256).reshape(1, 256)
ids = torch.tensor([[2, 0]], device="cuda", dtype=torch.int32)
tokens = torch.arange(2, device="cuda", dtype=torch.int32).reshape(1, 2)
weights = torch.full((1, 2), 0.5, device="cuda")
rows = []
outputs = []
graphs = []
metadata = []
functions = []
for enabled, policy in zip((False, True), policies):
    owner = NativeBindings(policy.values)
    permuted = torch.empty(2, 256, device="cuda", dtype=torch.float16)
    offsets = torch.empty(5, device="cuda", dtype=torch.int64)
    inverse = torch.empty(1, 2, device="cuda", dtype=torch.int32)
    index = torch.empty(1, 2, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)

    def forward(
        owner=owner,
        permuted=permuted,
        offsets=offsets,
        inverse=inverse,
        index=index,
        out=out,
    ):
        owner.moe_permute(
            x, ids, tokens, None, 4, 4, 2, permuted, offsets, inverse, index
        )
        owner.moe_unpermute(permuted, weights, inverse, None, 2, out)

    functions.append(forward)
    for _ in range(3):
        forward()
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        with NativeDispatchTrace() as observed:
            forward()
        torch.accelerator.synchronize()
    names = sorted(
        {
            e.name
            for e in profiler.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        }
    )
    assert any("singleTokenMoePermuteKernel" in name for name in names) == enabled, (
        names
    )
    assert any("singleTokenMoeUnpermuteKernel" in name for name in names) == enabled, (
        names
    )
    assert torch.equal(out, x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    outputs.append(out)
    graphs.append(graph)
    metadata.append((offsets, inverse, index))
    rows.append(
        {
            "enabled": enabled,
            "legacy_combined": "1",
            "cuda_kernels": names,
            "native_dispatch": observed.report(),
        }
    )
for i in range(3):
    x.mul_(0.5)
    ids.copy_(
        torch.tensor([[(i + 1) % 4, (i + 2) % 4]], device="cuda", dtype=torch.int32)
    )
    for graph in graphs:
        graph.replay()
    torch.accelerator.synchronize()
    for out in outputs:
        assert torch.equal(out, x)
    for old, new in zip(*metadata):
        assert torch.equal(old, new)
for i in (1, 0, 1, 0):
    functions[i]()
    torch.accelerator.synchronize()
    assert torch.equal(outputs[i], x)
args.output.write_text(
    json.dumps(
        {
            "policies": rows,
            "replays": 6,
            "alternating_owners": 4,
            "result": "bit_exact",
        },
        indent=2,
    )
    + "\n"
)
print("two explicit native policies, six replays and four alternating owners: exact")
