# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise and cold-cache checks for SM70 attention preparation and FP8 QDQ."""

import argparse
import functools
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from vllm.models.deepseek_v4.common.ops import fp8_software
from vllm.models.deepseek_v41.attn_fuse import attn_prep
from vllm.models.deepseek_v41.quant_rope import attention_qdq, rope_table
from vllm.triton_utils import tl, triton

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--save", action="store_true")
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
torch.manual_seed(519)
flush_buffer = torch.randn(16 * 1024 * 1024, device="cuda")
flush_result = torch.empty((), device="cuda")
digests = {}
benchmarks = {}


def check(name, operation):
    torch.sum(flush_buffer, dim=0, out=flush_result)
    result = operation()
    tensors = result if isinstance(result, tuple) else (result,)
    digest = hashlib.sha256()
    for tensor in tensors:
        tensor = tensor.contiguous().cpu()
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    digests[name] = digest.hexdigest()


for tokens in range(1, 7):
    for position in (0, 127, 255, 3000, 16378, 17000):
        for compressed in (False, True):
            positions = torch.arange(
                position, position + tokens, dtype=torch.int64, device="cuda"
            )
            table = rope_table(compressed=compressed)
            query = (torch.randn(tokens, 16, 512, device="cuda") * 0.3).half()
            kv = (torch.randn(tokens, 512, device="cuda") * 0.3).half()
            if position == 0:
                kv[-1].zero_()
            weight = torch.randn(512, device="cuda") * 0.1 + 1
            operation = functools.partial(
                attn_prep, query, kv, weight, 1e-20, positions, table
            )
            check(f"prep/T{tokens}/P{position}/C{compressed}", operation)
            if tokens == 6 and position == 3000:
                benchmarks[f"prep_T6_C{compressed}"] = operation

for tokens in (1, 6, 8, 64, 1024):
    for fp4, group, e4m3_scale in (
        (False, 128, False),
        (True, 16, True),
        (True, 32, False),
    ):
        x = (torch.randn(tokens, 512, device="cuda") * 0.5).half()
        operation = functools.partial(
            attention_qdq, x, fp4=fp4, group=group, e4m3_scale=e4m3_scale
        )
        check(f"qdq/T{tokens}/F{fp4}/G{group}", operation)
        if tokens == 6:
            benchmarks[f"qdq_T6_F{fp4}_G{group}"] = operation

if args.save:
    args.reference.write_text(json.dumps(digests, indent=2) + "\n")
else:
    expected = json.loads(args.reference.read_text())
    failures = [name for name in digests if expected.get(name) != digests[name]]
    if failures or set(expected) != set(digests):
        raise AssertionError(f"Bitwise mismatches: {failures[:20]}")
print(
    json.dumps(
        {"bitwise_cases": len(digests), "status": "saved" if args.save else "passed"}
    ),
    flush=True,
)

if hasattr(fp8_software, "fp32_to_fp8_e4m3fn_bits_bitcast"):

    @triton.jit
    def compare_encode(x, reference, actual, N: tl.constexpr, BLOCK: tl.constexpr):
        col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x + col, col < N, 0)
        old = fp8_software.fp32_to_fp8_e4m3fn_bits(value)
        new = fp8_software.fp32_to_fp8_e4m3fn_bits_bitcast(value)
        tl.store(reference + col, old, col < N)
        tl.store(actual + col, new, col < N)

    half_values = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.float16)
        .float()
    )
    half_values = half_values[torch.isfinite(half_values)]
    random_values = torch.randint(
        0, 0x7F800000, (1024 * 1024,), device="cuda", dtype=torch.int32
    ).view(torch.float32)
    inputs = torch.cat((half_values, random_values, -random_values)).contiguous()
    reference = torch.empty_like(inputs, dtype=torch.uint8)
    actual = torch.empty_like(reference)
    compare_encode[(triton.cdiv(inputs.numel(), 256),)](
        inputs, reference, actual, N=inputs.numel(), BLOCK=256
    )
    assert torch.equal(reference, actual), (reference != actual).sum().item()
    print(
        json.dumps({"fp8_bitwise_values": inputs.numel(), "status": "passed"}),
        flush=True,
    )

if args.profile:
    torch.cuda.cudart().cudaProfilerStart()
for name, operation in benchmarks.items():
    for _ in range(3):
        torch.sum(flush_buffer, dim=0, out=flush_result)
        operation()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    finish = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.sum(flush_buffer, dim=0, out=flush_result)
        start.record()
        result = operation()
        finish.record()
    samples = []
    torch.cuda.nvtx.range_push(name)
    for _ in range(40):
        graph.replay()
        torch.accelerator.synchronize()
        samples.append(start.elapsed_time(finish) * 1000)
    torch.cuda.nvtx.range_pop()
    print(
        json.dumps({"case": name, "cold_graph_us": statistics.median(samples)}),
        flush=True,
    )
if args.profile:
    torch.cuda.cudart().cudaProfilerStop()
