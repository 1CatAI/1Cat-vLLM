# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compact expert banks with current joint/vector gate/up dispatch.

Run under the shared GPU lock. Routing preparation and the rest of the FFN
are excluded; repeated routed activations match top-k token duplication.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import vllm._C as core
from benchmark_gguf_turbomind import elapsed, prepare_projection
from benchmark_gguf_turbomind_grouped import stack_prepared

from vllm.model_executor.kernels.gguf import (
    lattice_grouped_capabilities,
    raw_grouped_gate_up_capabilities,
)
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
    GGUFExpertBank,
    _expert_gate_up,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf")
    p.add_argument("--layer", type=int, default=35)
    p.add_argument("--m", type=int, nargs="+", default=[1, 4, 8, 16, 512])
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.iterations < 1 or any(m < 1 for m in a.m):
        p.error("Expected positive iteration and batch counts")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    reader = GGUFReader(a.gguf)
    tensors = {t.name: t for t in reader.tensors}
    n, k, experts, top_k = 160, 2560, 512, 10
    banks, controls, references, original = [], [], [], []
    for shard in ("gate", "up"):
        tensor = tensors[f"blk.{a.layer}.ffn_{shard}_exps.weight"]
        if int(tensor.tensor_type) != 18 or tensor.data.shape[0] != experts:
            raise ValueError("Expected IQ3_XXS E512 gate/up tensors")
        bank = GGUFExpertBank(18, experts, "cuda", torch.float16, prefer_compact=True)
        prepared, ref, rows = [], [], []
        for expert in range(experts):
            source = tensor.data[expert]
            bank.add(expert, torch.from_numpy(source), 0, 4, axis=0)
            raw = RawGGUFProjection.from_rows(source, 18).tp_slice(0, 4, axis=0)
            payload = raw.data[:, : raw.payload_bytes_per_row]
            prepared.append(prepare_projection(transcode_lattice(payload, 18)))
            ref.append(torch.from_numpy(dequantize(payload, 18)).half().cuda())
            rows.append(torch.from_numpy(raw.data).cuda())
        bank.finalize()
        if bank.storage_layout != "compact_original":
            raise ValueError(bank.compact_rejection_reason)
        banks.append(bank)
        controls.append((*stack_prepared(prepared), prepared[0][2].tolist()))
        references.append(ref)
        original.append(torch.stack(rows))
    # Active gate+up addresses, rather than the complete E512 bank, define
    # the cold-bank requirement for the smallest routing case.
    count = math.ceil(2 * 6291456 / (top_k * n * (k // 256) * 98 * 2))
    compact = [(banks[0].weights, banks[1].weights)]
    baseline = [(controls, original)]
    for _ in range(count - 1):
        compact.append(tuple(b.weights.clone() for b in banks))
        cloned = []
        for w, stats, _, meta in controls:
            cw, cs = w.clone(), stats.clone()
            ptrs = torch.ops._C.awq_moe_build_strided_ptrs(cw, cs, *meta, experts)
            cloned.append((cw, cs, ptrs, meta))
        baseline.append((cloned, [r.clone() for r in original]))
    raw_caps = raw_grouped_gate_up_capabilities(
        18,
        k,
        n,
        experts,
        torch.float16,
        is_sm70=True,
        original_storage_available=True,
    )
    raw_batches = [c.min_m for c in raw_caps if c.reason is None]
    caps = lattice_grouped_capabilities(18, k, n, experts, torch.float16)
    vector_bands = [
        v for c in caps[1:] if c.reason is None for v in (c.min_m, c.max_m or -1)
    ]
    report = dict(
        graph="FULL",
        tp=4,
        experts=experts,
        top_k=top_k,
        n=n,
        k=k,
        banks=count,
        raw_batches=raw_batches,
        vector_bands=vector_bands,
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        scope="gate/up pair; routing and remaining FFN excluded",
        results=[],
    )
    for m in a.m:
        selected = np.stack(
            [
                np.random.default_rng(9100 + m + t).choice(
                    experts, top_k, replace=False
                )
                for t in range(m)
            ]
        )
        flat = selected.ravel()
        order = np.argsort(flat, kind="stable")
        ids = torch.from_numpy(flat[order]).cuda()
        boundary = np.r_[0, np.bincount(flat, minlength=experts).cumsum()]
        offsets = torch.from_numpy(boundary.astype(np.int32)).cuda()
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        routed = x[torch.from_numpy(order // top_k).cuda()].contiguous()
        expected = [
            torch.empty(m * top_k, n, device="cuda", dtype=torch.float32)
            for _ in range(2)
        ]
        for shard in range(2):
            for expert in range(experts):
                begin, end = boundary[expert : expert + 2]
                expected[shard][begin:end] = (
                    routed[begin:end].float() @ references[shard][expert].float().T
                )

        def candidate(routed=routed, offsets=offsets, ids=ids):
            result = None
            for gate, up in compact:
                banks[0].weights, banks[1].weights = gate, up
                result = [b(routed, offsets, ids) for b in banks]
            return result

        def current(routed=routed, offsets=offsets, ids=ids):
            result = None
            for prepared, raw in baseline:
                result = _expert_gate_up(
                    routed,
                    offsets,
                    ids,
                    *raw,
                    *prepared[0][2],
                    *prepared[1][2],
                    18,
                    experts,
                    32,
                    n,
                    top_k,
                    raw_batches,
                    vector_bands,
                )
            return result

        errors = []
        for call in (candidate, current):
            output = call()
            error = max(
                float((o.float() - ref).norm() / ref.norm())
                for o, ref in zip(output, expected)
            )
            if not all(bool(torch.isfinite(o).all()) for o in output) or error > 0.003:
                raise AssertionError(error)
            errors.append(error)
        new_us = elapsed(candidate, a.iterations, capture=True) / count
        old_us = elapsed(current, a.iterations, capture=True) / count
        row = dict(
            m=m,
            compact_us=new_us,
            current_dispatch_us=old_us,
            compact_relative_l2=errors[0],
            current_relative_l2=errors[1],
        )
        report["results"].append(row)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
