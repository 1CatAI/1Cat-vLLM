# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare joint raw gate/up with canonical grouped projections on real weights."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import vllm._C as core
from benchmark_gguf_turbomind import elapsed, prepare_projection

import vllm
from vllm.model_executor.kernels.gguf import (
    lattice_grouped_capabilities,
    select_lattice_grouped_capability,
)
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf", type=Path)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--m", nargs="+", type=int, default=[1, 5, 20])
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert "site-packages" in vllm.__file__, vllm.__file__
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    reader = GGUFReader(str(a.gguf))
    names = [f"blk.{a.layer}.ffn_{name}_exps.weight" for name in ("gate", "up")]
    tensors = [next(t for t in reader.tensors if t.name == name) for name in names]
    kind = int(tensors[0].tensor_type)
    if kind not in (21, 22) or int(tensors[1].tensor_type) != kind:
        p.error("This grouped decoder requires matching IQ3_S or IQ2_S gate/up")
    experts = tensors[0].data.shape[0]
    raw_banks, canonical_banks, sources = [], [], []
    for tensor in tensors:
        raws, prepared, payloads = [], [], []
        for expert in range(experts):
            raw = RawGGUFProjection.from_rows(tensor.data[expert], kind).tp_slice(
                a.rank, 4, axis=0
            )
            payload = raw.data[:, : raw.payload_bytes_per_row]
            raws.append(raw.data)
            payloads.append(payload)
            prepared.append(prepare_projection(transcode_lattice(payload, kind)))
        raw_banks.append(torch.from_numpy(np.stack(raws)).cuda())
        sources.append(payloads)
        metadata = prepared[0][2].tolist()
        assert all(item[2].tolist() == metadata for item in prepared)
        weights, stats = (
            torch.stack([p[0] for p in prepared]),
            torch.stack([p[1] for p in prepared]),
        )
        pointers = torch.ops._C.awq_moe_build_strided_ptrs(
            weights, stats, *metadata, experts
        )
        canonical_banks.append((weights, stats, *pointers))
    n, k = raw.shape
    group = 32 if kind == 21 else 16
    capabilities = lattice_grouped_capabilities(kind, k, n, experts, torch.float16)
    results = []
    for m in a.m:
        if not 1 <= m <= 32:
            p.error("M must be in 1..32")
        torch.manual_seed(20261004 + m)
        top_k = 10
        ids = torch.randn(m, experts, device="cuda").topk(top_k, dim=1).indices
        sorted_ids, order = ids.flatten().sort(stable=True)
        offsets = torch.searchsorted(
            sorted_ids, torch.arange(experts + 1, device="cuda")
        ).int()
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        routed = x[order // top_k].contiguous()
        old_outputs = [
            torch.empty((m * top_k, n), device="cuda", dtype=x.dtype) for _ in range(2)
        ]
        new_outputs = [torch.empty_like(out) for out in old_outputs]
        selected = select_lattice_grouped_capability(capabilities, m * top_k)
        assert selected.reason is None

        def canonical(
            outputs=old_outputs,
            routed=routed,
            offsets=offsets,
            operator=selected.operator,
        ):
            for out, bank in zip(outputs, canonical_banks):
                getattr(torch.ops._C, operator)(
                    out, routed, offsets, bank[2], bank[3], kind, experts, group
                )
            return outputs[0]

        def candidate(
            outputs=new_outputs,
            routed=routed,
            offsets=offsets,
            sorted_ids=sorted_ids,
            top_k=top_k,
        ):
            torch.ops._C.gguf_lattice_raw_grouped_gate_up_sm70_out(
                *outputs, routed, *raw_banks, offsets, sorted_ids, kind, top_k
            )
            return outputs[0]

        canonical()
        candidate()
        active = sorted_ids.unique().tolist()
        error_rows = []
        for outputs, payloads in zip(zip(old_outputs, new_outputs), sources):
            decoded = {
                e: torch.from_numpy(dequantize(payloads[e], kind)).cuda()
                for e in active
            }
            reference = torch.empty_like(outputs[0], dtype=torch.float32)
            for e in active:
                lo, hi = offsets[e : e + 2].tolist()
                reference[lo:hi] = routed[lo:hi].float() @ decoded[e].T
            errors = []
            for out in outputs:
                difference = out.float() - reference
                errors.append(
                    {
                        "max_abs": difference.abs().max().item(),
                        "relative_l2": (difference.norm() / reference.norm()).item(),
                        "finite": bool(torch.isfinite(out).all()),
                    }
                )
                torch.testing.assert_close(
                    out.float(), reference, rtol=0.003, atol=0.03
                )
            error_rows.append(errors)
        old_us = elapsed(canonical, a.iterations, capture=True)
        new_us = elapsed(candidate, a.iterations, capture=True)
        results.append(
            {
                "m": m,
                "routes": m * top_k,
                "active_experts": len(active),
                "canonical_capability": asdict(selected),
                "canonical_gate_up_us": old_us,
                "raw_joint_gate_up_us": new_us,
                "saved_us": old_us - new_us,
                "errors_old_new_by_projection": error_rows,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    result = {
        "version": vllm.__version__,
        "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "tensor_names": names,
        "source_type": kind,
        "local_shape": [experts, n, k],
        "raw_gate_up_bytes": sum(w.numel() for w in raw_banks),
        "scope": "Real expert weights; synthetic routes; no complete model round",
        "cases": results,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
