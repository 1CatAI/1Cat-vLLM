# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare graph-captured local-vocabulary reranking with full FP32 logits."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["VLLM_SM70_DFLASH2_QPN8_RERANK"] = "1"
    os.environ["VLLM_SM70_DFLASH2_FP32_LOGITS"] = "1"
    from vllm import envs
    from vllm.model_executor.layers import vocab_parallel_embedding as vocab

    envs.disable_envs_cache()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    key = next(k for k in index["weight_map"] if k.endswith("lm_head.weight"))
    with safe_open(args.model / index["weight_map"][key], framework="pt") as tensors:
        weight = tensors.get_tensor(key)
    inputs = []
    for path in sorted(args.inputs.glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        inputs.append((str(path), record["input"]))
    if not inputs:
        raise ValueError("No captured model inputs found")
    report = {
        "model": str(args.model),
        "measurement": "CUDA Graph operator check",
        "input_hashes": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p, _ in inputs
        },
        "results": [],
    }
    try:
        for tp in (1, 2, 4):
            n = weight.shape[0] // tp
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(
                weight[:n].to(device="cuda", dtype=torch.float16), requires_grad=False
            )
            layer.tp_size = tp
            layer.shard_indices = SimpleNamespace(
                num_org_vocab_padding=0, org_vocab_start_index=0
            )
            assert vocab._prepare_sm70_dflash2_qpn8_rerank(layer)
            graphs = {}
            for path, hidden in inputs:
                rows = hidden.shape[0]
                if rows not in graphs:
                    x = hidden.cuda()
                    values, ids = vocab._maybe_sm70_dflash2_qpn8_rerank(layer, x, 21)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        values, ids = vocab._maybe_sm70_dflash2_qpn8_rerank(
                            layer, x, 21
                        )
                    graphs[rows] = x, values, ids, graph
                x, values, ids, graph = graphs[rows]
                x.copy_(hidden)
                graph.replay()
                dense = torch.mm(x, layer.weight.T, out_dtype=torch.float32)
                expected_values, expected_ids = dense.topk(21, sorted=True)
                same_ids = torch.equal(ids, expected_ids)
                exact_values = dense.gather(1, ids)
                error = float((values - exact_values).abs().max())
                fp64 = (x.double()[:, None, :] * layer.weight[ids].double()).sum(-1)
                row = {
                    "tp": tp,
                    "rows": rows,
                    "input": path,
                    "candidates": layer._sm70_dflash2_qpn8_ids.shape[1],
                    "top21_ids_equal": same_ids,
                    "max_abs_logit_diff": error,
                    "candidate_fp64_max_abs": float(
                        (values.double() - fp64).abs().max()
                    ),
                    "dense_fp64_max_abs": float(
                        (exact_values.double() - fp64).abs().max()
                    ),
                    "missing_top21": int(
                        (~(expected_ids[:, :, None] == ids[:, None, :]).any(-1)).sum()
                    ),
                }
                report["results"].append(row)
                print(json.dumps(row), flush=True)
                assert torch.isfinite(values).all()
                assert same_ids, row
                # cuBLAS and indexed FP32 use different reduction orders. An
                # FP64 dot is the independent accuracy oracle; IDs must still
                # match the complete vocabulary's top-21 exactly.
                torch.testing.assert_close(values.double(), fp64, atol=3e-6, rtol=3e-6)
            del graphs, x, values, ids, graph, layer, dense
            torch.cuda.empty_cache()
        report["passed"] = True
    finally:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
