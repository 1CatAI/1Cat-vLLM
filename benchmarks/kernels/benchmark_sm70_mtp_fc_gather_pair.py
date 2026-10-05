# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight draft input fusion: two full gathers vs one residual gather.

Includes both Gemma norms, two FP16 projections and four HC streams. Only
the placement of the residual addition changes. Launch with TP4 torchrun;
the candidate has no model dispatch, new kernel or acceleration variable.
"""

import argparse
import hashlib
import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from benchmarks.kernels.benchmark_sm70_mtp_fp32_dense import elapsed
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import graph_capture
from vllm.model_executor.layers.layernorm import GemmaRMSNorm


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise ValueError("This screen requires TP4")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4))
    ):
        init_distributed_environment(4, rank, "env://", rank, "nccl")
        initialize_model_parallel(4)
        index_path = args.model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())["weight_map"]
        with ExitStack() as stack:
            files = {}

            def get(suffix, sharded=False):
                name = next(n for n in index if n.endswith("mtp." + suffix))
                file = index[name]
                if file not in files:
                    files[file] = stack.enter_context(
                        safe_open(args.model / file, framework="pt")
                    )
                view = files[file].get_slice(name)
                value = view[rank * 640 : (rank + 1) * 640] if sharded else view[:]
                return value.cuda().half().contiguous()

            we = get("fc_embedding.weight", True)
            wh = get("fc_hidden.weight", True)
            cfg = json.loads((args.model / "config.json").read_text())
            cfg = cfg.get("text_config", cfg)
            eps = cfg["rms_norm_eps"]
            ne = GemmaRMSNorm(2560, eps=eps).cuda().half()
            nh = GemmaRMSNorm(10240, eps=eps).cuda().half()
            ne.weight.data.copy_(get("pre_fc_norm_embedding.weight"))
            nh.weight.data.copy_(get("pre_fc_norm_hidden.weight"))
        assert we.shape == wh.shape == (640, 2560)
        xs = [
            (
                torch.randn(m, 2560, device="cuda", dtype=torch.float16),
                torch.randn(m, 4, 2560, device="cuda", dtype=torch.float16),
            )
            for m in (5, 1, 1, 1)
        ]

        def launch(combined):
            out = []
            for e, h in xs:
                el = torch.nn.functional.linear(ne(e), we)
                hl = torch.nn.functional.linear(nh(h.flatten(-2)).view_as(h), wh)
                if combined:
                    y = tensor_model_parallel_all_gather(el.unsqueeze(-2) + hl)
                else:
                    ef = tensor_model_parallel_all_gather(el)
                    hf = tensor_model_parallel_all_gather(hl)
                    y = ef.unsqueeze(-2) + hf
                out.append(y.flatten(-2))
            return out

        graphs, outputs = {}, {}
        for arm in ("control", "combined"):
            for _ in range(3):
                launch(arm == "combined")
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with (
                graph_capture(
                    device=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
                ) as context,
                torch.cuda.graph(graph, stream=context.stream),
            ):
                saved = launch(arm == "combined")
            graphs[arm], outputs[arm] = graph, saved
        errors = []
        for scale in (0, 0.03, 1, 3):
            for e, h in xs:
                e.normal_(0, scale)
                h.normal_(0, scale)
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            for control, combined in zip(outputs["control"], outputs["combined"]):
                assert torch.isfinite(combined).all()
                errors.append((control - combined).abs().max().item())
            assert max(errors) == 0, "Moving the gather changed residual bytes"
        samples = {arm: [] for arm in graphs}
        for trial in range(7):
            order = ("control", "combined") if trial % 2 else ("combined", "control")
            for arm in order:
                dist.barrier()
                value = elapsed(graphs[arm])
                times = [None] * 4
                dist.all_gather_object(times, value, group=get_tp_group().cpu_group)
                samples[arm].append(max(times))
        report = dict(
            model_admission=False,
            shapes=[5, 1, 1, 1],
            hc_streams=4,
            includes_norms_projections_residual=True,
            gathers_per_four_steps={"control": 8, "combined": 4},
            max_error=max(errors),
            samples_ms=samples,
            median_ms={k: statistics.median(v) for k, v in samples.items()},
            index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
            native_sha256=hashlib.sha256(
                Path("vllm/_C.abi3.so").read_bytes()
            ).hexdigest(),
        )
        del graphs, outputs, graph, saved
        torch.cuda.synchronize()
        if rank == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
