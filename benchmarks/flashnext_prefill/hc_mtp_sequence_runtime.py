# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real MTP/HC adapters with replicated producers on four SM70 GPUs.

Synthetic FC/attention/MoE isolate row ownership, padding, returned multi-stream
state, and changed-input graph replay. Paired max-row L2 must remain <=0.001;
this is an adapter gate, not checkpoint quality or throughput evidence.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
from hc_sequence_runtime import Group, make_hc
from moe_tiles import relative_l2
from torch import nn

from vllm import distributed
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.config.execution_policy import CommunicationPolicy
from vllm.config.kernel import KernelConfig
from vllm.distributed import parallel_state
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.models.qwen4_exp.nvidia import model as adapter
from vllm.models.qwen4_exp.nvidia import mtp, sm70_mtp_fc


class Attention(nn.Module):
    def __init__(self, group, rank):
        super().__init__()
        self.group, self.rank = group, rank

    def forward(self, *, hidden_states, positions, output):
        return self.group.all_reduce(hidden_states * ((self.rank + 1) * 0.025))


class MoE(adapter.Qwen4ExpSparseMoeBlock):
    def __init__(self, group, rank):
        nn.Module.__init__(self)
        self.group, self.rank = group, rank

    def forward(self, x):
        return self.group.all_reduce(x * ((self.rank + 1) * 0.02) + x * 0.01)

    def forward_partials(self, x):
        raise AssertionError("Draft must preserve its original producer reduction")


def make_predictor(group, rank, project_fc=False):
    owner = mtp.Qwen4ExpMultiTokenPredictor.__new__(mtp.Qwen4ExpMultiTokenPredictor)
    nn.Module.__init__(owner)
    owner.hc_count, owner.hidden_size, owner.num_mtp_layers = 4, 2560, 1
    for name in (
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "fc_embedding",
        "fc_hidden",
    ):
        setattr(owner, name, nn.Identity())
    if project_fc:
        for name in ("fc_embedding", "fc_hidden"):
            layer = object.__new__(linear.ColumnParallelLinear)
            nn.Module.__init__(layer)
            layer.weight = nn.Parameter(
                torch.randn(640, 2560, device="cuda", dtype=torch.half) * 0.01,
                requires_grad=False,
            )
            layer.bias, layer.skip_bias_add = None, False
            layer.tp_size, layer.gather_output, layer.return_bias = 4, True, False
            layer.quant_method = linear.UnquantizedLinearMethod()
            layer._layer_trace = NS(unquant_debug=False)
            setattr(owner, name, layer)
        owner.pre_fc_norm_embedding = GemmaRMSNorm(2560).half().cuda()
        owner.pre_fc_norm_hidden = GemmaRMSNorm(10240).half().cuda()
    layer = adapter.Qwen4ExpDecoderLayer.__new__(adapter.Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer.ple, layer.layer_type = None, "full_attention"
    layer.attn_hyper_connection = make_hc(rank, 20, partial=False)
    layer.mlp_hyper_connection = make_hc(rank, 21, partial=False)
    layer.self_attn = Attention(group, rank)
    layer.mlp = MoE(group, rank)
    owner.layers = nn.ModuleList([layer])
    owner.hyper_connection_mixer = make_hc(rank, 22, combine=False, partial=False)
    return owner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-fc", action="store_true")
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[4096, 4097, 8192, 16384]
    )
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    dist.init_process_group("gloo")
    group = Group(rank)

    def gather_columns(x):
        gathered = group.all_gather(x, dim=0)
        return (
            gathered.reshape(4, *x.shape)
            .movedim(0, x.ndim - 1)
            .reshape(*x.shape[:-1], 4 * x.shape[-1])
        )

    cfg = NS(
        kernel_config=KernelConfig(),
        parallel_config=NS(communication=CommunicationPolicy()),
        compilation_config=CompilationConfig(mode=0, custom_ops=["all"]),
    )
    report = {"cases": [], "precision_reduction": True}
    try:
        with (
            patch.object(parallel_state, "get_tp_group", lambda: group),
            patch.object(distributed, "get_tp_group", lambda: group),
            patch.object(mtp, "get_tp_group", lambda: group),
            patch.object(sm70_mtp_fc, "get_tp_group", lambda: group),
            patch.object(linear, "tensor_model_parallel_all_gather", gather_columns),
            patch.object(sm70_mtp_fc, "maybe_combine_fc", lambda *args: None),
            set_current_vllm_config(cfg),
        ):
            owner = make_predictor(group, rank, args.project_fc)
            view = mtp._Qwen4ExpMTPPrefillSequenceModel(
                target_model=owner, vllm_config=cfg
            )
            assert not list(view.parameters()) and not list(view.buffers())
            compiled = torch.compile(
                view.forward, backend="eager", fullgraph=True, dynamic=True
            )
            for rows in args.tokens:
                torch.manual_seed(1900 + rows)
                hidden = torch.randn(rows, 10240, device="cuda", dtype=torch.half) * 0.2
                embeds = torch.randn(rows, 2560, device="cuda", dtype=torch.half) * 0.1
                source = hidden.clone()
                ids = torch.arange(rows, device="cuda", dtype=torch.int32)
                positions = ids.long()

                def expected(
                    ids=ids, positions=positions, hidden=hidden, embeds=embeds
                ):
                    return owner.forward(ids, positions, hidden, inputs_embeds=embeds)

                def check(actual, reference, rows=rows, hidden=hidden, source=source):
                    errors = [relative_l2(a, b) for a, b in zip(actual, reference)]
                    assert max(errors) <= 0.001, errors
                    assert all(bool(torch.isfinite(a).all()) for a in actual)
                    assert actual[0].shape == (rows, 2560) and actual[1].shape == (
                        rows,
                        10240,
                    )
                    assert torch.equal(hidden, source), "Borrowed target state mutated"
                    assert actual[1].data_ptr() != hidden.data_ptr()
                    return errors

                torch.accelerator.synchronize()
                base = torch.accelerator.memory_allocated()
                torch.accelerator.reset_peak_memory_stats()
                reference = expected()
                torch.accelerator.synchronize()
                baseline_peak = torch.accelerator.max_memory_allocated() - base
                group.counts = {"ar": 0, "rs": 0, "ag": 0}
                base = torch.accelerator.memory_allocated()
                torch.accelerator.reset_peak_memory_stats()
                actual = view(ids, positions, hidden, inputs_embeds=embeds)
                torch.accelerator.synchronize()
                candidate_peak = torch.accelerator.max_memory_allocated() - base
                counters = group.counts.copy()
                assert counters["ar"] == 2 and counters["rs"] == 0, counters
                errors = check(actual, reference)
                check(compiled(ids, positions, hidden, inputs_embeds=embeds), reference)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = view(ids, positions, hidden, inputs_embeds=embeds)
                embeds.mul_(-0.5)
                reference = expected()
                graph.replay()
                replay_errors = check(captured, reference)
                report["cases"].append(
                    dict(
                        rows=rows,
                        errors=errors,
                        replay_errors=replay_errors,
                        collectives=counters,
                        baseline_peak_mib=baseline_peak / 2**20,
                        candidate_peak_mib=candidate_peak / 2**20,
                    )
                )
                del graph, captured, actual, reference, hidden, source, embeds
                del expected, check
                torch.accelerator.empty_cache()
                print("MTP_HC_ROW_PASS", rank, rows, errors, flush=True)
            if rank == 0:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
