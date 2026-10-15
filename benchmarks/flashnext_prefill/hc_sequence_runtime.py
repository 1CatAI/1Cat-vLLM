# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual HC row stages and model adapters on four idle SM70 GPUs.

A two-layer backbone reuses the real decoder/model forward, HC operators and
producer overrides. Synthetic attention/MoE/PLE isolate row and state ownership.
Require finite paired max-row L2 <= 0.001, injection-factor error <= 0.0005,
preserved MTP/PLE capacity tails, and actual ReduceScatter with no producer AR.
This does not replace checkpoint serving and output-quality qualification.
"""

import argparse
import json
import os
import weakref
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
from moe_tiles import relative_l2
from torch import nn

from vllm import distributed
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.config.execution_policy import CommunicationPolicy
from vllm.config.kernel import KernelConfig
from vllm.distributed import parallel_state
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.models.qwen4_exp.nvidia import hyperconnection as hc
from vllm.models.qwen4_exp.nvidia import model as adapter
from vllm.models.qwen4_exp.nvidia import sm70_hcx as hcx
from vllm.models.qwen4_exp.nvidia.sm70_hc_prefill_peer import (
    prepare_hc_prefill_peer_runtime,
)
from vllm.runtime_resources import release_runtime_resources


class Group:
    world_size = 4
    unique_name = "hc_sequence_runtime"

    def __init__(self, rank):
        self.rank_in_group = rank
        self.cpu_group = dist.group.WORLD
        self.nccl = PyNcclCommunicator(dist.group.WORLD, rank)
        self.device_communicator = NS(
            hc_ll_comm=NS(status={"enabled": True}, order=(0, 1, 2, 3)),
            pynccl_comm=self.nccl,
            ca_comm=NS(fully_connected=True),
        )
        self.counts = {"ar": 0, "rs": 0, "ag": 0}
        parallel_state._groups[self.unique_name] = weakref.ref(self)

    def all_reduce(self, x):
        return torch.ops.vllm.all_reduce(x, self.unique_name)

    def all_gather(self, x, dim):
        return torch.ops.vllm.all_gather(x, dim, 4, self.unique_name)

    def reduce_scatter(self, x, dim):
        return torch.ops.vllm.reduce_scatter(x, dim, 4, self.unique_name)

    def _all_reduce_out_place(self, x):
        self.counts["ar"] += 1
        out = torch.empty_like(x)
        self.nccl.all_reduce(x, out)
        return out

    def _all_gather_out_place(self, x, dim):
        assert dim == 0
        self.counts["ag"] += 1
        out = x.new_empty((4 * x.shape[0], *x.shape[1:]))
        self.nccl.all_gather(out, x)
        return out

    def _reduce_scatter_out_place(self, x, dim):
        assert dim == 0
        self.counts["rs"] += 1
        out = x.new_empty((x.shape[0] // 4, *x.shape[1:]))
        self.nccl.reduce_scatter(out, x)
        return out


def make_hc(rank, number, *, combine=True, payload=False, partial=True):
    torch.manual_seed(5000 + number)
    obj = hc.GatedResidual.__new__(hc.GatedResidual)
    nn.Module.__init__(obj)
    obj.config = NS(rms_norm_eps=1e-6)
    obj.hc_count, obj.hidden_size, obj.lora_rank, obj.use_combine = (
        4,
        2560,
        320,
        combine,
    )
    obj.hc_norm = nn.Module()
    obj.hc_norm.register_parameter(
        "weight",
        nn.Parameter(torch.randn(10240, device="cuda", dtype=torch.half) * 0.01),
    )
    down = torch.randn(336, 10240, device="cuda", dtype=torch.half) * 0.01
    up = torch.randn(10240, 320, device="cuda", dtype=torch.half) * 0.01
    obj.register_buffer("hc_shard_down", hcx.pack_down(down, rank))
    obj.register_buffer("hc_shard_up", hcx.pack_up(up, rank))
    if partial:
        obj.enable_partial_inputs(f"test{number}")
    obj._hcx_moe_payload = payload
    return obj


class Attention(nn.Module):
    def __init__(self, rank, name):
        super().__init__()
        self.name = name
        hcx.register_output_projection(name, NS(output_size=2560), defer=False)

        class Projection:
            output_size = 2560

            def __call__(self, x):
                return x * ((rank + 1) * 0.025), None

        hcx.register_output_projection(name, Projection(), defer=False)

    def forward(self, *, hidden_states, output):
        result = hcx.partial_projection(self.name, hidden_states)
        if output is not None:
            output.copy_(result)
        return result


class MoE(adapter.Qwen4ExpSparseMoeBlock):
    def __init__(self, group, rank, name):
        nn.Module.__init__(self)
        self.rank, self.name = rank, name
        hcx.register_moe_runner(
            name,
            NS(
                _maybe_sm70_moe_sum2_allreduce=lambda *args: None,
                _maybe_reduce_final_output=lambda x, n: group.all_reduce(x)[..., :n],
            ),
        )

    def forward(self, x):
        return hcx.partial_moe_output(
            self.name, x * 0.01, x * ((self.rank + 1) * 0.02), 2560
        )

    def forward_partials(self, x):
        return x * ((self.rank + 1) * 0.02), x * 0.01


class Ple(nn.Module):
    def __init__(self):
        super().__init__()
        self._sm70_hcx_diagnostics = False
        self.prefix = "toy_ple"
        self.register_buffer(
            "state", torch.full((16384,), -99, device="cuda", dtype=torch.half)
        )

    def forward(self, hidden, ids, starts, context, *, hc_sequence_parallel=False):
        column = hidden[:, 0]
        if hc_sequence_parallel:
            column = distributed.get_tp_group().all_gather(column.contiguous(), dim=0)
        self.state[: ids.shape[0]].copy_(column[: ids.shape[0]])
        output = hidden * 0.125
        return output


def make_model(group, rank):
    obj = adapter.Qwen4ExpModel.__new__(adapter.Qwen4ExpModel)
    nn.Module.__init__(obj)
    obj.config = NS(hc_count=4)
    obj.start_layer, obj.end_layer = 0, 2
    obj.layers = nn.ModuleList()
    for i in range(2):
        layer = adapter.Qwen4ExpDecoderLayer.__new__(adapter.Qwen4ExpDecoderLayer)
        nn.Module.__init__(layer)
        layer.ple = Ple() if i == 1 else None
        layer.layer_type = "linear_attention"
        layer.attn_hyper_connection = make_hc(rank, 2 * i, payload=i > 0, partial=i > 0)
        layer.mlp_hyper_connection = make_hc(rank, 2 * i + 1)
        layer.linear_attn = Attention(rank, f"attn{i}")
        layer.mlp = MoE(group, rank, f"moe{i}")
        obj.layers.append(layer)
    obj.hyper_connection_mixer = make_hc(rank, 4, combine=False, payload=True)
    obj.register_buffer(
        "_mtp_hidden_buffer",
        torch.full((16384, 10240), -99, device="cuda", dtype=torch.half),
    )
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peer", action="store_true")
    parser.add_argument("--reduced-precision", action="store_true")
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 7344, 11424])
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    torch.set_grad_enabled(False)
    dist.init_process_group("gloo")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        args.reduced_precision
    )
    group = Group(rank)
    pp = NS(is_first_rank=True, is_last_rank=True)
    cfg = NS(
        kernel_config=KernelConfig(),
        parallel_config=NS(communication=CommunicationPolicy()),
        compilation_config=CompilationConfig(mode=0),
        scheduler_config=NS(max_num_batched_tokens=8192),
    )
    cfg.kernel_config.layer_execution.hc_prefill_sequence_parallel = args.peer
    cfg.kernel_config.layer_execution.hc_prefill_peer_norm = args.peer
    report = {"cases": []}
    try:
        with (
            patch.object(parallel_state, "get_tp_group", lambda: group),
            patch.object(distributed, "get_tp_group", lambda: group),
            patch.object(
                adapter,
                "get_pp_group",
                lambda: pp,
            ),
            patch.object(
                distributed, "tensor_model_parallel_all_reduce", group.all_reduce
            ),
            set_current_vllm_config(cfg),
        ):
            prepare_hc_prefill_peer_runtime(cfg, torch.device("cuda", rank))
            owner = make_model(group, rank)
            view = adapter._Qwen4ExpPrefillSequenceModel(
                target_model=owner, vllm_config=cfg
            )
            assert not list(view.parameters()) and not list(view.buffers())
            compiled = None
            for rows in args.tokens:
                torch.manual_seed(8000 + rows)
                embedding = (
                    torch.randn(rows, 2560, device="cuda", dtype=torch.half) * 0.3
                )
                ids = torch.arange(rows, device="cuda", dtype=torch.int32)
                positions = ids.long()
                kwargs = dict(
                    inputs_embeds=embedding,
                    query_start_loc=torch.tensor(
                        [0, rows], device="cuda", dtype=torch.int32
                    ),
                    ngram_context=ids,
                )
                with hcx.hc_prefill_sequence_parallel_context(False):
                    expected = owner.forward(ids, positions, **kwargs)
                hidden_expected = owner._mtp_hidden_buffer[:rows].clone()
                ple_expected = owner.layers[1].ple.state[:rows].clone()
                group.counts = {"ar": 0, "rs": 0, "ag": 0}
                with hcx.hc_prefill_sequence_parallel_context(True):
                    actual = view(ids, positions, **kwargs)
                counters = group.counts.copy()
                padded_rows = (rows + 3) // 4 * 4
                expected_rs = 0 if args.peer and 4096 <= padded_rows <= 8192 else 4
                assert (
                    counters["ar"] == 0
                    and counters["rs"] == expected_rs
                    and counters["ag"] == 16
                ), counters

                def check(output, targets, rows=rows):
                    expected, hidden_expected, ple_expected = targets
                    errors = [
                        relative_l2(output, expected),
                        relative_l2(owner._mtp_hidden_buffer[:rows], hidden_expected),
                        relative_l2(
                            owner.layers[1].ple.state[:rows, None],
                            ple_expected[:, None],
                        ),
                    ]
                    # A one-element PLE snapshot can be near zero. Verify its
                    # entire state scale instead of a per-scalar ratio.
                    errors[2] = float(
                        (
                            owner.layers[1].ple.state[:rows].float()
                            - ple_expected.float()
                        ).norm()
                        / ple_expected.float().norm()
                    )
                    assert max(errors) <= 0.001, errors
                    assert bool(torch.isfinite(output).all())
                    assert bool((owner._mtp_hidden_buffer[rows:] == -99).all())
                    assert bool((owner.layers[1].ple.state[rows:] == -99).all())
                    return errors

                targets = (expected, hidden_expected, ple_expected)
                errors = check(actual, targets)
                if rows == 7344:
                    compiled = torch.compile(
                        view.forward, backend="eager", fullgraph=True, dynamic=True
                    )
                if compiled is not None:
                    with hcx.hc_prefill_sequence_parallel_context(True):
                        check(compiled(ids, positions, **kwargs), targets)
                if rows in (7344, 7345):
                    graph = torch.cuda.CUDAGraph()
                    with (
                        hcx.hc_prefill_sequence_parallel_context(True),
                        torch.cuda.graph(graph),
                    ):
                        captured = view(ids, positions, **kwargs)
                    embedding.mul_(-0.5)
                    with hcx.hc_prefill_sequence_parallel_context(False):
                        expected = owner.forward(ids, positions, **kwargs)
                    hidden_expected = owner._mtp_hidden_buffer[:rows].clone()
                    ple_expected = owner.layers[1].ple.state[:rows].clone()
                    graph.replay()
                    check(captured, (expected, hidden_expected, ple_expected))
                    del graph, captured
                checks = [None] * 4
                dist.all_gather_object(checks, errors)
                report["cases"].append(
                    {
                        "rows": rows,
                        "all_rank_errors": checks,
                        "route_counts": counters,
                        "passed": True,
                    }
                )
                if rank == 0:
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(report["cases"][-1]), flush=True)
    finally:
        torch.accelerator.synchronize()
        release_runtime_resources(cfg)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
