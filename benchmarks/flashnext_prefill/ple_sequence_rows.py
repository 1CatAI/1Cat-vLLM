# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pair real PLE forward/conv/state with TP4 row-local token math.

Require finite paired max-row L2 <=0.001 for output and committed state;
padding and unused state slots must remain unchanged. Synthetic embeddings
and replicated FP16 projections isolate the qualified PLE adapter.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
from hc_sequence_runtime import Group
from moe_tiles import relative_l2
from torch import nn

from vllm import distributed
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.distributed import parallel_state
from vllm.models.qwen4_exp.nvidia import ple_layer as ple
from vllm.models.qwen4_exp.nvidia.sm70_hc_prefill_peer import hc_sequence_shard_rows


class Projection(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "data", torch.empty(16384, 2560, device="cuda", dtype=torch.half)
        )

    def forward(self, hidden, ids, starts, context):
        return self.data[: ids.shape[0]]


def make_layer():
    torch.manual_seed(190015)
    obj = ple.Qwen4ExpPLELayer.__new__(ple.Qwen4ExpPLELayer)
    nn.Module.__init__(obj)
    obj.prefix, obj._sm70_hcx_diagnostics = "ple_row_screen", False
    obj.hc_count, obj.hidden_size, obj.conv_state_len = 4, 2560, 9
    obj.num_spec_tokens = 4
    obj.short_conv_dilation, obj._prefill_short_conv = 3, True
    obj.ple_embedding = Embedding()
    obj.key_proj = Projection(2560, 10240, bias=False, device="cuda", dtype=torch.half)
    obj.value_proj = Projection(2560, 2560, bias=False, device="cuda", dtype=torch.half)
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = ple.Qwen4ExpPLEGroupedNorm(10240, 1e-6, 2560, torch.half).cuda()
        norm.weight.data.normal_(0, 0.01)
        setattr(obj, name, norm)
    obj.register_buffer(
        "state", torch.randn(6, 10240, 13, device="cuda", dtype=torch.half) * 0.1
    )
    obj.register_buffer(
        "weights", torch.randn(10240, 4, device="cuda", dtype=torch.half) * 0.03
    )
    obj.kv_cache = (obj.state,)
    obj.conv1d = NS(weight=obj.weights.unsqueeze(1))
    obj.state_args = None
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[4096, 4097, 7344, 7345, 8192, 16384]
    )
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    dist.init_process_group("gloo")
    group, report = Group(rank), {"cases": []}
    cfg = NS(kernel_config=KernelConfig(), compilation_config=CompilationConfig(mode=0))
    try:
        with (
            patch.object(distributed, "get_tp_group", lambda: group),
            patch.object(parallel_state, "get_tp_group", lambda: group),
            set_current_vllm_config(cfg),
        ):
            layer = make_layer()

            with (
                patch.object(ple, "is_conv_state_dim_first", lambda: True),
                patch.object(
                    ple,
                    "get_forward_context",
                    lambda: NS(
                        no_compile_layers={layer.prefix: layer},
                        attn_metadata={layer.prefix: layer.state_args[0]},
                    ),
                ),
            ):
                for rows in args.tokens:
                    torch.manual_seed(1700 + rows)
                    hidden = (
                        torch.randn(rows, 10240, device="cuda", dtype=torch.half) * 0.2
                    )
                    ids = torch.arange(rows, device="cuda", dtype=torch.int32)
                    starts = torch.tensor(
                        [0, rows - 7, rows], device="cuda", dtype=torch.int32
                    )
                    indices = torch.tensor([1, 4], device="cuda", dtype=torch.int32)
                    metadata = ple.PleShortConvAttentionMetadata.__new__(
                        ple.PleShortConvAttentionMetadata
                    )
                    for name, value in dict(
                        num_prefills=2,
                        num_decodes=0,
                        num_decode_tokens=0,
                        num_prefill_tokens=rows,
                        num_actual_tokens=rows,
                        spec_sequence_masks=None,
                        state_indices_tensor=indices,
                        non_spec_query_start_loc=starts,
                        has_initial_states_p=torch.tensor([True, False], device="cuda"),
                        max_prefill_query_len=rows - 7,
                    ).items():
                        setattr(metadata, name, value)
                    layer.state_args = metadata, indices, rows
                    layer.ple_embedding.data[:rows].normal_(0, 0.2)
                    hidden_local = hc_sequence_shard_rows(hidden)
                    initial = layer.state.clone()
                    expected = layer(hidden, ids, starts, ids)
                    expected_state = layer.state.clone()
                    layer.state.copy_(initial)
                    actual_local = layer(
                        hidden_local, ids, starts, ids, hc_sequence_parallel=True
                    )
                    actual = group.all_gather(actual_local, dim=0)[:rows]
                    errors = [
                        relative_l2(actual, expected),
                        relative_l2(layer.state.flatten(1), expected_state.flatten(1)),
                    ]
                    assert max(errors) <= 0.001, errors
                    assert bool(torch.isfinite(actual).all()) and bool(
                        torch.isfinite(layer.state).all()
                    )
                    assert torch.equal(layer.state[[0, 2, 3, 5]], initial[[0, 2, 3, 5]])
                    assert torch.equal(layer.state[..., 9:], initial[..., 9:])
                    padded = group.all_gather(actual_local, dim=0)[rows:]
                    assert bool((padded == 0).all())
                    case = dict(rows=rows, errors=errors, padded_rows=padded.shape[0])
                    if rows in (7344, 7345):
                        compiled = torch.compile(
                            layer.forward,
                            backend="inductor",
                            fullgraph=True,
                            dynamic=True,
                        )
                        # Both compiler branches use the real opaque convolution
                        # transaction. Measure live tensor peaks after compilation.
                        for sequence in (False, True):
                            layer.state.copy_(initial)
                            compiled(
                                hidden_local if sequence else hidden,
                                ids,
                                starts,
                                ids,
                                hc_sequence_parallel=sequence,
                            )
                        layer.state.copy_(initial)
                        compiled_expected = compiled(
                            hidden, ids, starts, ids, hc_sequence_parallel=False
                        ).clone()
                        compiled_expected_state = layer.state.clone()
                        case["compiled_control_vs_eager_errors"] = [
                            relative_l2(compiled_expected, expected),
                            relative_l2(
                                compiled_expected_state.flatten(1),
                                expected_state.flatten(1),
                            ),
                        ]
                        print(
                            "PLE_COMPILED_CONTROL",
                            rank,
                            rows,
                            case["compiled_control_vs_eager_errors"],
                            flush=True,
                        )
                        peaks = []
                        for sequence in (False, True):
                            layer.state.copy_(initial)
                            torch.accelerator.synchronize()
                            before = torch.accelerator.memory_allocated()
                            torch.accelerator.reset_peak_memory_stats()
                            result = compiled(
                                hidden_local if sequence else hidden,
                                ids,
                                starts,
                                ids,
                                hc_sequence_parallel=sequence,
                            )
                            torch.accelerator.synchronize()
                            peaks.append(
                                (torch.accelerator.max_memory_allocated() - before)
                                / 2**20
                            )
                            if sequence:
                                result = group.all_gather(result, dim=0)[:rows]
                            compiled_errors = [
                                relative_l2(result, compiled_expected),
                                relative_l2(
                                    layer.state.flatten(1),
                                    compiled_expected_state.flatten(1),
                                ),
                            ]
                            print(
                                "PLE_COMPILED_PAIR",
                                rank,
                                rows,
                                sequence,
                                compiled_errors,
                                peaks[-1],
                                flush=True,
                            )
                            assert max(compiled_errors) <= 0.001, compiled_errors
                            del result
                        case["compiled_live_peak_delta_mib"] = peaks
                        assert peaks[1] < peaks[0], peaks
                        layer.state.copy_(initial)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = compiled(
                                hidden_local,
                                ids,
                                starts,
                                ids,
                                hc_sequence_parallel=True,
                            )
                        hidden.mul_(0.75)
                        hidden_local.mul_(0.75)
                        layer.ple_embedding.data[:rows].mul_(-0.5)
                        layer.state.copy_(initial)
                        changed = compiled(
                            hidden, ids, starts, ids, hc_sequence_parallel=False
                        ).clone()
                        changed_state = layer.state.clone()
                        layer.state.copy_(initial)
                        graph.replay()
                        replay = group.all_gather(captured, dim=0)[:rows]
                        replay_errors = [
                            relative_l2(replay, changed),
                            relative_l2(
                                layer.state.flatten(1), changed_state.flatten(1)
                            ),
                        ]
                        assert max(replay_errors) <= 0.001, replay_errors
                        assert torch.equal(
                            layer.state[[0, 2, 3, 5]], initial[[0, 2, 3, 5]]
                        )
                        assert torch.equal(layer.state[..., 9:], initial[..., 9:])
                        case["replay_errors"] = replay_errors
                        del graph, captured, changed, changed_state, replay, compiled
                    report["cases"].append(case)
                    print("PLE_ROWS_PASS", rank, rows, errors, flush=True)
                if rank == 0:
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
