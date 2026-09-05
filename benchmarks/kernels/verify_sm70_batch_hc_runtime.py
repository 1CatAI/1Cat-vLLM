# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GatedResidual/TP channel route checks, not a model quality score."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual
from vllm.models.qwen4_exp.nvidia.sm70_batch_hc import prepare_sm70_batch_hc
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import Qwen38SM70FP16LinearMethod


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4))
    with set_current_vllm_config(config):
        init_distributed_environment(world_size=4, rank=rank, local_rank=rank)
        initialize_model_parallel(tensor_model_parallel_size=4)
        comm = get_tp_group().device_communicator
        assert comm.sm70_hc_batch_comm is not None
        assert comm.sm70_hc_batch_comm is not comm.ca_comm
        assert ops.supports_sm70_qwen38_hc_batch()
        assert ops.supports_sm70_qwen38_hc_up_mix_allgather()
        channel = comm.sm70_hc_batch_comm
        hc = GatedResidual(
            HyperConnectionConfig(
                hc_count=4,
                hidden_size=2560,
                hc_lowrank=320,
                params_dtype=torch.float16,
                hc_per_branch_norm=True,
            )
        ).cuda()
        hc.input_mix_weight_down_block_inject.quant_method = (
            Qwen38SM70FP16LinearMethod()
        )
        hc._sm70_qwen38_fp16_fused_hc = True
        index = json.loads((args.model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        prefix = "model.language_model.layers.0.attn_hyper_connection."

        def read_weight(suffix):
            name = prefix + suffix
            with safe_open(args.model / index[name], framework="pt", device="cpu") as f:
                return f.get_tensor(name).half().cuda()

        down = read_weight("input_mix_weight_down.weight")
        injection = read_weight("block_inject_weight.weight")
        hc.input_mix_weight_down_block_inject.weight.copy_(
            torch.cat((down, injection, down.new_zeros(12, 10240)))
        )
        hc.input_mix_weight_up.weight.copy_(read_weight("input_mix_weight_up.weight"))
        prepare_sm70_batch_hc(hc)
        packed = hc._sm70_batch_hc_up
        rows = []
        native_calls = [0]
        original_down = ops.sm70_qwen38_hc_batch_down

        def tracked_down(*args):
            native_calls[0] += 1
            return original_down(*args)

        ops.sm70_qwen38_hc_batch_down = tracked_down
        try:
            for m, prefill in (
                (1, False),
                (4, False),
                (8, False),
                (16, False),
                (4, True),
                (17, False),
                (32, True),
            ):
                metadata = {"attn": NS(max_query_len=m if prefill else 1)}
                torch.manual_seed(7)
                x = torch.randn(m, 10240, dtype=torch.float16, device="cuda")
                dist.broadcast(x, 0)
                with set_forward_context(metadata, config, num_tokens=m):
                    hc._sm70_batch_hc_up = None
                    reference = tuple(t.clone() for t in hc._project(x))
                    hc._sm70_batch_hc_up = packed
                    before = native_calls[0]
                    actual = tuple(t.clone() for t in hc._project(x))
                    hit = native_calls[0] > before
                    assert hit == (not prefill and 2 <= m <= 16)
                    if not hit:
                        for a, b in zip(actual, reference):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                    graph = torch.cuda.CUDAGraph()
                    # Warm every native/projection path before graph capture.
                    for _ in range(3):
                        hc._project(x)
                    torch.accelerator.synchronize()
                    with torch.cuda.graph(graph):
                        output = hc._project(x)
                    for cycle in range(8):
                        x.normal_().mul_(0.25 + cycle % 3)
                        dist.broadcast(x, 0)
                        expected = tuple(t.clone() for t in hc._project(x))
                        for t in output:
                            t.fill_(float("nan"))
                        graph.replay()
                        torch.accelerator.synchronize()
                        for a, b in zip(output, expected):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                    rows.append(
                        {
                            "rows": m,
                            "prefill": prefill,
                            "native_hit": hit,
                            "graph_eager_exact": True,
                            "reference_max_abs": [
                                (a.float() - b.float()).abs().max().item()
                                for a, b in zip(actual, reference)
                            ],
                        }
                    )
            # An initial prefill trace must not bake away the opaque decode op.
            compiled = torch.compile(hc._project, backend="eager", dynamic=True)
            for m, prefill in ((32, True), (4, False), (1, False), (16, True)):
                x = torch.randn(m, 10240, dtype=torch.float16, device="cuda")
                dist.broadcast(x, 0)
                with set_forward_context(
                    {"attn": NS(max_query_len=m if prefill else 1)},
                    config,
                    num_tokens=m,
                ):
                    expected = tuple(t.clone() for t in hc._project(x))
                    before = native_calls[0]
                    result = compiled(x)
                    hit = native_calls[0] > before
                    assert hit == (not prefill and 2 <= m <= 16)
                    for a, b in zip(result, expected):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
            pointer = packed.data_ptr()
            prepare_sm70_batch_hc(hc)
            assert hc._sm70_batch_hc_up.data_ptr() == pointer
            gathered = [None] * 4
            dist.all_gather_object(gathered, rows, group=get_tp_group().cpu_group)
            if rank == 0:
                lib = Path(os.environ["VLLM_SM70_CUSTOM_AR_LIBRARY"])
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(
                    json.dumps(
                        {
                            "rows": gathered,
                            "native_sha256": hashlib.sha256(
                                lib.read_bytes()
                            ).hexdigest(),
                            "torch": torch.__version__,
                            "prefill_first_compile_passed": True,
                            "reload_preserved_pointer": True,
                            "M1_native_capabilities": True,
                            "scope": (
                                "Real HC component, not full-model quality "
                                "or throughput"
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
        finally:
            ops.sm70_qwen38_hc_batch_down = original_down
            hc._sm70_batch_hc_up = packed
            torch.accelerator.synchronize()
            destroy_model_parallel()
            assert channel._ptr == 0
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
