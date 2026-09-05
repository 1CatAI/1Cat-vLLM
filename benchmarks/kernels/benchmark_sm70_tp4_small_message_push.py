# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed-size graph correctness gate for the experimental TP4 push admission.

Launch with torchrun on four idle fully-connected V100 GPUs. This is a native
collective gate, not a model-quality test or an endpoint speed measurement.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from vllm import envs
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

FLAG = "VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES"
# Include partial CTAs, HC payloads, established sizes and the pull fallback.
SIZES = (
    16,
    2048,
    2064,
    2688,
    5120,
    5376,
    8192,
    10752,
    20480,
    25600,
    40960,
    81920,
    81936,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cycles", default=64, type=int)
    args = parser.parse_args()
    if args.cycles <= 0:
        raise ValueError("cycles must be positive")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    group = dist.new_group(backend="gloo")
    assert dist.get_world_size() == 4
    assert torch.cuda.get_device_capability() == (7, 0)
    ca = CustomAllreduce(group, rank, max_size=128 * 1024)
    old = os.environ.get(FLAG)
    try:
        assert not ca.disabled and ca.fully_connected
        assert ca.sm70_tp4_push_buffer_ptrs is not None
        inputs = [
            torch.zeros(n // 2, device="cuda", dtype=torch.float16) for n in SIZES
        ]
        graphs, storage = [], []
        for enabled, reverse, sum2 in (
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, False, True),
        ):
            os.environ[FLAG] = str(int(enabled))
            envs.disable_envs_cache()
            buffers = [x.new_full((x.numel() + 16,), -37) for x in inputs]
            outputs = [x[8:-8] for x in buffers]
            graph = torch.cuda.CUDAGraph()
            order = list(range(len(inputs)))
            if reverse:
                order.reverse()
            torch.cuda.synchronize()
            dist.barrier()
            with ca.capture(), torch.cuda.graph(graph):
                # Odd count, changing CTA counts, followed by another graph:
                # per-block epochs must stay valid across all these transitions.
                for i in order:
                    ca.all_reduce(inputs[i], out=outputs[i], registered=True)
                    if sum2:
                        # Same push storage, but sum2 retains its old launch
                        # policy. Exercise transitions between both protocols.
                        ca.all_reduce_sum2(inputs[i], inputs[i], out=outputs[i])
            graphs.append(graph)
            storage.append(buffers)

        checks = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260905 + rank)
        specials = torch.tensor(
            [
                0,
                -32768,
                1,
                -32767,
                15360,
                -17408,
                31743,
                -1025,
                31744,
                -1024,
                32639,
                32256,
                -512,
            ],
            device="cuda",
            dtype=torch.int16,
        ).view(torch.float16)
        for pattern in ("random", "signed_zero", "special"):
            for cycle in range(args.cycles):
                for x in inputs:
                    if pattern == "random":
                        x.normal_(generator=generator).mul_(0.03)
                    elif pattern == "signed_zero":
                        x.zero_()
                        x[::2] = -0.0
                    else:
                        idx = torch.arange(x.numel(), device="cuda") + cycle + rank
                        x.copy_(specials[idx % specials.numel()])
                # Distinct outputs and poison on every replay catch stale writes.
                for buffers in storage:
                    for b in buffers:
                        b[8:-8].fill_(float("nan"))
                dist.barrier()
                for which in (0, 1, 2, 3) if cycle % 2 else (3, 2, 0, 1):
                    if rank == cycle % 4:
                        torch.cuda._sleep(10000)
                    graphs[which].replay()
                torch.cuda.synchronize()
                for i, x in enumerate(inputs):
                    peers = [torch.empty_like(x) for _ in range(4)]
                    dist.all_gather(peers, x)
                    expected = peers[0].float()
                    for peer in peers[1:]:
                        expected.add_(peer.float())
                    expected = expected.half()
                    expected_sum2 = (peers[0] + peers[0]).float()
                    for peer in peers[1:]:
                        expected_sum2.add_((peer + peer).float())
                    expected_sum2 = expected_sum2.half()
                    for graph_id, buffers in enumerate(storage):
                        actual = buffers[i][8:-8]
                        reference = expected_sum2 if graph_id == 3 else expected
                        finite = torch.isfinite(reference)
                        torch.testing.assert_close(
                            actual, reference, rtol=0, atol=0, equal_nan=True
                        )
                        assert torch.equal(
                            actual.view(torch.int16)[finite],
                            reference.view(torch.int16)[finite],
                        )
                        assert (buffers[i][:8] == -37).all()
                        assert (buffers[i][-8:] == -37).all()
            checks.append(
                dict(
                    pattern=pattern,
                    cycles=args.cycles,
                    graph_orders=len(graphs),
                    sizes=len(SIZES),
                )
            )
            if rank == 0:
                print(f"passed {pattern}: {args.cycles} cycles", flush=True)
        if rank == 0:
            library = os.environ.get("VLLM_SM70_CUSTOM_AR_LIBRARY")
            result = dict(
                checks=checks,
                bytes=SIZES,
                torch=torch.__version__,
                library=library,
                finite_bits_exact=True,
                sum2_interleaved=True,
                nan_payload_identity_required=False,
                model_quality_test=False,
            )
            if library:
                result["library_sha256"] = hashlib.sha256(
                    Path(library).read_bytes()
                ).hexdigest()
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if old is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = old
        envs.disable_envs_cache()
        ca.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
