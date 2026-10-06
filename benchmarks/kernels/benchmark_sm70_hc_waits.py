# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only TP4 HC attribution; the diagnostic DSO is not a model route."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.distributed.device_communicators.sm70_hc_ll import Sm70HcLLCommunicator
from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import _pack_hc_batch_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=8)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    source = Path(__file__).parents[1] / "csrc/sm70_hc_ll_diagnostic.cu"
    ext = load(
        name="hc_ll_diagnostic",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-arch=sm_70", "--ptxas-options=-v"],
    )
    cfg = SimpleNamespace(kernel_config=KernelConfig(hc_ll_shard=True))
    with set_current_vllm_config(cfg):
        owner = Sm70HcLLCommunicator(
            dist.group.WORLD, torch.device("cuda", rank), "hc_wait_attribution"
        )
    assert owner.status["enabled"], owner.status
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    packed = []
    for record in weights[: args.pairs]:
        down = torch.cat(
            (record["down"].float(), record["inj"].float(), torch.zeros(12, 10240))
        ).half()
        packed.append(
            (
                _pack_hc_batch_weight(down, "down", owner.logical_rank).cuda(),
                _pack_hc_batch_weight(
                    record["up"].half(), "up", owner.logical_rank
                ).cuda(),
            )
        )
    empty = torch.empty(0, dtype=torch.int64, device="cuda")
    rows = []
    for m in (5, 20):
        torch.manual_seed(20261006)
        x = (torch.randn(m, 10240, device="cuda") * 0.5).half()
        out = torch.empty(m, 2560, dtype=torch.float16, device="cuda")
        lora = torch.empty(m, 320, dtype=torch.float16, device="cuda")
        inj = torch.empty(m, 4, dtype=torch.float16, device="cuda")
        debug_down = torch.zeros(240, 4, dtype=torch.int64, device="cuda")
        debug_up = torch.zeros(240, 4, dtype=torch.int64, device="cuda")

        def down(w, compute=False, dbg=empty, x=x):
            ext.down(
                x,
                w[0],
                owner.partial,
                owner.down_counter,
                owner.down_pointers,
                owner.down_seq,
                owner.logical_rank,
                1,
                compute,
                dbg,
            )

        def up(w, compute=False, dbg=empty, x=x, out=out, lora=lora, inj=inj):
            ext.up(
                owner.down_pointers[owner.logical_rank],
                w[1],
                x,
                owner.up_counter,
                owner.up_pointers,
                owner.up_seq,
                owner.down_seq,
                owner.logical_rank,
                out,
                lora,
                inj,
                5,
                compute,
                dbg,
            )

        def chain(w):
            down(w)
            up(w)

        expected = owner.apply(x, *packed[0])
        torch.cuda.synchronize()
        dist.barrier()
        chain(packed[0])
        torch.cuda.synchronize()
        assert expected is not None
        torch.testing.assert_close(out, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(inj, expected[1], rtol=0, atol=0)

        timings = {}
        for name, fn in (
            ("production", lambda w, x=x: owner.apply(x, *w)),
            ("diagnostic_full", chain),
            ("down_full", down),
            ("down_compute", lambda w: down(w, True)),
            ("up_full", up),
            ("up_compute", lambda w: up(w, True)),
        ):
            for w in packed:
                chain(w)  # Publish valid LL operands before compute-only probes.
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(8):
                    for w in packed:
                        fn(w)
            for _ in range(4):
                g.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(5):
                dist.barrier()
                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(40):
                    g.replay()
                b.record()
                b.synchronize()
                samples.append(a.elapsed_time(b) * 1000 / 40 / 8 / len(packed))
            timings[name] = samples
            del g
        down(packed[0], dbg=debug_down)
        up(packed[0], dbg=debug_up)
        torch.cuda.synchronize()
        rows.append(
            dict(
                m=m,
                samples_us=timings,
                phase_cycles=dict(
                    down=debug_down.cpu().tolist(), up=debug_up.cpu().tolist()
                ),
            )
        )
        if rank == 0:
            print(
                m,
                {k: round(statistics.median(v), 3) for k, v in timings.items()},
                flush=True,
            )
    records = [None] * 4
    dist.all_gather_object(records, rows)
    if rank == 0:
        args.output.write_text(
            json.dumps(
                dict(
                    scope="TP4 operator attribution, not end-to-end model latency",
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    weights=str(args.weights),
                    pairs=len(packed),
                    capability=owner.status,
                    rank_records=records,
                ),
                indent=2,
            )
            + "\n"
        )
    owner.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
