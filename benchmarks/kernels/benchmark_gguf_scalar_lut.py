# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-cache ABBA of retained and candidate IQ expert gate/up decoders.

Run under the GPU ownership locks, with an installed source-complete wheel.
CUDA events are graph nodes surrounding one gate/up launch. A 64-MiB cache
scrub precedes every replay and lies outside the measured interval.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pynvml
import torch
import vllm._C as core

import vllm
from vllm.model_executor.layers.quantization.gguf_lattice_lut import pack_lattice_lut
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.platforms import current_platform
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, quant_size


def run(which, output, activation, ids, weights, kind, decoder):
    if which == 0:
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            output, activation, ids, *weights, kind, True
        )
    elif decoder == "bank-aware":
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            output, activation, ids, *weights, kind, True, 16, True
        )
    else:
        torch.ops._C.gguf_dp4a_scalar_lut_gate_up_sm70_out(
            output, activation, ids, *weights, kind, True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument(
        "--decoder", choices=["scalar-lut", "bank-aware"], default="scalar-lut"
    )
    parser.add_argument("--rank", type=int, default=0, choices=range(4))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.accelerator.set_device_index(args.device)
    if current_platform.get_device_capability(args.device) != (7, 0):
        raise RuntimeError("SM70 required")
    if "site-packages" not in vllm.__file__:
        raise RuntimeError("An installed wheel is required")
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    selected = {}
    readers = [GGUFReader(path) for path in args.models]
    for reader in readers:
        tensors = {t.name: t for t in reader.tensors}
        for tensor in reader.tensors:
            kind = int(tensor.tensor_type)
            if (
                kind in (18, 21, 22)
                and kind not in selected
                and ".ffn_gate_exps." in tensor.name
            ):
                other = tensors[tensor.name.replace(".ffn_gate_exps.", ".ffn_up_exps.")]
                if int(other.tensor_type) == kind:
                    selected[kind] = (tensor, other)
    if selected.keys() != {18, 21, 22}:
        raise RuntimeError(f"Missing representative IQ tensors: {selected.keys()}")
    report = dict(
        wheel_version=vllm.__version__,
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        rank=args.rank,
        candidate_decoder=args.decoder,
        graph_kernel_count=1,
        cache_scrub_bytes=64 * 1024 * 1024,
        bandwidth_scope=(
            "Logical routed and unique expert payload; not measured DRAM counters"
        ),
        rows=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    eviction = torch.zeros(16 * 1024 * 1024, device="cuda", dtype=torch.float32)
    rng = np.random.default_rng(20261007)
    for kind, tensors in sorted(selected.items()):
        original, normalized, shas = [], [], []
        for tensor in tensors:
            e, full_n, _ = tensor.data.shape
            k = int(tensor.shape[0])
            if full_n % 4:
                raise RuntimeError("Invalid TP4 output shard")
            n = full_n // 4
            rows = np.ascontiguousarray(
                tensor.data[:, args.rank * n : (args.rank + 1) * n]
            ).reshape(e * n, -1)
            shas.append(hashlib.sha256(rows.tobytes()).hexdigest())
            raw = RawGGUFProjection.from_rows(rows, kind)
            packed = (
                pack_lattice_lut(rows, kind)
                if args.decoder == "scalar-lut"
                else raw.data
            )
            original.append(torch.from_numpy(raw.data.reshape(e, n, -1)).cuda())
            normalized.append(
                torch.from_numpy(packed.reshape(e, n, -1)).cuda()
                if args.decoder == "scalar-lut"
                else original[-1]
            )
            del rows, raw, packed
        for m in (1, 5, 20):
            top_k = 10
            routes = rng.choice(e, (m, top_k), replace=m > 5)
            if m == 5:
                routes.flat[-3:] = routes.flat[:3]
            ids = torch.from_numpy(routes.astype(np.int32)).cuda()
            x = (torch.randn(m, k, device="cuda") * 0.5).half()
            q8 = torch.empty(m, k // 32, 36, device="cuda", dtype=torch.uint8)
            torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
            outputs = [
                torch.empty(m, top_k, n // 32, 36, device="cuda", dtype=torch.uint8)
                for _ in range(2)
            ]

            for which in (0, 1):
                run(
                    which,
                    outputs[which],
                    q8,
                    ids,
                    original if which == 0 else normalized,
                    kind,
                    args.decoder,
                )
            torch.accelerator.synchronize()
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            graphs, events = [], []
            for which in (0, 1):
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    start.record()
                    run(
                        which,
                        outputs[which],
                        q8,
                        ids,
                        original if which == 0 else normalized,
                        kind,
                        args.decoder,
                    )
                    end.record()
                graphs.append(graph)
                events.append((start, end))
            timings = [[], []]
            clocks = []
            for cycle in range(args.cycles + 5):
                for which in (0, 1, 1, 0):
                    eviction.add_(1)
                    graphs[which].replay()
                    torch.accelerator.synchronize()
                    if cycle >= 5:
                        timings[which].append(
                            events[which][0].elapsed_time(events[which][1]) * 1000
                        )
                if cycle >= 5:
                    clocks.append(
                        pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    )
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            _, block_bytes = quant_size(kind)
            unique = len(np.unique(routes))
            bytes_per_expert = [
                2 * n * (k // 256) * block_bytes,
                2 * n * (k // 32) * 20
                if args.decoder == "scalar-lut"
                else 2 * n * (k // 256) * block_bytes,
            ]
            medians = [float(np.median(t)) for t in timings]
            row = dict(
                kind=kind,
                tensors=[t.name for t in tensors],
                shard_sha256=shas,
                M=m,
                N=n,
                K=k,
                experts=e,
                unique_experts=unique,
                output_bytes_equal=True,
                clocks_mhz=sorted(set(clocks)),
                retained_us=medians[0],
                candidate_us=medians[1],
                saving_us=medians[0] - medians[1],
                unique_payload_GBs=[
                    unique * b / t / 1000 for b, t in zip(bytes_per_expert, medians)
                ],
                routed_payload_GBs=[
                    m * top_k * b / t / 1000 for b, t in zip(bytes_per_expert, medians)
                ],
                samples_us=timings,
            )
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                flush=True,
            )
            del graphs, events, outputs, q8, x, ids
        del original, normalized
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
