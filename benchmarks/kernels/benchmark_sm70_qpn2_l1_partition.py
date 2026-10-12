# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen shared-memory/L1 partition with identical projection arithmetic.

Keep grids, tile sizes, weights, decoder and accumulation unchanged. Compare
CUDA's automatic carveout against 25% and 100% shared-memory preference to
isolate L1 activation reuse; this does not register a serving route.
"""

import argparse
import hashlib
import json
import runpy
from functools import partial
from pathlib import Path

import torch
from benchmark_sm70_qpn2_effective_scale import extract_kernel, generate, graph_pair
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def source_text(source):
    text = generate(source)
    calls = []
    for gated in (False, True):
        name = "nvfp4_qpn2_" + ("gated_" if gated else "") + "sm70_kernel"
        for mode in (1, 2):
            old = extract_kernel(text, name + f"_scale{mode}")
            new = extract_kernel(text, name + "_scale0")
            new = new.replace(name + "_scale0", name + f"_scale{mode}")
            text = text.replace(old, new)
        template = "8,1,1,false,true" if gated else "16,2,1,false,false,true"
        for mode, carveout in ((0, None), (1, 25), (2, 100)):
            function = name + f"_scale{mode}<{template}>"
            if carveout is not None:
                calls.append(
                    f"C10_CUDA_CHECK(cudaFuncSetAttribute({function}, "
                    f"cudaFuncAttributePreferredSharedMemoryCarveout, {carveout}));"
                )
            calls.append(f"C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, {function}));")
            calls.append(
                "values.push_back(attr.numRegs); "
                "values.push_back(attr.sharedSizeBytes); "
                "values.push_back(attr.preferredShmemCarveout);"
            )
    text = text.replace(
        ", reinterpret_cast<const half*>(table.data_ptr<at::Half>())", ""
    )
    code = "std::vector<int> configure() { cudaFuncAttributes attr; "
    code += "std::vector<int> values;\n" + "\n".join(calls) + "\nreturn values; }\n"
    text = text.replace("PYBIND11_MODULE(", code + "PYBIND11_MODULE(")
    return text.replace(
        'm.def("prepare", &prepare);', 'm.def("configure", &configure);'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source_root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    generated = args.out / "l1_partition.cu"
    text = source_text(source)
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    extension = load(
        name="qpn2_l1_partition_screen",
        sources=[str(generated)],
        extra_include_paths=[str(source.parent)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )
    if args.compile_only:
        return
    helpers = runpy.run_path(
        str(Path(__file__).with_name("benchmark_sm70_nvfp4_qpn2.py"))
    )
    torch.manual_seed(123)
    torch.set_grad_enabled(False)
    empty = torch.empty(0, device="cuda", dtype=torch.float16)
    attributes = extension.configure()
    operands = []
    for p in helpers["_load_projection_shards"](args.model, 0, 0, 4):
        codes, scales = ops.nvfp4_qpn2_prepare_sm70(p.packed.cuda(), p.scales.cuda())
        bundle = torch.cat((codes.view(-1, 256), scales.view(-1, 32)), 1).contiguous()
        operands.append((p, codes, scales, bundle))
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
    original = x.clone()
    middle = [x.new_empty(8, 4352) for _ in range(3)]
    output = [torch.empty_like(x) for _ in range(3)]

    def projection(index, mode):
        p, _, scales, bundle = operands[index]
        extension.launch(
            middle[mode] if index == 0 else output[mode],
            x if index == 0 else middle[mode],
            bundle,
            scales,
            empty,
            p.inverse_global_scale,
            index == 0,
            mode,
        )

    def mlp(mode):
        projection(0, mode)
        projection(1, mode)

    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.copy_(original * amplitude)
        mlp(0)
        mlp(1)
        mlp(2)
        assert all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in (
                (middle[0], middle[1]),
                (output[0], output[1]),
                (middle[0], middle[2]),
                (output[0], output[2]),
            )
        )
        p, codes, scales, _ = operands[0]
        ref = torch.empty_like(middle[0])
        ops.nvfp4_qpn2_gated_sm70_out(
            ref, x, codes, scales, p.inverse_global_scale, 8, 1
        )
        assert torch.equal(ref.view(torch.int16), middle[0].view(torch.int16))
    x.copy_(original * 0.125)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    results = []
    for name, call in (
        ("gate", partial(projection, 0)),
        ("down", partial(projection, 1)),
        ("mlp", mlp),
    ):
        for mode in (1, 2):
            result = graph_pair(
                partial(call, 0), partial(call, mode), eviction, args.iters
            )
            results.append(
                {"name": name, "carveout_percent": 25 if mode == 1 else 100, **result}
            )
    result = {
        "scope": "Real layer0 TP4 shard, M8 FP16, cold L2 graph",
        "bitwise": True,
        "cuda_function_attributes": attributes,
        "results": results,
        "cuda_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "samples_us"} for r in results]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
