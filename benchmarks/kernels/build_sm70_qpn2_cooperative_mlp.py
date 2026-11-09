# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Literal CUDA templates retain the validated generated-source spelling.
# ruff: noqa: E501
"""Private cooperative gate/up -> row publication, with the original consumer."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--build", action="store_true")
    args = p.parse_args()
    W = Path(__file__).resolve().parents[2]
    D = args.output_dir.resolve()
    subprocess.run(
        [
            sys.executable,
            str(W / "benchmarks/kernels/build_sm70_qpn2_publish_candidate.py"),
            "--output-dir",
            str(D),
        ],
        check=True,
    )
    path = D / "sources/qpn2-publish.cu"
    s = path.read_text()
    parent_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    def function(name):
        start = s.index(
            "template <int SplitK, int NAcc, int RowTiles = 1>\n__global__ void " + name
        )
        opening = s.index("{", start)
        level = 1
        i = opening + 1
        while level:
            if s[i] == "{":
                level += 1
            elif s[i] == "}":
                level -= 1
            i += 1
        return s[start:i]

    gate = function("nvfp4_qpn2_gated_sm70_kernel")
    publish = function("nvfp4_qpn2_publish_sm70_kernel")
    gate = gate.replace(
        "__global__ void nvfp4_qpn2_gated_sm70_kernel",
        "__device__ __forceinline__ void coop_mlp_gate_device",
    )
    publish = publish.replace(
        "__global__ void nvfp4_qpn2_publish_sm70_kernel",
        "__device__ __forceinline__ void coop_mlp_publish_device",
    )
    # Fixed q8 only. All accumulators, K partitions, activation boundaries and
    # packet payloads are taken from the exact existing source.
    for old, new in [
        (
            "const int row_base = blockIdx.y * kQpn2RowsPerCta * RowTiles;",
            "const int row_base = 0;",
        )
    ]:
        assert gate.count(old) == publish.count(old) == 1
        gate = gate.replace(old, new)
        publish = publish.replace(old, new)
    s = "#include <cooperative_groups.h>\n" + s
    s += (
        "\nnamespace {\n"
        + gate
        + "\n"
        + publish
        + dedent(r"""
    __global__ __launch_bounds__(512,2) void qpn2_coop_mlp_kernel(
        const uint8_t* gate_codes, const uint8_t* gate_scales,
        const uint8_t* down_codes, const uint8_t* down_scales,
        const half* input, half* gate_output, half* output,
        float gate_global, float down_global, vllm::RankData peers,
        int rank, const uint32_t* epochs) {
      if (blockIdx.x < 136) {
        coop_mlp_gate_device<8,2,1>(gate_codes,gate_scales,input,gate_output,
                                   4352,5120,8,gate_global);
      }
      cooperative_groups::this_grid().sync();
      // Every producer finishes before the separate, frozen consumer launches.
      // There is no cross-rank wait or polling in this cooperative kernel.
      coop_mlp_publish_device<16,2,1>(down_codes,down_scales,gate_output,output,
                                     5120,4352,8,down_global,peers,rank,epochs);
    }
    }
    void qpn2_coop_mlp(torch::Tensor gate, torch::Tensor out, torch::Tensor input,
        torch::Tensor gate_codes, torch::Tensor gate_scales,
        torch::Tensor down_codes, torch::Tensor down_scales,
        double gate_global, double down_global, std::vector<int64_t> pointers, int64_t rank) {
      check_qpn2_tensors(gate,input,gate_codes,gate_scales,true);
      check_qpn2_tensors(out,gate,down_codes,down_scales,false);
      TORCH_CHECK(input.size(0)==8 && input.size(1)==5120 && gate.size(1)==4352 &&
          out.size(0)==8 && out.size(1)==5120, "exact TP4 q8 MLP required");
      const at::cuda::OptionalCUDAGuard guard(device_of(input));
      auto* props=at::cuda::getCurrentDeviceProperties();
      TORCH_CHECK(props->major==7 && props->minor==0 && props->cooperativeLaunch,
                   "SM70 cooperative launch required");
      int resident=0;
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,qpn2_coop_mlp_kernel,512,0));
      TORCH_CHECK(resident*props->multiProcessorCount>=160,"160 resident CTAs required");
      auto peers=qpn2_peer_pointers(pointers,rank);
      auto* gc=gate_codes.data_ptr<uint8_t>();auto* gs=gate_scales.data_ptr<uint8_t>();
      auto* dc=down_codes.data_ptr<uint8_t>();auto* ds=down_scales.data_ptr<uint8_t>();
      auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
      auto* g=reinterpret_cast<half*>(gate.data_ptr<at::Half>());
      auto* y=reinterpret_cast<half*>(out.data_ptr<at::Half>());
      float gg=static_cast<float>(gate_global),dg=static_cast<float>(down_global);
      int r=static_cast<int>(rank);
      auto* epochs=reinterpret_cast<const uint32_t*>(peers.ptrs[rank]);
      void* parameters[]={&gc,&gs,&dc,&ds,&x,&g,&y,&gg,&dg,&peers,&r,&epochs};
      C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(qpn2_coop_mlp_kernel),
          dim3(160),dim3(512),parameters,0,at::cuda::getCurrentCUDAStream()));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    std::vector<int64_t> qpn2_coop_resources() {
      cudaFuncAttributes attr{};int resident=0;
      C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,qpn2_coop_mlp_kernel));
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,qpn2_coop_mlp_kernel,512,0));
      return {attr.numRegs,static_cast<int64_t>(attr.sharedSizeBytes),static_cast<int64_t>(attr.localSizeBytes),resident};
    }
    TORCH_LIBRARY_FRAGMENT(_qpn2_candidate,ops) {
      ops.def("mlp(Tensor(a!) gate, Tensor(b!) out, Tensor input, Tensor gate_codes, Tensor gate_scales, Tensor down_codes, Tensor down_scales, float gate_global, float down_global, int[] pointers, int rank) -> ()");
      ops.impl("mlp",torch::kCUDA,&qpn2_coop_mlp);
      ops.def("resources() -> int[]");
      ops.impl("resources",&qpn2_coop_resources);
    }
    """)
    )
    # Avoid cross-DSO symbol preemption. The independently frozen publisher is
    # loaded separately by the benchmark, and retains its own operator namespace.
    s = s.replace("_qpn2_candidate", "_qpn2_coop_mlp")
    s = (
        s.replace("nvfp4_qpn2_", "coopbase_nvfp4_qpn2_")
        .replace("qpn2_publish", "coopbase_qpn2_publish")
        .replace("qpn2_consume", "coopbase_qpn2_consume")
        .replace("qpn2_peer_pointers", "coopbase_qpn2_peer_pointers")
    )
    path = D / "sources/qpn2-coop-mlp.cu"
    path.write_text(s)
    flags = [
        "-O3",
        "-lineinfo",
        "-gencode=arch=compute_70,code=sm_70",
        "-DVLLM_NVFP4_QPN2_STANDALONE",
        "-DVLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE",
        "-Xptxas=-v",
        "--maxrregcount=64",
    ]
    report = dict(
        source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parent_source_sha256=parent_sha,
        flags=flags,
        installed=False,
        scope="Private TP4 q8 gate/up -> published down, fixed 160 resident CTAs and one grid barrier; original arithmetic and peer packet protocol",
    )
    if args.build:
        from torch.utils.cpp_extension import load

        build = D / "build"
        build.mkdir(exist_ok=True)
        lib = Path(
            load(
                name="qpn2_coop_mlp",
                sources=[str(path)],
                build_directory=str(build),
                extra_cuda_cflags=flags,
                is_python_module=False,
                verbose=True,
            )
        )
        report.update(
            library=str(lib),
            library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
        )
    (D / "cooperative-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
