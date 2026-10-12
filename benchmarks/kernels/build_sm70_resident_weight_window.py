# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source anchors and CUDA strings retain their original spelling.
# ruff: noqa: E501
"""Screen next-gate weight reads in a producer/communication/norm window.

Prefetches start after the current FP8 projection publishes its peer payload.
They cover eight groups from each of the eight gate/up K slices (about 5 MB),
not an entire projection that exceeds Volta's L2 capacity. No activation or
sampling arithmetic changes. This is a research-only independent extension.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_product_lookup import function_end
from build_sm70_resident_projection_norm import generate as resident_source
from torch.utils.cpp_extension import load


def generate(root):
    text = resident_source(root, native_protocol=True)
    begin = text.index("void resident_out_norm(")
    end = function_end(text, begin)
    producer = text[begin:end]
    old = "float* residual_out) {"
    assert producer.count(old) == 1
    producer = producer.replace(old, "float* residual_out,const uint8_t* next_codes) {")
    old = "  if(blockIdx.x<40)\n    resident_down_norm_suffix<float>"
    assert producer.count(old) == 1
    producer = producer.replace(
        old,
        r"""
  const int line=blockIdx.x*blockDim.x+threadIdx.x;
  constexpr int PrefixGroups=8;
  if(line<272*8*PrefixGroups*3) {
    const int task=line/3;
    const int tile=task/(8*PrefixGroups);
    const int local=task%(8*PrefixGroups);
    const int group=(local/PrefixGroups)*40+local%PrefixGroups;
    const uint8_t* address=next_codes+
        (static_cast<size_t>(tile)*320+group)*288+(line%3)*128;
    asm volatile("prefetch.global.L2 [%0];" ::"l"(address):"memory");
  }
"""
        + old,
    )
    text = text[:begin] + producer + text[end:]
    begin = text.index("void fused_out(")
    end = function_end(text, begin)
    wrapper = text[begin:end]
    wrapper = wrapper.replace(
        "const std::vector<int64_t>& buffers,int rank) {",
        "const std::vector<int64_t>& buffers,int rank,torch::Tensor next_codes) {",
    )
    wrapper = wrapper.replace(
        "  void* args[]=",
        "  TORCH_CHECK(next_codes.is_contiguous() && next_codes.numel()==272*320*288);\n"
        "  const auto* next=next_codes.data_ptr<uint8_t>();\n  void* args[]=",
    ).replace(
        "&addresses,&rank,&local,&r,&w,&y,&ro};",
        "&addresses,&rank,&local,&r,&w,&y,&ro,&next};",
    )
    assert "&ro,&next}" in wrapper
    text = text[:begin] + wrapper + text[end:]
    # Keep private symbols distinct from previously imported screen DSOs.
    for name in (
        "resident_out_norm",
        "resident_down_norm",
        "resident_down_norm_suffix",
    ):
        text = text.replace(name, name + "_weight_window")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "resident_weight_window.cu"
    output.write_text(generate(args.source_root))
    load(
        name="sm70_resident_weight_window_screen",
        sources=[str(output)],
        extra_include_paths=[
            str(args.source_root / "csrc"),
            str(args.source_root / "csrc/sm70_turbomind/ops"),
        ],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
