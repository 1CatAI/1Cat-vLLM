# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen CTA order in the already fused qkvz/b/a launch, unchanged math.

The control appends 96 short b/a CTAs after 128 longer FP8 CTAs. Two candidate
orders move b/a first or distribute four FP8 and three b/a CTAs per group.
Only an independent extension is built; no serving dispatch is installed.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_effective_scale import extract_kernel
from benchmark_sm70_qpn8_two_phase import generate as qpn_source
from torch.utils.cpp_extension import load


def generate(root):
    source = root / "csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu"
    text = qpn_source(source)
    text = text[: text.index("PYBIND11_MODULE(")]
    original = extract_kernel(source.read_text(), "fp8_qpn8_sm70_kernel")
    for name, expression in (
        ("ba_first", "blockIdx.x < 96 ? 128 + blockIdx.x : blockIdx.x - 96"),
        (
            "ba_interleaved",
            (
                "blockIdx.x % 7 < 4 ? (blockIdx.x / 7) * 4 + blockIdx.x % 7 : "
                "128 + (blockIdx.x / 7) * 3 + blockIdx.x % 7 - 4"
            ),
        ),
    ):
        candidate = original.replace("fp8_qpn8_sm70_kernel", name)
        old = "  const int tile = blockIdx.x;"
        assert candidate.count(old) == 1
        text += candidate.replace(old, "  const int tile = " + expression + ";")
    return (
        text
        + r"""
void launch_ordered(torch::Tensor q, torch::Tensor z, torch::Tensor b, torch::Tensor a,
    torch::Tensor x, torch::Tensor codes, torch::Tensor scales, torch::Tensor ba,
    int mode) {
  TORCH_CHECK(x.sizes()==torch::IntArrayRef({8,5120}) && x.is_contiguous());
  TORCH_CHECK(q.sizes()==torch::IntArrayRef({8,2560}) && q.is_contiguous());
  TORCH_CHECK(z.sizes()==torch::IntArrayRef({8,1536}) && z.is_contiguous());
  TORCH_CHECK(ba.sizes()==torch::IntArrayRef({24,5120}) && ba.is_contiguous());
  auto kernel = mode==1 ? ba_first<16,2,true,false,false,true,true>
                       : ba_interleaved<16,2,true,false,false,true,true>;
  TORCH_CHECK(mode==1 || mode==2);
  kernel<<<224,512,0,at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<half*>(z.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(ba.data_ptr<at::Half>()), nullptr,
      reinterpret_cast<half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<half*>(a.data_ptr<at::Half>()),24,2560,4096,5120,8,true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("launch",&launch_ordered); }
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "ba_order.cu"
    text = generate(args.source_root)
    if not source.exists() or source.read_text() != text:
        source.write_text(text)
    load(
        name="qpn8_ba_order_screen",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
