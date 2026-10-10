# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen per-channel scaling after FP32 accumulation, not per FP16 weight.

Per-channel scale is constant over K. Moving it to the epilogue removes eight
half2 multiplications per K16 panel without reducing weight precision or
adding a weight copy. FP16 dequantization rounding changes: teacher-forcing
admission is required if the complete-layer speed gate passes.
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
    candidate = extract_kernel(source.read_text(), "fp8_qpn8_sm70_kernel")
    candidate = candidate.replace("fp8_qpn8_sm70_kernel", "epilogue_scale_qpn8")
    begin = candidate.index("    const half2 scale2 =")
    end = candidate.index("    const unsigned* b =", begin)
    candidate = candidate[:begin] + candidate[end:]
    old = "        const int col = tile * 32 + output_col;"
    assert candidate.count(old) == 1
    candidate = candidate.replace(
        old,
        old + "\n        value = __fmul_rn(value, __half2float(group_scales[col]));",
    )
    return (
        text
        + candidate
        + r"""
void launch_qkvz(torch::Tensor q, torch::Tensor z, torch::Tensor b, torch::Tensor a,
    torch::Tensor x, torch::Tensor codes, torch::Tensor scales, torch::Tensor ba) {
  TORCH_CHECK(x.sizes()==torch::IntArrayRef({8,5120}) && x.is_contiguous());
  TORCH_CHECK(q.sizes()==torch::IntArrayRef({8,2560}) && q.is_contiguous());
  TORCH_CHECK(z.sizes()==torch::IntArrayRef({8,1536}) && z.is_contiguous());
  TORCH_CHECK(ba.sizes()==torch::IntArrayRef({24,5120}) && ba.is_contiguous());
  TORCH_CHECK(scales.numel()==4096);
  epilogue_scale_qpn8<16,2,true,false,false,true,true>
      <<<224,512,0,at::cuda::getCurrentCUDAStream()>>>(
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
void launch_out(torch::Tensor y, torch::Tensor x, torch::Tensor codes,
    torch::Tensor scales) {
  TORCH_CHECK(x.sizes()==torch::IntArrayRef({8,1536}) && x.is_contiguous());
  TORCH_CHECK(y.sizes()==torch::IntArrayRef({8,5120}) && y.is_contiguous());
  TORCH_CHECK(scales.numel()==5120);
  epilogue_scale_qpn8<12,2,true,false>
      <<<160,384,0,at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()), nullptr, nullptr, nullptr,
      nullptr, nullptr,0,0,5120,1536,8,true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("qkvz",&launch_qkvz); m.def("out",&launch_out);
}
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "epilogue_scale.cu"
    text = generate(args.source_root)
    if not source.exists() or source.read_text() != text:
        source.write_text(text)
    load(
        name="qpn8_epilogue_scale_screen",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
