# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a research qkvz/a/b epilogue with convolution, gating and core zeroing.

The delta recurrence and weight geometry are unchanged. Shared projection
partials are reduced in the original order; raw FP16 tokens feed the original
four convolution taps. Complete-layer speed and numerical gates follow build.
"""

import argparse
from pathlib import Path

from benchmark_sm70_qpn2_effective_scale import extract_kernel
from benchmark_sm70_qpn8_two_phase import generate as qpn_source
from torch.utils.cpp_extension import load


def generate(root):
    source = root / "csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu"
    body = qpn_source(source)
    body = body[: body.index("PYBIND11_MODULE(")]
    fused = extract_kernel(source.read_text(), "fp8_qpn8_sm70_kernel")
    fused = fused.replace("fp8_qpn8_sm70_kernel", "projection_conv")
    fused = fused.replace(
        "bool channel_scales) {",
        """bool channel_scales, const half* conv, half* history,
    const int* conv_index, const int* accepted, const int* cu,
    const float* a_log, const half* bias, float* g, float* beta, half* core,
    int state_seq, int state_dim, int state_token) {
  __shared__ half raw_tokens[256];""",
    )
    bstore = """            b_output[static_cast<size_t>(token) * (ba_n / 2) + ba_row] =
                  __float2half(value);"""
    assert fused.count(bstore) == 1
    fused = fused.replace(
        bstore,
        bstore
        + """
              const float rounded = __half2float(__float2half(value));
              beta[token * 12 + ba_row] =
                  __fdividef(1.f, 1.f + __expf(-rounded));""",
    )
    astore = (
        "              a_output[static_cast<size_t>(token) * (ba_n / 2) + ba_row -\n"
        "                       ba_n / 2] = __float2half(value);"
    )
    assert fused.count(astore) == 1
    fused = fused.replace(
        astore,
        astore
        + """
              const int head = ba_row - ba_n / 2;
              const float x = __half2float(__float2half(value)) +
                              __half2float(bias[head]);
              const float soft = x <= 20.f ? __logf(1.f + __expf(x)) : x;
              g[token * 12 + head] = -__expf(a_log[head]) * soft;""",
    )
    marker = "  const int quadpair ="
    fused = fused.replace(
        marker,
        """
  const int history_slot = conv_index[0];
  const int valid_tokens = cu[1] - cu[0];
"""
        + marker,
    )
    store = """            output[static_cast<size_t>(output_row) * qkv_n + col] =
                __float2half(value);"""
    assert fused.count(store) == 1
    fused = fused.replace(
        store,
        """
            raw_tokens[element] = __float2half(value);
            if (history_slot < 0 || output_row >= valid_tokens)
              output[static_cast<size_t>(output_row) * qkv_n + col] =
                  __float2half(value);
            if (col < 1536)
              core[static_cast<size_t>(output_row) * 1536 + col] =
                  __float2half(0.f);
""",
    )
    end = fused.rfind("\n}")
    fused = (
        fused[:end]
        + r"""
  if (tile < qkv_n / 32) {
    __syncthreads();
    if (threadIdx.x < 32 && history_slot >= 0 && valid_tokens > 0) {
      const int col = tile * 32 + threadIdx.x;
      half* hist = history + static_cast<size_t>(history_slot) * state_seq +
                   col * state_dim;
      const int offset = accepted[0] - 1;
      half previous[3] = {hist[offset*state_token],
                          hist[(offset+1)*state_token],
                          hist[(offset+2)*state_token]};
      const half weights[4] = {conv[col*4],conv[col*4+1],
                               conv[col*4+2],conv[col*4+3]};
      hist[0] = previous[1]; hist[state_token] = previous[2];
      for (int t = 0; t < valid_tokens; ++t)
        hist[(t+2)*state_token] = raw_tokens[t*32 + threadIdx.x];
      for (int t = 0; t < valid_tokens; ++t) {
        const half current = raw_tokens[t*32 + threadIdx.x];
        float sum = 0.f;
        sum = __fadd_rn(sum,__half2float(__hmul(previous[0],weights[0])));
        sum = __fadd_rn(sum,__half2float(__hmul(previous[1],weights[1])));
        sum = __fadd_rn(sum,__half2float(__hmul(previous[2],weights[2])));
        sum = __fadd_rn(sum,__half2float(__hmul(current,weights[3])));
        previous[0]=previous[1]; previous[1]=previous[2]; previous[2]=current;
        const float silu = __fdividef(sum,1.f + __expf(-sum));
        output[static_cast<size_t>(t)*qkv_n + col] = __float2half_rn(silu);
      }
    }
  }
"""
        + fused[end:]
    )
    return (
        body
        + fused
        + r"""
void launch_gdn(torch::Tensor q, torch::Tensor z, torch::Tensor b, torch::Tensor a,
    torch::Tensor g, torch::Tensor beta, torch::Tensor core,
    torch::Tensor x, torch::Tensor codes, torch::Tensor scales, torch::Tensor ba,
    torch::Tensor conv, torch::Tensor history, torch::Tensor conv_index,
    torch::Tensor accepted, torch::Tensor cu, torch::Tensor a_log,
    torch::Tensor bias) {
  TORCH_CHECK(x.sizes()==torch::IntArrayRef({8,5120}) && x.is_contiguous());
  TORCH_CHECK(q.sizes()==torch::IntArrayRef({8,2560}) && q.is_contiguous());
  TORCH_CHECK(history.dim()==3 && history.size(1)==2560 &&
              history.size(2)>=10 && conv.is_contiguous());
  TORCH_CHECK(ba.sizes()==torch::IntArrayRef({24,5120}) && ba.is_contiguous());
  auto stream=at::cuda::getCurrentCUDAStream();
  projection_conv<16,2,true,false,false,true,true><<<224,512,0,stream>>>(
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<half*>(z.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(ba.data_ptr<at::Half>()), nullptr,
      reinterpret_cast<half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<half*>(a.data_ptr<at::Half>()),24,2560,4096,5120,8,true,
      reinterpret_cast<const half*>(conv.data_ptr<at::Half>()),
      reinterpret_cast<half*>(history.data_ptr<at::Half>()),
      conv_index.data_ptr<int>(),accepted.data_ptr<int>(),cu.data_ptr<int>(),
      a_log.data_ptr<float>(),
      reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
      g.data_ptr<float>(),beta.data_ptr<float>(),
      reinterpret_cast<half*>(core.data_ptr<at::Half>()),
      history.stride(0),history.stride(1),history.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("launch",&launch_gdn); }
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "projection_conv.cu"
    text = generate(args.source_root)
    if not source.exists() or source.read_text() != text:
        source.write_text(text)
    load(
        name="gdn_projection_conv_screen",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
