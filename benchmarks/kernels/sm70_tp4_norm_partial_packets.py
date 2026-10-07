# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remove the row-leader inverse handoff from the forty-CTA norm screen.

Each CTA publishes its variance and generation together in an aligned 64-bit
packet, then computes an ordered sum locally. Five parts preserve the original
arithmetic; ten/twenty parts change grouping. No extra peer/grid barrier is used.
"""

from benchmark_sm70_tp4_pull_norm import generate as baseline_source


def generate(root, parts=5):
    assert parts in (5, 10, 20)
    text = baseline_source(root)
    native = (root / "csrc/custom_all_reduce.cuh").read_text()
    begin = native.index("template <typename WeightT, bool Reference = false>")
    end = native.index("\nclass CustomAllreduce", begin)
    kernel = native[begin:end].replace(
        "sm70_push_allreduce_gemma_rms_norm", "partial_packet_norm"
    )
    begin = kernel.index("  using Meta =")
    end = kernel.index("  const uint32_t generation", begin)
    kernel = (
        kernel[:begin]
        + "  auto* meta = reinterpret_cast<volatile PartialPacketMeta*>(\n"
        "      const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[rank]))\n"
        "      + kSm70Tp4PushAllreduceBufferBytes + sizeof(PullSignals));\n"
        + kernel[end:]
    )
    kernel = kernel.replace(
        "const uint32_t generation = meta->generation[row] + 1;",
        "auto* generation_meta = reinterpret_cast<volatile Sm70PushNormMeta*>(local);\n"
        "  const uint32_t generation = generation_meta->generation[row] + 1;",
    )
    if parts != 5:
        threads = 640 // parts
        kernel = kernel.replace(
            "__launch_bounds__(128, 1)", f"__launch_bounds__({threads}, 1)"
        )
        kernel = kernel.replace(
            "constexpr int Threads = 128, Parts = kSm70PushNormParts;",
            f"constexpr int Threads = {threads}, Parts = {parts};",
        )
    begin = kernel.index("  if constexpr (Reference) {")
    end = kernel.index("  __syncthreads();\n  if (tid < PacksPerPart)", begin)
    kernel = (
        kernel[:begin]
        + r"""
  if (tid == 0) {
    const uint64_t packet=(uint64_t(generation)<<32)|__float_as_uint(variance);
    asm volatile("st.volatile.global.u64 [%0], %1;" ::
                 "l"(&meta->partial[row][part]), "l"(packet) : "memory");
  }
  __syncwarp();
  if (tid < 32) {
    float own_partial=0.f;
    if (tid < Parts) {
      uint64_t packet;
      do {
        asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(packet) :
                     "l"(&meta->partial[row][tid]) : "memory");
      } while (uint32_t(packet>>32)!=generation);
      own_partial=__uint_as_float(uint32_t(packet));
    }
    __syncwarp();
    float total=0.f;
#pragma unroll
    for (int p=0;p<Parts;++p)
      total+=__shfl_sync(0xffffffff,own_partial,p);
    if (tid==0) {
      inverse=rsqrtf(total/Width+epsilon);
      // Every producer has read this generation before publishing a packet.
      if (part==0) generation_meta->generation[row]=generation;
    }
  }
"""
        + kernel[end:]
    )
    packet_meta = r"""
struct alignas(8) PartialPacketMeta {
  uint32_t reserved[8];
  uint64_t partial[8][5];
};
static_assert(sizeof(PartialPacketMeta)==352,"Aligned packet metadata");
"""
    marker = "size_t buffer_bytes() {"
    assert text.count(marker) == 1
    if parts != 5:
        packet_meta = packet_meta.replace("partial[8][5]", f"partial[8][{parts}]")
        packet_meta = packet_meta.replace("==352", f"=={32 + 64 * parts}")
    text = text.replace(marker, packet_meta + kernel + "\n" + marker)
    text = text.replace(
        "return kSm70Tp4PushAllreduceBufferBytes + sizeof(PullSignals);",
        "return kSm70Tp4PushAllreduceBufferBytes + sizeof(PullSignals) +\n"
        "      sizeof(PartialPacketMeta);",
    )
    text = text.replace(
        "if (mode == 1) direct_pull_norm<float><<<40,128,0,stream>>>(\n"
        "      peers(buffers),peers(inputs),x,r,w,y,ro,rank,1e-6f);",
        f"if (mode == 1) partial_packet_norm<float><<<{8 * parts},{640 // parts},"
        "0,stream>>>(\n"
        "      peers(buffers),x,r,w,y,ro,rank,1e-6f);",
    )
    return text
