# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen atomic generation/data triples instead of sentinel payload resets.

Each aligned uint64 carries three FP16 values and a nonzero 16-bit generation.
Two epochs retain the existing reuse protection. This increases peer payload
by 50%, but removes sentinel polling and clearing. Keep five norm parts to
isolate the protocol with the original arithmetic; measure before admission.
"""

from sm70_tp4_norm_partial_packets import generate as packet_source


def generate(root):
    text = packet_source(root, 5)
    begin = text.index(
        "  if (tid < PacksPerPart) {", text.index("void partial_packet_norm")
    )
    end = text.index("    const P sum =", begin)
    text = (
        text[:begin]
        + r"""
  if (tid < PacksPerPart) {
    P value = reinterpret_cast<const P*>(input)[pack];
#pragma unroll
    for (int i=0;i<P::size;++i) vllm::sm70_push_escape_sentinel(value.data[i]);
    const uint64_t tag=uint64_t(generation % 65535U + 1U)<<48;
    const uint64_t words[3] = {
      tag | uint64_t(__half_as_ushort(value.data[0])) |
        (uint64_t(__half_as_ushort(value.data[1]))<<16) |
        (uint64_t(__half_as_ushort(value.data[2]))<<32),
      tag | uint64_t(__half_as_ushort(value.data[3])) |
        (uint64_t(__half_as_ushort(value.data[4]))<<16) |
        (uint64_t(__half_as_ushort(value.data[5]))<<32),
      tag | uint64_t(__half_as_ushort(value.data[6])) |
        (uint64_t(__half_as_ushort(value.data[7]))<<16)};
    constexpr int Packs=Elements/P::size;
    constexpr size_t WireOffset=kSm70Tp4PushAllreduceBufferBytes+
        sizeof(PullSignals)+sizeof(PartialPacketMeta);
    const size_t own_start=((generation&1)*4+rank)*Packs;
#pragma unroll
    for (int peer=0;peer<4;++peer) {
      auto* dst=reinterpret_cast<uint64_t*>(
          const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[peer]))
              + WireOffset);
#pragma unroll
      for (int word=0;word<3;++word)
        asm volatile("st.volatile.global.u64 [%0], %1;" ::
          "l"(dst+(own_start+pack)*3+word), "l"(words[word]):"memory");
    }
    const auto* src=reinterpret_cast<const uint64_t*>(
        reinterpret_cast<const char*>(buffers.ptrs[rank])+WireOffset);
    P peers[4];
#pragma unroll
    for (int peer=0;peer<4;++peer) {
      const size_t start=(((generation&1)*4+peer)*Packs+pack)*3;
      uint64_t w0,w1,w2;
      do {
        asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(w0) :
          "l"(src+start):"memory");
        asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(w1) :
          "l"(src+start+1):"memory");
        asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(w2) :
          "l"(src+start+2):"memory");
      } while ((w0&0xffff000000000000ULL)!=tag ||
               (w1&0xffff000000000000ULL)!=tag ||
               (w2&0xffff000000000000ULL)!=tag);
      peers[peer].data[0]=__ushort_as_half(uint16_t(w0));
      peers[peer].data[1]=__ushort_as_half(uint16_t(w0>>16));
      peers[peer].data[2]=__ushort_as_half(uint16_t(w0>>32));
      peers[peer].data[3]=__ushort_as_half(uint16_t(w1));
      peers[peer].data[4]=__ushort_as_half(uint16_t(w1>>16));
      peers[peer].data[5]=__ushort_as_half(uint16_t(w1>>32));
      peers[peer].data[6]=__ushort_as_half(uint16_t(w2));
      peers[peer].data[7]=__ushort_as_half(uint16_t(w2>>16));
    }
"""
        + text[end:]
    )
    begin = text.index("    P empty;", text.index("void partial_packet_norm"))
    end = text.index("\n  }\n  using Reduce", begin)
    text = text[:begin] + text[end:]
    old = "      sizeof(PartialPacketMeta);"
    assert text.count(old) == 1
    text = text.replace(old, old[:-1] + " + 2*4*(8*5120/8)*3*8;")
    return text
