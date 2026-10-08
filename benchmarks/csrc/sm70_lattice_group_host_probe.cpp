// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// CPU oracle probe for the exact decoder used by the SM70 persistent reader.
#include "gguf_lattice_group.cuh"

template <int Type, bool BankAware>
void decode(const uint8_t* blocks, int count, const uint32_t* book,
            const uint32_t* masks, int8_t* words, uint16_t* scales,
            uint8_t* subscales) {
  constexpr int bytes = Type == 18 ? 98 : Type == 21 ? 110 : 82;
  for (int b = 0; b < count; ++b) {
    for (int g = 0; g < 8; ++g) {
      const auto w = vllm::sm70_gguf::load_lattice_group<Type, BankAware>(
          blocks + b * bytes, g, book, masks);
      const int index = b * 8 + g;
      scales[index] = w.scale_bits;
      subscales[index * 2] = w.scale0;
      subscales[index * 2 + 1] = w.scale1;
      for (int i = 0; i < 32; ++i)
        words[index * 32 + i] =
            int8_t(uint32_t(w.words[i / 4]) >> ((i % 4) * 8));
    }
  }
}

extern "C" int lattice_groups(int type, int bank_aware, const uint8_t* blocks,
                              int count, const uint32_t* book,
                              const uint32_t* masks, int8_t* words,
                              uint16_t* scales, uint8_t* subscales) {
#define PROBE(TYPE)                                                  \
  case TYPE:                                                         \
    if (bank_aware)                                                  \
      decode<TYPE, true>(blocks, count, book, masks, words, scales,  \
                         subscales);                                 \
    else                                                             \
      decode<TYPE, false>(blocks, count, book, masks, words, scales, \
                          subscales);                                \
    return 0
  switch (type) {
    PROBE(18);
    PROBE(21);
    PROBE(22);
    default:
      return 1;
  }
#undef PROBE
}
