# IQ3_XXS reader for shared-activation pairs

The next seven IQ3_S/IQ3_XXS gate/up pairs reuse the two-reader M8 kernel
from the IQ3_S/IQ4_XS path and the source-sized records described in
`gguf_iq3_xxs_pair_records.md`. The IQ3_XXS reader loads three aligned
16-byte packets per K128 record, caches original d across K256 halves,
and advances pointers without recalculating block addresses in the loop.
It retains the original seven sign bits and four-bit small scale per K32.

Operand formation calls `LatticeRawDecoder<18>::table_fragment<half>`.
The existing original-block decoder owns the FP16 operand proof and scale
formula. This reader adds no quantization formula, scale expansion or
rounding policy. Dot products and split-K reduction retain FP32 arithmetic
in the existing shared-activation skeleton.

The raw operator builds both IQ3_XXS/IQ3_S orientations alongside the two
existing IQ3_S/IQ4_XS orientations. Byte checks distinguish 98/110/136-byte
blocks. Host launch code is shared; the existing two kernel instantiations
retain their implementation. Model capability declarations are unchanged:
IQ3_XXS pairs remain canonical pending actual-weight GPU checks and a
matched cold-L2 graph comparison.

Ten CPU record/cursor checks pass. The new cursor oracle covers all eight
split-K starts for K5120, including odd starts, two N32 tiles, all columns
and all sixteen octets per K128 record. Reconstructed index bytes, sign/scale
words and cached original d match the source. An initial oracle failure was
caused by using big-endian integer parsing in the test; specifying the
original little-endian byte order fixes the oracle without changing the
reader. GPU numerical, graph, resource and speed evidence remains pending.
