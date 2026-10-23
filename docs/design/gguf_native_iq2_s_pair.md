# Original IQ2_S shared-activation pair reader

The next two source-byte priorities pair IQ2_S with IQ3_S in layers5/9/17
and IQ3_XXS in layers7/18/33. All six are TP4 N4352/K5120. Both gate/up
orientations occur for each combination.

Preserve each original 82-byte K256 block as two K128 packet sets:
16-byte indices, 16-byte signs and eight bytes of high bits/scales per
column. Original d follows once per K256 block. Packet planes interleave
N32 columns with aligned 16/8-byte loads; no field expands or changes.
The reader caches original d across halves, including odd split-K starts,
and increments pointers in the K loop.

Reuse `LatticeRawDecoder<22>` codebook initialization and exact final FP16
operand formation, with the existing llama.cpp MIT attribution. No expanded
FP16 coefficient or second decode formula is introduced. The shared-A
skeleton, FP32 dot/reduction and gated epilogue are unchanged. Raw operators
compile both orientations; model capabilities remain canonical pending
actual-weight numerical and matched speed checks.

Ten CPU inverse/cursor checks pass, covering N32/64/96, complete K256 blocks,
strided inputs and all eight split-K starts at K5120. Six real IQ2_S tensors,
each sampled at N64/K5120, preserve 629,760 source bytes and 1,966,080 official
FP32 dequantized values bitwise. Native build, graph and speed checks are
pending; no end-to-end result is claimed.

The normal CUDA12.8 SM70 build and complete wheel pass. IQ2_S/IQ3_S
orientations use63 registers/41,984 shared bytes; IQ2_S/IQ3_XXS use
62–63 registers/33,792 shared bytes. All have zero stack/local memory.

Primary1290/877MHz,300W, actual TP4 M8/N4352/K5120,16MiB cold-L2
graph ABBA passes for all six real layers. IQ3_S/IQ2_S layer5 native
57.344–57.856us versus canonical91.136–92.160us; reverse layers9/17
59.392us versus93.184us. Payload16,711,680 bytes gives281.4–291.4GB/s.
IQ2_S/IQ3_XXS layer7 native61.440us versus92.160us; reverse layers18/33
61.440us versus91.136us. Payload15,667,200 bytes gives255.0GB/s.
Native relative L2 versus official FP32 GEMM is0.000488–0.000526.
Runtime M512/8/1/5/16/20/32/8 and graph replay pass; other M are
bitwise canonical.

Admit only these four measured SM70 FP16 M8/N4352/K5120 gate/up
orientations. Six layers save an estimated190.976–192.512us per M8
round, not an end-to-end measurement. Original-weight retention is
97,136,640 bytes/rank. Coverage reaches38/64 gate/up layers, including
30/40 mixed pairs and79.413% of mixed-pair bytes.
