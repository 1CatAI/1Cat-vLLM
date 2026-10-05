# Joint original-block expert model dispatch

Retain original GGUF gate/up blocks alongside canonical storage only for
measured IQ2_S and IQ3_S expert bands. Keep whole packed rows when slicing
TP4 output channels. Expert down projections keep the existing canonical
TP4 treatment of the 640-wide intermediate, including local K=160.
EP is not required. IQ3_XXS remains canonical by default: retaining its
original rows would add about 2.55 GiB per rank for modest M5 benefit.

The ordinary opaque operator resolves original M from routed rows/top-k
at execution time, so range compilation cannot freeze a large-prefill choice
for MTP verification. IQ2_S admits M1/5; IQ3_S admits M1/5/20. Other original
batch sizes keep canonical vector/GEMM scheduling. Mixed gate/up types, EP,
missing banks and incompatible shapes retain the canonical route. Admission
records reasons, measured original batches and actual retained bytes through
the existing kernel framework. The 30 useful layers add about 4.20 GiB/rank.
No environment variable or activation/accumulation precision change is added.

## Installed validation

Ordinary package `1.5.2.dev765+g45249d4d3` contains model source and the
whole core built from #937. Nine raw capability, 16 gate/up dispatch/bank
and 12 down dispatch checks pass (37 total CPU checks). Tests exercise all
four TP row slices, original rather than routed M and canonical fallback.
Ten GPU checks pass: four mixed-format MoE TP4 cases at M1/8/32/512,
four opaque gate/up graph cases for IQ2_S/IQ3_S M5/20, and two existing
loader/adapter cases. Graph cases change inputs and reproduce identical
results on repeated replay; IQ2_S M20 uses the canonical fallback.

Core SHA256:
`f1cf194ddd056195d690db8396ba81b7d73ea90cb8505981170f2fb25fba9b5e`.
Whole wheel SHA256:
`2dea0f1b3dadfc5dbf11df91429c07387e0648a28808826a9a4a0f258f582db8`.
All 213 dependencies pass. Fresh import selects the ordinary packaged core
without a private library override. Later changes add only tests/documentation
and merged main content; the measured native source remains identical.

The joint operator's real-weight cold-bank timings and numerical errors are
in `sm70_gguf_joint_original_experts.md`. The separately measured complete
model composition includes this dispatch, GDN copies and other merged
operators: Flash-Next IQ3_S + FP16 MTP4, TP4 four V100s, C1 I8192/O256
26.072 ms/round and 92.267 decode tokens/s; C4 I128/O1024 52.059 ms/round
and 280.441 aggregate decode tokens/s. Four natural EOS prompts retain the
original token IDs. These are combined composition results, not a new
isolated model restart or a claim that every delta comes from gate/up.
