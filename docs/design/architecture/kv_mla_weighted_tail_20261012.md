# Shared MLA weighted-partial qualification

The concatenated MLA stage's initial short-context cases did not establish
all valid partition lengths. An additional single-partition, 8192-token
case found a numerical failure in the existing shared grouped computation's
INT8 binding: small P times row-scale values disappeared at the Half MMA
input cast. FP16, BF16 and E4M3 controls passed. The parent INT8 case failed
all 4096 output elements, with maximum absolute error 0.0096364 and relative
error 0.05330 under the existing `atol=0.004, rtol=0.03` contract.

The compiled row-scale specialization now uses an exact binary factor before
its high/low MMA casts and removes it in FP32 after the two dot products.
The factor is bounded by each row's peak weighted value and input-type range,
so large finite values do not overflow the cast. Traversal, online softmax,
unweighted denominator, partitions, workspace and executor remain shared.
The FP8 numerical order and branch are unchanged. No model qualification or
new executor is introduced, and the failed error limit is not relaxed.

The corrected single-partition cases pass for all four representation codecs.
Four additional INT8 cases exercise the adjacent representable inputs
4062/4064 across the factor boundary, plus 5000/65504, and replay graphs with
changed writer values/scales across that boundary. All outputs remain finite.
Nine affected existing INT8 decode/graph cases also pass, including BF16
queries. These establish **eight new edge cases and nine affected regressions**,
not a representative engine, latency or task-quality result. The first bounded
candidate run records five passes; the stronger boundary/replay run records
13 passes, with the original first-row large-value test superseded by its
expanded parameterized cases rather than counted twice.

Source/component evidence uses the existing normally built stable component
from the concatenated stage and the changed shared Triton source, on V100
SXM2/SM70 with CUDA 12.8.93 and Torch 2.10.0+cu128. Whole-wheel and engine
qualification remain open. The installation build began with parent
`ac8be7ef26`; final artifacts must include this follow-up's source before
promotion. Keep the earlier parent failure and initial bounded support scope
visible in [concatenated MLA evidence](kv_triton_mla_20261012.md).

## Normal artifact build gap

The complete source build additionally exposes a missing aggregate initializer
in `bind_dense_scale_view`: QSA added the shared page-mapping field, but the
dense constructor still initialized only its two scale planes. The normal FA2
Split-D translation unit fails with `-Werror=missing-field-initializers`.
Initialize the existing third field with its default mapping, preserving the
dense address and scale contracts. This is a build integration fix, not a new
kernel path or numerical change. Keep the failed normal build separate from
previous standalone component successes and require the complete artifact
check and normal installation before release admission.
