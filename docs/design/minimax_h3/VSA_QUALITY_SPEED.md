# VSA quality and 31.3-second stage

The user resumed FastH3 VSA Data-Free development with original floating
weights, TP4 and the official top-k 64/four-step algorithm. This stage requires
independent FP32 full-sampling quality and human audiovisual review, plus a
full warmup and three unprofiled requests with median denoise <=31.3 seconds
and CV <=5%. Complete requests must beat a matched Dense control. Official
GPU-kernel validation and >80 useful TFLOP/s/card remain separate unfinished
objectives. No quality threshold is relaxed and no AUTO promotion is made.

## Frozen baseline and diagnosis

The source baseline is `970c5fb3f86d59440a3431e53029e98f8d690778`, which merges
the existing shared-kernel dependency into the owned VSA worktree. Commands,
source/binary hashes, input captures and diagnostic artifacts are retained in
`/home/ymzx/h3-sm70-artifacts-20260909/vsa-quality-speed-20260910/`.

The native host policy now matches the retained Dense configuration: pageable
shared VAE masters, prepared floating columns and explicit native TP4 peer
rows with a 4 GiB communication budget. A complete primary capture preserves
the previous VSA final video/audio latents, all pre-encoding RGB frames and
PCM bitwise. Its 34.895545-second denoise is a cold diagnostic including input
capture, not formal performance or a quality repair.

A separate profiled request records a 32.039245-second denoise NVTX span on
the slowest rank. Sparse Attention accounts for 7.343837 seconds and GEMM for
15.466638 seconds. The initial parser classified peer `reduce_rows` in other
kernels; its 3.654961 seconds must be included in communication, alongside
1.727143 seconds of other collectives. Profiler start/stop overhead is outside
this NVTX span but inside native stage timing; neither time is an unprofiled
acceptance result. No NCU occupancy/utilization claim is made.

The fixed-input diagnostic shows probability rounding contributes to error,
but a compensated-PV prototype only reduces actual operator relative L2 from
0.000203199 to 0.000135391. It fails its numerical admission criterion and is
not installed or run as a full-sampling candidate. Further localization keeps
QKV, selected blocks and gates fixed; FP32 output and QK/PV diagnostics are
separate from production precision.

## Deferred work accounting and internal validation

Dynamic selected-pair and selected-block counts stay as int64 device scalars
while layers execute. The request-owned counter retains their producing
tensors and step/layer association, then copies all counts to CPU together
after complete denoise. Public result fields remain Python integers and are
reconciled across steps, layers and totals. Counter completion remains inside
the complete-denoise timer and before the final TP barrier. Closing or failing
a request removes hooks and releases pending tensors.

H3-owned geometry and a nonempty mask produced by the official top-k/prefix
construction use a private prevalidated CUDA entrypoint. Device, layout,
dtype, shape, alignment and index-limit checks remain. The general sparse
operator still validates block-size values and rejects empty query rows.
Older wheels use its checked entrypoint until rebuilt; no user flag bypasses
validation. Kernel arithmetic is unchanged.

Validation so far: 73 CPU checks pass, 2 GPU checks skip in the CPU run;
23 leased-GPU checks pass, including real dense/sparse DiT work accounting,
tail/padding controls, strict public input rejection, exact public/private
output equality and absence of host scalar reads in the private entrypoint.
The rebuilt sparse kernel retains 215 registers and zero spills. Nine further
checks pass under both Compute Sanitizer memcheck and synccheck with zero
reported errors. The
complete primary capture preserves final video/audio latents, pre-encoding
RGB and PCM bitwise. Selected blocks, valid pairs and useful FLOPs also match
at every rank, layer and step. Its 34.458830-second cold captured denoise is
diagnostic only. This preserves the old native output, whose independent
FP32 quality still fails; it does not establish engineering quality.

## Full-sampling amplification diagnosis

A separate artifact-only prototype resets the FP32 PV accumulator for each
64-key block and adds a scaled low probability component. Fixed-input relative
L2 improves from 0.000203199 to 0.000046140, but this still misses the prototype's
fivefold numerical improvement criterion. It is not installed. A full run was
used specifically to localize amplification, not as acceptance or performance
evidence. The centered-exponent variant adds no useful numerical improvement.

The independent FP32 reference was run again and reproduces the original final
video and audio latents bitwise. With identical initial tensors, the block-local
candidate's video latent errors after the four steps are 0.003839, 0.016306,
0.056423 and 0.369510; final audio error is 0.050502. Both final latent gates fail.
Selection first differs in the second layer on all ranks. By the last layer of
the first step, 73.2--78.4% of query blocks have at least one changed selected
block. This is the fraction of affected queries, not the fraction of replaced
keys. Fixed-reference-map controls separate continuous arithmetic error from
dynamic-selection amplification. They cannot be shipped. Fixed reference maps
still give final video/audio errors of 0.189996/0.016187 with the native kernel
and 0.175657/0.014001 with compensated block-local PV. Exact FP32 prefix queries
with dynamic video selection give 0.401763/0.086767. All three fail; neither
fixed routing nor prefix precision alone solves the problem.

The native frontend using the same FP32 sparse operator matches the independent
official frontend bitwise on real inputs, including the official transport-only
partner block. The block-local diagnostic before final FP16 rounding has relative
L2 0.000005753 against FP32 (prefix 0.000015489, video 0.000005443). A further
artifact isolates exact FP32 QK/softmax and compensated 64-key PV partials; no
production sparse-math change is admitted from these local numbers.

## Exact layout fusion

One kernel gathers Q/K/V directly into their final padded tiles. Another
combines learned compression with the sparse output and restores the original
row order, preserving separate FP16 multiply/add rounding, including overflow.
The original pooling, top-k, sparse traversal and work reductions are retained.
An older extension or strided/unaligned inputs use the existing Python layout.

Only geometry indices are cached across requests. A denoise context owns QKV
scratch storage separately for each device/stream and releases it on success or
failure. Nested contexts restore their parent. Every returned output is freshly
allocated, so a later layer cannot overwrite an earlier result.

The artifact prototype preserves real primary outputs and useful counts bitwise.
The corrected comparison includes work counting on both sides: median 41.969666
ms baseline versus 37.358593 ms fused over seven paired operator measurements.
This is an operator result, not complete-denoise acceptance. The installed source
passes 30 GPU tests covering layout, sparse math and geometry; its eight layout
checks also pass memcheck with zero errors. Complete native output preservation
and formal timing remain pending. Engineering quality and the 31.3-second stage
remain incomplete; VSA is not promoted into default or AUTO selection.
