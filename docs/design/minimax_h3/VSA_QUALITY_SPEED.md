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
The rebuilt sparse kernel retains 215 registers and zero spills. Full native
preservation, sanitizer checks and end-to-end timing for these changes are
still pending. Engineering quality and speed are both incomplete.
