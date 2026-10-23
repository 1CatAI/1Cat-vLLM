# Reuse PLE speculative metadata request indices

PLE metadata previously copied speculative request indices from CPU to CUDA
once for state rows and again for accepted-token counts. Both defaulted to
blocking transfers. Reuse the device indices and submit remaining transfers
asynchronously on the current stream. Classification, ordering and persistent
graph buffers are unchanged; mixed decode/prefill retains its original sort.

Eight CPU/CUDA cases cover C1/C4, padded requests and interleaved speculative,
plain decode and prefill requests, checking exact state and acceptance order.
The ordinary artifact from `5b764a3d43` uses the owned SM70 native build that
includes the allreduce synchronization change. No precision changes occur.

A CUDA-only capture of100 C1 metadata builds records:

|API or transfer|Before|After|
|---|---:|---:|
|cudaStreamSynchronize|200|0|
|Eight-byte host-to-device index copy|200|100|
|All cudaMemcpyAsync calls|700|600|

The first separate-process host timing gave425.969/424.424us for C1/C4
before and570.013/611.665us after. This result did not support promotion and
is retained. A controlled same-runtime comparison loads the earlier Python
builder as a reference, uses the same native libraries, alternates order over
eight epochs and drains the stream outside timing. Every epoch favors the
new path:

|Requests|Earlier host build(us)|New host build(us)|Saved host time(ms)|
|---|---:|---:|---:|
|1|412.204|389.945|0.0223|
|4|411.186|386.561|0.0246|

These measurements cover metadata host work, excluding pending GPU completion.
They are not verifier-round speedups. Process timing variation is unresolved;
the same-process alternating result is the controlled implementation check.

Reproduce with `benchmarks/benchmark_ple_metadata_transfers.py OUTPUT.json`.
The optional `--reference-module` loads an earlier installed Python metadata
implementation only, without native overlays; `--profile --requests 1` allows
a single CUDA-only profiler range. Use V100, CUDA12.8, Torch2.10cu128 and the
ordinary installed artifact. Model replay spread and complete-round timings
are checked separately before claiming end-to-end benefit.

## Combined model check

The final ordinary artifact includes this change and prepared GGUF head order.
The full model's CUDA-only prefix API attribution has zero
cudaStreamSynchronize calls on all four ranks. GGUF C1 I8192/O256 gives
23.2973 and23.6183ms unobserved, with identical complete IDs across those
controls and the earlier head-order run. C4 I128/O600 gives48.1459ms,
with different trajectories from that earlier run; do not attribute its whole
difference to metadata host work. The same-artifact NVFP4 controls are
20.9488/20.5529ms C1 and46.3875ms C4.

Actual GPU target entry spread has49.856us median but407.928us p90 and a
14.722ms maximum under profiling. CPU replay submission still has1,366.582us
median spread, with rank0 latest. Asynchronous submission overlaps preceding
GPU work; eliminating these two stream waits does not eliminate all CPU
asymmetry. No artificial rank barrier was introduced. Eight natural600-token
prompts give paired acceptance44.246% versus45.204%, with difference95%
interval −3.347 to+0.929pp. This small set cannot establish equivalence.
