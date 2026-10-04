# Flash-Next GGUF MTP4 on a four-GPU NVLink ring

The target is IQ3_S GGUF with the original BF16 MTP weights, TP4, and four
speculative tokens. The complete-round objective is 10–11 ms, including target
verification, rejection sampling and four draft steps. This is an objective,
not a measured result. Reuse the V2 runner and current MTP4 graph scheduling.

## Collective scope

The four V100s have direct links 0–1, 0–2, 1–3 and 2–3; the two diagonals cross
NUMA through SYS. Recursive doubling can embed its two exchanges on these
direct links. System-scoped native peer atomics must be supported on every
accessed edge. Large messages retain NCCL. Small-message admission follows
measured limits rather than assuming an improvement at every concurrency.

The previous volatile-vector packet prototype is not admitted: its data/tag
visibility and atomicity were not established. The system-fenced block variant
is a retained negative result. The new research protocol uses naturally aligned
64-bit atomic packets containing two losslessly encoded FP32 values and a
two-bit generation tag, with system-scoped relaxed stores and loads. Each
packet contains its entire payload, so it does not publish a separate memory
object. Intermediate sums remain FP32 and
only the final output narrows to the existing FP16 activation boundary.

FP16 inputs and partial sums of up to eight inputs have FP32 exponent codes
0, 103–145 or 255. Renumbering the exponent takes six bits; all 23 mantissa
bits and the sign remain intact. Two 30-bit values fit alongside the tag.
Double buffering and independent counters for active value pairs prevent
stale packets when graph widths shrink and grow. A block-wide counter failed
that check and was rejected. The implementation unrolls peer exchanges and
does not require a CUDA 12.8 atomic builtin.

[NVIDIA's memory model](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html)
requires native peer atomic support for system-scoped atomic accesses to GPU
memory shared by GPU threads. Capability checks must enforce that condition.
Packet synchronization, buffer reuse and graph width changes require their
own correctness checks; atomicity alone does not prove the entire protocol.

Measure 5,120-byte and 25,600-byte collectives in CUDA graphs with multiple
calls per replay, alternating with NCCL. Include odd tails, subnormal inputs,
changed/poisoned replay and shrinking/growing widths. Report maximum-rank
latency; per-rank service is not complete-round latency. HC all-gather and the
MoE reduction epilogue will use the same communication implementation after
basic allreduce is qualified.

### Operator screen

Four V100 SXM2 32 GB GPUs at 300 W, CUDA 12.8.93, Torch 2.10.0+cu128,
FP16 input/output and FP32 partial sums. Each point uses six alternating
samples and 96 collectives per CUDA graph. Times below are the largest
per-rank median. These are research-extension results; installed framework
validation is still pending.

| M | Input bytes | Ring µs | NCCL µs | Remote packet stores per rank |
|---|---:|---:|---:|---:|
| 1 | 5,120 | 3.522 | 13.626 | 20,480 B |
| 2 | 10,240 | 3.606 | 13.349 | 40,960 B |
| 4 | 20,480 | 4.545 | 15.882 | 81,920 B |
| 5 | 25,600 | 4.999 | 16.652 | 102,400 B |
| 8 | 40,960 | 7.258 | 18.408 | 163,840 B |
| 16 | 81,920 | 13.130 | 18.926 | 327,680 B |

All four ranks passed 31 replay checks, including odd tails, changed and
poisoned outputs, width changes, subnormals and an independent FP64 oracle.
Packet-store bandwidth is 5.814 GB/s at M1 and 20.484 GB/s at M5; this excludes
polling and read traffic. Each 100 nonoverlapped calls projects a 1.010 ms M1
or 1.165 ms M5 saving. Actual speculative-round call counts and overlap have
not yet been collected, so no complete-round speedup is inferred.

A second screen using explicit PTX system-scoped accesses passed the same 31
checks per rank. Maximum-rank medians were 3.734 µs at M1 and 5.024 µs at M5;
the M5 result is slightly above the target and requires further validation.

The framework enables only a four-rank SM70 direct NVLink ring with native
peer atomics, contiguous local FP16 input and a payload of at most 25,600
bytes. Other hardware, full-mesh groups, larger inputs and batch-invariant
reduction retain existing dispatch. Startup observations report admission
and fallback reasons. HC all-gather and the weighted MoE epilogue are pending.
The capability, dispatch and existing acceleration-report suites pass 50 tests
from source and from a clean installed wheel. The complete `_C` and the current
SM70 sampler module are source-built through CMake; other unchanged native
modules use the normal precompiled package. The installed operators are present,
source/wheel/installed Python modules match, and the primary extension has no
private dependencies or RPATH. Installed graph/lifecycle checks pass: 25 cases per rank and two buffer
reopens after collective close. With the source-built installed extension,
maximum-rank medians are 3.160/3.636/4.593/5.095 µs for M1/M2/M4/M5,
versus NCCL 12.177/13.369/15.916/16.666 µs. The M5 result remains above
5 µs; the first screen alone does not establish that target. Model checks
are pending.

Further screens rejected a uniform 64-thread launch (M1 3.242 µs, M5
5.103 µs), a 96-thread launch (M1 3.509 µs, M5 5.122 µs), and Half2
input/output transport (M1 3.463 µs, M5 5.063 µs). The scalar 128-thread
implementation remains the baseline. A natural exponent-field packet layout
passes 1,065,536 CPU bit round trips and 31 GPU replay checks per rank, but its
3.545/5.016 µs M1/M5 result does not establish a decisive improvement. It is
retained as a research candidate.

Rejected variants remain useful bounds: uncompressed release/acquire packets
took about 26.7/44.1 µs at M1/M5; relaxed uncompressed packets took about
6.0/17.9 µs. Neither met the small-message target.

## Model and measurement gates

Finish standalone Flash-Next loading, packed PLE row transport and canonical
expert preparation before adding the separate BF16 MTP checkpoint. Shared
embedding and head remain those of the GGUF target. Do not quantize MTP or HC,
or reduce the draft vocabulary, without explicit approval.

Record real prompt routing for every layer and speculative round. Operator
measurements use M=5 verification and M=1 drafting, cold L2, actual TP4 weight
shapes, graph timing, bytes, bandwidth and numerical errors. Convert each
candidate win into a projected complete-round saving using trace call counts;
report stream overlap separately. End-to-end runs occur at merge gates or
when accumulated estimated savings exceed about one millisecond. Investigate
projection errors above 15% before using the cost model for the next change.

Final model gates use 256K capacity, 8K input, greedy and temperature 0.7,
teacher-forced distribution checks, eight prompts of at least 600 generated
tokens for acceptance statistics, the fixed quality set including 128K/258K
needles, and a C4 regression smoke. Numerical limits are mean/p99/max KL
0.001/0.01/0.05, top-1 agreement at least 99%, and maximum logit error 0.5
against FP32 dequantization of the same GGUF checkpoint. Rejection sampling
must preserve the reference target distribution.

The separate draft loader has passed focused sharing and quantization
configuration tests in a clean installed wheel. The original checkpoint
contains 31 BF16 MTP tensors, totaling 5,214,301,696 bytes. All values are
finite, but converting to FP16 changes 583,979 values. The draft path must
retain BF16 weights and use a compatible reader before model measurements.
An IQ3_S MTP4 baseline, acceptance statistics and model quality gates remain
pending.
