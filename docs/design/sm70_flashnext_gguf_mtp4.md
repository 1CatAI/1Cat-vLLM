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

### Matched dense-model check

On Qwen3.8-27B UD-Q4_K_M, TP4, FP16 activation/KV, graph execution,
8,320-token capacity and four sequence slots, a 1,024-in/128-out synthetic
cohort gives the following pure decode means across two repeats:

| Concurrency | Ring disabled tok/s | Default ring tok/s | Change |
|---|---:|---:|---:|
| C1 | 59.880 | 66.462 | +10.991% |
| C4 | 221.019 | 221.001 | -0.008% |

Four natural greedy prompts retain identical token IDs and natural EOS in
both arms: Paris, arithmetic, translation and a Chinese explanation. Worker
reports confirm automatic direct-ring admission; large padded graph shapes
report the calibrated-byte-range fallback. The C4 tensors exceed the small
message limit. This is a dense-model check, not Flash-Next MTP4 latency.
Startup and compilation are excluded from decode. This smoke is complete.
The collective merge check uses Flash-Next C1/C4, identical outputs and natural
prompt termination; long-form model quality belongs to the model integration.

Final model gates use 256K capacity, 8K input, greedy and temperature 0.7,
teacher-forced distribution checks, eight prompts of at least 600 generated
tokens for acceptance statistics, the fixed quality set including 128K/258K
needles, and a C4 regression smoke. Numerical limits are mean/p99/max KL
0.001/0.01/0.05, top-1 agreement at least 99%, and maximum logit error 0.5
against FP32 dequantization of the same GGUF checkpoint. Rejection sampling
must preserve the reference target distribution.

The separate draft loader passes focused sharing and quantization
configuration tests in a clean installed wheel. The original checkpoint
contains 31 BF16 MTP tensors, totaling 5,214,301,696 bytes. All values are
finite. The initial model comparison uses the existing FP16 loader; BF16
reader operators remain a separate precision follow-up.

### Initial Flash-Next MTP4 comparison

The IQ3_S target and FP16 MTP4 draft run on TP4 with FP16 activation/KV,
FP32 recurrent state, FULL decode graphs, 8,704 capacity, batch budget 512,
four sequence slots and 0.90 memory utilization. C1 uses 8K input and 256
fixed-length greedy output tokens; natural greedy prompts are separate.

| C1 measurement | Ring disabled | Default ring |
|---|---:|---:|
| Complete engine round mean | 46.822 ms | 41.499 ms |
| Pure decode | 71.566 tok/s | 75.212 tok/s |
| Mean acceptance length | 3.427 | 3.036 |
| Trimmed timing intervals | 57 | 66 |

Four natural prompts have identical complete token lists and EOS in both
arms. The observed complete-round reduction is 5.323 ms and pure decode
improves 5.095%. The synthetic acceptance trajectory differs, so the round
reduction is not an isolated collective saving. Initial timing reports did
not retain synthetic token lists. Trace attribution must use actual calls
and overlap before calibrating the projected saving.

Four 8K inputs do not form a stable C4 cohort with batch budget 512. Both C4
arms instead use 128 input tokens per request, preserving all model and
execution settings. The default-ring arm records 37 steady intervals,
66.413 ms per engine round and 240.918 aggregate tok/s. The matching control
records 66.615 ms and 253.171 tok/s, with mean acceptance 3.920 versus 3.861.
The trimmed cohorts emit 624 versus 592 tokens. This first C4 point decreases
pure decode 4.840% and does not pass non-regression; retain it as a negative
result. A 1024-output-token, three-repeat comparison with complete token
recording precedes the admission decision. Final larger-capacity acceptance
and model quality measurements remain separate follow-ups.

### Longer Flash-Next C4 smoke

At the same 128-input-token C4 configuration, use 1,024 output tokens and
three repeats per arm in a matched normal installed package. Pure decode is
228.918/229.421/233.225 tok/s without ring and
260.337/263.335/263.158 tok/s with ring. Summing emitted tokens and engine
seconds across all repeats gives 230.505 versus 262.269 tok/s (+13.780%).
Weighted round means are 67.673 versus 65.624 ms, an observed 2.049 ms reduction.

Within each arm, all complete timing token lists and acceptance counters are
identical across three repeats. Four separate natural greedy prompts have
identical complete token lists between arms and stop at EOS. This longer
C4 smoke does not regress; the negative 256-output-token point remains above.
Output identity for the collective merge check refers to these natural prompts.

The forced-length timing lists differ between arms at zero-based positions
34, 471, 495 and 669 in the four streams. Full-cohort mean acceptance changes
from 3.285 to 3.830, and steady emitted tokens per request-round from 3.900
to 4.303. Acceptance and timing have distinct statistical windows. These
changes prevent attributing the full throughput or round reduction to the
collective. The difference is retained for numerical attribution; final model
distribution and long-output checks remain separate. Actual trace calls,
waiting and overlap are the next input to the cost model.
