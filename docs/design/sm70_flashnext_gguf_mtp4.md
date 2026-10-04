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
64-bit atomic packets containing FP32 payload bits and a generation tag, with
system release stores and acquire loads. Intermediate sums remain FP32 and
only the final output narrows to the existing FP16 activation boundary.

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

Current status: research protocol compilation and model integration checks
are in progress. No new collective speedup or IQ3_S MTP4 baseline is claimed.
