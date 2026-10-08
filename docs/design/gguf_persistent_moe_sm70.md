# Resident GGUF MoE on SM70

Small speculative verification batches reuse experts across tokens. A routed
implementation can decode the same expert row several times and cross separate
gate/up and down kernel boundaries. This experiment gives each output-row
stripe a resident CTA and shares decoded integer words among tokens selecting
the same expert. It does not select a new model route until numerical, complete
chain and matched model benchmarks pass.

The implementation uses ordinary SM70 loads and `dp4a`. It shares its integer
reader with the existing GGUF operators. It requires no TMA, asynchronous copy
instruction, FP8 matrix instruction, token sorting or padded token tile.

## Supported contract

The prototype accepts five tokens, at most ten distinct experts per token,
at most 512 experts, K divisible by 256 up to 2560, and an intermediate width
divisible by 32 up to 160. Gate/up reads original IQ3_S, IQ3_XXS or IQ2_S blocks.
Down reads the existing canonical IQ4_NL or Q2_0 descriptors. In particular,
Q2_0 retains its group-32 canonical representation at a TP boundary inside a
source group of 64. No expanded FP16 expert weights are retained.

Each token's route IDs must be unique, as supplied by top-k routing. Negative
or out-of-range IDs are ignored. Epoch buffers start at zero; every invocation
uses the same workspace sizes and advances every CTA's epoch. Workspace must
belong to one non-overlapping execution stream. Concurrent invocations require
different workspace. Routes, quantized intermediates, counters and flags are
allocated before graph capture.

M=20 and the final IQ4_XS gate/up layer remain on their existing paths. This
prototype does not change router, shared expert, TP collective or draft work.

## Execution and progress

Each CTA constructs the same sorted expert-to-token incidence list. A gate/up
task owns an expert and 32 intermediate rows. The row's integer weight words
are decoded once and used for every token selecting that expert. The existing
FP16 gate/up, FP16 SiLU/multiply and Q8_1 intermediate boundaries are retained.
The first prototype quantizes input activations in each CTA's shared memory;
this is redundant work whose cost is included in the complete-chain benchmark.

After writing a quantized intermediate chunk, all writers fence their stores
and the CTA publishes a GPU-scope release flag containing the invocation epoch.
Down consumes only chunks whose acquire flag matches that epoch. It shares the
down weight decode across selecting tokens, stores per-group FP32 partials and
reduces groups in fixed order before the FP16 projection boundary. Weighted
unroute retains the previous four-warp FP32 reduction order. Small differences
from multiply/add contraction still require a GPU numerical check.

The launch reserves more than half of the SM's shared memory per CTA and uses
at most one CTA per SM. It checks device shared memory limits, block occupancy
and grid size before launch. This avoids queued producers behind a fully
occupied grid of waiting consumers.

A CTA prioritizes ready down chunks. If none are ready and it still owns a
gate/up task, it runs that producer instead of waiting. Suppose all resident
CTAs were waiting with no executable work. Every producer queue would then
have been exhausted, so all gate/up chunks would have been published. Any
unfinished down chunk would be ready, contradicting the supposed blocked
state. A CTA cannot finish until it has consumed every selected expert's
chunks; therefore it cannot exit while it still owns an unpublished task.
This argument depends on the validated residency and workspace contract.

## Validation and admission

The portable source reader has 18 CPU cases comparing signed codes, base
scales and subscales with official GGUF decoding. These cover three formats,
both shared-book layouts, random/zero/ones source bytes and finite FP16 scale
boundaries. Reconstruction is elementwise exact. Six CUDA format pairs compile
for SM70 with no register spills; this is compilation evidence only.

Twelve GPU tests in the normally installed wheel pass. They change routes,
inputs and probabilities between replays, poison
intermediates and readiness, exercise expert sharing and empty routing, then
resume on the same graph. They compare the quantized hidden bytes with the
previous chain and the final projection with its FP32-accumulating output.
The real-weight benchmark additionally uses official FP32 down weights with
the same Q8 activation and FP16 projection boundary.

The real-weight screen measures the complete local chain in one process and
on one GPU with alternating timing order. It includes routing preparation,
input quantization, gate/up, activation/intermediate quantization, down,
readiness and weighted unroute. It uses recorded top-10 routes and real TP4
IQ3_S, IQ3_XXS and IQ2_S weights. Random activations are a microbenchmark
limitation. Report compulsory bytes and route-issued bytes separately; their
ratio is not a measured HBM byte count.

Before model admission, separate the contributions of shared weight decode
and resident scheduling. A faster isolated GEMV does not qualify the chain.
The accepted route must use a normally packaged module and pass same-wheel
C1/C4, target output, teacher-forcing and acceptance checks. Profile the
accepted configuration again before selecting the next structural change.
No end-to-end latency or model-quality result is claimed by the compilation,
CPU tests or isolated-chain tests.

## First complete-chain screens

The first resident schedule passed changing-route graph replay numerics and
preserved every hidden Q8 byte. Its final relative L2 error against the existing
chain was at most 1.28e-5. It was slower on all three real-weight cases and is
rejected for model admission:

| TP4 rank-0 weights, M5 | Existing chain | Resident chain |
| --- | ---: | ---: |
| Layer 17, IQ3_S / IQ4_NL | 56.22 us | 156.23 us |
| Layer 0, IQ3_XXS / IQ4_NL | 53.41 us | 155.84 us |
| Layer 1, IQ2_S / Q2_0 | 54.58 us | 167.76 us |

These are same-process graph measurements on one V100-SXM2-32GB, CUDA 12.8,
Torch 2.10.0. Each arm has eight alternating-order timing samples. Original
Q8_1/FP16 boundaries and FP32 accumulation are retained. Activations are
synthetic; routes and weights are recorded from the real model.

Separate gate/up weight reuse is numerically exact against the previous
chain. It saves only 0.86 us for IQ3_S and 1.25 us for IQ3_XXS, and loses 3.83 us
for IQ2_S in the matched screen. This does not meet the structural improvement
budget. Waiting for whole down batches still costs 149--162 us and is rejected.

The combined resident kernel uses 72--80 registers/thread, compared with
38--44 for the previous down kernels. Its 512-thread CTA reserves one SM while
many down phases activate only a subset of its warps. Removing launches alone
does not solve that resource and dependency structure.

The third experiment uses 128-thread workers, six independent expert sequences
per output stripe and one readiness word per completed expert. Eight-row
gate/up tasks combine through four-producer FP16 chunk joins before the
existing Q8_1 quantization boundary. Input quantization and routing preparation
share an initial kernel; weighted unroute remains a final kernel. Both kernels
give ordered workspace reset and output reduction without polling between
invocations. The worker pool retains shared integer decoders and checks that
all workers can reside before launching. It compiles without spills and passes
the real-weight changing-route numerical screen, including exact hidden Q8
bytes. Its final relative L2 error remains below 1.28e-5. It also loses the
complete-chain comparison and is rejected for model admission:

| TP4 rank-0 weights, M5 | Existing chain | Six-worker task pool |
| --- | ---: | ---: |
| Layer 17, IQ3_S / IQ4_NL | 61.08 us | 80.88 us |
| Layer 0, IQ3_XXS / IQ4_NL | 57.07 us | 81.19 us |
| Layer 1, IQ2_S / Q2_0 | 57.92 us | 86.49 us |

These measurements use the same GPU and alternating graph timing protocol as
the earlier screens. Each comparison uses its own matched control; controls
from different runs must not be used to infer gains. The smaller workers
improve the resident prototype substantially but do not recover the original
chain's performance. Fewer kernel boundaries and fewer logical weight reads
are insufficient admission criteria. Further scheduler changes require
instruction, memory and dependency counter evidence from this failed chain.

The experimental module is included in the normal CMake and wheel build. The
clean-wheel audit preserves all 16 previous native modules byte-for-byte and
adds only the resident MoE module. This packaging check does not promote the
experimental operators to model dispatch or establish an end-to-end gain.

## References

Weight-major ownership and readiness scheduling are informed by
[MonoMoE](https://arxiv.org/abs/2609.04244) and
[Mirage MPK](https://arxiv.org/abs/2512.22219). These implementations target
different hardware and workload contracts; this kernel reuses the existing
SM70 numerical reader and explicit deterministic reductions. Source integer
layouts follow the packaged MIT-licensed llama.cpp definitions. The new
scheduler is Apache-2.0 code.
