# Flash-Next GGUF MTP4 trace

The first node trace explains the IQ3_S target and FP16 MTP4 draft under
TP4, 8,192 input tokens, 256 output tokens, 8,704 capacity, batch budget 512
and four sequence slots. Activation and KV are FP16; recurrent state is FP32.
Workers use FULL decode graphs and automatic SM70 ring admission.

The normal installed source is `3720df1225`, wheel SHA256
`def49fcd6c7ce3aad1150d898a65a4e461eba70c9b4147e72adcf3a2626599e9`,
with native core SHA256
`0bb550c1cde2a901f08224ef01b70f7abe52134c4996dd30c7cb7b6b20e8f933`.
Nsight Systems 2024.6.2 collects CUDA graph nodes during one generation only;
two subsequent generations run with capture disabled but Nsight attached.
Loading, compilation, natural prompts and warmup precede collection.

## Measurement scopes

| Measurement | Complete engine round mean | Mean full-request acceptance |
|---|---:|---:|
| Node capture enabled | 54.232 ms | 2.393 |
| Capture disabled afterward, repeat 1 | 43.043 ms | 2.393 |
| Capture disabled afterward, repeat 2 | 43.569 ms | 2.393 |

All three complete timing token lists and speculative counters are identical.
Each reports 107 drafts, 428 proposed tokens and 149 accepted tokens.
Four separate natural prompts retain the same complete token lists and EOS
as the earlier ring comparison. The capture-off mean is 43.306 ms; enabling
node capture adds 25.228% relative to it. Attachment and diagnostic-marker
overhead are not removed by turning collection off.

The prior unattached automatic-ring C1 result remains 41.499 ms with a
different 3.036-token acceptance trajectory. Do not substitute this trace
for that endpoint result, or proportionally scale node service times into
an unprofiled wall table.

## Select actual target replays

Both target and draft initially emit the same FULL replay label. Aggregating
all 584 replays per worker gives mixed target, draft and prefill statistics;
that aggregate is rejected as a speculative-round measurement.

CUDA launch correlation identifies graph 14 as the full target: 107 complete
launches per rank, each with 3,194 nodes. Draft graphs have 81 and 64 nodes.
A clipped final 31-node target launch is excluded. A derived SQLite copy
labels only the 107 full target ranges; original events and timestamps remain
unchanged. Pair each worker's target ordinal with the next target ordinal,
then remove eight leading and trailing ranges. This yields 91 diagnostic
rounds per rank. The engine benchmark uses 89 trimmed output intervals;
these statistical windows are close but distinct.

| Diagnostic measurement | Result |
|---|---:|
| Mean maximum rank target-to-next-target interval | 54.660 ms |
| Mean maximum rank summed kernel/copy service | 44.934 ms |
| Mean maximum rank GPU activity envelope | 54.468 ms |
| Mean rank target graph GPU envelope | 46.873 ms |
| Four draft graphs, mean rank summed kernel service | 4.900 ms |

Service may overlap and collectives include arrival waits. The target graph
envelope excludes graph-external logits, sampling and state work. The draft
row is service, not complete four-draft wall time. None of these rows forms
an additive endpoint decomposition.

## Initial operator ledger

The following are rank-average kernel service and calls per complete
diagnostic round. Kernel categories identify implementation families; generic
FP16 GEMMs still need parameter ownership and exact byte attribution.

| Scope | Family | Calls/rank/round | Rank-mean service |
|---|---|---:|---:|
| Target | FP16 GEMM/GEMV | 328 | 8.545 ms |
| Target | GGUF lattice projections | 94 | 6.781 ms |
| Target | GGUF LUT4 projections | 126 | 3.134 ms |
| Target | GGUF bitplane projections | 131 | 2.994 ms |
| Target | Ring all-reduce | 98 | 2.503 ms |
| Target | Copy/cast kernels | 507 | 1.987 ms |
| Target | Index/reduce/scatter | 440 | 1.714 ms |
| Target | QSA/indexer | 72 | 1.595 ms |
| Target | HC combine/gates | 290 | 1.265 ms |
| Target | GGUF affine projections | 50 | 1.158 ms |
| Target | Router/shared gate | 96 | 0.464 ms |
| Draft | FP16 GEMM/GEMV | 34 | 2.262 ms |
| Draft | Ring all-reduce | 12 | 0.395 ms |
| Draft | QSA/indexer | 15 | 0.291 ms |
| Draft | HC combine/gates | 32 | 0.228 ms |
| Draft | NCCL all-gather | 12 | 0.224 ms |

The actual round contains 110 ring reductions, not the earlier rough
300-collective estimate. NCCL gathers and graph-external reductions remain
separate. Ring service includes readiness waiting and node-capture overhead;
its trace mean is not link-transfer latency and cannot replace the installed
3.160/5.095 µs microbenchmark. Endpoint ring comparisons also change
acceptance trajectories, so their full savings are not isolated collective
savings. Reconcile exact call shapes and overlap before the next projection.

The expert projection ledger is shared with the GGUF operator work. Reuse its
kernel changes rather than starting a second lattice implementation.

## Packed PLE path

The PLE table is IQ4_NL, with logical shape 320,001,536 by 160 and
28,800,138,240 payload bytes. Each row has five 32-element blocks at 18 bytes
per block: 90 packed bytes and 320 decoded FP16 bytes. TP4 holds 6.71 GiB
of packed pinned-host storage per rank, with a prepared UVA view.

Each M5 target has one packed gather of 80 rows followed by IQ4_NL dequant.
The logical packet is 7,200 bytes per rank, and the output is 25,600 bytes.
This excludes transaction amplification and duplicate/masked rows; measured
host-memory traffic and valid unique row counts are still needed.

| Operator | Calls/rank/round | Rank-mean diagnostic service |
|---|---:|---:|
| Packed UVA gather | 1 | 97.862 µs |
| IQ4_NL dequant | 1 | 2.811 µs |

The packed UVA path executes in the captured target graph. This proves row
transport selection, not complete elimination of all host waits. NUMA
placement and cold-row timing precede a placement or decoder change.

## Next measurements

Confirm PLE host-wait behavior and measure the current 90-byte row path on
local and remote NUMA placement with real table rows. Record exact projection
parameters and route distributions before assigning read bytes or a
750 GB/s lower bound. Inspect the generic FP16 projection ownership, HC,
QSA and router grids, and compare exact-shape candidates with cold L2.
Only measured call counts and overlap enter endpoint saving projections.

Raw reports, the target-selection record, complete kernel names and grids,
and per-rank interval tables are retained with the benchmark artifacts.
The initial unavailable CUDA wrapper, unsupported old CLI option and
missing old importer are retained as tooling failures; none supplied model
timing. A five-node probe validated the complete profiler and SQLite parser
before the model capture.
