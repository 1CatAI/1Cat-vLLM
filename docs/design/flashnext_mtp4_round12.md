# Flash-Next GGUF MTP4 complete-round optimization

The performance objective is C1 at or below 12 ms per round in the existing
acceptance benchmark, with correct target output, unchanged acceptance and
no C4 regression. A kernel service-time saving is a hypothesis until the same
installed wheel's end-to-end A/B confirms it.

## Measurement contract

Use Flash-Next GSQ-RCO IQ3_S, FP16-loaded MTP4, TP4 on four SM70 V100s,
FP16 KV and FP32 recurrent state. Preserve the benchmark's eight natural
prompts, greedy sampling, natural EOS, 600-token limit, 8192-input/256-output
C1 probe and 128-input/600-output C4 probe. Report tokens per round together
with round latency. A result obtained after acceptance collapses is rejected.

Freeze source and installed core hashes, graph policy, topology, clocks and
configuration for each arm. Obtain the unprofiled baseline first; use a
separate short node trace to inspect order and overlap. Profiler overhead must
not be subtracted from or reported as accepted latency.

The per-round ledger must distinguish target verification, drafting,
sampling/state updates, host preparation, peer skew and GPU activity gaps.
Union intervals before reporting busy/idle time; summed concurrent kernel
service is not a wall-clock ledger. Leave unattributed time explicit.

## Starting evidence

The previously accepted main path measured 18.534 ms/round at C1 with
4.886 emitted tokens/round. This is a historical reference; a new normal-wheel
baseline and trace are required before choosing the next implementation.

The operator-integration branch reports 18.602 ms/round with its new switches
off and 18.394 with all switches on. Its HCX-off arm reports 19.780, but
the off arm's individual cohorts range from 18.540 to 20.583 ms/round.
Consequently the reported 1.4-ms HCX contribution requires a matched repeat.
Other retained experimental arms reduce tokens/round to approximately one;
these are numerical/dispatch failures, not eligible speed comparisons.

The branch is imported into an isolated source tree for normal-wheel tests.
All new routes retain explicit configuration switches for ablation. No route
is promoted based on the aggregate microbenchmark estimate.

## Normal-wheel HC load qualification

Source `4309a6e9e5ab787184d31f494711d801e25d0815` built core SHA256
`9f3f9264edcb80998e92ba71cfbe2f52ae077725a86e3771d0bb4df210532cf3`.
Same-process, same-wheel four-rank ABBA across eight real HC weight pairs:

| Batch | Original, µs/pair | Optimized, µs/pair | Saving |
| --- | ---: | ---: | ---: |
| M=5 | 21.428 | 20.820 | 2.84% |
| M=20 | 31.690 | 31.035 | 2.07% |

These are maximum rank median graph times. The M=5 delta estimates only
0.058 ms across 96 pairs; it does not substantiate an end-to-end speed claim.
The research DSO's larger saving must not be substituted for this result.

Output and injection match by raw FP16 bit pattern on all ranks at
M=1,2,4,5,8,10,20, including changed-input graph replay. Tag-wrap/batch-transition
stress against the replicated reference has block relative maximum error
2.13e-4 and zero injection error. Four CPU policy tests and installed dependency
checks pass. Clean-process ABI, core hash and loaded-library checks exclude
private kernel DSOs.

## Next decision

Rebuild the integrated operators as a source-complete installed wheel. Run
the current-path baseline and trace, then choose ablations from the measured
critical path. Investigate acceptance failures before comparing affected arms.
HCX must preserve FP16 norm materialization and avoid keeping duplicate HC
weight packs before it is eligible for promotion.
