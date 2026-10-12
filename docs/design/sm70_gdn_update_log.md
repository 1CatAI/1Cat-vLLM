# FP32 GDN update-log screen on SM70

The M8 speculative verifier stores eight FP32 recurrent-state matrices per
GDN layer although the next forward consumes only the accepted prefix. This
research benchmark replaces those stores with a round base and eight FP32
rank-one update logs. It preserves the existing q/k reduction, FP32 decay
rounding and fused update, and FP16 output. The control is the installed,
unchanged recurrent kernel. The generated research variant retains the
Flash Linear Attention authors' MIT notice.

The workload is 48 independent GDN layers, TP4, four V100-SXM2-32GB GPUs
with all-pair NV2, CUDA 12.8, Torch 2.10, FP16 activations, FP32 state,
H=4, HV=12, K=V=128, M=8. The selected-state screen preserves the packed
projection's 4120-element physical row stride. Both arms use the same
installed runtime. All six GPU ownership locks are acquired before execution.
These are state-chain measurements, not model round measurements.

## Traffic ceiling

One layer writes 6,291,456 bytes of snapshots. The log variant writes
786,432 bytes of round base plus 98,688 bytes of FP32 update logs. Across
48 layers the nominal write reduction is 259,504,128 bytes, or 0.289 ms at
898.048 GB/s theoretical HBM bandwidth with the 877 MHz memory clock.
This is a payload estimate: reconstruction adds state reads, logs, FP32
instructions, and synchronization. It is not an end-to-end saving.

Two persistence structures were tested:

- Replay the previously accepted prefix inside the next recurrent kernel.
  Two log banks prevent cross-CTA overwrites of the preceding round's logs.
- Read the ordinary selected cache entry during verification, then run one
  grouped restoration after the accepted count is known. Only the chosen
  physical snapshot slot is written. Restoration is included in chain timing.

| 48-layer state chain | Original us | Candidate us | Saving us |
| --- | ---: | ---: | ---: |
| Replay at next forward | 741.382 | 664.018 | 77.363 |
| Post-sampling selected restoration | 734.876 | 685.529 | 49.347 |

Values are four-rank means of same-run ABBA samples, not differences between
machines or separate wheel builds. The maximum rank mean changes from
742.650 to 665.364 us for replay, and from 736.238 to 686.711 us for selected
restoration. Both experiments record SM/memory clocks and timing intervals;
post-timing samples are 1530/877 MHz. Original and candidate advance the same
number of recurrence steps. Forty-eight independent layers exceed L2 in
both arms. The replay graph executes two complete layer chains before reuse;
the selected-state graph executes one chain and its grouped restoration.

## Correctness and decision

Four ranks pass 20 successive numerical rounds, all eight accepted-prefix
states, and bitwise FP16 output comparisons. Selected-state numerical probes
also vary gates over [-10, -0.001] and beta over [0.001, 0.999]. Captured chains
pass eight changed-input replays covering accepted lengths 1 through 8.
Selected restoration leaves hypothetical unaccepted slots unwritten and the
next forward correctly reads the selected slot.

The saved traffic does not translate into the optimistic 0.289 ms saving:
FP32 reconstruction takes work. Both measured gains are below 0.1 ms per
48-layer state chain. They do not justify prioritizing serving integration
over the larger projection bottleneck. No serving dispatch, source-complete
candidate wheel, model A/B, acceptance claim, or accepted after-trace is
introduced by this screen. The <=12 ms model-round objective remains unmet.
A future asynchronous restoration proposal must include its cache lifecycle,
stream dependencies, and same-wheel acceptance/C4 evidence before admission.

The [measurement record](data/sm70_gdn_update_log_54633_20261008.json)
retains all four rank samples, timing intervals, numerical checks, and source
hashes. The benchmark is `benchmarks/benchmark_sm70_gdn_lazy_state.py`.
Performance-counter access on this machine is denied; no actual DRAM-byte or
instruction-floor claim is inferred from theoretical payload traffic.
