# HCX exchange on a full TP4 NVLink mesh

HCX selects direct three-peer exchange when every pair has NVLink access.
On a ring it selects recursive doubling for the input sum and XOR forwarding
for the LoRA and hidden slices. A full mesh supports both protocols, so
topology admission alone does not establish which is faster.

The full-mesh control uses Flash-Next IQ3_S, FP16 MTP4, TP4, V100-SXM2-32GB,
CUDA 12.8, Torch 2.10.0 and FULL target graphs. Target history is a device
E4M3 reference with protected FP16 hot/staging rows; draft history is FP16.
With I8192/O256 the two unobserved C1 samples are 19.817 and 19.801 ms/round.
C4 I128/O600 costs 43.113 ms/round. These are a new host's controls, not an
optimization gain relative to the ring host.

In a separate 32-round GPU-event window the critical target envelope is
15.637 ms and the same rank's four-step draft envelope is 3.407 ms. CPU target
submission spread has median 0.254 ms and p90 0.371 ms. These are current-stream
envelopes with nested spans, not additive kernel service measurements.

For M5 the direct input-sum protocol sends 307,200 bytes/rank/boundary,
including epoch tags; recursive doubling sends 204,800. The latter adds a
communication dependency. This byte difference by itself does not explain
the target latency difference between hosts. Both unprojected native variants
use 92 registers/thread, 13,824 static shared bytes and no spills in the
frozen module. Register spills do not explain a difference between them.

`benchmark_sm70_hcx_protocols.py` compares both protocols using the same
installed module, eight real HC weight pairs and graph workspace. It checks
three changing eager inputs and three changing graph inputs for exact output
equality on every rank. Alternating timing order amortizes initial CPU entry
skew over 512 HC boundaries. Each sample reports the maximum rank envelope;
protocol service times must not be added across ranks.

The protocol screen passes exact eager and graph output checks on all four
ranks. Both arms use the frozen native module with SHA256 prefix `784d1447`.
At M5, eight real weight pairs, 64 chain repeats and 32 timing samples per
protocol, the critical medians are:

| Protocol | Maximum-rank coupled graph envelope per boundary |
| --- | ---: |
| Direct three-peer exchange | 23.650 us |
| Recursive doubling and XOR forwarding | 24.892 us |

Changing to XOR forwarding loses 1.242 us/boundary in this screen. The
existing full-mesh choice is retained. This experiment does not identify the
cause of the model target's cross-host latency increase. There is no protocol
change or model gain to admit; a future change still requires same-wheel model
C1/C4, teacher-forcing and acceptance checks.
