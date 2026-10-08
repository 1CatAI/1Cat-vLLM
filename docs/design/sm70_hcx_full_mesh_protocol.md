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

The protocol screen has no model dispatch changes. GPU measurements are
pending. A protocol change requires the isolated numerical gate, complete
chain timing and same-wheel model C1/C4, teacher-forcing and acceptance gates.
