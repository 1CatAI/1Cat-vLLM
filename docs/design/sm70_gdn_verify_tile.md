# Single-request SM70 GDN verification tiles

The FP32-state TP4 GDN verifier can split each head's independent value
columns into two-column tiles. Its K128 reduction and recurrent arithmetic
remain the same. The smaller tile shortens register dependency chains for
the eight-token single-request verifier.

Admission requires SM70, the Qwen GDN shape with four query/key heads and
twelve value heads, FP16 projection/output transport, precomputed FP32
gating, FP32 state snapshots, and the existing recurrent numerical contract.
Batched requests use their existing geometry, and legacy explicit tile
overrides keep their existing behavior.

Validation compares both output bits and every state snapshot against the
prior tile. Each replay restores its pre-forward state before changing
projection and gating inputs; acceptance and model measurements use the
same fixed-prefix numerical and fixed-seed prompt gates as other decode
changes.

The serving workload uses CUDA 12.8, Torch 2.10, four V100-SXM2-32GB GPUs at
300 W, TP4, original Qwen3.8-27B NVFP4 with FP8 head, DFlash2 draft7, maximum
length 262144, memory utilization 0.8, FP8 E4M3 target KV storage, and
1024/8192-token inputs.
